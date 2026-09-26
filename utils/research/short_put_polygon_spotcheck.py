"""Polygon free-tier spot-check helpers for short-put BS validation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from utils.research.last_15m_short_put_bs import _busday_on_or_before


def polygon_option_ticker(
    underlying: str,
    expiry: str,
    strike: float,
    right: str = "P",
) -> str:
    """OCC-style Polygon ticker, e.g. O:AVGO250411P00210000."""
    ymd = str(expiry).replace("-", "")[2:]
    strike_i = int(round(float(strike) * 1000))
    r = str(right).upper()[0]
    return "O:%s%s%s%08d" % (str(underlying).upper(), ymd, r, strike_i)


def _as_date(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    ts = pd.Timestamp(val)
    if pd.isna(ts):
        return None
    return ts.date()


def bars_from_aggs_response(resp: Dict[str, Any]) -> pd.DataFrame:
    rows = resp.get("results") or []
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out = []
    for b in rows:
        ts = b.get("t")
        if ts is None:
            continue
        d = datetime.utcfromtimestamp(int(ts) / 1000.0).date()
        out.append(
            {
                "date": d,
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
            }
        )
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def pick_entry_exit_bars(
    bars: pd.DataFrame,
    buy_date: date,
    exit_date: date,
    *,
    entry_slack_days: int = 3,
    exit_slack_days: int = 2,
) -> Tuple[Optional[dict], Optional[dict], str]:
    """Pick entry bar on/after buy, exit bar on/after exit (within slack)."""
    if bars is None or bars.empty:
        return None, None, "no_bars"
    entry_end = buy_date + timedelta(days=int(entry_slack_days) + 2)
    exit_end = exit_date + timedelta(days=int(exit_slack_days) + 2)
    entry_cands = bars[(bars["date"] >= buy_date) & (bars["date"] <= entry_end)]
    if entry_cands.empty:
        return None, None, "no_entry_bar"
    entry = entry_cands.iloc[0].to_dict()
    exit_cands = bars[
        (bars["date"] >= max(exit_date, entry["date"])) & (bars["date"] <= exit_end)
    ]
    if exit_cands.empty:
        return entry, None, "no_exit_bar"
    exit_row = exit_cands.iloc[0].to_dict()
    if exit_row["date"] < entry["date"]:
        return entry, None, "exit_before_entry"
    # Thin tape: first print is on/after the stop day — use that bar's open as
    # sell credit and close as buyback (same-session stop).
    if exit_row["date"] == entry["date"] and entry.get("open") is not None:
        entry = dict(entry)
        entry["close"] = entry.get("open")
    return entry, exit_row, "ok"


def closest_listed_expiry(
    expiries: Sequence[str],
    target: date,
) -> Optional[str]:
    """Pick listed YYYY-MM-DD expiry closest to target date."""
    best = None
    best_abs = None
    for e in expiries or []:
        try:
            d = date.fromisoformat(str(e)[:10])
        except ValueError:
            continue
        dist = abs((d - target).days)
        if best_abs is None or dist < best_abs:
            best_abs = dist
            best = d.isoformat()
    return best


@dataclass
class SpotRow:
    stock: str
    buy_date: date
    exit_date: date
    strike: float
    expiry: str
    ticker: str
    bs_credit: float
    bs_exit: float
    bs_gain_cs: float
    live_entry: Optional[float] = None
    live_exit: Optional[float] = None
    live_entry_date: Optional[date] = None
    live_exit_date: Optional[date] = None
    live_gain_cs: Optional[float] = None
    exit_vs_bs: Optional[float] = None
    status: str = "pending"


def select_hard_stop_sample(
    trades: pd.DataFrame,
    *,
    n: int = 12,
    min_buy: str = "2024-10-01",
    min_adv: float = 1_000_000.0,
) -> pd.DataFrame:
    """Recent hard-stop rows, prefer higher ADV, diverse symbols."""
    df = trades.copy()
    df = df[df["put_exit_reason"].astype(str) == "hard_stop"].copy()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    df = df[df["buy_date"] >= pd.Timestamp(min_buy)]
    if "adv_20" in df.columns:
        df = df[pd.to_numeric(df["adv_20"], errors="coerce").fillna(0) >= float(min_adv)]
    need = ["stock", "buy_date", "put_exit_date", "put_strike", "put_credit", "put_exit_premium"]
    df = df.dropna(subset=need)
    if df.empty:
        return df
    asc = [False, False]
    cols = ["adv_20", "buy_date"] if "adv_20" in df.columns else ["buy_date"]
    if "adv_20" not in df.columns:
        asc = [False]
    df = df.sort_values(cols, ascending=asc, na_position="last")
    picked = df.drop_duplicates("stock", keep="first")
    if len(picked) < int(n):
        rest = df[~df.index.isin(picked.index)]
        need_n = int(n) - len(picked)
        picked = pd.concat([picked, rest.head(need_n)], axis=0)
    return picked.head(int(n)).reset_index(drop=True)


def build_spot_plan(row: pd.Series, *, dte_fallback: int = 45) -> SpotRow:
    buy = _as_date(row.get("buy_date"))
    exit_d = _as_date(row.get("put_exit_date"))
    strike = float(row["put_strike"])
    dte = int(row["put_dte"]) if pd.notna(row.get("put_dte")) else int(dte_fallback)
    if buy is None or exit_d is None:
        raise ValueError("missing buy/exit date")
    expiry = _busday_on_or_before(buy + timedelta(days=dte)).isoformat()
    ticker = polygon_option_ticker(str(row["stock"]), expiry, strike, "P")
    gain = float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else float("nan")
    return SpotRow(
        stock=str(row["stock"]).upper(),
        buy_date=buy,
        exit_date=exit_d,
        strike=strike,
        expiry=expiry,
        ticker=ticker,
        bs_credit=float(row["put_credit"]),
        bs_exit=float(row["put_exit_premium"]),
        bs_gain_cs=gain,
    )
