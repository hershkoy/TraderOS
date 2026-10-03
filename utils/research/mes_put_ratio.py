"""dtoptions 56 DTE 2x1 put-ratio claim-check (SPXW proxy for /MES).

Source video: https://youtu.be/_Vd7WKRSiV4 (dtoptions, 2026-10-02).

Stated rules:
  - Every Friday, open a 2x1 put ratio ~56 DTE
  - Sell 2 puts near 10 delta, buy 1 put near 14 delta
  - Claim: 242 trades, 2 losses (99% win rate), no losing trade in 2022

The video does not state a profit target or stop. The same author's published
90 DTE version takes 50% of the credit and exits by 21 DTE, with no stop.
Those are optional exits here, next to hold-to-expiration.

Listed marks are SPXW (weekly PM-settled SPX), not MES. Theta has no MES
option root. Win/loss shape should match a delta-matched S&P ratio; SPAN
buying-power returns on micros are not reproduced.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger(__name__)

CLAIMED: Dict[str, Any] = {
    "trades": 242,
    "losses": 2,
    "win_rate_pct": 99.0,
    "bear_2022_losses": 0,
    "structure": "long 1x ~14-delta put, short 2x ~10-delta put",
    "cadence": "every Friday",
    "dte": 56,
    "source": "dtoptions youtu.be/_Vd7WKRSiV4",
}

DEFAULT_LONG_DELTA = 0.14
DEFAULT_SHORT_DELTA = 0.10
DEFAULT_DTE = 56
DEFAULT_DTE_TOL = 10
DEFAULT_MULT = 100.0  # SPX
DEFAULT_COMM_PER_LEG = 0.65
EXIT_EXPIRY = "expiry"
EXIT_TP = "tp"
EXIT_DTE = "dte"
EXIT_OPEN = "open"


@dataclass
class RatioLeg:
    entry_date: date
    expiration: date
    spot: float
    long_k: float
    short_k: float
    long_delta: float
    short_delta: float


@dataclass
class SimTrade:
    entry_date: date
    expiration: date
    exit_date: Optional[date]
    spot: float
    long_k: float
    short_k: float
    long_delta: float
    short_delta: float
    width: float
    credit_pts: float
    exit_value_pts: float
    pnl_pts: float
    pnl_dollars: float
    exit_reason: str
    hold_days: int
    dte_entry: int
    dte_exit: Optional[int]
    mae_pts: float
    mfe_pts: float
    mark_mode: str
    leg_slip: float
    skipped: bool = False
    skip_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["entry_date"] = self.entry_date.isoformat()
        row["expiration"] = self.expiration.isoformat()
        row["exit_date"] = self.exit_date.isoformat() if self.exit_date else None
        row.pop("extra", None)
        row.update(self.extra)
        return row


def package_value(long_px: float, short_px: float) -> float:
    """Mark of short 2x / long 1x. Positive means the shorts are richer (a debit to close)."""
    return 2.0 * float(short_px) - float(long_px)


def expiry_package_value(spot: float, long_k: float, short_k: float) -> float:
    """Cash settlement of short 2x / long 1x. Positive means we pay."""
    long_i = max(float(long_k) - float(spot), 0.0)
    short_i = max(float(short_k) - float(spot), 0.0)
    return package_value(long_i, short_i)


def _mid(bid: float, ask: float) -> float:
    return 0.5 * (float(bid) + float(ask))


def transacted_prices(
    mark: Dict[str, float],
    *,
    mode: str,
    leg_slip: float,
    closing: bool,
) -> Tuple[float, float]:
    """Return (long_px, short_px) on the side we trade.

    Open: buy the long, sell the shorts. Close: sell the long, buy the shorts.
    ``leg_slip`` is applied to each mid (against us). Natural uses the NBBO side.
    """
    if mode == "natural":
        if closing:
            return float(mark["long_bid"]), float(mark["short_ask"])
        return float(mark["long_ask"]), float(mark["short_bid"])
    slip = float(leg_slip)
    long_mid = _mid(mark["long_bid"], mark["long_ask"])
    short_mid = _mid(mark["short_bid"], mark["short_ask"])
    if closing:
        return long_mid - slip, short_mid + slip
    return long_mid + slip, short_mid - slip


def _usable_mark(mark: Dict[str, Any]) -> bool:
    try:
        lb = float(mark["long_bid"])
        la = float(mark["long_ask"])
        sb = float(mark["short_bid"])
        sa = float(mark["short_ask"])
    except (KeyError, TypeError, ValueError):
        return False
    if min(lb, la, sb, sa) <= 0:
        return False
    if la < lb or sa < sb:
        return False
    return True


def candidate_expiries(
    expirations: Sequence[date],
    day: date,
    target_dte: int = DEFAULT_DTE,
    tol: int = 21,
) -> List[date]:
    """Friday expirations nearest ``target_dte``, closest first.

    The exact 56 DTE Friday is often not listed yet. Callers should walk this
    list and keep the first expiration that has a chain that day.
    """
    ranked: List[Tuple[int, int, date]] = []
    for ed in expirations:
        if ed.weekday() != 4:
            continue
        dte = (ed - day).days
        if dte < 21:
            continue
        err = abs(dte - int(target_dte))
        if err > int(tol):
            continue
        ranked.append((err, dte, ed))
    ranked.sort()
    return [ed for _, _, ed in ranked]


def pick_friday_expiry(
    expirations: Sequence[date],
    day: date,
    target_dte: int = DEFAULT_DTE,
    tol: int = DEFAULT_DTE_TOL,
) -> Optional[date]:
    """Friday expiration closest to ``target_dte`` calendar days."""
    cands = candidate_expiries(expirations, day, target_dte=target_dte, tol=tol)
    return cands[0] if cands else None


def friday_sessions(days: Iterable[date], start: date, end: date) -> List[date]:
    out: List[date] = []
    for d in days:
        if d < start or d > end:
            continue
        if d.weekday() == 4:
            out.append(d)
    return sorted(set(out))


def pick_put_by_delta(
    chain: pd.DataFrame,
    target_abs: float,
    *,
    spot: Optional[float] = None,
    min_strike: Optional[float] = None,
    max_abs_delta: float = 0.45,
    min_abs_delta: float = 0.02,
) -> Optional[pd.Series]:
    """Listed put whose |delta| is closest to ``target_abs``."""
    if chain is None or chain.empty:
        return None
    df = chain.copy()
    for col in ("strike", "bid", "ask", "delta"):
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (
        df["strike"].notna()
        & df["delta"].notna()
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
    )
    if spot is not None and float(spot) > 0:
        mask &= df["strike"] < float(spot)
    if min_strike is not None:
        mask &= df["strike"] > float(min_strike)
    sub = df.loc[mask].copy()
    if sub.empty:
        return None
    sub["ad"] = sub["delta"].abs()
    sub = sub[(sub["ad"] >= float(min_abs_delta)) & (sub["ad"] <= float(max_abs_delta))]
    if sub.empty:
        return None
    sub["err"] = (sub["ad"] - float(target_abs)).abs()
    sub = sub.sort_values(["err", "strike"], ascending=[True, True])
    return sub.iloc[0]


def pick_ratio_strikes(
    chain: pd.DataFrame,
    *,
    long_delta: float = DEFAULT_LONG_DELTA,
    short_delta: float = DEFAULT_SHORT_DELTA,
    max_delta_err: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Long the higher-strike ~14 delta, short 2x the lower-strike ~10 delta."""
    if chain is None or chain.empty:
        return None
    spot = None
    if "underlying_price" in chain.columns:
        px = pd.to_numeric(chain["underlying_price"], errors="coerce").dropna()
        if not px.empty:
            spot = float(px.median())
    short_row = pick_put_by_delta(chain, short_delta, spot=spot)
    if short_row is None:
        return None
    short_k = float(short_row["strike"])
    long_row = pick_put_by_delta(chain, long_delta, spot=spot, min_strike=short_k)
    if long_row is None:
        return None
    long_k = float(long_row["strike"])
    if long_k <= short_k:
        return None
    short_err = abs(abs(float(short_row["delta"])) - float(short_delta))
    long_err = abs(abs(float(long_row["delta"])) - float(long_delta))
    if short_err > float(max_delta_err) or long_err > float(max_delta_err):
        return None
    if spot is None:
        spot = float("nan")
    return {
        "spot": spot,
        "long_k": long_k,
        "short_k": short_k,
        "long_delta": float(long_row["delta"]),
        "short_delta": float(short_row["delta"]),
        "long_bid": float(long_row["bid"]),
        "long_ask": float(long_row["ask"]),
        "short_bid": float(short_row["bid"]),
        "short_ask": float(short_row["ask"]),
        "long_delta_err": long_err,
        "short_delta_err": short_err,
    }


def simulate_trade(
    *,
    entry_date: date,
    expiration: date,
    spot_entry: float,
    long_k: float,
    short_k: float,
    long_delta: float,
    short_delta: float,
    marks: Sequence[Dict[str, Any]],
    settle_spot: Optional[float],
    mode: str = "mid",
    leg_slip: float = 0.0,
    tp_frac: Optional[float] = None,
    exit_dte: Optional[int] = None,
    comm_per_leg: float = DEFAULT_COMM_PER_LEG,
    multiplier: float = DEFAULT_MULT,
) -> SimTrade:
    """One Friday ratio. ``marks`` are session rows with bid/ask for both legs."""
    width = float(long_k) - float(short_k)
    dte_entry = (expiration - entry_date).days
    usable = [m for m in marks if _usable_mark(m)]
    usable.sort(key=lambda m: m["day"])
    entry_mark = next((m for m in usable if m["day"] == entry_date), None)
    base = dict(
        entry_date=entry_date,
        expiration=expiration,
        exit_date=None,
        spot=float(spot_entry),
        long_k=float(long_k),
        short_k=float(short_k),
        long_delta=float(long_delta),
        short_delta=float(short_delta),
        width=width,
        credit_pts=0.0,
        exit_value_pts=0.0,
        pnl_pts=0.0,
        pnl_dollars=0.0,
        exit_reason=EXIT_OPEN,
        hold_days=0,
        dte_entry=dte_entry,
        dte_exit=None,
        mae_pts=0.0,
        mfe_pts=0.0,
        mark_mode=mode,
        leg_slip=float(leg_slip),
    )
    if entry_mark is None or not np.isfinite(spot_entry):
        return SimTrade(skipped=True, skip_reason="no_entry_quote", **base)

    long_open, short_open = transacted_prices(
        entry_mark, mode=mode, leg_slip=leg_slip, closing=False
    )
    credit = package_value(long_open, short_open)
    # MAE/MFE use the same close-side marks as the exit, after the entry bar
    # (the entry bar's close side is the round-trip spread, not a market move).
    path_pnl: List[float] = []
    exit_day: Optional[date] = None
    exit_value: Optional[float] = None
    reason: Optional[str] = None
    for mark in usable:
        day = mark["day"]
        if day <= entry_date or day > expiration:
            continue
        long_c, short_c = transacted_prices(
            mark, mode=mode, leg_slip=leg_slip, closing=True
        )
        close_val = package_value(long_c, short_c)
        path_pnl.append(credit - close_val)
        dte_left = (expiration - day).days
        if tp_frac is not None and credit > 0 and close_val <= float(tp_frac) * credit:
            exit_day, exit_value, reason = day, close_val, EXIT_TP
            break
        if exit_dte is not None and dte_left <= int(exit_dte):
            exit_day, exit_value, reason = day, close_val, EXIT_DTE
            break

    if reason is None:
        if settle_spot is None or not np.isfinite(settle_spot):
            return SimTrade(
                skipped=True,
                skip_reason="no_settlement",
                credit_pts=credit,
                **{k: v for k, v in base.items() if k != "credit_pts"},
            )
        exit_value = expiry_package_value(float(settle_spot), long_k, short_k)
        exit_day = expiration
        reason = EXIT_EXPIRY
        path_pnl.append(credit - exit_value)

    pnl_pts = float(credit - exit_value)
    # 3 contracts to open and 3 to close / settle.
    fees = float(comm_per_leg) * 6.0
    pnl_dollars = pnl_pts * float(multiplier) - fees
    mae = float(min(path_pnl)) if path_pnl else pnl_pts
    mfe = float(max(path_pnl)) if path_pnl else pnl_pts
    return SimTrade(
        entry_date=entry_date,
        expiration=expiration,
        exit_date=exit_day,
        spot=float(spot_entry),
        long_k=float(long_k),
        short_k=float(short_k),
        long_delta=float(long_delta),
        short_delta=float(short_delta),
        width=width,
        credit_pts=float(credit),
        exit_value_pts=float(exit_value),
        pnl_pts=pnl_pts,
        pnl_dollars=pnl_dollars,
        exit_reason=reason,
        hold_days=(exit_day - entry_date).days if exit_day else 0,
        dte_entry=dte_entry,
        dte_exit=(expiration - exit_day).days if exit_day else None,
        mae_pts=mae,
        mfe_pts=mfe,
        mark_mode=mode,
        leg_slip=float(leg_slip),
        extra={"fees": fees},
    )


def _pf(pnls: np.ndarray) -> Optional[float]:
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss <= 1e-9:
        return None if gross_win <= 0 else float("inf")
    return gross_win / gross_loss


def summarize_trades(trades: Sequence[SimTrade]) -> Dict[str, Any]:
    kept = [t for t in trades if not t.skipped]
    if not kept:
        return {"n": 0, "skipped": sum(1 for t in trades if t.skipped)}
    pnls = np.array([t.pnl_dollars for t in kept], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    scratches = pnls[pnls == 0]
    reasons: Dict[str, int] = {}
    for t in kept:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    by_year: Dict[str, Dict[str, Any]] = {}
    for year in sorted({t.entry_date.year for t in kept}):
        sub = [t for t in kept if t.entry_date.year == year]
        arr = np.array([t.pnl_dollars for t in sub], dtype=float)
        by_year[str(year)] = {
            "n": len(sub),
            "losses": int((arr < 0).sum()),
            "win_rate_pct": round(100.0 * float((arr > 0).mean()), 1),
            "mean_pnl": round(float(arr.mean()), 2),
            "total_pnl": round(float(arr.sum()), 2),
            "pf": _round_pf(_pf(arr)),
            "worst": round(float(arr.min()), 2),
        }
    overlap_2022 = [
        t
        for t in kept
        if t.pnl_dollars < 0
        and t.entry_date <= date(2022, 12, 31)
        and (t.exit_date or t.expiration) >= date(2022, 1, 1)
    ]
    entry_2022 = [t for t in kept if t.entry_date.year == 2022 and t.pnl_dollars < 0]
    pf = _pf(pnls)
    mean_win = float(wins.mean()) if len(wins) else 0.0
    mean_loss = float(losses.mean()) if len(losses) else 0.0
    worst = min(kept, key=lambda t: t.pnl_dollars)
    return {
        "n": len(kept),
        "skipped": sum(1 for t in trades if t.skipped),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "scratches": int(len(scratches)),
        "win_rate_pct": round(100.0 * float((pnls > 0).mean()), 1),
        "mean_pnl": round(float(pnls.mean()), 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "pf": _round_pf(pf),
        "mean_win": round(mean_win, 2),
        "mean_loss": round(mean_loss, 2),
        "payoff": round(abs(mean_win / mean_loss), 2) if mean_loss < 0 else None,
        "worst": round(float(pnls.min()), 2),
        "best": round(float(pnls.max()), 2),
        "worst_trade": {
            "entry": worst.entry_date.isoformat(),
            "exit": worst.exit_date.isoformat() if worst.exit_date else None,
            "pnl": round(worst.pnl_dollars, 2),
            "reason": worst.exit_reason,
            "spot": worst.spot,
            "long_k": worst.long_k,
            "short_k": worst.short_k,
        },
        "avg_credit_pts": round(float(np.mean([t.credit_pts for t in kept])), 2),
        "avg_width": round(float(np.mean([t.width for t in kept])), 1),
        "avg_mae_pts": round(float(np.mean([t.mae_pts for t in kept])), 2),
        "credit_only_n": sum(1 for t in kept if t.credit_pts > 0),
        "losses_entry_2022": len(entry_2022),
        "losses_overlapping_2022": len(overlap_2022),
        "exit_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "by_entry_year": by_year,
    }


def _round_pf(pf: Optional[float]) -> Optional[float]:
    if pf is None:
        return None
    if pf == float("inf"):
        return pf
    return round(float(pf), 2)


def overlapping_book_equity(
    trades: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """End-of-day equity of overlapping packages.

    Each trade dict has ``entry``, ``exit``, ``realized`` (booked on the exit
    date), and ``marks`` as ``(day, mtm_dollars)`` for sessions still open.
    A trade's last mark is carried forward until the day before exit.
    """
    if not trades:
        return {"max_dd": 0.0, "worst_equity": 0.0, "worst_open": 0.0, "worst_day": None, "open_on_worst": 0}

    days = set()
    parsed = []
    for trade in trades:
        marks = {day: float(pnl) for day, pnl in trade.get("marks") or []}
        parsed.append(
            {
                "entry": trade["entry"],
                "exit": trade["exit"],
                "realized": float(trade["realized"]),
                "marks": marks,
            }
        )
        days.add(trade["exit"])
        days.update(marks)
    ordered_days = sorted(days)
    realized_cum = 0.0
    peak = 0.0
    max_dd = 0.0
    worst_equity = 0.0
    worst_open = 0.0
    worst_day: Optional[date] = None
    open_on_worst = 0
    for day in ordered_days:
        realized_today = 0.0
        open_sum = 0.0
        open_n = 0
        for trade in parsed:
            if trade["exit"] == day:
                realized_today += trade["realized"]
                continue
            if not (trade["entry"] < day < trade["exit"]):
                continue
            prior = [d for d in trade["marks"] if d <= day]
            if not prior:
                continue
            open_sum += trade["marks"][max(prior)]
            open_n += 1
        realized_cum += realized_today
        equity = realized_cum + open_sum
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            worst_equity = equity
            worst_open = open_sum
            worst_day = day
            open_on_worst = open_n
    return {
        "max_dd": round(max_dd, 2),
        "worst_equity": round(worst_equity, 2),
        "worst_open": round(worst_open, 2),
        "worst_day": None if worst_day is None else worst_day.isoformat(),
        "open_on_worst": open_on_worst,
    }


def realized_equity(points: Sequence[Tuple[date, float]]) -> List[Dict[str, Any]]:
    """Cumulative realized dollars, ordered by exit date. Same-day exits sum in input order."""
    ordered = sorted(points, key=lambda item: item[0])
    total = 0.0
    out: List[Dict[str, Any]] = []
    for day, pnl in ordered:
        total += float(pnl)
        out.append(
            {
                "date": day.isoformat(),
                "pnl": round(float(pnl), 2),
                "equity": round(total, 2),
            }
        )
    return out


def trades_to_frame(trades: Sequence[SimTrade]) -> pd.DataFrame:
    return pd.DataFrame([t.as_row() for t in trades])


def _retry_get(session, url: str, params: Dict[str, Any], timeout: float = 120.0):
    last = None
    for attempt in range(4):
        r = session.get(url, params=params, timeout=timeout)
        last = r
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        return r
    return last


def fetch_put_chain(
    client,
    *,
    symbol: str,
    expiration: date,
    day: date,
    cache_dir: Path,
) -> pd.DataFrame:
    """One-day SPXW put greeks chain (listed delta, bid, ask)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"chain_{symbol}_{expiration.isoformat()}_{day.isoformat()}.csv"
    if path.exists():
        if path.stat().st_size < 15:
            return pd.DataFrame()
        try:
            cached = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            path.unlink(missing_ok=True)
            cached = None
        if cached is not None:
            return cached
    session = requests.Session()
    r = _retry_get(
        session,
        f"{client.base_url}/v3/option/history/greeks/eod",
        {
            "symbol": str(symbol).upper(),
            "expiration": expiration.isoformat(),
            "right": "put",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
        },
    )
    if r is None or r.status_code == 472 or not r.content:
        pd.DataFrame(columns=["strike", "bid", "ask", "delta"]).to_csv(path, index=False)
        return pd.DataFrame()
    if r.status_code != 200:
        LOG.warning("chain %s %s -> %s", expiration, day, r.status_code)
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    df.to_csv(path, index=False)
    return df


def fetch_leg_eod(
    client,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    start: date,
    end: date,
    cache_dir: Path,
) -> pd.DataFrame:
    """EOD bid/ask for one put from entry through expiration."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    strike_s = f"{float(strike):.3f}"
    path = cache_dir / f"eod_{symbol}_{expiration.isoformat()}_{strike_s}.csv"
    if path.exists():
        try:
            cached = pd.read_csv(path)
            if not cached.empty and "session_date" in cached.columns:
                days = cached["session_date"].astype(str).str[:10]
                covers = days.min() <= start.isoformat() and days.max() >= (
                    end - timedelta(days=7)
                ).isoformat()
                if covers:
                    return cached
        except Exception:  # noqa: BLE001
            path.unlink(missing_ok=True)
    session = requests.Session()
    r = _retry_get(
        session,
        f"{client.base_url}/v3/option/history/eod",
        {
            "symbol": str(symbol).upper(),
            "expiration": expiration.isoformat(),
            "strike": strike_s,
            "right": "put",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
    )
    if r is None or r.status_code == 472 or not getattr(r, "content", None):
        if r is not None and r.status_code == 472:
            pd.DataFrame().to_csv(path, index=False)
        return pd.DataFrame()
    if r.status_code != 200:
        LOG.warning("eod %s %s -> %s", expiration, strike_s, r.status_code)
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        df.to_csv(path, index=False)
        return df
    if "created" in df.columns:
        ts = pd.to_datetime(df["created"], errors="coerce")
    elif "last_trade" in df.columns:
        ts = pd.to_datetime(df["last_trade"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    out = pd.DataFrame(
        {
            "session_date": ts.dt.date.astype(str),
            "bid": pd.to_numeric(df.get("bid"), errors="coerce"),
            "ask": pd.to_numeric(df.get("ask"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["session_date"])
    out = out.drop_duplicates("session_date", keep="last")
    out.to_csv(path, index=False)
    return out


def marks_from_legs(
    entry: date,
    expiration: date,
    long_eod: pd.DataFrame,
    short_eod: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if long_eod is None or short_eod is None or long_eod.empty or short_eod.empty:
        return []
    long_df = long_eod.copy()
    short_df = short_eod.copy()
    long_df["session_date"] = long_df["session_date"].astype(str).str[:10]
    short_df["session_date"] = short_df["session_date"].astype(str).str[:10]
    merged = long_df.merge(short_df, on="session_date", suffixes=("_long", "_short"))
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        try:
            day = date.fromisoformat(str(row["session_date"])[:10])
        except ValueError:
            continue
        if day < entry or day > expiration:
            continue
        rows.append(
            {
                "day": day,
                "long_bid": float(row["bid_long"]),
                "long_ask": float(row["ask_long"]),
                "short_bid": float(row["bid_short"]),
                "short_ask": float(row["ask_short"]),
            }
        )
    rows.sort(key=lambda r: r["day"])
    return rows


def settlement_spot(spot_by_day: Dict[date, float], expiration: date) -> Optional[float]:
    """SPX close on expiration, or the last session on or before it once that date has passed."""
    if not spot_by_day or expiration > max(spot_by_day):
        return None
    if expiration in spot_by_day:
        return float(spot_by_day[expiration])
    prior = [d for d in spot_by_day if d <= expiration]
    if not prior:
        return None
    return float(spot_by_day[max(prior)])


def load_spx_eod(client, start: date, end: date) -> pd.DataFrame:
    """SPX daily OHLC from Theta, with Yahoo ^GSPC if the index endpoint is empty."""
    r = _retry_get(
        requests.Session(),
        f"{client.base_url}/v3/index/history/eod",
        {
            "symbol": "SPX",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
    )
    if r is not None and r.status_code == 200 and r.content:
        df = pd.read_csv(StringIO(r.text))
        if not df.empty and "close" in df.columns:
            if "created" in df.columns:
                ts = pd.to_datetime(df["created"], errors="coerce")
            elif "last_trade" in df.columns:
                ts = pd.to_datetime(df["last_trade"], errors="coerce")
            else:
                ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            out = pd.DataFrame(
                {"close": pd.to_numeric(df["close"], errors="coerce")},
                index=pd.DatetimeIndex(ts.dt.normalize()),
            )
            out = out.dropna(subset=["close"]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            if not out.empty:
                return out
    LOG.warning("Theta SPX EOD unavailable; falling back to Yahoo ^GSPC")
    import yfinance as yf

    raw = yf.download(
        "^GSPC",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = pd.DataFrame({"close": pd.to_numeric(raw["Close"], errors="coerce")})
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    return out.dropna(subset=["close"]).sort_index()


def index_closes(frame: pd.DataFrame) -> Dict[date, float]:
    out: Dict[date, float] = {}
    if frame is None or frame.empty:
        return out
    for ts, row in frame.iterrows():
        d = pd.Timestamp(ts).date()
        px = row["close"] if "close" in frame.columns else row.iloc[0]
        if pd.notna(px):
            out[d] = float(px)
    return out


def build_schedule(
    sessions: Sequence[date],
    expirations: Sequence[date],
    *,
    start: date,
    end: date,
    target_dte: int = DEFAULT_DTE,
    tol: int = DEFAULT_DTE_TOL,
) -> List[Tuple[date, date]]:
    rows: List[Tuple[date, date]] = []
    for day in friday_sessions(sessions, start, end):
        exp = pick_friday_expiry(expirations, day, target_dte=target_dte, tol=tol)
        if exp is None:
            continue
        rows.append((day, exp))
    return rows
