"""QQQ LEAPS gap claim-check with ThetaData listed NBBO / EOD marks.

Same signals as ``qqq_leaps_gap`` (gap-down + optional SMA100). Contract pick:
Jan/Mar/Jun/Sep LEAPS-cycle expiry near 365 DTE, strike nearest BS 60-delta
among listed strikes that have a nonzero RTH NBBO on the entry day.

Marks:
  - Entry: first nonzero 1m NBBO after 09:30 ET (mid or ask).
  - Path / exit: Theta ``history/eod`` bid/ask (max 365d range) walked daily;
    TP when exit mark >= entry * (1 + tp_frac); else last EOD or intrinsic.

Research books: mid/mid (claim-like), ask/bid (marketable).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.options.greeks import calculate_greeks
from utils.research.qqq_leaps_gap import (
    CLAIMED,
    DEFAULT_MAX_OPEN,
    DEFAULT_Q,
    DEFAULT_R,
    DEFAULT_TARGET_DELTA,
    DEFAULT_TP_FRAC,
    EXIT_EXPIRY,
    EXIT_PANEL_END,
    EXIT_TP,
    REASON_OK,
    REASON_SKIP_OCC,
    LeapTrade,
    _as_date,
    book_stats,
    iv_from_atr,
    signal_mask,
    strike_for_call_delta,
    trades_to_frame,
    year_bucket_stats,
)

LOG = logging.getLogger(__name__)

REASON_NO_CONTRACT = "no_contract"
REASON_NO_ENTRY_NBBO = "no_entry_nbbo"
REASON_NO_EOD = "no_eod"
LEAPS_MONTHS = {1, 3, 6, 9}


@dataclass
class ThetaContract:
    expiration: date
    strike: float
    dte: int
    entry_bid: float
    entry_ask: float
    entry_mid: float
    entry_ts: Any
    delta_bs: float
    iv_atr: float


@dataclass
class ThetaBookResult:
    mode: str
    mark_mode: str  # mid_mid | ask_bid
    trades: List[LeapTrade] = field(default_factory=list)
    signals_raw: int = 0
    filled: int = 0
    skipped_occ: int = 0
    skipped_other: int = 0


def list_qqq_expirations(client: ThetaDataClient) -> List[date]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": "QQQ"},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    out = []
    for x in df[col].tolist():
        try:
            out.append(date.fromisoformat(str(x)[:10]))
        except ValueError:
            continue
    return out


def list_strikes(client: ThetaDataClient, exp: date) -> List[float]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": "QQQ", "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    return sorted({float(x) for x in df[col].tolist() if pd.notna(x)})


def leaps_cycle_candidates(exps: Sequence[date], day: date) -> List[date]:
    rows = []
    for ed in exps:
        dte = (ed - day).days
        if not (300 <= dte <= 450):
            continue
        if ed.month not in LEAPS_MONTHS:
            continue
        mid = 0 if 14 <= ed.day <= 22 else 1
        rows.append((ed, mid, abs(dte - 365)))
    rows.sort(key=lambda x: (x[1], x[2]))
    return [r[0] for r in rows]


def fetch_eod(
    client: ThetaDataClient,
    *,
    expiration: date,
    strike: float,
    start: date,
    end: date,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
) -> pd.DataFrame:
    """Daily EOD bid/ask for one contract (Theta max range 365d)."""
    if end < start:
        return pd.DataFrame()
    # Clamp to 365 calendar days inclusive-ish
    if (end - start).days > 364:
        end = start + timedelta(days=364)
    key = (expiration.isoformat(), float(strike), start.isoformat(), end.isoformat())
    if key in cache:
        return cache[key].copy()

    r = client.session.get(
        f"{client.base_url}/v3/option/history/eod",
        params={
            "symbol": "QQQ",
            "expiration": expiration.isoformat(),
            "strike": f"{float(strike):.3f}",
            "right": "call",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=120,
    )
    if r.status_code == 472 or (r.content and b"No data" in r.content[:80]):
        cache[key] = pd.DataFrame()
        return pd.DataFrame()
    if r.status_code != 200 or not r.content:
        LOG.debug("EOD HTTP %s for %s %s", r.status_code, expiration, strike)
        cache[key] = pd.DataFrame()
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        cache[key] = df
        return df.copy()
    # Prefer created/last_trade date column
    if "created" in df.columns:
        ts = pd.to_datetime(df["created"], errors="coerce")
    elif "last_trade" in df.columns:
        ts = pd.to_datetime(df["last_trade"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 4], errors="coerce")
    df = df.copy()
    df["session_date"] = ts.dt.date
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.dropna(subset=["session_date", "bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df.drop_duplicates("session_date", keep="last").sort_values("session_date")
    cache[key] = df
    return df.copy()


def select_contract(
    client: ThetaDataClient,
    *,
    day: date,
    spot_open: float,
    atr: float,
    exps: Sequence[date],
    target_delta: float = DEFAULT_TARGET_DELTA,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
) -> Optional[ThetaContract]:
    iv = iv_from_atr(atr, spot_open)
    if iv is None:
        iv = 0.22
    for exp in leaps_cycle_candidates(exps, day)[:10]:
        dte = (exp - day).days
        T = max(dte, 1) / 365.25
        k_bs = strike_for_call_delta(spot_open, T, r, iv, q, target_delta=target_delta)
        strikes = list_strikes(client, exp)
        if not strikes:
            continue
        # Try nearest few strikes in case the exact BS snap has no quotes
        ranked = sorted(strikes, key=lambda x: abs(x - k_bs))[:5]
        for strike in ranked:
            nbbo = client.nbbo_at_or_after(
                symbol="QQQ",
                expiration=exp,
                strike=float(strike),
                right="call",
                day=day,
                clock=time(9, 30),
                end_time="12:00:00",
            )
            if not nbbo:
                continue
            g = calculate_greeks(spot_open, float(strike), T, r, iv, "C", q=q)
            return ThetaContract(
                expiration=exp,
                strike=float(strike),
                dte=dte,
                entry_bid=float(nbbo["bid"]),
                entry_ask=float(nbbo["ask"]),
                entry_mid=float(nbbo["mid"]),
                entry_ts=nbbo.get("timestamp"),
                delta_bs=float(g["delta"]),
                iv_atr=float(iv),
            )
    return None


def _exit_mark(row: pd.Series, mark_mode: str) -> float:
    if mark_mode == "ask_bid":
        return float(row["bid"])
    return float(row["mid"])


def simulate_theta_trade(
    client: ThetaDataClient,
    *,
    day: date,
    spot_open: float,
    atr: float,
    contracts: int,
    mode: str,
    mark_mode: str,
    exps: Sequence[date],
    panel: pd.DataFrame,
    eod_cache: Dict[Tuple[Any, ...], pd.DataFrame],
    tp_frac: float = DEFAULT_TP_FRAC,
    target_delta: float = DEFAULT_TARGET_DELTA,
) -> LeapTrade:
    """One LEAPS trade marked with Theta."""
    contract = select_contract(
        client,
        day=day,
        spot_open=spot_open,
        atr=atr,
        exps=exps,
        target_delta=target_delta,
    )
    if contract is None:
        return LeapTrade(
            mode=mode,
            entry_date=day,
            exit_date=None,
            contracts=contracts,
            entry_px=spot_open,
            strike=0.0,
            entry_premium=0.0,
            exit_premium=None,
            entry_delta=0.0,
            entry_iv=0.0,
            dte=0,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_CONTRACT,
            skipped=True,
        )

    if mark_mode == "ask_bid":
        entry_prem = contract.entry_ask
    else:
        entry_prem = contract.entry_mid
    if not math.isfinite(entry_prem) or entry_prem <= 0:
        return LeapTrade(
            mode=mode,
            entry_date=day,
            exit_date=None,
            contracts=contracts,
            entry_px=spot_open,
            strike=contract.strike,
            entry_premium=0.0,
            exit_premium=None,
            entry_delta=contract.delta_bs,
            entry_iv=contract.iv_atr,
            dte=contract.dte,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_ENTRY_NBBO,
            skipped=True,
        )

    tp_level = entry_prem * (1.0 + float(tp_frac))
    end = min(contract.expiration, day + timedelta(days=364))
    eod = fetch_eod(
        client,
        expiration=contract.expiration,
        strike=contract.strike,
        start=day,
        end=end,
        cache=eod_cache,
    )
    if eod.empty:
        return LeapTrade(
            mode=mode,
            entry_date=day,
            exit_date=None,
            contracts=contracts,
            entry_px=spot_open,
            strike=contract.strike,
            entry_premium=entry_prem,
            exit_premium=None,
            entry_delta=contract.delta_bs,
            entry_iv=contract.iv_atr,
            dte=contract.dte,
            hold_days=None,
            exit_reason=None,
            pnl_cash=None,
            gain_pct=None,
            reason=REASON_NO_EOD,
            skipped=True,
        )

    # Walk sessions after entry day
    after = eod.loc[eod["session_date"] > day]
    exit_prem: Optional[float] = None
    exit_day: Optional[date] = None
    exit_reason: Optional[str] = None

    for _, bar in after.iterrows():
        d = bar["session_date"]
        mark = _exit_mark(bar, mark_mode)
        if mark >= tp_level - 1e-9:
            exit_prem = float(tp_level)  # assume fill at target
            exit_day = d
            exit_reason = EXIT_TP
            break

    if exit_prem is None:
        # Expiry / panel end: last EOD mark, or intrinsic if on/after expiration
        last = after.iloc[-1] if not after.empty else eod.iloc[-1]
        exit_day = last["session_date"]
        if exit_day >= contract.expiration:
            # intrinsic from underlying close if available
            hits = panel.loc[panel["session_date"] == exit_day]
            if not hits.empty:
                cl = float(hits.iloc[0]["close"])
                exit_prem = max(0.0, cl - contract.strike)
            else:
                exit_prem = max(0.0, _exit_mark(last, mark_mode))
            exit_reason = EXIT_EXPIRY
        else:
            exit_prem = _exit_mark(last, mark_mode)
            exit_reason = EXIT_PANEL_END

    assert exit_day is not None and exit_prem is not None
    pnl_cash = (exit_prem - entry_prem) * 100.0 * int(contracts)
    gain_pct = 100.0 * (exit_prem - entry_prem) / entry_prem
    return LeapTrade(
        mode=mode,
        entry_date=day,
        exit_date=exit_day,
        contracts=int(contracts),
        entry_px=spot_open,
        strike=contract.strike,
        entry_premium=float(entry_prem),
        exit_premium=float(exit_prem),
        entry_delta=contract.delta_bs,
        entry_iv=contract.iv_atr,
        dte=contract.dte,
        hold_days=(exit_day - day).days,
        exit_reason=exit_reason,
        pnl_cash=float(pnl_cash),
        gain_pct=float(gain_pct),
        reason=REASON_OK,
        skipped=False,
    )


def run_theta_book(
    panel: pd.DataFrame,
    *,
    mode: str,
    gap_pct: float,
    contracts: int,
    start: date,
    end: date,
    mark_mode: str = "mid_mid",
    sma_rule: str = "prior",
    max_open: Optional[int] = DEFAULT_MAX_OPEN,
    client: Optional[ThetaDataClient] = None,
    exps: Optional[Sequence[date]] = None,
    eod_cache: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
    tp_frac: float = DEFAULT_TP_FRAC,
    target_delta: float = DEFAULT_TARGET_DELTA,
) -> ThetaBookResult:
    client = client or get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal not reachable on :25503")
    if exps is None:
        exps = list_qqq_expirations(client)
    eod_cache = eod_cache if eod_cache is not None else {}

    mask = signal_mask(panel, mode=mode, gap_pct=gap_pct, sma_rule=sma_rule)
    dates = [_as_date(x) for x in panel["session_date"].tolist()]
    idxs = [
        i
        for i, ok in enumerate(mask.tolist())
        if ok and dates[i] is not None and start <= dates[i] <= end
    ]
    book = ThetaBookResult(mode=mode, mark_mode=mark_mode, signals_raw=len(idxs))
    open_lots: List[Tuple[date, LeapTrade]] = []

    for n, i in enumerate(idxs):
        day = dates[i]
        assert day is not None
        open_lots = [(ed, t) for ed, t in open_lots if ed is None or ed > day]
        if max_open is not None and len(open_lots) >= int(max_open):
            book.skipped_occ += 1
            book.trades.append(
                LeapTrade(
                    mode=mode,
                    entry_date=day,
                    exit_date=None,
                    contracts=contracts,
                    entry_px=float(panel.iloc[i]["open"]),
                    strike=0.0,
                    entry_premium=0.0,
                    exit_premium=None,
                    entry_delta=0.0,
                    entry_iv=0.0,
                    dte=0,
                    hold_days=None,
                    exit_reason=None,
                    pnl_cash=None,
                    gain_pct=None,
                    reason=REASON_SKIP_OCC,
                    skipped=True,
                )
            )
            continue

        row = panel.iloc[i]
        atr = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
        LOG.info(
            "[%s %s %d/%d] %s gap=%.2f",
            mode,
            mark_mode,
            n + 1,
            len(idxs),
            day,
            float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else float("nan"),
        )
        tr = simulate_theta_trade(
            client,
            day=day,
            spot_open=float(row["open"]),
            atr=atr,
            contracts=contracts,
            mode=mode,
            mark_mode=mark_mode,
            exps=exps,
            panel=panel,
            eod_cache=eod_cache,
            tp_frac=tp_frac,
            target_delta=target_delta,
        )
        book.trades.append(tr)
        if tr.skipped:
            book.skipped_other += 1
            LOG.info("  skip %s", tr.reason)
            continue
        book.filled += 1
        open_lots.append((tr.exit_date, tr))
        LOG.info(
            "  fill K=%.1f dte=%d entry=%.2f exit=%.2f pnl=$%.0f %s hold=%sd",
            tr.strike,
            tr.dte,
            tr.entry_premium,
            tr.exit_premium or 0.0,
            tr.pnl_cash or 0.0,
            tr.exit_reason,
            tr.hold_days,
        )

    return book


__all__ = [
    "CLAIMED",
    "ThetaBookResult",
    "ThetaContract",
    "book_stats",
    "fetch_eod",
    "leaps_cycle_candidates",
    "list_qqq_expirations",
    "list_strikes",
    "run_theta_book",
    "select_contract",
    "simulate_theta_trade",
    "trades_to_frame",
    "year_bucket_stats",
]
