"""Low-VIX hedge overlays on the QQQ LEAPS gap Theta book.

Structures (plan locks):
  H1 — QQQ put debit vertical (~45 DTE, long ~5% OTM / short ~10% OTM);
       arm prior VIX < 15; size = 1 vertical per open LEAPS lot.
  H2 — VIX call debit vertical (~45 DTE, long ~18 / short ~25);
       arm prior VIX < 14; size = 1 vertical per 2 open lots (floor 1).
  H3 — Zero-cost QQQ collar (~45 DTE, long put ~5% OTM + short call ~10% OTM);
       arm prior VIX < 15 and >=1 LEAPS open; skip if net debit mid > $0.10.

Attach modes:
  while_open  — hedge only while >=1 LEAPS open and VIX gate on; size tracks lots.
  continuous  — rolling hedge whenever VIX gate on (min 1 unit even with 0 LEAPS).
  entry_day   — one hedge lot per LEAPS fill day if gate on; exit with that LEAPS.

Marks: mid/mid or ask/bid via Theta NBBO entry + EOD path. Exit on LEAPS exit,
hedge expiry, or prior VIX >= 20 (close hedge only).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.research.cta_trend import load_macro_closes
from utils.research.qqq_leaps_gap import LeapTrade, _as_date, book_stats

LOG = logging.getLogger(__name__)

ATTACH_WHILE_OPEN = "while_open"
ATTACH_CONTINUOUS = "continuous"
ATTACH_ENTRY_DAY = "entry_day"
ATTACH_MODES = (ATTACH_WHILE_OPEN, ATTACH_CONTINUOUS, ATTACH_ENTRY_DAY)

H1 = "H1"
H2 = "H2"
H3 = "H3"
HEDGE_IDS = (H1, H2, H3)

VIX_ARM_H1 = 15.0
VIX_ARM_H2 = 14.0
VIX_ARM_H3 = 15.0
VIX_EXIT = 20.0

TARGET_DTE = 45
DTE_MIN = 30
DTE_MAX = 60
PUT_LONG_OTM = 0.05
PUT_SHORT_OTM = 0.10
CALL_SHORT_OTM = 0.10
VIX_LONG_K = 18.0
VIX_SHORT_K = 25.0
COLLAR_MAX_DEBIT = 0.10

EXIT_LEAPS = "leaps_exit"
EXIT_VIX_SPIKE = "vix_spike"
EXIT_EXPIRY = "expiry"
EXIT_PANEL_END = "panel_end"
EXIT_TARGET_DOWN = "target_down"

REASON_OK = "ok"
REASON_NO_EXP = "no_expiration"
REASON_NO_STRIKE = "no_strike"
REASON_NO_NBBO = "no_nbbo"
REASON_NO_EOD = "no_eod"
REASON_COLLAR_RICH = "collar_debit_too_rich"
REASON_SKIP_GATE = "skip_vix_gate"


@dataclass
class HedgeTrade:
    structure: str
    attach: str
    mark_mode: str
    entry_date: date
    exit_date: Optional[date]
    contracts: int
    symbol: str
    expiration: date
    long_strike: float
    short_strike: float
    long_right: str
    short_right: str
    entry_debit: float
    exit_credit: Optional[float]
    exit_reason: Optional[str]
    pnl_cash: Optional[float]
    hold_days: Optional[int]
    reason: str = REASON_OK
    skipped: bool = False
    linked_leaps_entry: Optional[str] = None
    open_lots_at_entry: int = 0

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("entry_date", "exit_date", "expiration"):
            v = d.get(k)
            d[k] = v.isoformat() if isinstance(v, date) else v
        return d


@dataclass
class HedgeBookResult:
    structure: str
    attach: str
    mark_mode: str
    max_open: Optional[int]
    hedges: List[HedgeTrade] = field(default_factory=list)
    leaps_filled: int = 0
    hedge_filled: int = 0
    hedge_skipped: int = 0
    notes: str = ""


def arm_threshold(structure: str) -> float:
    if structure == H2:
        return VIX_ARM_H2
    if structure == H3:
        return VIX_ARM_H3
    return VIX_ARM_H1


def load_vix_prior_close(
    start: date,
    end: date,
) -> pd.Series:
    """Prior-session cash VIX close indexed by session date (causal gate)."""
    warm = start - timedelta(days=10)
    frames = load_macro_closes(
        ["VIX"],
        start=datetime(warm.year, warm.month, warm.day),
        end=datetime(end.year, end.month, end.day),
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    s = frames.get("VIX")
    if s is None or s.empty:
        raise RuntimeError("No VIX daily closes for hedge gate")
    closes = s.astype(float).dropna().sort_index()
    idx_dates = [pd.Timestamp(ts).date() for ts in closes.index]
    ser = pd.Series(closes.values, index=idx_dates, name="vix_close")
    prior = ser.shift(1)
    prior.name = "vix_prior"
    return prior


def leaps_from_csv(path) -> List[LeapTrade]:
    """Load filled LeapTrade rows from a Theta claim-check CSV."""
    df = pd.read_csv(path)
    if "skipped" in df.columns:
        df = df.loc[~df["skipped"].astype(bool)]
    df = df.dropna(subset=["pnl_cash", "entry_date", "exit_date"])
    out: List[LeapTrade] = []
    for _, r in df.iterrows():
        out.append(
            LeapTrade(
                mode=str(r.get("mode") or ""),
                entry_date=_as_date(r["entry_date"]),
                exit_date=_as_date(r["exit_date"]),
                contracts=int(r.get("contracts") or 1),
                entry_px=float(r.get("entry_px") or 0.0),
                strike=float(r.get("strike") or 0.0),
                entry_premium=float(r.get("entry_premium") or 0.0),
                exit_premium=float(r["exit_premium"])
                if pd.notna(r.get("exit_premium"))
                else None,
                entry_delta=float(r.get("entry_delta") or 0.0),
                entry_iv=float(r.get("entry_iv") or 0.0),
                dte=int(r.get("dte") or 0),
                hold_days=int(r["hold_days"]) if pd.notna(r.get("hold_days")) else None,
                exit_reason=str(r.get("exit_reason") or ""),
                pnl_cash=float(r["pnl_cash"]),
                gain_pct=float(r["gain_pct"]) if pd.notna(r.get("gain_pct")) else None,
                reason=str(r.get("reason") or REASON_OK),
                skipped=False,
            )
        )
    return out


def apply_max_open(trades: Sequence[LeapTrade], max_open: Optional[int]) -> List[LeapTrade]:
    """FCFS occupancy filter on already-filled LEAPS (entry order)."""
    if max_open is None:
        return [t for t in trades if not t.skipped and t.pnl_cash is not None]
    kept: List[LeapTrade] = []
    open_lots: List[Tuple[date, LeapTrade]] = []
    for t in sorted(trades, key=lambda x: (x.entry_date or date.min, x.mode)):
        if t.skipped or t.pnl_cash is None or t.entry_date is None or t.exit_date is None:
            continue
        day = t.entry_date
        open_lots = [(ed, x) for ed, x in open_lots if ed is None or ed > day]
        if len(open_lots) >= int(max_open):
            continue
        kept.append(t)
        open_lots.append((t.exit_date, t))
    return kept


def open_leaps_on_day(trades: Sequence[LeapTrade], day: date) -> List[LeapTrade]:
    """LEAPS lots open on ``day`` (entry <= day < exit). Exit day closes at EOD."""
    out = []
    for t in trades:
        if t.entry_date is None or t.exit_date is None:
            continue
        if t.entry_date <= day < t.exit_date:
            out.append(t)
    return out


def target_hedge_units(
    structure: str,
    attach: str,
    n_open: int,
    vix_prior: Optional[float],
) -> int:
    """Desired number of open hedge units for calendar attach modes."""
    if vix_prior is None or not math.isfinite(float(vix_prior)):
        return 0
    arm = arm_threshold(structure)
    if float(vix_prior) >= arm:
        return 0
    if attach == ATTACH_WHILE_OPEN:
        if n_open <= 0:
            return 0
        if structure == H2:
            return max(1, int(math.ceil(n_open / 2.0)))
        return int(n_open)
    if attach == ATTACH_CONTINUOUS:
        if structure == H2:
            return max(1, int(math.ceil(max(n_open, 1) / 2.0))) if n_open > 0 else 1
        return max(1, int(n_open)) if n_open > 0 else 1
    return 0


def list_expirations(client: ThetaDataClient, symbol: str) -> List[date]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": str(symbol).upper()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    out = []
    for x in df[col].tolist():
        try:
            out.append(date.fromisoformat(str(x)[:10]))
        except ValueError:
            continue
    return out


_STRIKES_CACHE: Dict[Tuple[str, str], List[float]] = {}


def list_strikes(client: ThetaDataClient, symbol: str, exp: date) -> List[float]:
    key = (str(symbol).upper(), exp.isoformat())
    if key in _STRIKES_CACHE:
        return list(_STRIKES_CACHE[key])
    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": str(symbol).upper(), "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        _STRIKES_CACHE[key] = []
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    out = sorted({float(x) for x in df[col].tolist() if pd.notna(x)})
    _STRIKES_CACHE[key] = out
    return list(out)


def pick_expiration(exps: Sequence[date], day: date) -> Optional[date]:
    cands = []
    for ed in exps:
        dte = (ed - day).days
        if DTE_MIN <= dte <= DTE_MAX:
            cands.append((abs(dte - TARGET_DTE), ed))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def snap_strike(strikes: Sequence[float], target: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda x: abs(x - float(target))))


def fetch_eod_leg(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    start: date,
    end: date,
    cache: Dict[Tuple[Any, ...], pd.DataFrame],
) -> pd.DataFrame:
    if end < start:
        return pd.DataFrame()
    if (end - start).days > 364:
        end = start + timedelta(days=364)
    right_s = "put" if str(right).lower().startswith("p") else "call"
    key = (
        str(symbol).upper(),
        expiration.isoformat(),
        float(strike),
        right_s,
        start.isoformat(),
        end.isoformat(),
    )
    if key in cache:
        return cache[key].copy()
    r = client.session.get(
        f"{client.base_url}/v3/option/history/eod",
        params={
            "symbol": str(symbol).upper(),
            "expiration": expiration.isoformat(),
            "strike": f"{float(strike):.3f}",
            "right": right_s,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=120,
    )
    if r.status_code == 472 or (r.content and b"No data" in r.content[:80]):
        cache[key] = pd.DataFrame()
        return pd.DataFrame()
    if r.status_code != 200 or not r.content:
        cache[key] = pd.DataFrame()
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        cache[key] = df
        return df.copy()
    if "created" in df.columns:
        ts = pd.to_datetime(df["created"], errors="coerce")
    elif "last_trade" in df.columns:
        ts = pd.to_datetime(df["last_trade"], errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, min(4, len(df.columns) - 1)], errors="coerce")
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


def _nbbo_leg(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    day: date,
) -> Optional[Dict[str, float]]:
    return client.nbbo_at_or_after(
        symbol=str(symbol).upper(),
        expiration=expiration,
        strike=float(strike),
        right=right,
        day=day,
        clock=time(9, 30),
        end_time="12:00:00",
    )


def _debit_from_nbbo(
    long_nbbo: Dict[str, float],
    short_nbbo: Dict[str, float],
    mark_mode: str,
) -> float:
    if mark_mode == "ask_bid":
        return float(long_nbbo["ask"]) - float(short_nbbo["bid"])
    return float(long_nbbo["mid"]) - float(short_nbbo["mid"])


def _credit_from_row(
    long_row: pd.Series,
    short_row: pd.Series,
    mark_mode: str,
) -> float:
    if mark_mode == "ask_bid":
        return float(long_row["bid"]) - float(short_row["ask"])
    return float(long_row["mid"]) - float(short_row["mid"])


def _spot_on_day(panel: pd.DataFrame, day: date) -> Optional[float]:
    hits = panel.loc[panel["session_date"] == day]
    if hits.empty:
        return None
    row = hits.iloc[0]
    if pd.notna(row.get("open")):
        return float(row["open"])
    if pd.notna(row.get("close")):
        return float(row["close"])
    return None


def select_h1_legs(
    client: ThetaDataClient,
    *,
    day: date,
    spot: float,
    exps: Sequence[date],
) -> Optional[Tuple[date, float, float]]:
    exp = pick_expiration(exps, day)
    if exp is None:
        return None
    strikes = list_strikes(client, "QQQ", exp)
    long_k = snap_strike(strikes, spot * (1.0 - PUT_LONG_OTM))
    short_k = snap_strike(strikes, spot * (1.0 - PUT_SHORT_OTM))
    if long_k is None or short_k is None:
        return None
    if short_k >= long_k:
        # Ensure short is further OTM
        lower = [k for k in strikes if k < long_k]
        if not lower:
            return None
        short_k = max(lower)
    return exp, float(long_k), float(short_k)


def select_h2_legs(
    client: ThetaDataClient,
    *,
    day: date,
    exps: Sequence[date],
) -> Optional[Tuple[date, float, float]]:
    exp = pick_expiration(exps, day)
    if exp is None:
        return None
    strikes = list_strikes(client, "VIX", exp)
    long_k = snap_strike(strikes, VIX_LONG_K)
    short_k = snap_strike(strikes, VIX_SHORT_K)
    if long_k is None or short_k is None:
        return None
    if short_k <= long_k:
        higher = [k for k in strikes if k > long_k]
        if not higher:
            return None
        short_k = min(higher)
    return exp, float(long_k), float(short_k)


def select_h3_legs(
    client: ThetaDataClient,
    *,
    day: date,
    spot: float,
    exps: Sequence[date],
) -> Optional[Tuple[date, float, float]]:
    exp = pick_expiration(exps, day)
    if exp is None:
        return None
    strikes = list_strikes(client, "QQQ", exp)
    put_k = snap_strike(strikes, spot * (1.0 - PUT_LONG_OTM))
    call_k = snap_strike(strikes, spot * (1.0 + CALL_SHORT_OTM))
    if put_k is None or call_k is None:
        return None
    if call_k <= put_k:
        higher = [k for k in strikes if k > put_k]
        if not higher:
            return None
        call_k = min(higher)
    return exp, float(put_k), float(call_k)


def simulate_vertical(
    client: ThetaDataClient,
    *,
    structure: str,
    attach: str,
    mark_mode: str,
    day: date,
    contracts: int,
    symbol: str,
    expiration: date,
    long_strike: float,
    short_strike: float,
    long_right: str,
    short_right: str,
    leaps_exit: Optional[date],
    vix_prior: pd.Series,
    panel_end: date,
    eod_cache: Dict[Tuple[Any, ...], pd.DataFrame],
    open_lots_at_entry: int = 0,
    linked_leaps_entry: Optional[date] = None,
    collar_max_debit: float = COLLAR_MAX_DEBIT,
) -> HedgeTrade:
    """Open one vertical/collar unit and walk EOD until exit."""
    base = dict(
        structure=structure,
        attach=attach,
        mark_mode=mark_mode,
        entry_date=day,
        exit_date=None,
        contracts=int(contracts),
        symbol=symbol,
        expiration=expiration,
        long_strike=float(long_strike),
        short_strike=float(short_strike),
        long_right=long_right,
        short_right=short_right,
        entry_debit=0.0,
        exit_credit=None,
        exit_reason=None,
        pnl_cash=None,
        hold_days=None,
        open_lots_at_entry=int(open_lots_at_entry),
        linked_leaps_entry=linked_leaps_entry.isoformat() if linked_leaps_entry else None,
    )
    long_nbbo = _nbbo_leg(
        client,
        symbol=symbol,
        expiration=expiration,
        strike=long_strike,
        right=long_right,
        day=day,
    )
    short_nbbo = _nbbo_leg(
        client,
        symbol=symbol,
        expiration=expiration,
        strike=short_strike,
        right=short_right,
        day=day,
    )
    if not long_nbbo or not short_nbbo:
        return HedgeTrade(**base, reason=REASON_NO_NBBO, skipped=True)

    entry_debit = _debit_from_nbbo(long_nbbo, short_nbbo, mark_mode)
    if structure == H3 and mark_mode == "mid_mid" and entry_debit > collar_max_debit + 1e-9:
        return HedgeTrade(
            **{**base, "entry_debit": float(entry_debit)},
            reason=REASON_COLLAR_RICH,
            skipped=True,
        )
    if structure == H3 and mark_mode == "ask_bid" and entry_debit > collar_max_debit + 1e-9:
        return HedgeTrade(
            **{**base, "entry_debit": float(entry_debit)},
            reason=REASON_COLLAR_RICH,
            skipped=True,
        )
    if not math.isfinite(entry_debit):
        return HedgeTrade(**base, reason=REASON_NO_NBBO, skipped=True)

    end = min(expiration, panel_end, day + timedelta(days=120))
    if leaps_exit is not None:
        end = min(end, leaps_exit)
    long_eod = fetch_eod_leg(
        client,
        symbol=symbol,
        expiration=expiration,
        strike=long_strike,
        right=long_right,
        start=day,
        end=end,
        cache=eod_cache,
    )
    short_eod = fetch_eod_leg(
        client,
        symbol=symbol,
        expiration=expiration,
        strike=short_strike,
        right=short_right,
        start=day,
        end=end,
        cache=eod_cache,
    )
    if long_eod.empty or short_eod.empty:
        return HedgeTrade(
            **{**base, "entry_debit": float(entry_debit)},
            reason=REASON_NO_EOD,
            skipped=True,
        )
    merged = long_eod.merge(
        short_eod,
        on="session_date",
        how="inner",
        suffixes=("_l", "_s"),
    )
    if merged.empty:
        return HedgeTrade(
            **{**base, "entry_debit": float(entry_debit)},
            reason=REASON_NO_EOD,
            skipped=True,
        )

    exit_credit: Optional[float] = None
    exit_day: Optional[date] = None
    exit_reason: Optional[str] = None
    after = merged.loc[merged["session_date"] > day]

    def _row_credit(row: pd.Series) -> float:
        long_row = pd.Series(
            {"bid": row["bid_l"], "ask": row["ask_l"], "mid": row["mid_l"]}
        )
        short_row = pd.Series(
            {"bid": row["bid_s"], "ask": row["ask_s"], "mid": row["mid_s"]}
        )
        return _credit_from_row(long_row, short_row, mark_mode)

    for _, row in after.iterrows():
        d = row["session_date"]
        vp = vix_prior.get(d) if d in vix_prior.index else None
        if vp is not None and math.isfinite(float(vp)) and float(vp) >= VIX_EXIT:
            exit_credit = _row_credit(row)
            exit_day = d
            exit_reason = EXIT_VIX_SPIKE
            break
        if leaps_exit is not None and d >= leaps_exit:
            exit_credit = _row_credit(row)
            exit_day = d
            exit_reason = EXIT_LEAPS
            break

    if exit_credit is None:
        last = after.iloc[-1] if not after.empty else merged.iloc[-1]
        exit_day = last["session_date"]
        exit_credit = _row_credit(last)
        if exit_day >= expiration:
            exit_reason = EXIT_EXPIRY
        elif leaps_exit is not None and exit_day >= leaps_exit:
            exit_reason = EXIT_LEAPS
        else:
            exit_reason = EXIT_PANEL_END

    assert exit_day is not None and exit_credit is not None
    pnl = (float(exit_credit) - float(entry_debit)) * 100.0 * int(contracts)
    return HedgeTrade(
        structure=structure,
        attach=attach,
        mark_mode=mark_mode,
        entry_date=day,
        exit_date=exit_day,
        contracts=int(contracts),
        symbol=symbol,
        expiration=expiration,
        long_strike=float(long_strike),
        short_strike=float(short_strike),
        long_right=long_right,
        short_right=short_right,
        entry_debit=float(entry_debit),
        exit_credit=float(exit_credit),
        exit_reason=exit_reason,
        pnl_cash=float(pnl),
        hold_days=(exit_day - day).days,
        reason=REASON_OK,
        skipped=False,
        linked_leaps_entry=linked_leaps_entry.isoformat() if linked_leaps_entry else None,
        open_lots_at_entry=int(open_lots_at_entry),
    )


def _structure_legs(
    client: ThetaDataClient,
    structure: str,
    day: date,
    spot: Optional[float],
    qqq_exps: Sequence[date],
    vix_exps: Sequence[date],
) -> Optional[Tuple[str, date, float, float, str, str]]:
    """Return (symbol, exp, long_k, short_k, long_right, short_right)."""
    if structure == H1:
        if spot is None:
            return None
        picked = select_h1_legs(client, day=day, spot=spot, exps=qqq_exps)
        if not picked:
            return None
        exp, lk, sk = picked
        return ("QQQ", exp, lk, sk, "put", "put")
    if structure == H2:
        picked = select_h2_legs(client, day=day, exps=vix_exps)
        if not picked:
            return None
        exp, lk, sk = picked
        return ("VIX", exp, lk, sk, "call", "call")
    if structure == H3:
        if spot is None:
            return None
        picked = select_h3_legs(client, day=day, spot=spot, exps=qqq_exps)
        if not picked:
            return None
        exp, put_k, call_k = picked
        return ("QQQ", exp, put_k, call_k, "put", "call")
    return None


def run_entry_day_hedges(
    client: ThetaDataClient,
    *,
    structure: str,
    mark_mode: str,
    leaps: Sequence[LeapTrade],
    panel: pd.DataFrame,
    vix_prior: pd.Series,
    qqq_exps: Sequence[date],
    vix_exps: Sequence[date],
    eod_cache: Dict[Tuple[Any, ...], pd.DataFrame],
    panel_end: date,
) -> List[HedgeTrade]:
    arm = arm_threshold(structure)
    out: List[HedgeTrade] = []
    for t in leaps:
        day = t.entry_date
        if day is None:
            continue
        vp = vix_prior.get(day) if day in vix_prior.index else None
        if vp is None or not math.isfinite(float(vp)) or float(vp) >= arm:
            out.append(
                HedgeTrade(
                    structure=structure,
                    attach=ATTACH_ENTRY_DAY,
                    mark_mode=mark_mode,
                    entry_date=day,
                    exit_date=None,
                    contracts=int(t.contracts),
                    symbol="QQQ" if structure != H2 else "VIX",
                    expiration=day,
                    long_strike=0.0,
                    short_strike=0.0,
                    long_right="put",
                    short_right="put",
                    entry_debit=0.0,
                    exit_credit=None,
                    exit_reason=None,
                    pnl_cash=None,
                    hold_days=None,
                    reason=REASON_SKIP_GATE,
                    skipped=True,
                    linked_leaps_entry=day.isoformat(),
                )
            )
            continue
        spot = _spot_on_day(panel, day)
        legs = _structure_legs(client, structure, day, spot, qqq_exps, vix_exps)
        if legs is None:
            out.append(
                HedgeTrade(
                    structure=structure,
                    attach=ATTACH_ENTRY_DAY,
                    mark_mode=mark_mode,
                    entry_date=day,
                    exit_date=None,
                    contracts=int(t.contracts),
                    symbol="QQQ" if structure != H2 else "VIX",
                    expiration=day,
                    long_strike=0.0,
                    short_strike=0.0,
                    long_right="put",
                    short_right="put",
                    entry_debit=0.0,
                    exit_credit=None,
                    exit_reason=None,
                    pnl_cash=None,
                    hold_days=None,
                    reason=REASON_NO_EXP,
                    skipped=True,
                    linked_leaps_entry=day.isoformat(),
                )
            )
            continue
        symbol, exp, lk, sk, lr, sr = legs
        # H2 entry_day: 1 vertical per fill (contracts=1); H1/H3 match LEAPS size
        ctr = 1 if structure == H2 else int(t.contracts)
        tr = simulate_vertical(
            client,
            structure=structure,
            attach=ATTACH_ENTRY_DAY,
            mark_mode=mark_mode,
            day=day,
            contracts=ctr,
            symbol=symbol,
            expiration=exp,
            long_strike=lk,
            short_strike=sk,
            long_right=lr,
            short_right=sr,
            leaps_exit=t.exit_date,
            vix_prior=vix_prior,
            panel_end=panel_end,
            eod_cache=eod_cache,
            open_lots_at_entry=1,
            linked_leaps_entry=day,
        )
        out.append(tr)
        LOG.info(
            "entry_day %s %s leaps=%s hedge=%s pnl=%s",
            structure,
            day,
            t.pnl_cash,
            "skip" if tr.skipped else tr.exit_reason,
            tr.pnl_cash,
        )
    return out


def run_calendar_hedges(
    client: ThetaDataClient,
    *,
    structure: str,
    attach: str,
    mark_mode: str,
    leaps: Sequence[LeapTrade],
    panel: pd.DataFrame,
    vix_prior: pd.Series,
    qqq_exps: Sequence[date],
    vix_exps: Sequence[date],
    eod_cache: Dict[Tuple[Any, ...], pd.DataFrame],
    panel_end: date,
) -> List[HedgeTrade]:
    """Walk session days and open/close hedge units to match target size.

    Active units store legs only; Theta EOD walk runs once at close/flush.
    """
    if not leaps:
        return []
    start = min(t.entry_date for t in leaps if t.entry_date)
    end = max(
        (t.exit_date for t in leaps if t.exit_date),
        default=panel_end,
    )
    end = min(end, panel_end)
    sessions = sorted(
        {
            d
            for d in panel["session_date"].tolist()
            if isinstance(d, date) and start <= d <= end
        }
    )
    active: List[Dict[str, Any]] = []
    closed: List[HedgeTrade] = []

    for day in sessions:
        n_open = len(open_leaps_on_day(leaps, day))
        vp = vix_prior.get(day) if day in vix_prior.index else None

        # Expire units on/after listed expiration
        kept_exp = []
        for a in active:
            if a["expiration"] <= day:
                closed.append(
                    _close_active_unit(
                        client,
                        active_meta=a,
                        structure=structure,
                        attach=attach,
                        mark_mode=mark_mode,
                        leaps_exit=day,
                        vix_prior=vix_prior,
                        panel_end=panel_end,
                        eod_cache=eod_cache,
                        exit_reason_hint=EXIT_EXPIRY,
                    )
                )
            else:
                kept_exp.append(a)
        active = kept_exp

        # Force close all on VIX spike
        if vp is not None and math.isfinite(float(vp)) and float(vp) >= VIX_EXIT and active:
            for a in list(active):
                closed.append(
                    _close_active_unit(
                        client,
                        active_meta=a,
                        structure=structure,
                        attach=attach,
                        mark_mode=mark_mode,
                        leaps_exit=day,
                        vix_prior=vix_prior,
                        panel_end=panel_end,
                        eod_cache=eod_cache,
                    )
                )
            active = []
            continue

        target = target_hedge_units(structure, attach, n_open, vp)
        while len(active) > target:
            a = active.pop(0)
            closed.append(
                _close_active_unit(
                    client,
                    active_meta=a,
                    structure=structure,
                    attach=attach,
                    mark_mode=mark_mode,
                    leaps_exit=day,
                    vix_prior=vix_prior,
                    panel_end=panel_end,
                    eod_cache=eod_cache,
                    exit_reason_hint=EXIT_TARGET_DOWN,
                )
            )

        while len(active) < target:
            spot = _spot_on_day(panel, day)
            legs = _structure_legs(client, structure, day, spot, qqq_exps, vix_exps)
            if legs is None:
                LOG.info("%s %s %s no legs on %s", structure, attach, mark_mode, day)
                break
            symbol, exp, lk, sk, lr, sr = legs
            active.append(
                {
                    "entry_date": day,
                    "symbol": symbol,
                    "expiration": exp,
                    "long_strike": lk,
                    "short_strike": sk,
                    "long_right": lr,
                    "short_right": sr,
                    "n_open": n_open,
                }
            )
            LOG.info(
                "calendar open %s %s %s lots=%d target=%d active=%d",
                structure,
                attach,
                day,
                n_open,
                target,
                len(active),
            )

    for a in active:
        closed.append(
            _close_active_unit(
                client,
                active_meta=a,
                structure=structure,
                attach=attach,
                mark_mode=mark_mode,
                leaps_exit=None,
                vix_prior=vix_prior,
                panel_end=panel_end,
                eod_cache=eod_cache,
            )
        )
    return closed


def _close_active_unit(
    client: ThetaDataClient,
    *,
    active_meta: Dict[str, Any],
    structure: str,
    attach: str,
    mark_mode: str,
    leaps_exit: Optional[date],
    vix_prior: pd.Series,
    panel_end: date,
    eod_cache: Dict[Tuple[Any, ...], pd.DataFrame],
    exit_reason_hint: Optional[str] = None,
) -> HedgeTrade:
    """Simulate an active unit from its entry through ``leaps_exit`` / natural path."""
    entry = active_meta["entry_date"]
    tr = simulate_vertical(
        client,
        structure=structure,
        attach=attach,
        mark_mode=mark_mode,
        day=entry,
        contracts=1,
        symbol=active_meta["symbol"],
        expiration=active_meta["expiration"],
        long_strike=active_meta["long_strike"],
        short_strike=active_meta["short_strike"],
        long_right=active_meta["long_right"],
        short_right=active_meta["short_right"],
        leaps_exit=leaps_exit,
        vix_prior=vix_prior,
        panel_end=panel_end,
        eod_cache=eod_cache,
        open_lots_at_entry=int(active_meta.get("n_open") or 0),
    )
    if not tr.skipped and exit_reason_hint and tr.exit_reason == EXIT_LEAPS:
        tr.exit_reason = exit_reason_hint
    return tr


def run_hedge_book(
    *,
    structure: str,
    attach: str,
    mark_mode: str,
    leaps: Sequence[LeapTrade],
    panel: pd.DataFrame,
    vix_prior: pd.Series,
    max_open: Optional[int],
    client: Optional[ThetaDataClient] = None,
    qqq_exps: Optional[Sequence[date]] = None,
    vix_exps: Optional[Sequence[date]] = None,
    eod_cache: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
    panel_end: Optional[date] = None,
) -> HedgeBookResult:
    client = client or get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal not reachable on :25503")
    filled = apply_max_open(leaps, max_open)
    eod_cache = eod_cache if eod_cache is not None else {}
    if qqq_exps is None:
        qqq_exps = list_expirations(client, "QQQ")
    if vix_exps is None:
        vix_exps = list_expirations(client, "VIX")
    if panel_end is None:
        panel_end = max(panel["session_date"])

    book = HedgeBookResult(
        structure=structure,
        attach=attach,
        mark_mode=mark_mode,
        max_open=max_open,
        leaps_filled=len(filled),
    )
    if attach == ATTACH_ENTRY_DAY:
        hedges = run_entry_day_hedges(
            client,
            structure=structure,
            mark_mode=mark_mode,
            leaps=filled,
            panel=panel,
            vix_prior=vix_prior,
            qqq_exps=qqq_exps,
            vix_exps=vix_exps,
            eod_cache=eod_cache,
            panel_end=panel_end,
        )
    else:
        hedges = run_calendar_hedges(
            client,
            structure=structure,
            attach=attach,
            mark_mode=mark_mode,
            leaps=filled,
            panel=panel,
            vix_prior=vix_prior,
            qqq_exps=qqq_exps,
            vix_exps=vix_exps,
            eod_cache=eod_cache,
            panel_end=panel_end,
        )
    book.hedges = hedges
    book.hedge_filled = sum(1 for h in hedges if not h.skipped and h.pnl_cash is not None)
    book.hedge_skipped = sum(1 for h in hedges if h.skipped)
    if structure == H2:
        book.notes = "Theta VIX options available (probe 2026-10-01)"
    return book


def hedge_stats(hedges: Sequence[HedgeTrade]) -> Dict[str, Any]:
    rows = [h for h in hedges if not h.skipped and h.pnl_cash is not None]
    if not rows:
        return {
            "n": 0,
            "win_rate_pct": None,
            "total_pnl": 0.0,
            "pf": None,
            "avg_days": None,
            "max_dd": 0.0,
            "avg_entry_debit": None,
            "gross_win": 0.0,
            "gross_loss": 0.0,
        }
    pnls = [float(h.pnl_cash) for h in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else None
    # cash MDD on hedge-only chronological
    ordered = sorted(rows, key=lambda h: (h.exit_date or h.entry_date, h.entry_date))
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for h in ordered:
        eq += float(h.pnl_cash or 0.0)
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    days = [h.hold_days for h in rows if h.hold_days is not None]
    return {
        "n": len(rows),
        "win_rate_pct": round(100.0 * len(wins) / len(rows), 1),
        "total_pnl": round(float(sum(pnls)), 2),
        "pf": round(pf, 2) if pf is not None else None,
        "avg_days": round(float(np.mean(days)), 1) if days else None,
        "max_dd": round(float(mdd), 2),
        "avg_entry_debit": round(
            float(np.mean([h.entry_debit for h in rows])), 4
        ),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
    }


def combine_leaps_hedge_stats(
    leaps: Sequence[LeapTrade],
    hedges: Sequence[HedgeTrade],
) -> Dict[str, Any]:
    """Pooled cash stats: LEAPS + filled hedges as one P&L stream."""
    ls = book_stats(list(leaps))
    hs = hedge_stats(hedges)
    events: List[Tuple[date, float]] = []
    for t in leaps:
        if t.skipped or t.pnl_cash is None or t.exit_date is None:
            continue
        events.append((t.exit_date, float(t.pnl_cash)))
    for h in hedges:
        if h.skipped or h.pnl_cash is None or h.exit_date is None:
            continue
        events.append((h.exit_date, float(h.pnl_cash)))
    events.sort(key=lambda x: x[0])
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for _, pnl in events:
        eq += pnl
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    total = float(sum(p for _, p in events))
    wins = [p for _, p in events if p > 0]
    losses = [p for _, p in events if p < 0]
    gw = float(sum(wins))
    gl = float(abs(sum(losses)))
    return {
        "leaps": ls,
        "hedge": hs,
        "combined_n_events": len(events),
        "combined_pnl": round(total, 2),
        "combined_pf": round(gw / gl, 2) if gl > 1e-12 else None,
        "combined_max_dd": round(float(mdd), 2),
        "hedge_cost_bleed": hs.get("total_pnl"),
        "leaps_pnl": ls.get("total_pnl"),
    }


def hedges_to_frame(hedges: Sequence[HedgeTrade]) -> pd.DataFrame:
    return pd.DataFrame([h.to_row() for h in hedges])


__all__ = [
    "ATTACH_MODES",
    "ATTACH_CONTINUOUS",
    "ATTACH_ENTRY_DAY",
    "ATTACH_WHILE_OPEN",
    "H1",
    "H2",
    "H3",
    "HEDGE_IDS",
    "HedgeBookResult",
    "HedgeTrade",
    "apply_max_open",
    "combine_leaps_hedge_stats",
    "hedge_stats",
    "hedges_to_frame",
    "leaps_from_csv",
    "list_expirations",
    "load_vix_prior_close",
    "open_leaps_on_day",
    "pick_expiration",
    "run_hedge_book",
    "snap_strike",
    "target_hedge_units",
]
