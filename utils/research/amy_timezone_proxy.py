"""Improved proxy for Amy Meissner TimeZone (calendar + credit spread).

Public roundtable (Aeromir 2025-10-02 / YouTube pvRJ7lnubM8) discloses structure,
Thursday cadence, +/-5% capital targets, and proactive adjustments — not the
exact workshop strike/term/adjust recipe.

v2 improvements vs the first no-adjust EOD proxy:
  * Mark modes: mid | mid_slip (nickel/dime off mid, clamped to NBBO) | ask_bid
  * 1m Theta NBBO path for entry timing and intraday TP/SL
  * Optional IV term-structure gate (skip steep front>back backwardation)
  * Basic +/-2% underlying move adjustments (roll puts / roll calendar ATM; max 3)

Still not her exact system — listed proxy only.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.options.greeks import implied_volatility

LOG = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

CLAIMED = {
    "source": "Amy Meissner TimeZone refresh — Aeromir roundtable 2025-10-02 (pvRJ7lnubM8)",
    "structure": "calendar + credit spread; positive theta + positive vega; market-neutral entry",
    "underlying": "RUT 2-lot (~$7-10k capital); also SPX / ETF",
    "entry": "Thursday fixed weekly (or 1-2 weeks/month cadence)",
    "hold": "~3-5 trading days (avg ~5)",
    "targets": "+5% / -5% of planned capital",
    "windows": {
        "2016_2019_1wk_mo": {"wr": 0.75, "note": "one week/month"},
        "2016_2019_more": {"wr": 0.83, "pf": 10.0},
        "2020_jan_aug": {"wr": 0.80},
        "2020_mar": {"wins": 3, "losses": 1, "net": "profit"},
        "last_3y_1wk_mo": {
            "n": 38,
            "wr": 0.87,
            "pf": 5.0,
            "ret_pct_capital": 130.0,
            "days_in": 154,
        },
        "2025_jan_aug_weekly": {
            "n": 30,
            "wr": 0.80,
            "pf": 4.0,
            "ret_pct_capital": 100.0,
            "avg_profit_2lot": 400.0,
        },
    },
}

ROOT = "RUTW"
MULT = 100.0
DEFAULT_CAPITAL = 8000.0
DEFAULT_LOTS = 2
DEFAULT_TP_FRAC = 0.05
DEFAULT_SL_FRAC = 0.05
DEFAULT_MAX_HOLD_DAYS = 5
DEFAULT_PUT_OTM_FRAC = 0.03
DEFAULT_PUT_WIDTH = 10.0
DEFAULT_FRONT_DTE = (5, 12)
DEFAULT_SLIP = 0.05
DEFAULT_ADJUST_MOVE = 0.02
DEFAULT_MAX_ADJUSTS = 3
DEFAULT_IV_BACKWARD_MAX = 0.08
DEFAULT_ENTRY_START = time(14, 0)
DEFAULT_ENTRY_END = time(15, 45)


@dataclass
class LegQuote:
    expiration: date
    strike: float
    right: str
    bid: float
    ask: float
    timestamp: Optional[pd.Timestamp] = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class StructureState:
    atm: float
    put_short: float
    put_long: float
    front: date
    back: date


@dataclass
class ProxyTrade:
    entry_date: date
    exit_date: date
    entry_ts: Optional[str]
    exit_ts: Optional[str]
    spot: float
    atm: float
    front: date
    back: date
    put_short: float
    put_long: float
    lots: int
    capital: float
    mark_mode: str
    entry_debit: float
    exit_credit: float
    pnl: float
    pnl_pct_capital: float
    hold_days: int
    exit_reason: str
    calendar_entry: float
    put_credit_entry: float
    n_adjusts: int = 0
    front_iv: Optional[float] = None
    back_iv: Optional[float] = None
    slip: float = 0.0


@dataclass
class BookResult:
    mode: str
    mark_mode: str
    cadence: str
    trades: List[ProxyTrade] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def list_friday_expirations(client: ThetaDataClient, root: str = ROOT) -> List[date]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": root},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    out: List[date] = []
    for x in df[col].tolist():
        try:
            d = date.fromisoformat(str(x)[:10])
        except ValueError:
            continue
        if d.weekday() == 4:
            out.append(d)
    return sorted(set(out))


def list_strikes(client: ThetaDataClient, exp: date, root: str = ROOT) -> List[float]:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": root, "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        return []
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    return sorted({float(x) for x in df[col].tolist() if pd.notna(x)})


def nearest_strike(strikes: Sequence[float], spot: float) -> Optional[float]:
    if not strikes:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - float(spot))))


def strike_at_or_below(strikes: Sequence[float], level: float) -> Optional[float]:
    below = [float(s) for s in strikes if float(s) <= float(level) + 1e-9]
    if not below:
        return None
    return max(below)


def load_rut_spots(start: date, end: date) -> pd.Series:
    df = yf.download(
        "^RUT",
        start=start.isoformat(),
        end=(end + timedelta(days=2)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]
    s = close.astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).date
    return s


def thursdays_between(start: date, end: date) -> List[date]:
    d = start
    while d.weekday() != 3:
        d += timedelta(days=1)
        if d > end:
            return []
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def first_thursday_of_month(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 3:
        d += timedelta(days=1)
    return d


def monthly_thursdays(start: date, end: date) -> List[date]:
    out = []
    y, m = start.year, start.month
    while True:
        t = first_thursday_of_month(y, m)
        if t > end:
            break
        if t >= start:
            out.append(t)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        if date(y, m, 1) > end + timedelta(days=31):
            break
    return out


def pick_front_back(
    fridays: Sequence[date],
    entry: date,
    front_dte: Tuple[int, int] = DEFAULT_FRONT_DTE,
) -> Optional[Tuple[date, date]]:
    lo, hi = front_dte
    fronts = [e for e in fridays if lo <= (e - entry).days <= hi]
    if not fronts:
        return None
    front = fronts[0]
    backs = [e for e in fridays if e > front]
    if not backs:
        return None
    return front, backs[0]


def _normalize_quote_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["minute", "timestamp", "bid", "ask", "mid"])
    out = df.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame(columns=["minute", "timestamp", "bid", "ask", "mid"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(ET)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(ET)
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out = out.dropna(subset=["timestamp", "bid", "ask"])
    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])]
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["minute"] = out["timestamp"].dt.floor("min")
    out = out.sort_values("minute").drop_duplicates("minute", keep="last")
    return out[["minute", "timestamp", "bid", "ask", "mid"]]


def fetch_1m_day(
    client: ThetaDataClient,
    *,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    root: str = ROOT,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    cache: Optional[Dict[Tuple[Any, ...], pd.DataFrame]] = None,
) -> pd.DataFrame:
    key = (root, expiration, float(strike), right.lower()[:1], day, start_time, end_time)
    if cache is not None and key in cache:
        return cache[key]
    raw = client.option_history_quote(
        symbol=root,
        expiration=expiration,
        strike=strike,
        right=right,
        day=day,
        interval="1m",
        start_time=start_time,
        end_time=end_time,
    )
    out = _normalize_quote_df(raw)
    if cache is not None:
        cache[key] = out
    return out


def fill_px(mid: float, bid: float, ask: float, *, side: str, mark_mode: str, slip: float) -> float:
    mm = mark_mode.lower()
    if mm in ("ask_bid", "marketable"):
        px = ask if side == "buy" else bid
    elif mm in ("mid", "mid_mid"):
        px = mid
    elif mm in ("mid_slip", "slip", "nickel", "dime"):
        px = mid + slip if side == "buy" else mid - slip
    else:
        raise ValueError(f"unknown mark_mode={mark_mode}")
    return float(min(max(px, bid), ask))


def package_open_debit(
    sfc: LegQuote,
    lbc: LegQuote,
    sp: LegQuote,
    lp: LegQuote,
    *,
    mark_mode: str,
    lots: int,
    slip: float,
) -> Tuple[float, float, float]:
    back_buy = fill_px(lbc.mid, lbc.bid, lbc.ask, side="buy", mark_mode=mark_mode, slip=slip)
    front_sell = fill_px(sfc.mid, sfc.bid, sfc.ask, side="sell", mark_mode=mark_mode, slip=slip)
    put_sell = fill_px(sp.mid, sp.bid, sp.ask, side="sell", mark_mode=mark_mode, slip=slip)
    put_buy = fill_px(lp.mid, lp.bid, lp.ask, side="buy", mark_mode=mark_mode, slip=slip)
    cal = back_buy - front_sell
    put_c = put_sell - put_buy
    debit = (cal - put_c) * MULT * lots
    return float(debit), float(cal * MULT * lots), float(put_c * MULT * lots)


def package_close_credit(
    sfc: LegQuote,
    lbc: LegQuote,
    sp: LegQuote,
    lp: LegQuote,
    *,
    mark_mode: str,
    lots: int,
    slip: float,
) -> float:
    back_sell = fill_px(lbc.mid, lbc.bid, lbc.ask, side="sell", mark_mode=mark_mode, slip=slip)
    front_buy = fill_px(sfc.mid, sfc.bid, sfc.ask, side="buy", mark_mode=mark_mode, slip=slip)
    put_buy = fill_px(sp.mid, sp.bid, sp.ask, side="buy", mark_mode=mark_mode, slip=slip)
    put_sell = fill_px(lp.mid, lp.bid, lp.ask, side="sell", mark_mode=mark_mode, slip=slip)
    cal = back_sell - front_buy
    put_c = put_buy - put_sell
    return float((cal - put_c) * MULT * lots)


def spot_from_parity(call_mid: float, put_mid: float, strike: float) -> float:
    return float(strike + call_mid - put_mid)


def estimate_iv(
    *,
    mid: float,
    spot: float,
    strike: float,
    dte_days: float,
    right: str,
) -> Optional[float]:
    t = max(float(dte_days), 0.25) / 365.0
    if mid <= 0 or spot <= 0 or t <= 0:
        return None
    try:
        iv = implied_volatility(
            float(mid),
            float(spot),
            float(strike),
            t,
            0.05,
            "call" if right.lower().startswith("c") else "put",
        )
    except Exception:
        return None
    if iv is None or not math.isfinite(iv) or iv <= 0.01 or iv > 2.5:
        return None
    return float(iv)


def _leg_at_minute(
    df: pd.DataFrame, minute: pd.Timestamp, exp: date, strike: float, right: str
) -> Optional[LegQuote]:
    if df is None or df.empty:
        return None
    rows = df[df["minute"] == minute]
    if rows.empty:
        prior = df[df["minute"] <= minute]
        if prior.empty:
            return None
        row = prior.iloc[-1]
    else:
        row = rows.iloc[-1]
    return LegQuote(
        expiration=exp,
        strike=float(strike),
        right=right,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        timestamp=row["timestamp"],
    )


def align_package_minutes(legs: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    sets = []
    for df in legs.values():
        if df is None or df.empty:
            return []
        sets.append(set(df["minute"].tolist()))
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def run_proxy_book(
    *,
    start: date,
    end: date,
    cadence: str = "weekly",
    mark_mode: str = "mid_slip",
    capital: float = DEFAULT_CAPITAL,
    lots: int = DEFAULT_LOTS,
    tp_frac: float = DEFAULT_TP_FRAC,
    sl_frac: float = DEFAULT_SL_FRAC,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
    put_otm_frac: float = DEFAULT_PUT_OTM_FRAC,
    put_width: float = DEFAULT_PUT_WIDTH,
    front_dte: Tuple[int, int] = DEFAULT_FRONT_DTE,
    slip: float = DEFAULT_SLIP,
    use_1m: bool = True,
    allow_adjust: bool = True,
    adjust_move: float = DEFAULT_ADJUST_MOVE,
    max_adjusts: int = DEFAULT_MAX_ADJUSTS,
    iv_filter: bool = True,
    iv_backward_max: float = DEFAULT_IV_BACKWARD_MAX,
    entry_start: time = DEFAULT_ENTRY_START,
    entry_end: time = DEFAULT_ENTRY_END,
    clamp_sl: bool = True,
    client: Optional[ThetaDataClient] = None,
    root: str = ROOT,
) -> BookResult:
    client = client or get_thetadata_client()
    fridays = list_friday_expirations(client, root=root)
    spots = load_rut_spots(start - timedelta(days=5), end + timedelta(days=20))
    if spots.empty:
        raise RuntimeError("no ^RUT spots from Yahoo")

    if cadence == "weekly":
        entries = thursdays_between(start, end)
    elif cadence in ("monthly", "1wk_mo"):
        entries = monthly_thursdays(start, end)
    else:
        raise ValueError(f"unknown cadence={cadence}")

    mm_in = mark_mode.lower()
    if mm_in in ("slip", "mid_slip", "mid_slip05", "nickel"):
        mark_mode = "mid_slip"
        slip = 0.05
    elif mm_in in ("mid_slip10", "dime"):
        mark_mode = "mid_slip"
        slip = 0.10
    elif mm_in in ("mid", "mid_mid"):
        mark_mode = "mid"
        slip = 0.0
    elif mm_in in ("ask_bid", "marketable"):
        mark_mode = "ask_bid"
        slip = 0.0

    tp_usd = capital * tp_frac
    sl_usd = capital * sl_frac
    cache_1m: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    mode = "proxy_1m_adjust" if (use_1m and allow_adjust) else (
        "proxy_1m" if use_1m else "proxy_eod"
    )
    result = BookResult(mode=mode, mark_mode=mark_mode, cadence=cadence)

    def load_leg_day(exp: date, strike: float, right: str, day: date) -> pd.DataFrame:
        return fetch_1m_day(
            client,
            expiration=exp,
            strike=strike,
            right=right,
            day=day,
            root=root,
            cache=cache_1m,
        )

    for entry in entries:
        prior = [d for d in spots.index if d <= entry]
        if not prior:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_spot"})
            continue
        spot0 = float(spots.loc[prior[-1]])

        fb = pick_front_back(fridays, entry, front_dte=front_dte)
        if fb is None:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_front_back"})
            continue
        front, back = fb
        strikes = list_strikes(client, front, root=root)
        if not strikes:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_strikes"})
            continue
        atm = nearest_strike(strikes, spot0)
        put_short = strike_at_or_below(strikes, spot0 * (1.0 - put_otm_frac))
        if atm is None or put_short is None:
            result.skipped.append({"entry": entry.isoformat(), "reason": "strike_pick"})
            continue
        put_long = strike_at_or_below(strikes, put_short - put_width)
        if put_long is None or put_long >= put_short:
            below = [s for s in strikes if s < put_short]
            if not below:
                result.skipped.append({"entry": entry.isoformat(), "reason": "put_width"})
                continue
            put_long = max(below)

        state = StructureState(
            atm=float(atm),
            put_short=float(put_short),
            put_long=float(put_long),
            front=front,
            back=back,
        )

        if not use_1m:
            trade = _run_eod_trade(
                client=client,
                entry=entry,
                spot0=spot0,
                state=state,
                spots=spots,
                mark_mode=mark_mode,
                slip=slip,
                lots=lots,
                capital=capital,
                tp_usd=tp_usd,
                sl_usd=sl_usd,
                max_hold_days=max_hold_days,
                clamp_sl=clamp_sl,
                root=root,
                iv_filter=iv_filter,
                iv_backward_max=iv_backward_max,
            )
            if isinstance(trade, dict):
                result.skipped.append(trade)
            else:
                result.trades.append(trade)
                _log_trade(trade)
            continue

        day_legs = {
            "sfc": load_leg_day(front, state.atm, "call", entry),
            "lbc": load_leg_day(back, state.atm, "call", entry),
            "sp": load_leg_day(front, state.put_short, "put", entry),
            "lp": load_leg_day(front, state.put_long, "put", entry),
        }
        minutes = align_package_minutes(day_legs)
        entry_minutes = [
            m
            for m in minutes
            if time(m.hour, m.minute) >= entry_start and time(m.hour, m.minute) <= entry_end
        ]
        if not entry_minutes:
            entry_minutes = [
                m for m in minutes if time(10, 0) <= time(m.hour, m.minute) <= time(15, 55)
            ]
        if not entry_minutes:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_1m_entry"})
            continue

        entry_min = entry_minutes[0]
        legs_e = {
            "sfc": _leg_at_minute(day_legs["sfc"], entry_min, front, state.atm, "call"),
            "lbc": _leg_at_minute(day_legs["lbc"], entry_min, back, state.atm, "call"),
            "sp": _leg_at_minute(day_legs["sp"], entry_min, front, state.put_short, "put"),
            "lp": _leg_at_minute(day_legs["lp"], entry_min, front, state.put_long, "put"),
        }
        if any(v is None for v in legs_e.values()):
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_entry_leg"})
            continue

        atm_put_df = load_leg_day(front, state.atm, "put", entry)
        atm_put = _leg_at_minute(atm_put_df, entry_min, front, state.atm, "put")
        if atm_put is not None:
            spot_entry = spot_from_parity(legs_e["sfc"].mid, atm_put.mid, state.atm)
        else:
            spot_entry = spot0

        dte_f = max((front - entry).days, 1)
        dte_b = max((back - entry).days, 1)
        front_iv = estimate_iv(
            mid=legs_e["sfc"].mid, spot=spot_entry, strike=state.atm, dte_days=dte_f, right="call"
        )
        back_iv = estimate_iv(
            mid=legs_e["lbc"].mid, spot=spot_entry, strike=state.atm, dte_days=dte_b, right="call"
        )
        if iv_filter and front_iv is not None and back_iv is not None:
            if (front_iv - back_iv) > iv_backward_max:
                result.skipped.append(
                    {
                        "entry": entry.isoformat(),
                        "reason": "iv_backwardation",
                        "front_iv": front_iv,
                        "back_iv": back_iv,
                    }
                )
                continue

        entry_debit, cal_e, put_e = package_open_debit(
            legs_e["sfc"],
            legs_e["lbc"],
            legs_e["sp"],
            legs_e["lp"],
            mark_mode=mark_mode,
            lots=lots,
            slip=slip,
        )
        if entry_debit <= 0:
            result.skipped.append(
                {"entry": entry.isoformat(), "reason": "non_positive_debit", "debit": entry_debit}
            )
            continue

        adjust_cash = 0.0
        n_adjusts = 0
        ref_spot = spot_entry
        sess_days = [d for d in spots.index if entry <= d < front][: max_hold_days + 1]
        if not sess_days:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_path"})
            continue

        exit_date = entry
        exit_ts = str(entry_min)
        exit_credit = entry_debit
        exit_reason = "max_hold"
        hold_days = 0
        got = False
        pnl_i = 0.0
        finished = False

        for day in sess_days:
            day_pack = {
                "sfc": load_leg_day(state.front, state.atm, "call", day),
                "lbc": load_leg_day(state.back, state.atm, "call", day),
                "sp": load_leg_day(state.front, state.put_short, "put", day),
                "lp": load_leg_day(state.front, state.put_long, "put", day),
            }
            mins = align_package_minutes(day_pack)
            if day == entry:
                mins = [m for m in mins if m >= entry_min]
            else:
                hold_days += 1
            if not mins:
                continue

            stop_day = False
            for minute in mins:
                legs = {
                    "sfc": _leg_at_minute(day_pack["sfc"], minute, state.front, state.atm, "call"),
                    "lbc": _leg_at_minute(day_pack["lbc"], minute, state.back, state.atm, "call"),
                    "sp": _leg_at_minute(
                        day_pack["sp"], minute, state.front, state.put_short, "put"
                    ),
                    "lp": _leg_at_minute(
                        day_pack["lp"], minute, state.front, state.put_long, "put"
                    ),
                }
                if any(v is None for v in legs.values()):
                    continue

                atm_p = _leg_at_minute(
                    load_leg_day(state.front, state.atm, "put", day),
                    minute,
                    state.front,
                    state.atm,
                    "put",
                )
                if atm_p is not None:
                    spot_now = spot_from_parity(legs["sfc"].mid, atm_p.mid, state.atm)
                else:
                    spot_now = spot_from_parity(legs["sfc"].mid, legs["sp"].mid, state.put_short)

                mark = package_close_credit(
                    legs["sfc"],
                    legs["lbc"],
                    legs["sp"],
                    legs["lp"],
                    mark_mode=mark_mode,
                    lots=lots,
                    slip=slip,
                )
                pnl_i = mark - entry_debit + adjust_cash
                exit_date, exit_credit, exit_ts = day, mark, str(minute)
                got = True

                if pnl_i >= tp_usd:
                    exit_reason = "tp"
                    finished = True
                    stop_day = True
                    break
                if pnl_i <= -sl_usd:
                    exit_reason = "sl"
                    if clamp_sl:
                        exit_credit = entry_debit - adjust_cash - sl_usd
                        pnl_i = -sl_usd
                    finished = True
                    stop_day = True
                    break

                if allow_adjust and n_adjusts < max_adjusts and ref_spot > 0:
                    move = (spot_now - ref_spot) / ref_spot
                    if abs(move) >= adjust_move:
                        new_state, adj_cash, ok = _apply_adjust(
                            client=client,
                            day=day,
                            minute=minute,
                            spot_now=spot_now,
                            state=state,
                            strikes=strikes,
                            legs=legs,
                            mark_mode=mark_mode,
                            slip=slip,
                            lots=lots,
                            put_otm_frac=put_otm_frac,
                            put_width=put_width,
                            move=move,
                            root=root,
                            cache_1m=cache_1m,
                        )
                        if ok and new_state is not None:
                            adjust_cash += adj_cash
                            state = new_state
                            n_adjusts += 1
                            ref_spot = spot_now
                            day_pack = {
                                "sfc": load_leg_day(state.front, state.atm, "call", day),
                                "lbc": load_leg_day(state.back, state.atm, "call", day),
                                "sp": load_leg_day(state.front, state.put_short, "put", day),
                                "lp": load_leg_day(state.front, state.put_long, "put", day),
                            }
                            LOG.info(
                                "%s adjust#%d move=%.2f%% atm=%.0f puts=%.0f/%.0f cash=$%.0f",
                                day,
                                n_adjusts,
                                100 * move,
                                state.atm,
                                state.put_short,
                                state.put_long,
                                adj_cash,
                            )

            if finished or stop_day:
                break
            if hold_days >= max_hold_days and day != entry:
                exit_reason = "max_hold"
                break
            if day == sess_days[-1]:
                exit_reason = "pre_front"
                break

        if not got:
            result.skipped.append({"entry": entry.isoformat(), "reason": "no_path_marks"})
            continue

        pnl = pnl_i if finished else (exit_credit - entry_debit + adjust_cash)
        trade = ProxyTrade(
            entry_date=entry,
            exit_date=exit_date,
            entry_ts=str(entry_min),
            exit_ts=exit_ts,
            spot=spot_entry,
            atm=float(atm),
            front=front,
            back=back,
            put_short=float(put_short),
            put_long=float(put_long),
            lots=lots,
            capital=capital,
            mark_mode=mark_mode,
            entry_debit=entry_debit,
            exit_credit=exit_credit,
            pnl=pnl,
            pnl_pct_capital=100.0 * pnl / capital,
            hold_days=max(1, hold_days if hold_days else 1),
            exit_reason=exit_reason,
            calendar_entry=cal_e,
            put_credit_entry=put_e,
            n_adjusts=n_adjusts,
            front_iv=front_iv,
            back_iv=back_iv,
            slip=slip,
        )
        result.trades.append(trade)
        _log_trade(trade)

    return result


def _apply_adjust(
    *,
    client: ThetaDataClient,
    day: date,
    minute: pd.Timestamp,
    spot_now: float,
    state: StructureState,
    strikes: Sequence[float],
    legs: Dict[str, LegQuote],
    mark_mode: str,
    slip: float,
    lots: int,
    put_otm_frac: float,
    put_width: float,
    move: float,
    root: str,
    cache_1m: Dict[Tuple[Any, ...], pd.DataFrame],
) -> Tuple[Optional[StructureState], float, bool]:
    if move > 0:
        new_short = strike_at_or_below(strikes, spot_now * (1.0 - put_otm_frac))
        if new_short is None or new_short <= state.put_short:
            return None, 0.0, False
        new_long = strike_at_or_below(strikes, new_short - put_width)
        if new_long is None or new_long >= new_short:
            below = [s for s in strikes if s < new_short]
            if not below:
                return None, 0.0, False
            new_long = max(below)
        old_sp, old_lp = legs["sp"], legs["lp"]
        close_cost = (
            fill_px(old_sp.mid, old_sp.bid, old_sp.ask, side="buy", mark_mode=mark_mode, slip=slip)
            - fill_px(old_lp.mid, old_lp.bid, old_lp.ask, side="sell", mark_mode=mark_mode, slip=slip)
        ) * MULT * lots
        sp_df = fetch_1m_day(
            client, expiration=state.front, strike=new_short, right="put", day=day, root=root, cache=cache_1m
        )
        lp_df = fetch_1m_day(
            client, expiration=state.front, strike=new_long, right="put", day=day, root=root, cache=cache_1m
        )
        new_sp = _leg_at_minute(sp_df, minute, state.front, new_short, "put")
        new_lp = _leg_at_minute(lp_df, minute, state.front, new_long, "put")
        if new_sp is None or new_lp is None:
            return None, 0.0, False
        open_credit = (
            fill_px(new_sp.mid, new_sp.bid, new_sp.ask, side="sell", mark_mode=mark_mode, slip=slip)
            - fill_px(new_lp.mid, new_lp.bid, new_lp.ask, side="buy", mark_mode=mark_mode, slip=slip)
        ) * MULT * lots
        cash = open_credit - close_cost
        return (
            StructureState(
                atm=state.atm,
                put_short=float(new_short),
                put_long=float(new_long),
                front=state.front,
                back=state.back,
            ),
            float(cash),
            True,
        )

    new_atm = nearest_strike(strikes, spot_now)
    if new_atm is None or abs(new_atm - state.atm) < 1e-9:
        return None, 0.0, False
    old_sfc, old_lbc = legs["sfc"], legs["lbc"]
    close_cal = (
        fill_px(old_lbc.mid, old_lbc.bid, old_lbc.ask, side="sell", mark_mode=mark_mode, slip=slip)
        - fill_px(old_sfc.mid, old_sfc.bid, old_sfc.ask, side="buy", mark_mode=mark_mode, slip=slip)
    ) * MULT * lots
    sfc_df = fetch_1m_day(
        client, expiration=state.front, strike=new_atm, right="call", day=day, root=root, cache=cache_1m
    )
    lbc_df = fetch_1m_day(
        client, expiration=state.back, strike=new_atm, right="call", day=day, root=root, cache=cache_1m
    )
    new_sfc = _leg_at_minute(sfc_df, minute, state.front, new_atm, "call")
    new_lbc = _leg_at_minute(lbc_df, minute, state.back, new_atm, "call")
    if new_sfc is None or new_lbc is None:
        return None, 0.0, False
    open_cal = (
        fill_px(new_lbc.mid, new_lbc.bid, new_lbc.ask, side="buy", mark_mode=mark_mode, slip=slip)
        - fill_px(new_sfc.mid, new_sfc.bid, new_sfc.ask, side="sell", mark_mode=mark_mode, slip=slip)
    ) * MULT * lots
    cash = close_cal - open_cal
    put_short = state.put_short
    new_long = state.put_long
    new_short = strike_at_or_below(strikes, spot_now * (1.0 - put_otm_frac))
    if new_short is not None and new_short < state.put_short:
        new_long_c = strike_at_or_below(strikes, new_short - put_width)
        if new_long_c is None or new_long_c >= new_short:
            below = [s for s in strikes if s < new_short]
            new_long_c = max(below) if below else None
        if new_long_c is not None:
            old_sp, old_lp = legs["sp"], legs["lp"]
            close_puts = (
                fill_px(old_sp.mid, old_sp.bid, old_sp.ask, side="buy", mark_mode=mark_mode, slip=slip)
                - fill_px(old_lp.mid, old_lp.bid, old_lp.ask, side="sell", mark_mode=mark_mode, slip=slip)
            ) * MULT * lots
            sp_df = fetch_1m_day(
                client, expiration=state.front, strike=new_short, right="put", day=day, root=root, cache=cache_1m
            )
            lp_df = fetch_1m_day(
                client, expiration=state.front, strike=new_long_c, right="put", day=day, root=root, cache=cache_1m
            )
            nsp = _leg_at_minute(sp_df, minute, state.front, new_short, "put")
            nlp = _leg_at_minute(lp_df, minute, state.front, new_long_c, "put")
            if nsp is not None and nlp is not None:
                open_puts = (
                    fill_px(nsp.mid, nsp.bid, nsp.ask, side="sell", mark_mode=mark_mode, slip=slip)
                    - fill_px(nlp.mid, nlp.bid, nlp.ask, side="buy", mark_mode=mark_mode, slip=slip)
                ) * MULT * lots
                cash += open_puts - close_puts
                put_short = float(new_short)
                new_long = float(new_long_c)

    return (
        StructureState(
            atm=float(new_atm),
            put_short=float(put_short),
            put_long=float(new_long),
            front=state.front,
            back=state.back,
        ),
        float(cash),
        True,
    )


def _run_eod_trade(
    *,
    client: ThetaDataClient,
    entry: date,
    spot0: float,
    state: StructureState,
    spots: pd.Series,
    mark_mode: str,
    slip: float,
    lots: int,
    capital: float,
    tp_usd: float,
    sl_usd: float,
    max_hold_days: int,
    clamp_sl: bool,
    root: str,
    iv_filter: bool,
    iv_backward_max: float,
) -> Any:
    cache: Dict[Tuple[Any, ...], Optional[LegQuote]] = {}

    def eod(exp: date, strike: float, right: str, day: date) -> Optional[LegQuote]:
        key = (root, exp, float(strike), right, day)
        if key in cache:
            return cache[key]
        r = client.session.get(
            f"{client.base_url}/v3/option/history/eod",
            params={
                "symbol": root,
                "expiration": exp.isoformat(),
                "strike": f"{float(strike):.3f}",
                "right": "call" if right.lower().startswith("c") else "put",
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
            timeout=90,
        )
        if r.status_code != 200 or not r.content:
            cache[key] = None
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            cache[key] = None
            return None
        bid = float(pd.to_numeric(df.get("bid"), errors="coerce").iloc[-1] or 0)
        ask = float(pd.to_numeric(df.get("ask"), errors="coerce").iloc[-1] or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            cache[key] = None
            return None
        q = LegQuote(exp, float(strike), right, bid, ask)
        cache[key] = q
        return q

    legs_e = {
        "sfc": eod(state.front, state.atm, "call", entry),
        "lbc": eod(state.back, state.atm, "call", entry),
        "sp": eod(state.front, state.put_short, "put", entry),
        "lp": eod(state.front, state.put_long, "put", entry),
    }
    if any(v is None for v in legs_e.values()):
        return {"entry": entry.isoformat(), "reason": "no_entry_quotes"}

    front_iv = estimate_iv(
        mid=legs_e["sfc"].mid,
        spot=spot0,
        strike=state.atm,
        dte_days=max((state.front - entry).days, 1),
        right="call",
    )
    back_iv = estimate_iv(
        mid=legs_e["lbc"].mid,
        spot=spot0,
        strike=state.atm,
        dte_days=max((state.back - entry).days, 1),
        right="call",
    )
    if iv_filter and front_iv is not None and back_iv is not None:
        if (front_iv - back_iv) > iv_backward_max:
            return {
                "entry": entry.isoformat(),
                "reason": "iv_backwardation",
                "front_iv": front_iv,
                "back_iv": back_iv,
            }

    entry_debit, cal_e, put_e = package_open_debit(
        legs_e["sfc"], legs_e["lbc"], legs_e["sp"], legs_e["lp"],
        mark_mode=mark_mode, lots=lots, slip=slip,
    )
    if entry_debit <= 0:
        return {"entry": entry.isoformat(), "reason": "non_positive_debit", "debit": entry_debit}

    sess = [d for d in spots.index if entry < d < state.front]
    if not sess:
        return {"entry": entry.isoformat(), "reason": "no_path"}
    exit_date = sess[0]
    exit_credit = entry_debit
    exit_reason = "max_hold"
    hold = 0
    got = False
    for d in sess:
        hold += 1
        legs_d = {
            "sfc": eod(state.front, state.atm, "call", d),
            "lbc": eod(state.back, state.atm, "call", d),
            "sp": eod(state.front, state.put_short, "put", d),
            "lp": eod(state.front, state.put_long, "put", d),
        }
        if any(v is None for v in legs_d.values()):
            if hold >= max_hold_days:
                break
            continue
        mark = package_close_credit(
            legs_d["sfc"], legs_d["lbc"], legs_d["sp"], legs_d["lp"],
            mark_mode=mark_mode, lots=lots, slip=slip,
        )
        pnl_i = mark - entry_debit
        exit_date, exit_credit = d, mark
        got = True
        if pnl_i >= tp_usd:
            exit_reason = "tp"
            break
        if pnl_i <= -sl_usd:
            exit_reason = "sl"
            if clamp_sl:
                exit_credit = entry_debit - sl_usd
            break
        if hold >= max_hold_days:
            exit_reason = "max_hold"
            break
        if d == sess[-1]:
            exit_reason = "pre_front"
            break
    if not got:
        return {"entry": entry.isoformat(), "reason": "no_path_marks"}
    pnl = exit_credit - entry_debit
    return ProxyTrade(
        entry_date=entry,
        exit_date=exit_date,
        entry_ts=None,
        exit_ts=None,
        spot=spot0,
        atm=state.atm,
        front=state.front,
        back=state.back,
        put_short=state.put_short,
        put_long=state.put_long,
        lots=lots,
        capital=capital,
        mark_mode=mark_mode,
        entry_debit=entry_debit,
        exit_credit=exit_credit,
        pnl=pnl,
        pnl_pct_capital=100.0 * pnl / capital,
        hold_days=max(1, hold),
        exit_reason=exit_reason,
        calendar_entry=cal_e,
        put_credit_entry=put_e,
        front_iv=front_iv,
        back_iv=back_iv,
        slip=slip,
    )


def _log_trade(t: ProxyTrade) -> None:
    LOG.info(
        "%s %s entry=$%.0f pnl=$%.0f (%.1f%%) hold=%d %s atm=%.0f adj=%d slip=%.2f",
        t.entry_date,
        t.mark_mode,
        t.entry_debit,
        t.pnl,
        t.pnl_pct_capital,
        t.hold_days,
        t.exit_reason,
        t.atm,
        t.n_adjusts,
        t.slip,
    )


def book_stats(trades: Sequence[ProxyTrade], capital: float = DEFAULT_CAPITAL) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "wr": None,
            "pf": None,
            "pnl": 0.0,
            "ret_pct_capital": 0.0,
            "avg_pnl": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_hold": None,
            "avg_adjusts": None,
            "tp_n": 0,
            "sl_n": 0,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (None if gross_win <= 0 else float("inf"))
    return {
        "n": int(len(trades)),
        "wr": float((pnls > 0).mean()),
        "pf": pf,
        "pnl": float(pnls.sum()),
        "ret_pct_capital": float(100.0 * pnls.sum() / capital),
        "avg_pnl": float(pnls.mean()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_hold": float(np.mean([t.hold_days for t in trades])),
        "avg_adjusts": float(np.mean([t.n_adjusts for t in trades])),
        "tp_n": int(sum(1 for t in trades if t.exit_reason == "tp")),
        "sl_n": int(sum(1 for t in trades if t.exit_reason == "sl")),
        "exit_reasons": {
            k: int(sum(1 for t in trades if t.exit_reason == k))
            for k in sorted({t.exit_reason for t in trades})
        },
    }


def trades_to_frame(trades: Sequence[ProxyTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "spot": t.spot,
                "atm": t.atm,
                "front": t.front.isoformat(),
                "back": t.back.isoformat(),
                "put_short": t.put_short,
                "put_long": t.put_long,
                "mark_mode": t.mark_mode,
                "slip": t.slip,
                "entry_debit": t.entry_debit,
                "exit_credit": t.exit_credit,
                "calendar_entry": t.calendar_entry,
                "put_credit_entry": t.put_credit_entry,
                "pnl": t.pnl,
                "pnl_pct_capital": t.pnl_pct_capital,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "n_adjusts": t.n_adjusts,
                "front_iv": t.front_iv,
                "back_iv": t.back_iv,
            }
        )
    return pd.DataFrame(rows)
