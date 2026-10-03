"""Earnings expected-move premium sale (Options With Ravish), multi-name.

Video: https://youtu.be/FtX1JshVLvg (2026-02-24)

Shared clocks and strike helpers. NVDA-only claim constants stay in
``nvda_earnings_em``. Walk-forward picks a structure from trailing stock
facts, then prices only the next report.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient

LOG = logging.getLogger(__name__)

MULT = 100.0
COMM_PER_LEG = 0.65
WIDTH_PCT = 5.0 / 190.0
EXAMPLE_WIDTH = 5.0

ENTRY_CLOCK = dtime(15, 45)
ENTRY_END = dtime(15, 50)
EXIT_CLOCK = dtime(9, 30)
EXIT_END = dtime(9, 40)

TRAIL_WINDOW = 12
TRAIL_MIN = 6
INSIDE_MIN = 0.75

DEFAULT_UNIVERSE: Tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "AVGO",
    "AMD",
    "NFLX",
    "COST",
    "CRM",
)

MODES: Tuple[Tuple[str, float], ...] = (
    ("mid", 0.0),
    ("mid", 0.05),
    ("mid", 0.10),
    ("natural", 0.0),
)

STRUCT_PCS = "pcs"
STRUCT_IC = "ic"
STRUCT_SKIP = "skip"

Split = Tuple[date, float]


def mode_key(mode: str, slip: float) -> str:
    if mode == "natural":
        return "natural"
    if slip <= 0:
        return "mid"
    return "mid_slip%g" % slip


def pick_expiry_after(expirations: Sequence[date], day: date) -> Optional[date]:
    """First listed expiry strictly after an after-close report."""
    later = [e for e in expirations if e > day]
    return min(later) if later else None


def pick_expiry_for_report(
    expirations: Sequence[date],
    earn: date,
    *,
    amc: bool,
) -> Optional[date]:
    """First listed expiry after the print: > earn for AMC, >= earn for BMO."""
    if amc:
        return pick_expiry_after(expirations, earn)
    later = [e for e in expirations if e >= earn]
    return min(later) if later else None


def nearest_strike(strikes: Sequence[float], spot: float) -> Optional[float]:
    if not strikes or spot <= 0:
        return None
    return float(min(strikes, key=lambda s: (abs(float(s) - float(spot)), float(s))))


def strike_just_below(strikes: Sequence[float], level: float) -> Optional[float]:
    below = [float(s) for s in strikes if float(s) < float(level) - 1e-6]
    return max(below) if below else None


def strike_just_above(strikes: Sequence[float], level: float) -> Optional[float]:
    above = [float(s) for s in strikes if float(s) > float(level) + 1e-6]
    return min(above) if above else None


def target_wing(spot: float) -> float:
    """Dollar wing matching 5 points when the stock is $190."""
    return max(0.5, float(spot) * WIDTH_PCT)


def wing_strike(
    strikes: Sequence[float],
    short: float,
    width: float,
    right: str,
) -> Optional[float]:
    """Listed hedge closest to ``width`` points beyond ``short``."""
    right_s = "put" if str(right).lower().startswith("p") else "call"
    if right_s == "put":
        target = float(short) - float(width)
        cands = [float(s) for s in strikes if float(s) < float(short) - 1e-6]
        if not cands:
            return None
        return min(cands, key=lambda s: (abs(s - target), s))
    target = float(short) + float(width)
    cands = [float(s) for s in strikes if float(s) > float(short) + 1e-6]
    if not cands:
        return None
    return min(cands, key=lambda s: (abs(s - target), -s))


def usable_quote(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    try:
        bid = float(raw.get("bid") or 0)
        ask = float(raw.get("ask") or 0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask + 1e-9 < bid:
        return None
    mid = 0.5 * (bid + ask)
    return {"bid": bid, "ask": ask, "mid": mid, "spread": ask - bid}


def synthetic_forward(strike: float, call_q: Dict[str, float], put_q: Dict[str, float]) -> float:
    return float(strike) + float(call_q["mid"]) - float(put_q["mid"])


def sell_price(q: Dict[str, float], mode: str, slip: float) -> float:
    if mode == "natural":
        return float(q["bid"])
    px = float(q["mid"]) - float(slip)
    return min(float(q["ask"]), max(float(q["bid"]), px))


def buy_price(q: Dict[str, float], mode: str, slip: float) -> float:
    if mode == "natural":
        return float(q["ask"])
    px = float(q["mid"]) + float(slip)
    return min(float(q["ask"]), max(float(q["bid"]), px))


def package_pnl(
    legs: Sequence[Dict[str, Any]],
    mode: str,
    slip: float,
    *,
    comm_per_leg: float = COMM_PER_LEG,
) -> Dict[str, float]:
    credit = 0.0
    debit = 0.0
    for leg in legs:
        entry = leg["entry"]
        exit_q = leg["exit"]
        if leg["role"] == "short":
            credit += sell_price(entry, mode, slip)
            debit += buy_price(exit_q, mode, slip)
        else:
            credit -= buy_price(entry, mode, slip)
            debit -= sell_price(exit_q, mode, slip)
    fees = float(comm_per_leg) * len(legs) * 2.0
    gross = credit - debit
    return {
        "credit": credit,
        "exit_debit": debit,
        "gross_ps": gross,
        "fees": fees,
        "net": gross * MULT - fees,
    }


def defined_risk(width: float, credit: float) -> float:
    return float(width) - float(credit)


def put_intrinsic(strike: float, spot: float) -> float:
    return max(0.0, float(strike) - float(spot))


def call_intrinsic(strike: float, spot: float) -> float:
    return max(0.0, float(spot) - float(strike))


def spread_intrinsic(short_k: float, long_k: Optional[float], spot: float, right: str) -> float:
    right_s = "put" if str(right).lower().startswith("p") else "call"
    if right_s == "put":
        short_i = put_intrinsic(short_k, spot)
        long_i = put_intrinsic(long_k, spot) if long_k is not None else 0.0
    else:
        short_i = call_intrinsic(short_k, spot)
        long_i = call_intrinsic(long_k, spot) if long_k is not None else 0.0
    return short_i - long_i


def amc_sessions(
    earn: date,
    sessions: Sequence[date],
) -> Optional[Tuple[date, date]]:
    """AMC: entry on the report session, exit the next session."""
    days = sorted(set(sessions))
    if earn not in days:
        prior = [d for d in days if d <= earn]
        if not prior:
            return None
        entry = prior[-1]
    else:
        entry = earn
    later = [d for d in days if d > entry]
    if not later:
        return None
    return entry, later[0]


def bmo_sessions(
    earn: date,
    sessions: Sequence[date],
) -> Optional[Tuple[date, date]]:
    """BMO: entry the prior session 15:45, exit the report-morning open."""
    days = sorted(set(sessions))
    exit_day = earn if earn in days else None
    if exit_day is None:
        later = [d for d in days if d >= earn]
        if not later:
            return None
        exit_day = later[0]
    prior = [d for d in days if d < exit_day]
    if not prior:
        return None
    return prior[-1], exit_day


def earn_sessions(
    earn: date,
    sessions: Sequence[date],
    *,
    amc: bool,
) -> Optional[Tuple[date, date]]:
    if amc:
        return amc_sessions(earn, sessions)
    return bmo_sessions(earn, sessions)


def classify_open(
    *,
    prior_close: float,
    next_open: float,
    spot: float,
    em: float,
) -> Dict[str, Any]:
    gap = float(next_open) / float(prior_close) - 1.0 if prior_close else None
    down = float(spot) - float(em)
    up = float(spot) + float(em)
    return {
        "gap_pct": None if gap is None else 100.0 * gap,
        "opened_green": bool(next_open >= prior_close - 1e-9),
        "inside_em": bool(down - 1e-9 <= next_open <= up + 1e-9),
        "broke_down": bool(next_open < down - 1e-6),
        "broke_up": bool(next_open > up + 1e-6),
        "em_down": down,
        "em_up": up,
    }


def held_short_put(next_open: float, short_k: float) -> bool:
    return float(next_open) >= float(short_k) - 1e-9


def held_iron_condor(next_open: float, short_put: float, short_call: float) -> bool:
    return float(short_put) - 1e-9 <= float(next_open) <= float(short_call) + 1e-9


def book_stats(rows: Sequence[Dict[str, Any]], pnl_key: str) -> Dict[str, Any]:
    vals = []
    rors = []
    for row in rows:
        if row.get(pnl_key) is None:
            continue
        vals.append(float(row[pnl_key]))
        rkey = pnl_key.replace("net_", "ror_")
        if row.get(rkey) is not None:
            rors.append(float(row[rkey]))
    n = len(vals)
    if n == 0:
        return {"n": 0}
    wins = [v for v in vals if v > 1e-9]
    losses = [v for v in vals if v < -1e-9]
    gross_win = sum(wins)
    gross_loss = sum(losses)
    pf = None
    if gross_loss < 0:
        pf = gross_win / abs(gross_loss)
    return {
        "n": n,
        "win_rate_pct": round(100.0 * len(wins) / n, 1),
        "mean_pnl": round(sum(vals) / n, 2),
        "med_pnl": round(float(pd.Series(vals).median()), 2),
        "total_pnl": round(sum(vals), 2),
        "pf": None if pf is None else round(pf, 2),
        "mean_ror_pct": None if not rors else round(sum(rors) / len(rors), 2),
        "avg_credit": None,
    }


def parse_earn_date(ts: Any) -> Optional[date]:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("America/New_York")
    else:
        t = t.tz_convert("America/New_York")
    return t.date()


def is_amc(ts: Any) -> bool:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("America/New_York")
    else:
        t = t.tz_convert("America/New_York")
    return int(t.hour) >= 16


def unadjusted_price(session: date, adjusted: float, splits: Sequence[Split]) -> float:
    """Map a split-adjusted print onto the listed-strike scale of that session."""
    factor = 1.0
    for split_day, ratio in splits:
        if session < split_day and float(ratio) > 0:
            factor *= float(ratio)
    return float(adjusted) * factor


def yahoo_splits(symbol: str) -> List[Split]:
    import yfinance as yf

    raw = yf.Ticker(str(symbol).upper()).splits
    if raw is None or len(raw) == 0:
        return []
    out: List[Split] = []
    for ts, ratio in raw.items():
        d = parse_earn_date(ts)
        if d is None:
            continue
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            continue
        if r <= 0:
            continue
        out.append((d, r))
    out.sort(key=lambda x: x[0])
    return out


def load_stock(
    symbol: str,
    start: date,
    end: date,
    splits: Optional[Sequence[Split]] = None,
) -> pd.DataFrame:
    """Yahoo OHLC, scaled back onto the listed-strike price."""
    import yfinance as yf

    raw = yf.download(
        str(symbol).upper(),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["session", "open", "high", "low", "close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.reset_index()
    out = pd.DataFrame(
        {
            "session": pd.to_datetime(frame["Date"]).dt.date,
            "open": pd.to_numeric(frame["Open"], errors="coerce"),
            "high": pd.to_numeric(frame["High"], errors="coerce"),
            "low": pd.to_numeric(frame["Low"], errors="coerce"),
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["session", "open", "close"])
    if splits is None:
        splits = yahoo_splits(symbol)
    for col in ("open", "high", "low", "close"):
        out[col] = [
            unadjusted_price(session, px, splits)
            for session, px in zip(out["session"], out[col])
        ]
    return out.reset_index(drop=True)


def fetch_earnings(symbol: str, limit: int = 40) -> List[Dict[str, Any]]:
    import yfinance as yf

    frame = yf.Ticker(str(symbol).upper()).get_earnings_dates(limit=int(limit))
    if frame is None or frame.empty:
        return []
    out: List[Dict[str, Any]] = []
    for ts, row in frame.iterrows():
        reported = row.get("Reported EPS")
        if reported is None or reported != reported:
            continue
        d = parse_earn_date(ts)
        if d is None:
            continue
        out.append(
            {
                "symbol": str(symbol).upper(),
                "earn_date": d,
                "amc": bool(is_amc(ts)),
            }
        )
    out.sort(key=lambda r: r["earn_date"])
    return out


def fetch_clock_nbbo(
    client: ThetaDataClient,
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    clock: dtime,
    end: dtime,
) -> Optional[Dict[str, float]]:
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            raw = client.nbbo_at_or_after(
                symbol=symbol,
                expiration=expiration,
                strike=float(strike),
                right=right,
                day=day,
                clock=clock,
                end_time=end.strftime("%H:%M:%S"),
            )
            return usable_quote(raw)
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "429" in msg or "478" in msg or "timed out" in msg.lower():
                time.sleep(1.5 * (attempt + 1))
                continue
            LOG.warning("nbbo fail %s %s %s %s: %s", symbol, expiration, strike, day, exc)
            return None
    if last_err:
        LOG.warning("nbbo retries exhausted %s %s %s: %s", symbol, strike, day, last_err)
    return None


def quote_key(
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    which: str,
) -> str:
    return "%s|%s|%.3f|%s|%s|%s" % (
        str(symbol).upper(),
        expiration.isoformat(),
        float(strike),
        right[:1].upper(),
        day.isoformat(),
        which,
    )


def load_quote_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("cache read failed: %s", exc)
        return {}


def save_quote_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def cached_quote(
    client: ThetaDataClient,
    cache: Dict[str, Any],
    *,
    symbol: str,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    which: str,
) -> Optional[Dict[str, float]]:
    key = quote_key(symbol, expiration, strike, right, day, which)
    if key in cache:
        raw = cache[key]
        return None if not raw else raw
    clock = ENTRY_CLOCK if which == "entry" else EXIT_CLOCK
    end = ENTRY_END if which == "entry" else EXIT_END
    q = fetch_clock_nbbo(
        client,
        symbol=symbol,
        expiration=expiration,
        strike=float(strike),
        right=right,
        day=day,
        clock=clock,
        end=end,
    )
    cache[key] = q
    return q


def strikes_retry(client: ThetaDataClient, symbol: str, expiration: date) -> List[float]:
    from utils.research.forward_factor import list_strikes

    last: Optional[Exception] = None
    for attempt in range(4):
        try:
            return list_strikes(client, str(symbol).upper(), expiration)
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    if last:
        LOG.warning("strikes failed %s %s: %s", symbol, expiration, last)
    return []


def trailing_stats(
    prior: Sequence[Dict[str, Any]],
    *,
    window: int = TRAIL_WINDOW,
    min_n: int = TRAIL_MIN,
) -> Dict[str, Any]:
    """Containment on completed reports before the event being classified."""
    use = list(prior)[-int(window) :]
    n = len(use)
    if n < int(min_n):
        return {"n": n, "ready": False}
    inside = sum(1 for r in use if r.get("inside_em"))
    broke_down = sum(1 for r in use if r.get("broke_down"))
    opened_green = sum(1 for r in use if r.get("opened_green"))
    broke_up = sum(1 for r in use if r.get("broke_up"))
    return {
        "n": n,
        "ready": True,
        "inside": inside,
        "inside_frac": inside / n,
        "broke_down": broke_down,
        "opened_green": opened_green,
        "broke_up": broke_up,
    }


def pick_structure(
    trail: Dict[str, Any],
    *,
    inside_min: float = INSIDE_MIN,
) -> str:
    """Causal structure for the *next* report. Stock facts only."""
    if not trail.get("ready"):
        return STRUCT_SKIP
    inside_frac = float(trail.get("inside_frac") or 0.0)
    broke_down = int(trail.get("broke_down") or 0)
    if inside_frac + 1e-12 < float(inside_min):
        return STRUCT_SKIP
    if broke_down == 0:
        return STRUCT_PCS
    return STRUCT_IC


def attach_trailing(
    rows: Sequence[Dict[str, Any]],
    *,
    window: int = TRAIL_WINDOW,
    min_n: int = TRAIL_MIN,
    inside_min: float = INSIDE_MIN,
) -> List[Dict[str, Any]]:
    """Fill trail_* and wf_structure from prior events of the same symbol."""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda r: (str(r.get("symbol") or ""), str(r.get("earn_date") or "")),
    )
    for row in ordered:
        item = dict(row)
        sym = str(item.get("symbol") or "")
        prior = by_sym.setdefault(sym, [])
        trail = trailing_stats(prior, window=window, min_n=min_n)
        item["trail_n"] = trail.get("n")
        item["trail_ready"] = bool(trail.get("ready"))
        item["trail_inside"] = trail.get("inside")
        item["trail_inside_frac"] = (
            None if trail.get("inside_frac") is None else round(float(trail["inside_frac"]), 4)
        )
        item["trail_broke_down"] = trail.get("broke_down")
        item["trail_opened_green"] = trail.get("opened_green")
        item["wf_structure"] = pick_structure(trail, inside_min=inside_min)
        out.append(item)
        prior.append(item)
    return out


def _ror(net: float, risk_ps: float) -> Optional[float]:
    if risk_ps <= 0.05:
        return None
    return 100.0 * float(net) / (float(risk_ps) * 100.0)


def mark_row(
    legs: Sequence[Dict[str, Any]],
    *,
    risk_width: Optional[float],
    naked_strike: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mode, slip in MODES:
        key = mode_key(mode, slip)
        pnl = package_pnl(legs, mode, slip, comm_per_leg=COMM_PER_LEG)
        risk_ps = None
        if risk_width is not None:
            risk_ps = defined_risk(risk_width, pnl["credit"])
        elif naked_strike is not None:
            risk_ps = float(naked_strike) - pnl["credit"]
        out["credit_%s" % key] = round(pnl["credit"], 4)
        out["debit_%s" % key] = round(pnl["exit_debit"], 4)
        out["net_%s" % key] = round(pnl["net"], 2)
        ror = None if risk_ps is None else _ror(pnl["net"], risk_ps)
        out["ror_%s" % key] = None if ror is None else round(ror, 2)
        if key == "mid":
            out["risk_ps_mid"] = None if risk_ps is None else round(risk_ps, 4)
    return out


def legs_from(
    pairs: Sequence[Tuple[str, float, str]],
    quotes_entry: Dict[Tuple[float, str], Dict[str, float]],
    quotes_exit: Dict[Tuple[float, str], Dict[str, float]],
) -> Optional[List[Dict[str, Any]]]:
    legs = []
    for role, strike, right in pairs:
        key = (float(strike), right)
        entry = quotes_entry.get(key)
        exit_q = quotes_exit.get(key)
        if not entry or not exit_q:
            return None
        legs.append({"role": role, "entry": entry, "exit": exit_q})
    return legs


def atm_straddle_em(
    client: ThetaDataClient,
    cache: Dict[str, Any],
    *,
    symbol: str,
    expiry: date,
    strikes: Sequence[float],
    seed_spot: float,
    entry_day: date,
) -> Optional[Dict[str, Any]]:
    """15:45 ATM call+put of the first post-report expiry. Not report-day EOD."""
    atm = nearest_strike(strikes, seed_spot)
    if atm is None:
        return None

    def grab(strike: float, right: str):
        return cached_quote(
            client,
            cache,
            symbol=symbol,
            expiration=expiry,
            strike=strike,
            right=right,
            day=entry_day,
            which="entry",
        )

    call_q = grab(atm, "call")
    put_q = grab(atm, "put")
    if not call_q or not put_q:
        return None
    spot = synthetic_forward(atm, call_q, put_q)
    atm2 = nearest_strike(strikes, spot)
    if atm2 is not None and abs(atm2 - atm) > 1e-6:
        call2 = grab(atm2, "call")
        put2 = grab(atm2, "put")
        if call2 and put2:
            atm = atm2
            call_q, put_q = call2, put2
            spot = synthetic_forward(atm, call_q, put_q)
    em = float(call_q["mid"]) + float(put_q["mid"])
    if em <= 0 or spot <= 0:
        return None
    return {"atm": float(atm), "spot": float(spot), "em": float(em), "call": call_q, "put": put_q}


def inventory_event(
    client: ThetaDataClient,
    cache: Dict[str, Any],
    *,
    symbol: str,
    event: Dict[str, Any],
    stock: pd.DataFrame,
    expirations: Sequence[date],
) -> Optional[Dict[str, Any]]:
    """ATM 15:45 expected-move row. No wings, no exit quotes."""
    earn: date = event["earn_date"]
    amc = bool(event.get("amc", True))
    sessions = list(stock["session"])
    pair = earn_sessions(earn, sessions, amc=amc)
    if pair is None:
        LOG.info("%s %s skipped: no entry/exit session", symbol, earn)
        return None
    entry, exit_day = pair
    if entry not in set(stock["session"]) or exit_day not in set(stock["session"]):
        LOG.info("%s %s skipped: missing OHLC", symbol, earn)
        return None
    entry_row = stock.loc[stock["session"] == entry].iloc[-1]
    exit_row = stock.loc[stock["session"] == exit_day].iloc[-1]
    prior_close = float(entry_row["close"])
    next_open = float(exit_row["open"])
    expiry = pick_expiry_for_report(expirations, earn, amc=amc)
    if expiry is None:
        LOG.info("%s %s skipped: no expiry", symbol, earn)
        return None
    strikes = strikes_retry(client, symbol, expiry)
    if not strikes:
        LOG.info("%s %s skipped: no strikes %s", symbol, earn, expiry)
        return None
    em_pack = atm_straddle_em(
        client,
        cache,
        symbol=symbol,
        expiry=expiry,
        strikes=strikes,
        seed_spot=prior_close,
        entry_day=entry,
    )
    if em_pack is None:
        LOG.info("%s %s skipped: ATM 15:45 quote missing", symbol, earn)
        return None
    spot = float(em_pack["spot"])
    em = float(em_pack["em"])
    flags = classify_open(prior_close=prior_close, next_open=next_open, spot=spot, em=em)
    em_put = strike_just_below(strikes, spot - em)
    em_call = strike_just_above(strikes, spot + em)
    row = {
        "symbol": str(symbol).upper(),
        "earn_date": earn.isoformat() if isinstance(earn, date) else str(earn),
        "amc": amc,
        "session": "amc" if amc else "bmo",
        "entry_date": entry.isoformat(),
        "exit_date": exit_day.isoformat(),
        "expiry": expiry.isoformat(),
        "prior_close": round(prior_close, 4),
        "next_open": round(next_open, 4),
        "spot_1545": round(spot, 4),
        "atm": em_pack["atm"],
        "em": round(em, 4),
        "em_pct": round(100.0 * em / spot, 2) if spot else None,
        "gap_pct": None if flags["gap_pct"] is None else round(flags["gap_pct"], 2),
        "opened_green": flags["opened_green"],
        "inside_em": flags["inside_em"],
        "broke_down": flags["broke_down"],
        "broke_up": flags["broke_up"],
        "em_put": em_put,
        "em_call": em_call,
    }
    LOG.info(
        "%s %s %s spot=%.2f EM=%.2f%% open=%.2f gap=%+.2f%% inside=%s down=%s",
        symbol,
        earn,
        "AMC" if amc else "BMO",
        spot,
        row["em_pct"] or 0.0,
        next_open,
        row["gap_pct"] or 0.0,
        row["inside_em"],
        row["broke_down"],
    )
    return row


def price_structure(
    client: ThetaDataClient,
    cache: Dict[str, Any],
    *,
    symbol: str,
    inventory: Dict[str, Any],
    structure: str,
    stock: pd.DataFrame,
    expirations: Sequence[date],
) -> Optional[Dict[str, Any]]:
    """Price expected-move PCS or iron condor at 15:45 / next 09:30."""
    if structure not in (STRUCT_PCS, STRUCT_IC):
        return None
    earn = date.fromisoformat(str(inventory["earn_date"])[:10])
    amc = bool(inventory.get("amc", True))
    sessions = list(stock["session"])
    pair = earn_sessions(earn, sessions, amc=amc)
    if pair is None:
        return None
    entry, exit_day = pair
    expiry = date.fromisoformat(str(inventory["expiry"])[:10])
    strikes = strikes_retry(client, symbol, expiry)
    if not strikes:
        return None
    spot = float(inventory["spot_1545"])
    next_open = float(inventory["next_open"])

    def _strike(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return float(val)

    em_put = _strike(inventory.get("em_put"))
    em_call = _strike(inventory.get("em_call"))
    if em_put is None:
        em_put = strike_just_below(strikes, spot - float(inventory["em"]))
    if em_call is None:
        em_call = strike_just_above(strikes, spot + float(inventory["em"]))
    if em_put is None:
        return None
    wing = target_wing(spot)
    long_p = wing_strike(strikes, float(em_put), wing, "put")
    if long_p is None:
        return None
    pairs: List[Tuple[str, float, str]] = [
        ("short", float(em_put), "put"),
        ("long", float(long_p), "put"),
    ]
    width = float(em_put) - float(long_p)
    if structure == STRUCT_IC:
        if em_call is None:
            return None
        long_c = wing_strike(strikes, float(em_call), wing, "call")
        if long_c is None:
            return None
        pairs.extend(
            [
                ("short", float(em_call), "call"),
                ("long", float(long_c), "call"),
            ]
        )
        width = max(width, float(long_c) - float(em_call))

    entry_q: Dict[Tuple[float, str], Dict[str, float]] = {}
    exit_q: Dict[Tuple[float, str], Dict[str, float]] = {}
    for role, strike, right in pairs:
        qe = cached_quote(
            client,
            cache,
            symbol=symbol,
            expiration=expiry,
            strike=strike,
            right=right,
            day=entry,
            which="entry",
        )
        qx = cached_quote(
            client,
            cache,
            symbol=symbol,
            expiration=expiry,
            strike=strike,
            right=right,
            day=exit_day,
            which="exit",
        )
        if qe:
            entry_q[(float(strike), right)] = qe
        if qx:
            exit_q[(float(strike), right)] = qx
    legs = legs_from(pairs, entry_q, exit_q)
    row = dict(inventory)
    row["wf_structure"] = structure
    row["width"] = round(width, 4)
    row["strikes"] = "/".join(
        "%s%.1f%s" % ("S" if role == "short" else "L", strike, right[:1].upper())
        for role, strike, right in pairs
    )
    if legs is None:
        row["missing_quote"] = True
        return row
    row["missing_quote"] = False
    row.update(mark_row(legs, risk_width=width))
    if structure == STRUCT_PCS:
        row["spot_held"] = held_short_put(next_open, float(em_put))
    else:
        row["spot_held"] = held_iron_condor(next_open, float(em_put), float(em_call))
    return row
