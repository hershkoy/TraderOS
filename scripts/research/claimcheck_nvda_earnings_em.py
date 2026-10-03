#!/usr/bin/env python3
"""Claim-check the Ravish NVDA earnings expected-move sale on Theta NBBO.

Video: https://youtu.be/FtX1JshVLvg
Entry 15:45 ET on the report day, exit first 09:30 ET NBBO the next session.
Spot seed is Theta unadjusted stock EOD (intraday stock quotes are 403 here).
The 15:45 spot used for the expected move is put-call parity on that expiry.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_nvda_earnings_em.py
  python scripts\\research\\claimcheck_nvda_earnings_em.py --only 2025-11-19
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations, list_strikes  # noqa: E402
from utils.research.nvda_earnings_em import (  # noqa: E402
    CLAIMED,
    COMM_PER_LEG,
    EXAMPLE_WIDTH,
    MODES,
    amc_sessions,
    book_stats,
    classify_open,
    defined_risk,
    fetch_clock_nbbo,
    held_iron_condor,
    held_short_put,
    is_amc,
    mode_key,
    nearest_strike,
    package_pnl,
    parse_earn_date,
    pick_expiry_after,
    put_intrinsic,
    spread_intrinsic,
    strike_just_above,
    strike_just_below,
    synthetic_forward,
    target_wing,
    unadjusted_price,
    wing_strike,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_nvda_earnings_em")

DEFAULT_OUT = ROOT / "reports" / "research" / "nvda_earnings_em"
CLAIM_START = date(2023, 2, 22)
CLAIM_END = date(2025, 11, 19)
ENTRY_CLOCK = dtime(15, 45)
ENTRY_END = dtime(15, 50)
EXIT_CLOCK = dtime(9, 30)
EXIT_END = dtime(9, 40)

STRUCTURES = (
    "short_put_em",
    "pcs_em",
    "pcs_em_w5",
    "ic_em",
    "ic_em_w5",
    "short_put_near",
    "pcs_near",
)


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _load_stock(start: date, end: date):
    """Split-adjusted Yahoo OHLC, scaled back onto the listed-strike price."""
    import yfinance as yf

    import pandas as pd

    raw = yf.download(
        "NVDA",
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
    for col in ("open", "high", "low", "close"):
        out[col] = [
            unadjusted_price(session, px)
            for session, px in zip(out["session"], out[col])
        ]
    return out.reset_index(drop=True)


def _earnings(limit: int) -> List[Dict[str, Any]]:
    import yfinance as yf

    frame = yf.Ticker("NVDA").get_earnings_dates(limit=int(limit))
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
        out.append({"earn_date": d, "amc": bool(is_amc(ts))})
    out.sort(key=lambda r: r["earn_date"])
    return out


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("cache read failed: %s", exc)
        return {}


def _quote_key(expiration: date, strike: float, right: str, day: date, which: str) -> str:
    return "%s|%.3f|%s|%s|%s" % (
        expiration.isoformat(),
        float(strike),
        right[:1].upper(),
        day.isoformat(),
        which,
    )


def _cached_quote(
    client,
    cache: Dict[str, Any],
    *,
    expiration: date,
    strike: float,
    right: str,
    day: date,
    which: str,
) -> Optional[Dict[str, float]]:
    key = _quote_key(expiration, strike, right, day, which)
    if key in cache:
        raw = cache[key]
        return None if not raw else raw
    clock = ENTRY_CLOCK if which == "entry" else EXIT_CLOCK
    end = ENTRY_END if which == "entry" else EXIT_END
    q = fetch_clock_nbbo(
        client,
        symbol="NVDA",
        expiration=expiration,
        strike=float(strike),
        right=right,
        day=day,
        clock=clock,
        end=end,
    )
    cache[key] = q
    return q


def _strikes_retry(client, expiration: date) -> List[float]:
    last: Optional[Exception] = None
    for attempt in range(4):
        try:
            return list_strikes(client, "NVDA", expiration)
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    if last:
        LOG.warning("strikes failed %s: %s", expiration, last)
    return []


def _ror(net: float, risk_ps: float) -> Optional[float]:
    if risk_ps <= 0.05:
        return None
    return 100.0 * float(net) / (float(risk_ps) * 100.0)


def _mark_row(
    legs: Sequence[Dict[str, Any]],
    *,
    risk_width: Optional[float],
    naked_strike: Optional[float],
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


def _legs_from(
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


def _intrinsic_net(credit: float, debit: float, n_legs: int) -> float:
    fees = COMM_PER_LEG * n_legs * 2.0
    return (float(credit) - float(debit)) * 100.0 - fees


def _build_row(
    *,
    earn: date,
    entry: date,
    exit_day: date,
    expiry: date,
    prior_close: float,
    next_open: float,
    spot: float,
    em: float,
    atm: float,
    structures: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    flags = classify_open(prior_close=prior_close, next_open=next_open, spot=spot, em=em)
    row: Dict[str, Any] = {
        "earn_date": earn.isoformat(),
        "entry_date": entry.isoformat(),
        "exit_date": exit_day.isoformat(),
        "expiry": expiry.isoformat(),
        "sample": "claim" if CLAIM_START <= earn <= CLAIM_END else "oos",
        "prior_close": round(prior_close, 4),
        "next_open": round(next_open, 4),
        "spot_1545": round(spot, 4),
        "atm": atm,
        "em": round(em, 4),
        "em_pct": round(100.0 * em / spot, 2) if spot else None,
        "gap_pct": None if flags["gap_pct"] is None else round(flags["gap_pct"], 2),
        "opened_green": flags["opened_green"],
        "inside_em": flags["inside_em"],
        "broke_down": flags["broke_down"],
        "broke_up": flags["broke_up"],
    }
    for name, payload in structures.items():
        prefix = name + "_"
        for k, v in payload.items():
            row[prefix + k] = v
    return row


def _one_event(client, cache, stock, expirations, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    earn: date = event["earn_date"]
    if not event["amc"]:
        LOG.info("%s skipped: before-open report is not in this NVDA sample", earn)
        return None
    sessions = list(stock["session"])
    pair = amc_sessions(earn, sessions)
    if pair is None:
        LOG.info("%s skipped: no next session", earn)
        return None
    entry, exit_day = pair
    entry_row = stock.loc[stock["session"] == entry].iloc[-1]
    exit_row = stock.loc[stock["session"] == exit_day].iloc[-1]
    prior_close = float(entry_row["close"])
    next_open = float(exit_row["open"])
    expiry = pick_expiry_after(expirations, earn)
    if expiry is None:
        LOG.info("%s skipped: no expiry", earn)
        return None
    strikes = _strikes_retry(client, expiry)
    if not strikes:
        LOG.info("%s skipped: no strikes %s", earn, expiry)
        return None
    atm = nearest_strike(strikes, prior_close)
    if atm is None:
        return None

    def grab(strike: float, right: str, which: str):
        return _cached_quote(
            client, cache, expiration=expiry, strike=strike, right=right, day=entry if which == "entry" else exit_day, which=which
        )

    call_q = grab(atm, "call", "entry")
    put_q = grab(atm, "put", "entry")
    if not call_q or not put_q:
        LOG.info("%s skipped: ATM quote missing at %s", earn, atm)
        return None
    spot = synthetic_forward(atm, call_q, put_q)
    atm2 = nearest_strike(strikes, spot)
    if atm2 is not None and abs(atm2 - atm) > 1e-6:
        call2 = grab(atm2, "call", "entry")
        put2 = grab(atm2, "put", "entry")
        if call2 and put2:
            atm = atm2
            call_q, put_q = call2, put2
            spot = synthetic_forward(atm, call_q, put_q)
    em = float(call_q["mid"]) + float(put_q["mid"])
    if em <= 0 or spot <= 0:
        LOG.info("%s skipped: bad EM", earn)
        return None

    em_put = strike_just_below(strikes, spot - em)
    em_call = strike_just_above(strikes, spot + em)
    near_put = strike_just_below(strikes, spot)
    wing = target_wing(spot)
    chosen: Dict[str, Optional[float]] = {
        "em_put": em_put,
        "em_call": em_call,
        "near_put": near_put,
        "em_put_long": wing_strike(strikes, em_put, wing, "put") if em_put else None,
        "em_put_long5": wing_strike(strikes, em_put, EXAMPLE_WIDTH, "put") if em_put else None,
        "em_call_long": wing_strike(strikes, em_call, wing, "call") if em_call else None,
        "em_call_long5": wing_strike(strikes, em_call, EXAMPLE_WIDTH, "call") if em_call else None,
        "near_long": wing_strike(strikes, near_put, wing, "put") if near_put else None,
    }

    needed: List[Tuple[float, str]] = [(atm, "call"), (atm, "put")]
    for key, right in (
        ("em_put", "put"),
        ("em_put_long", "put"),
        ("em_put_long5", "put"),
        ("em_call", "call"),
        ("em_call_long", "call"),
        ("em_call_long5", "call"),
        ("near_put", "put"),
        ("near_long", "put"),
    ):
        strike = chosen[key]
        if strike is None:
            continue
        needed.append((float(strike), right))
    # unique
    uniq = []
    seen = set()
    for item in needed:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)

    entry_q: Dict[Tuple[float, str], Dict[str, float]] = {}
    exit_q: Dict[Tuple[float, str], Dict[str, float]] = {}
    for strike, right in uniq:
        qe = grab(strike, right, "entry")
        qx = grab(strike, right, "exit")
        if qe:
            entry_q[(strike, right)] = qe
        if qx:
            exit_q[(strike, right)] = qx

    structures: Dict[str, Dict[str, Any]] = {}

    def add_structure(name: str, pairs, width: Optional[float], naked: Optional[float], intrinsic_debit: float):
        base: Dict[str, Any] = {
            "width": None if width is None else round(width, 4),
            "strikes": "/".join(
                "%s%.1f%s" % ("S" if role == "short" else "L", strike, right[:1].upper())
                for role, strike, right in pairs
            ),
        }
        legs = _legs_from(pairs, entry_q, exit_q)
        if legs is None:
            base["missing_quote"] = True
            structures[name] = base
            return
        base["missing_quote"] = False
        base.update(_mark_row(legs, risk_width=width, naked_strike=naked))
        mid_credit = base.get("credit_mid")
        if mid_credit is not None:
            base["intrinsic_net"] = round(_intrinsic_net(mid_credit, intrinsic_debit, len(legs)), 2)
        structures[name] = base

    if em_put is not None:
        add_structure(
            "short_put_em",
            [("short", em_put, "put")],
            None,
            em_put,
            put_intrinsic(em_put, next_open),
        )
        long_p = chosen["em_put_long"]
        if long_p is not None:
            add_structure(
                "pcs_em",
                [("short", em_put, "put"), ("long", long_p, "put")],
                em_put - long_p,
                None,
                spread_intrinsic(em_put, long_p, next_open, "put"),
            )
        long5 = chosen["em_put_long5"]
        if long5 is not None and long5 != long_p:
            add_structure(
                "pcs_em_w5",
                [("short", em_put, "put"), ("long", long5, "put")],
                em_put - long5,
                None,
                spread_intrinsic(em_put, long5, next_open, "put"),
            )
    if em_put is not None and em_call is not None:
        long_p = chosen["em_put_long"]
        long_c = chosen["em_call_long"]
        if long_p is not None and long_c is not None:
            width = max(em_put - long_p, long_c - em_call)
            debit = spread_intrinsic(em_put, long_p, next_open, "put") + spread_intrinsic(
                em_call, long_c, next_open, "call"
            )
            add_structure(
                "ic_em",
                [
                    ("short", em_put, "put"),
                    ("long", long_p, "put"),
                    ("short", em_call, "call"),
                    ("long", long_c, "call"),
                ],
                width,
                None,
                debit,
            )
        long_p5 = chosen["em_put_long5"]
        long_c5 = chosen["em_call_long5"]
        if (
            long_p5 is not None
            and long_c5 is not None
            and (long_p5 != chosen["em_put_long"] or long_c5 != chosen["em_call_long"])
        ):
            width = max(em_put - long_p5, long_c5 - em_call)
            debit = spread_intrinsic(em_put, long_p5, next_open, "put") + spread_intrinsic(
                em_call, long_c5, next_open, "call"
            )
            add_structure(
                "ic_em_w5",
                [
                    ("short", em_put, "put"),
                    ("long", long_p5, "put"),
                    ("short", em_call, "call"),
                    ("long", long_c5, "call"),
                ],
                width,
                None,
                debit,
            )
    if near_put is not None:
        add_structure(
            "short_put_near",
            [("short", near_put, "put")],
            None,
            near_put,
            put_intrinsic(near_put, next_open),
        )
        long_n = chosen["near_long"]
        if long_n is not None:
            add_structure(
                "pcs_near",
                [("short", near_put, "put"), ("long", long_n, "put")],
                near_put - long_n,
                None,
                spread_intrinsic(near_put, long_n, next_open, "put"),
            )

    # Spot-held flags live on the row via structure fields.
    if em_put is not None and "short_put_em" in structures:
        structures["short_put_em"]["spot_held"] = held_short_put(next_open, em_put)
    if "ic_em" in structures and em_put is not None and em_call is not None:
        structures["ic_em"]["spot_held"] = held_iron_condor(next_open, em_put, em_call)
    if near_put is not None and "short_put_near" in structures:
        structures["short_put_near"]["spot_held"] = held_short_put(next_open, near_put)
    if "pcs_em" in structures and em_put is not None:
        structures["pcs_em"]["spot_held"] = held_short_put(next_open, em_put)
    if "pcs_em_w5" in structures and em_put is not None:
        structures["pcs_em_w5"]["spot_held"] = held_short_put(next_open, em_put)
    if "pcs_near" in structures and near_put is not None:
        structures["pcs_near"]["spot_held"] = held_short_put(next_open, near_put)
    if "ic_em_w5" in structures and em_put is not None and em_call is not None:
        structures["ic_em_w5"]["spot_held"] = held_iron_condor(next_open, em_put, em_call)

    row = _build_row(
        earn=earn,
        entry=entry,
        exit_day=exit_day,
        expiry=expiry,
        prior_close=prior_close,
        next_open=next_open,
        spot=spot,
        em=em,
        atm=atm,
        structures=structures,
    )
    LOG.info(
        "%s spot=%.2f close=%.2f EM=%.2f (%.2f%%) open=%.2f gap=%+.2f%% inside=%s down=%s up=%s",
        earn,
        spot,
        prior_close,
        em,
        row["em_pct"],
        next_open,
        row["gap_pct"] or 0.0,
        row["inside_em"],
        row["broke_down"],
        row["broke_up"],
    )
    return row


def _subset_stats(rows: Sequence[Dict[str, Any]], structure: str) -> Dict[str, Any]:
    slim = []
    held = 0
    held_n = 0
    credits = []
    for row in rows:
        item = {
            "net_mid": row.get(structure + "_net_mid"),
            "ror_mid": row.get(structure + "_ror_mid"),
            "net_mid_slip0.05": row.get(structure + "_net_mid_slip0.05"),
            "ror_mid_slip0.05": row.get(structure + "_ror_mid_slip0.05"),
            "net_mid_slip0.1": row.get(structure + "_net_mid_slip0.1"),
            "ror_mid_slip0.1": row.get(structure + "_ror_mid_slip0.1"),
            "net_natural": row.get(structure + "_net_natural"),
            "ror_natural": row.get(structure + "_ror_natural"),
            "credit_mid": row.get(structure + "_credit_mid"),
            "intrinsic_net": row.get(structure + "_intrinsic_net"),
        }
        if item["net_mid"] is None and item["net_natural"] is None:
            continue
        slim.append(item)
        if row.get(structure + "_spot_held") is not None:
            held_n += 1
            held += int(bool(row.get(structure + "_spot_held")))
        if item["credit_mid"] is not None:
            credits.append(float(item["credit_mid"]))
    out = {
        "mid": book_stats(slim, "net_mid"),
        "mid_slip0.05": book_stats(slim, "net_mid_slip0.05"),
        "mid_slip0.1": book_stats(slim, "net_mid_slip0.1"),
        "natural": book_stats(slim, "net_natural"),
        "intrinsic": book_stats(
            [{"net_mid": r["intrinsic_net"], "ror_mid": None} for r in slim if r["intrinsic_net"] is not None],
            "net_mid",
        ),
    }
    if credits:
        out["avg_credit"] = round(sum(credits) / len(credits), 3)
    if held_n:
        out["spot_held"] = "%d/%d" % (held, held_n)
    return out


def _fmt_stats(label: str, stats: Dict[str, Any]) -> str:
    mid = stats.get("mid") or {}
    nat = stats.get("natural") or {}
    intr = stats.get("intrinsic") or {}
    return (
        "%s: n=%s spot_held=%s avgCredit=%s | mid WR=%s%% mean$=%s PF=%s meanRoR=%s total$=%s | "
        "natural WR=%s%% mean$=%s PF=%s total$=%s | intrinsic WR=%s%% mean$=%s"
        % (
            label,
            mid.get("n"),
            stats.get("spot_held"),
            stats.get("avg_credit"),
            mid.get("win_rate_pct"),
            mid.get("mean_pnl"),
            mid.get("pf"),
            mid.get("mean_ror_pct"),
            mid.get("total_pnl"),
            nat.get("win_rate_pct"),
            nat.get("mean_pnl"),
            nat.get("pf"),
            nat.get("total_pnl"),
            intr.get("win_rate_pct"),
            intr.get("mean_pnl"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-02-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--only", default="", help="Comma-separated earn dates")
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = DEFAULT_OUT / "quote_cache.json"
    cache = _load_cache(cache_path)

    t0 = time.perf_counter()
    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal is not up on %s", client.base_url)
        return 1
    events = [
        e
        for e in _earnings(40)
        if start <= e["earn_date"] <= end and (not only or e["earn_date"].isoformat() in only)
    ]
    LOG.info("events %d", len(events))
    if not events:
        return 1
    stock = _load_stock(start, end + timedelta(days=7))
    LOG.info("stock sessions %d %s -> %s", len(stock), stock["session"].min() if len(stock) else None, stock["session"].max() if len(stock) else None)
    expirations = list_expirations(client, "NVDA")
    LOG.info("expirations %d", len(expirations))

    rows: List[Dict[str, Any]] = []
    for event in events:
        try:
            row = _one_event(client, cache, stock, expirations, event)
        except Exception as exc:
            LOG.exception("%s failed: %s", event["earn_date"], exc)
            row = None
        if row is not None:
            rows.append(row)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "trades.csv", index=False)
    claim_rows = [r for r in rows if r.get("sample") == "claim"]
    oos_rows = [r for r in rows if r.get("sample") == "oos"]

    def stock_blurb(sample_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(sample_rows)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "opened_green": sum(1 for r in sample_rows if r["opened_green"]),
            "inside_em": sum(1 for r in sample_rows if r["inside_em"]),
            "broke_down": sum(1 for r in sample_rows if r["broke_down"]),
            "broke_up": sum(1 for r in sample_rows if r["broke_up"]),
        }

    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "n": len(rows),
        "stock_claim": stock_blurb(claim_rows),
        "stock_oos": stock_blurb(oos_rows),
        "stock_all": stock_blurb(rows),
        "structures_claim": {name: _subset_stats(claim_rows, name) for name in STRUCTURES},
        "structures_oos": {name: _subset_stats(oos_rows, name) for name in STRUCTURES},
        "structures_all": {name: _subset_stats(rows, name) for name in STRUCTURES},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("stock claim %s", summary["stock_claim"])
    LOG.info("stock oos %s", summary["stock_oos"])
    for name in STRUCTURES:
        block = summary["structures_claim"][name]
        if (block.get("mid") or {}).get("n"):
            LOG.info("CLAIM %s", _fmt_stats(name, block))
    for name in ("short_put_em", "pcs_em", "ic_em", "short_put_near", "pcs_near"):
        block = summary["structures_oos"][name]
        if (block.get("mid") or {}).get("n"):
            LOG.info("OOS %s", _fmt_stats(name, block))
    LOG.info("wrote %s in %.1fs", outdir, summary["wall_clock_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
