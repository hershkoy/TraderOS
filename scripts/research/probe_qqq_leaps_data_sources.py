"""Probe ThetaData + Polygon/Massive for historical QQQ LEAPS quotes.

Picks a few known gap-down days, finds ~12mo ~60-delta call candidates via BS
strike guess + chain listing, then tries:
  - Theta: 1m NBBO at open / 15:45
  - Polygon: day aggs, minute aggs, NBBO quotes

Usage:
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\probe_qqq_leaps_data_sources.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.options.greeks import calculate_greeks  # noqa: E402
from utils.research.qqq_leaps_gap import (  # noqa: E402
    iv_from_atr,
    prepare_qqq_daily,
    round_qqq_strike,
    strike_for_call_delta,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("probe_qqq_leaps_data")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

# Representative claim-window gap days (bull / bear / recent).
PROBE_DAYS = [
    date(2020, 3, 9),   # COVID crash gap
    date(2022, 5, 5),   # 2022 bear gap cluster
    date(2024, 7, 24),  # recent tech selloff gap
    date(2025, 4, 4),   # 2025 tariff-gap window (if in panel)
]

OUTDIR = ROOT / "reports" / "research" / "qqq_leaps_gap" / date.today().isoformat() / "data_probe"


def _occ(symbol: str, exp: date, strike: float, right: str = "C") -> str:
    # O:QQQ250620C00400000
    yy = exp.strftime("%y%m%d")
    k = int(round(float(strike) * 1000))
    return f"O:{symbol.upper()}{yy}{right.upper()}{k:08d}"


def _nearest_friday_exp(anchor: date, *, min_dte: int = 330, max_dte: int = 400) -> date:
    """Rough LEAPS target: Friday ~12 months out (monthly/LEAPS cadence approx)."""
    target = anchor + timedelta(days=365)
    # walk to Friday
    while target.weekday() != 4:
        target += timedelta(days=1)
    # clamp into band vs anchor
    dte = (target - anchor).days
    if dte < min_dte:
        target += timedelta(days=7 * ((min_dte - dte + 6) // 7))
    if (target - anchor).days > max_dte:
        target -= timedelta(days=7)
    while target.weekday() != 4:
        target -= timedelta(days=1)
    return target


def load_qqq() -> pd.DataFrame:
    frames = load_ohlcv_many(
        ["QQQ"],
        timeframe="1d",
        provider="IB",
        start=datetime(2019, 1, 1),
        end=datetime(2025, 8, 1),
        use_cache=True,
        workers=1,
        fallback_provider="ALPACA",
        merge_mode="prefix",
    )
    raw = frames.get("QQQ")
    if raw is None or raw.empty:
        raise RuntimeError("QQQ daily missing")
    return prepare_qqq_daily(raw)


def row_for_day(d: pd.DataFrame, day: date) -> Optional[pd.Series]:
    hits = d.loc[d["session_date"] == day]
    if hits.empty:
        return None
    return hits.iloc[0]


def bs_plan(row: pd.Series, day: date) -> Dict[str, Any]:
    S = float(row["open"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
    iv = iv_from_atr(atr, S) or 0.22
    T = 365.0 / 365.25
    K = strike_for_call_delta(S, T, 0.05, iv, 0.005, target_delta=0.60)
    g = calculate_greeks(S, K, T, 0.05, iv, "C", q=0.005)
    exp = _nearest_friday_exp(day)
    return {
        "day": str(day),
        "S_open": S,
        "iv_atr": round(iv, 4),
        "strike_bs": K,
        "delta_bs": round(float(g["delta"]), 3),
        "exp_guess": str(exp),
        "dte_guess": (exp - day).days,
        "occ_guess": _occ("QQQ", exp, K, "C"),
    }


def probe_theta(plan: Dict[str, Any]) -> Dict[str, Any]:
    client = get_thetadata_client()
    out: Dict[str, Any] = {"ready": client.is_ready()}
    if not out["ready"]:
        out["error"] = "Theta Terminal not reachable on :25503"
        return out

    # list expirations
    try:
        r = client.session.get(
            f"{client.base_url}/v3/option/list/expirations",
            params={"symbol": "QQQ"},
            timeout=30,
        )
        out["expirations_status"] = r.status_code
        if r.status_code == 200:
            payload = r.json() if r.content else {}
            exps = payload.get("response") or payload.get("results") or payload
            if isinstance(exps, dict):
                exps = exps.get("data") or exps.get("expirations") or []
            if isinstance(exps, list):
                # normalize to strings
                exp_strs = []
                for e in exps:
                    if isinstance(e, dict):
                        exp_strs.append(str(e.get("expiration") or e.get("date") or e))
                    else:
                        exp_strs.append(str(e)[:10])
                out["n_expirations"] = len(exp_strs)
                day = date.fromisoformat(plan["day"])
                # pick closest ~365 DTE
                best = None
                best_abs = 10**9
                for es in exp_strs:
                    try:
                        ed = date.fromisoformat(es[:10])
                    except ValueError:
                        continue
                    dte = (ed - day).days
                    if dte < 250 or dte > 500:
                        continue
                    if abs(dte - 365) < best_abs:
                        best_abs = abs(dte - 365)
                        best = ed
                out["exp_picked"] = str(best) if best else None
                out["exp_dte"] = (best - day).days if best else None
            else:
                out["expirations_raw_type"] = type(exps).__name__
        else:
            out["expirations_body"] = (r.text or "")[:300]
    except Exception as exc:
        out["expirations_error"] = str(exc)
        return out

    exp = out.get("exp_picked") or plan["exp_guess"]
    strike = float(plan["strike_bs"])
    day = plan["day"]

    # try list strikes for that expiration
    try:
        r = client.session.get(
            f"{client.base_url}/v3/option/list/strikes",
            params={"symbol": "QQQ", "expiration": exp},
            timeout=30,
        )
        out["strikes_status"] = r.status_code
        if r.status_code == 200 and r.content:
            payload = r.json()
            strikes = payload.get("response") or payload.get("results") or []
            if isinstance(strikes, dict):
                strikes = strikes.get("data") or strikes.get("strikes") or []
            nums = []
            for s in strikes if isinstance(strikes, list) else []:
                try:
                    nums.append(float(s if not isinstance(s, dict) else s.get("strike")))
                except (TypeError, ValueError):
                    pass
            if nums:
                nearest = min(nums, key=lambda x: abs(x - strike))
                out["strike_listed"] = nearest
                out["n_strikes"] = len(nums)
                strike = nearest
    except Exception as exc:
        out["strikes_error"] = str(exc)

    # first RTH NBBO call
    try:
        q = client.first_rth_nbbo(
            symbol="QQQ",
            expiration=exp,
            strike=strike,
            right="call",
            day=day,
        )
        out["first_rth"] = q
    except Exception as exc:
        out["first_rth_error"] = str(exc)

    # 15:45 NBBO same day
    try:
        from datetime import time as dtime

        q2 = client.nbbo_at_or_after(
            symbol="QQQ",
            expiration=exp,
            strike=strike,
            right="call",
            day=day,
            clock=dtime(15, 45),
            end_time="16:00:00",
        )
        out["nbbo_1545"] = q2
    except Exception as exc:
        out["nbbo_1545_error"] = str(exc)

    # EOD-ish: try daily interval if supported
    try:
        df = client.option_history_quote(
            symbol="QQQ",
            expiration=exp,
            strike=strike,
            right="call",
            day=day,
            interval="1d",
            start_time="09:30:00",
            end_time="16:00:00",
        )
        out["hist_1d_rows"] = 0 if df is None else int(len(df))
        if df is not None and not df.empty:
            out["hist_1d_sample"] = df.head(2).to_dict(orient="records")
    except Exception as exc:
        out["hist_1d_error"] = str(exc)

    out["used"] = {"expiration": exp, "strike": strike, "right": "call", "day": day}
    return out


def probe_polygon(plan: Dict[str, Any], theta_exp: Optional[str] = None) -> Dict[str, Any]:
    client = get_polygon_client()
    out: Dict[str, Any] = {}
    day = plan["day"]
    strike = float(plan["strike_bs"])
    exp = date.fromisoformat((theta_exp or plan["exp_guess"])[:10])

    # 1) contract discovery as_of
    try:
        chain = client.get_contracts_for_expiry(
            "QQQ",
            expiry=exp.isoformat(),
            as_of=day,
            contract_type="call",
            min_strike=strike - 20,
            max_strike=strike + 20,
            limit=1000,
        )
        results = chain.get("results") or []
        out["contracts_n"] = len(results)
        if results:
            # pick nearest strike
            best = min(
                results,
                key=lambda r: abs(float(r.get("strike_price") or 0) - strike),
            )
            out["contract"] = {
                "ticker": best.get("ticker"),
                "strike": best.get("strike_price"),
                "expiration": best.get("expiration_date"),
                "as_of_ok": True,
            }
            ticker = best.get("ticker")
        else:
            # fallback OCC guess
            ticker = _occ("QQQ", exp, strike, "C")
            out["contract"] = {"ticker": ticker, "fallback_occ": True}
    except Exception as exc:
        out["contracts_error"] = str(exc)
        ticker = _occ("QQQ", exp, strike, "C")
        out["contract"] = {"ticker": ticker, "fallback_occ": True, "error": str(exc)}

    # 2) day aggregates (Starter+)
    try:
        eod = client.get_option_eod(ticker, day)
        results = eod.get("results") or []
        out["day_aggs_status"] = eod.get("status")
        out["day_aggs_n"] = len(results)
        if results:
            bar = results[0]
            out["day_aggs"] = {
                "o": bar.get("o"),
                "h": bar.get("h"),
                "l": bar.get("l"),
                "c": bar.get("c"),
                "v": bar.get("v"),
                "vw": bar.get("vw"),
                "n": bar.get("n"),
            }
    except Exception as exc:
        out["day_aggs_error"] = str(exc)

    # 3) minute aggs first hour
    try:
        mins = client.get_option_minute_aggs(ticker, day)
        results = mins.get("results") or []
        out["minute_aggs_status"] = mins.get("status")
        out["minute_aggs_n"] = len(results)
        if results:
            out["minute_aggs_first"] = results[0]
            out["minute_aggs_last"] = results[-1]
    except Exception as exc:
        out["minute_aggs_error"] = str(exc)

    # 4) NBBO quotes window (Advanced)
    try:
        # morning window as dates — Polygon quotes want timestamps; use day bounds
        q = client.get_options_quotes(
            ticker,
            from_date=f"{day}T13:30:00Z",  # ~09:30 ET
            to_date=f"{day}T14:30:00Z",
            limit=50,
        )
        results = q.get("results") or []
        out["quotes_status"] = q.get("status")
        out["quotes_n"] = len(results)
        if results:
            r0 = results[0]
            out["quotes_first"] = {
                "bid": r0.get("bid_price") or r0.get("bp"),
                "ask": r0.get("ask_price") or r0.get("ap"),
                "t": r0.get("sip_timestamp") or r0.get("t"),
            }
        # capture http-ish error body if present
        if not results and q.get("error"):
            out["quotes_error_field"] = q.get("error")
        if q.get("message"):
            out["quotes_message"] = q.get("message")
    except Exception as exc:
        out["quotes_error"] = str(exc)

    # 5) historical snapshot (often paid)
    try:
        snap = client.get_options_snapshot(ticker, as_of=day)
        out["snapshot_keys"] = list(snap.keys())[:12] if isinstance(snap, dict) else None
        out["snapshot_status"] = snap.get("status") if isinstance(snap, dict) else None
        if isinstance(snap, dict) and snap.get("results"):
            out["snapshot_has_results"] = True
        elif isinstance(snap, dict) and snap.get("error"):
            out["snapshot_error"] = snap.get("error")
        elif isinstance(snap, dict) and snap.get("message"):
            out["snapshot_message"] = snap.get("message")
    except Exception as exc:
        out["snapshot_error"] = str(exc)

    out["ticker"] = ticker
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = load_qqq()
    rows: List[Dict[str, Any]] = []

    # Theta readiness once
    th = get_thetadata_client()
    theta_ready = th.is_ready()
    LOG.info("Theta Terminal ready=%s base=%s", theta_ready, th.base_url)

    poly_ok = True
    try:
        get_polygon_client()
        LOG.info("Polygon client constructed")
    except Exception as exc:
        poly_ok = False
        LOG.error("Polygon client failed: %s", exc)

    for day in PROBE_DAYS:
        row = row_for_day(d, day)
        if row is None:
            LOG.warning("No QQQ bar for %s — skip", day)
            rows.append({"day": str(day), "skip": "no_bar"})
            continue
        gap = float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None
        plan = bs_plan(row, day)
        plan["gap_pct"] = None if gap is None else round(gap, 3)
        LOG.info(
            "Day %s gap=%s open=%.2f strike_bs=%.1f exp=%s",
            day,
            plan["gap_pct"],
            plan["S_open"],
            plan["strike_bs"],
            plan["exp_guess"],
        )

        theta = probe_theta(plan) if theta_ready else {"ready": False, "error": "not_ready"}
        time.sleep(0.2)
        poly = probe_polygon(plan, theta_exp=theta.get("exp_picked")) if poly_ok else {"error": "no_client"}
        time.sleep(0.35)

        rec = {"plan": plan, "theta": theta, "polygon": poly}
        rows.append(rec)
        LOG.info(
            "  Theta first_rth=%s 1545=%s | Poly day_aggs_n=%s minute_n=%s quotes_n=%s",
            bool(theta.get("first_rth")),
            bool(theta.get("nbbo_1545")),
            poly.get("day_aggs_n"),
            poly.get("minute_aggs_n"),
            poly.get("quotes_n"),
        )

    out_path = OUTDIR / "qqq_leaps_source_probe.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s", out_path)

    # concise markdown summary
    md = [
        "# QQQ LEAPS data-source probe",
        "",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        f"- Theta ready: `{theta_ready}`",
        f"- Polygon client: `{poly_ok}`",
        "",
        "| Day | Gap% | Strike | Theta 09:30 mid | Theta 15:45 mid | Poly day C | Poly min n | Poly quotes n |",
        "|-----|-----:|-------:|----------------:|----------------:|-----------:|-----------:|--------------:|",
    ]
    for rec in rows:
        if rec.get("skip"):
            md.append(f"| {rec['day']} | — | — | skip | — | — | — | — |")
            continue
        p = rec["plan"]
        th = rec["theta"]
        po = rec["polygon"]
        t0 = th.get("first_rth") or {}
        t1 = th.get("nbbo_1545") or {}
        day_c = (po.get("day_aggs") or {}).get("c")
        md.append(
            f"| {p['day']} | {p.get('gap_pct')} | {p['strike_bs']} | "
            f"{t0.get('mid')} | {t1.get('mid')} | {day_c} | "
            f"{po.get('minute_aggs_n')} | {po.get('quotes_n')} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- Theta Standard: 1m NBBO hist via local Terminal — usable for LEAPS if expirations/strikes resolve.",
            "- Polygon/Massive Options: day/minute aggs need Starter+; NBBO quotes need Advanced.",
            "- Empty cells mean no data / plan / error — see JSON for error bodies.",
            "",
            f"JSON: `{out_path.as_posix()}`",
            "",
        ]
    )
    md_path = OUTDIR / "qqq_leaps_source_probe.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    LOG.info("Wrote %s", md_path)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
