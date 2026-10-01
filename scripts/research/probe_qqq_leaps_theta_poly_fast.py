"""Fast probe: Theta NBBO for QQQ LEAPS + single-shot Polygon (no retry storm)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.config.env_loader import get_env_var  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.options.greeks import calculate_greeks  # noqa: E402
from utils.research.qqq_leaps_gap import (  # noqa: E402
    iv_from_atr,
    prepare_qqq_daily,
    strike_for_call_delta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("probe_fast")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

PROBE_DAYS = [
    date(2020, 3, 9),
    date(2022, 5, 5),
    date(2024, 7, 24),
    date(2025, 4, 4),
]
OUTDIR = (
    ROOT
    / "reports"
    / "research"
    / "qqq_leaps_gap"
    / date.today().isoformat()
    / "data_probe"
)


def _occ(exp: date, strike: float) -> str:
    return f"O:QQQ{exp.strftime('%y%m%d')}C{int(round(strike * 1000)):08d}"


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
    return prepare_qqq_daily(frames["QQQ"])


def list_theta_exps(client) -> List[str]:
    from io import StringIO

    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": "QQQ"},
        timeout=60,
    )
    LOG.info("Theta expirations HTTP %s ctype=%s", r.status_code, r.headers.get("content-type"))
    if r.status_code != 200 or not r.content:
        LOG.warning("body=%s", (r.text or "")[:400])
        return []
    # Terminal v3 list endpoints return CSV.
    df = pd.read_csv(StringIO(r.text))
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    return [str(x)[:10] for x in df[col].tolist()]


def pick_exp(exps: List[str], day: date) -> Optional[date]:
    best = None
    best_abs = 10**9
    for es in exps:
        try:
            ed = date.fromisoformat(es[:10])
        except ValueError:
            continue
        dte = (ed - day).days
        if 300 <= dte <= 450:
            if abs(dte - 365) < best_abs:
                best_abs = abs(dte - 365)
                best = ed
    return best


def list_strikes(client, exp: date) -> List[float]:
    from io import StringIO

    r = client.session.get(
        f"{client.base_url}/v3/option/list/strikes",
        params={"symbol": "QQQ", "expiration": exp.isoformat()},
        timeout=60,
    )
    if r.status_code != 200 or not r.content:
        return []
    ctype = (r.headers.get("content-type") or "").lower()
    if "json" in ctype:
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
        return nums
    df = pd.read_csv(StringIO(r.text))
    col = "strike" if "strike" in df.columns else df.columns[-1]
    return [float(x) for x in df[col].tolist() if pd.notna(x)]


def polygon_once(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Single GET; no retries. Never log the key."""
    key = get_env_var("POLYGON_API_KEY")
    p = dict(params)
    p["apiKey"] = key
    url = f"https://api.polygon.io{path}"
    try:
        r = requests.get(url, params=p, timeout=30)
        body: Any
        try:
            body = r.json()
        except Exception:
            body = {"raw": (r.text or "")[:300]}
        # strip any accidental key echo
        return {
            "http": r.status_code,
            "status": body.get("status") if isinstance(body, dict) else None,
            "error": body.get("error") if isinstance(body, dict) else None,
            "message": body.get("message") if isinstance(body, dict) else None,
            "results_n": len(body.get("results") or []) if isinstance(body, dict) else 0,
            "sample": (body.get("results") or [None])[0] if isinstance(body, dict) else None,
        }
    except Exception as exc:
        return {"http": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = load_qqq()
    client = get_thetadata_client()
    LOG.info("Theta ready=%s", client.is_ready())
    exps = list_theta_exps(client)
    LOG.info("Theta QQQ expirations listed: %d", len(exps))
    if exps:
        LOG.info("exp range %s .. %s", exps[0], exps[-1])

    rows: List[Dict[str, Any]] = []
    for day in PROBE_DAYS:
        hits = d.loc[d["session_date"] == day]
        if hits.empty:
            rows.append({"day": str(day), "skip": "no_bar"})
            continue
        row = hits.iloc[0]
        S = float(row["open"])
        gap = float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None
        atr = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
        iv = iv_from_atr(atr, S) or 0.22
        K = strike_for_call_delta(S, 1.0, 0.05, iv, 0.005, 0.60)
        delta = float(calculate_greeks(S, K, 1.0, 0.05, iv, "C", q=0.005)["delta"])

        exp = pick_exp(exps, day)
        strikes = list_strikes(client, exp) if exp else []
        strike = min(strikes, key=lambda x: abs(x - K)) if strikes else K

        rec: Dict[str, Any] = {
            "day": str(day),
            "gap_pct": None if gap is None else round(gap, 3),
            "S_open": S,
            "strike_bs": K,
            "delta_bs": round(delta, 3),
            "iv_atr": round(iv, 4),
            "exp": str(exp) if exp else None,
            "dte": (exp - day).days if exp else None,
            "strike_listed": strike if strikes else None,
            "n_strikes": len(strikes),
        }

        if exp is None:
            rec["theta_error"] = "no_leaps_expiration_in_list"
        else:
            q0 = client.first_rth_nbbo(
                symbol="QQQ", expiration=exp, strike=float(strike), right="call", day=day
            )
            q1 = client.nbbo_at_or_after(
                symbol="QQQ",
                expiration=exp,
                strike=float(strike),
                right="call",
                day=day,
                clock=time(15, 45),
                end_time="16:00:00",
            )
            rec["theta_0930"] = q0
            rec["theta_1545"] = q1

            # multi-day sample: next ~10 sessions at 15:45 (Theta range ~1mo cap)
            end = day + timedelta(days=14)
            try:
                clock_df = client.daily_clock_quotes(
                    symbol="QQQ",
                    expiration=exp,
                    strike=float(strike),
                    right="call",
                    start_date=day,
                    end_date=end,
                    clock=time(15, 45),
                )
                rec["theta_clock_14d_n"] = int(len(clock_df))
                if not clock_df.empty:
                    rec["theta_clock_14d_sample"] = clock_df.head(3).to_dict(orient="records")
            except Exception as exc:
                rec["theta_clock_error"] = f"{type(exc).__name__}: {exc}"

            ticker = _occ(exp, float(strike))
            rec["occ"] = ticker
            # Polygon/Massive single-shot
            rec["poly_contracts"] = polygon_once(
                "/v3/reference/options/contracts",
                {
                    "underlying_ticker": "QQQ",
                    "expiration_date": exp.isoformat(),
                    "as_of": day.isoformat(),
                    "expired": "true",
                    "contract_type": "call",
                    "strike_price.gte": float(strike) - 5,
                    "strike_price.lte": float(strike) + 5,
                    "limit": 50,
                },
            )
            rec["poly_day_aggs"] = polygon_once(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{day}/{day}",
                {"adjusted": "true", "limit": 1},
            )
            rec["poly_min_aggs"] = polygon_once(
                f"/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}",
                {"adjusted": "true", "limit": 5},
            )
            rec["poly_quotes"] = polygon_once(
                f"/v3/quotes/{ticker}",
                {
                    "timestamp.gte": f"{day}T13:30:00Z",
                    "timestamp.lte": f"{day}T14:00:00Z",
                    "limit": 5,
                },
            )

        LOG.info(
            "%s gap=%s exp=%s K=%s theta0930=%s theta1545=%s poly_day=%s poly_min=%s poly_q=%s",
            day,
            rec.get("gap_pct"),
            rec.get("exp"),
            rec.get("strike_listed") or rec.get("strike_bs"),
            bool(rec.get("theta_0930")),
            bool(rec.get("theta_1545")),
            (rec.get("poly_day_aggs") or {}).get("http"),
            (rec.get("poly_min_aggs") or {}).get("http"),
            (rec.get("poly_quotes") or {}).get("http"),
        )
        rows.append(rec)

    out_json = OUTDIR / "qqq_leaps_theta_poly_probe.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    md = [
        "# QQQ LEAPS — ThetaData vs Polygon/Massive probe",
        "",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        f"- Theta QQQ expirations listed: **{len(exps)}**",
        "",
        "| Day | Gap% | Exp | Strike | Theta 09:30 mid (b/a) | Theta 15:45 mid | Poly day | Poly min | Poly quotes |",
        "|-----|-----:|-----|-------:|----------------------:|----------------:|---------:|---------:|------------:|",
    ]
    for r in rows:
        if r.get("skip"):
            md.append(f"| {r['day']} | — | — | — | skip | — | — | — | — |")
            continue
        t0 = r.get("theta_0930") or {}
        t1 = r.get("theta_1545") or {}
        t0s = (
            f"{t0.get('mid')} ({t0.get('bid')}/{t0.get('ask')})"
            if t0
            else "none"
        )
        t1s = f"{t1.get('mid')}" if t1 else "none"
        pd_ = r.get("poly_day_aggs") or {}
        pm = r.get("poly_min_aggs") or {}
        pq = r.get("poly_quotes") or {}
        md.append(
            f"| {r['day']} | {r.get('gap_pct')} | {r.get('exp')} | "
            f"{r.get('strike_listed') or r.get('strike_bs')} | {t0s} | {t1s} | "
            f"{pd_.get('http')}/{pd_.get('results_n')} | "
            f"{pm.get('http')}/{pm.get('results_n')} | "
            f"{pq.get('http')}/{pq.get('results_n')} |"
        )
    md.extend(
        [
            "",
            "## Read",
            "",
            "- **Theta**: Standard plan NBBO via local Terminal. `none` = 472/empty for that contract/day.",
            "- **Polygon/Massive**: `http` / `results_n`. 403 = plan forbids that endpoint/lookback; 200/0 = allowed but empty.",
            "- Day/minute option aggs typically need Options **Starter+**; NBBO quotes need **Advanced**.",
            "",
            f"JSON: `{out_json.as_posix()}`",
            "",
        ]
    )
    out_md = OUTDIR / "qqq_leaps_theta_poly_probe.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    LOG.info("Wrote %s and %s", out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
