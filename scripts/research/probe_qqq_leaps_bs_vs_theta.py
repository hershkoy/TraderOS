"""Spot-check BS LEAPS debit vs Theta NBBO mid on QQQ gap days.

Picks monthly-ish expirations (~365 DTE), nearest listed strike to 60-delta,
first nonzero RTH NBBO mid vs BS mark at open.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.options.greeks import black_scholes_call, calculate_greeks  # noqa: E402
from utils.research.qqq_leaps_gap import (  # noqa: E402
    iv_from_atr,
    prepare_qqq_daily,
    strike_for_call_delta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("qqq_leaps_theta_spot")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

PROBE_DAYS = [
    date(2020, 3, 9),
    date(2020, 9, 3),
    date(2022, 5, 5),
    date(2022, 11, 2),
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


def list_exps(client) -> pd.DataFrame:
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": "QQQ"},
        timeout=60,
    )
    df = pd.read_csv(StringIO(r.text))
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    return df


def pick_monthly_leaps(exps: pd.DataFrame, day: date) -> List[date]:
    """Rank Jan/Mar/Jun/Sep LEAPS-cycle expirations near +365 DTE."""
    leaps_months = {1, 3, 6, 9}
    rows = []
    for ed in exps["expiration"].tolist():
        dte = (ed - day).days
        if not (300 <= dte <= 450):
            continue
        if ed.month not in leaps_months:
            continue
        # Prefer 3rd-Friday-ish (14-22)
        mid_month = 0 if 14 <= ed.day <= 22 else 1
        rows.append((ed, abs(dte - 365), mid_month))
    rows.sort(key=lambda x: (x[2], x[1]))
    return [r[0] for r in rows]


def list_strikes(client, exp: date) -> List[float]:
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


def first_nonzero_nbbo(client, *, exp: date, strike: float, day: date) -> Optional[Dict[str, Any]]:
    # Full morning through lunch — skip leading zero quotes.
    return client.nbbo_at_or_after(
        symbol="QQQ",
        expiration=exp,
        strike=strike,
        right="call",
        day=day,
        clock=time(9, 30),
        end_time="12:00:00",
    )


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = load_qqq()
    client = get_thetadata_client()
    assert client.is_ready()
    exps = list_exps(client)
    LOG.info("QQQ expirations: %d (%s .. %s)", len(exps), exps["expiration"].iloc[0], exps["expiration"].iloc[-1])

    rows: List[Dict[str, Any]] = []
    for day in PROBE_DAYS:
        hits = d.loc[d["session_date"] == day]
        if hits.empty:
            rows.append({"day": str(day), "skip": "no_bar"})
            continue
        row = hits.iloc[0]
        S = float(row["open"])
        gap = float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None
        exp_cands = pick_monthly_leaps(exps, day)
        if not exp_cands:
            rows.append({"day": str(day), "skip": "no_exp"})
            continue

        atr = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
        iv = iv_from_atr(atr, S) or 0.22

        nbbo = None
        chosen = None
        K = None
        bs = None
        delta = None
        T = None
        for exp in exp_cands[:8]:
            T = (exp - day).days / 365.25
            K_bs = strike_for_call_delta(S, T, 0.05, iv, 0.005, 0.60)
            strikes = list_strikes(client, exp)
            if not strikes:
                continue
            K_try = min(strikes, key=lambda x: abs(x - K_bs))
            q = first_nonzero_nbbo(client, exp=exp, strike=K_try, day=day)
            if not q:
                continue
            chosen = exp
            K = K_try
            bs = float(black_scholes_call(S, K, T, 0.05, iv, q=0.005))
            delta = float(calculate_greeks(S, K, T, 0.05, iv, "C", q=0.005)["delta"])
            nbbo = q
            break

        if chosen is None or nbbo is None:
            rows.append(
                {
                    "day": str(day),
                    "gap_pct": None if gap is None else round(gap, 3),
                    "S_open": round(S, 2),
                    "skip": "no_theta_nbbo",
                    "tried_exps": [str(x) for x in exp_cands[:8]],
                }
            )
            LOG.info("%s no Theta NBBO among %s", day, [str(x) for x in exp_cands[:5]])
            continue

        rec: Dict[str, Any] = {
            "day": str(day),
            "gap_pct": None if gap is None else round(gap, 3),
            "S_open": round(S, 2),
            "exp": str(chosen),
            "dte": (chosen - day).days,
            "strike": K,
            "iv_atr": round(iv, 4),
            "delta_bs": round(float(delta), 3),
            "bs_mid": round(float(bs), 2),
            "theta_nbbo": nbbo,
            "theta_mid": round(float(nbbo["mid"]), 2),
            "theta_spread": round(float(nbbo["spread"]), 2),
            "bs_vs_mid_pct": round(100.0 * (float(bs) - float(nbbo["mid"])) / float(nbbo["mid"]), 1),
            "half_spread_pct": round(
                100.0 * (0.5 * float(nbbo["spread"])) / float(nbbo["mid"]), 1
            ),
        }
        LOG.info(
            "%s gap=%s exp=%s K=%s bs=%.2f theta_mid=%.2f delta=%.2f vs=%.1f%%",
            day,
            rec["gap_pct"],
            chosen,
            K,
            bs,
            rec["theta_mid"],
            delta,
            rec["bs_vs_mid_pct"],
        )
        rows.append(rec)

    out_json = OUTDIR / "qqq_leaps_bs_vs_theta.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    md = [
        "# QQQ LEAPS BS vs Theta NBBO spot-check",
        "",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        "- Theta Standard via local Terminal; first nonzero RTH NBBO (09:30-12:00 ET).",
        "- Strike nearest listed to BS 60-delta; monthly-ish exp ~365 DTE.",
        "",
        "| Day | Gap% | Exp | K | BS mid | Theta mid | BS-mid% | Half-sprd% | Delta |",
        "|-----|-----:|-----|--:|-------:|----------:|--------:|-----------:|------:|",
    ]
    for r in rows:
        if r.get("skip"):
            md.append(f"| {r['day']} | — | — | — | skip | — | — | — | — |")
            continue
        md.append(
            f"| {r['day']} | {r.get('gap_pct')} | {r.get('exp')} | {r.get('strike')} | "
            f"{r.get('bs_mid')} | {r.get('theta_mid')} | {r.get('bs_vs_mid_pct')} | "
            f"{r.get('half_spread_pct')} | {r.get('delta_bs')} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive `BS-mid%` = BS richer than listed mid (optimistic long entry if using BS).",
            "- Negative = BS cheaper than mid (BS understates debit on that day).",
            "- COVID-era zeros / missing NBBO still possible on some LEAPS; treat as coverage holes.",
            "",
            f"JSON: `{out_json.as_posix()}`",
            "",
        ]
    )
    out_md = OUTDIR / "qqq_leaps_bs_vs_theta.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
