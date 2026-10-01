#!/usr/bin/env python3
"""HTML matrix report for QQQ LEAPS gap + low-VIX hedges.

Reads matrix_summary.json from the hedge claim-check outdir and embeds a
LEAPS+hedge equity walk for the selected combo (default: best MDD lift at
max_open=10 mid_mid).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_qqq_leaps_gap_hedge_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.qqq_leaps_gap_hedge import (  # noqa: E402
    apply_max_open,
    leaps_from_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("generate_qqq_leaps_gap_hedge_report")

DEFAULT_HEDGE_DIR = (
    ROOT / "reports" / "research" / "qqq_leaps_gap" / "2026-10-01_hedge"
)
DEFAULT_LEAPS_CSV = (
    ROOT
    / "reports"
    / "research"
    / "qqq_leaps_gap"
    / "2026-09-30_theta"
    / "mid_mid_uncapped_combined.csv"
)
WINDOW_START = date(2020, 1, 3)
WINDOW_END = date(2025, 6, 5)


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9bb4; --accent:#5b9fd4; --good:#3ecf8e; --bad:#f07178; }
  body { margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }
  header { padding:1.25rem 1.5rem; border-bottom:1px solid #2a3548; }
  h1 { margin:0 0 .35rem; font-size:1.35rem; font-weight:650; }
  .sub { color:var(--muted); font-size:.9rem; }
  main { padding:1rem 1.5rem 2rem; }
  .kpis { display:flex; flex-wrap:wrap; gap:.6rem; margin:1rem 0; }
  .kpi { background:var(--card); border-radius:8px; padding:.65rem .85rem; min-width:7.5rem; }
  .kpi .l { color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; }
  .kpi .v { font-size:1.05rem; font-weight:600; margin-top:.15rem; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; background:var(--card); border-radius:8px; overflow:hidden; }
  th, td { padding:.45rem .55rem; text-align:left; border-bottom:1px solid #2a3548; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0; background:#152033; }
  tr:hover td { background:#1f2b3d; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .good { color:var(--good); }
  .bad { color:var(--bad); }
  #eq { width:100%; height:320px; background:var(--card); border-radius:8px; margin:1rem 0; }
  .note { color:var(--muted); font-size:.8rem; margin-top:.75rem; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__SUB__</div>
</header>
<main>
  <div class="kpis">
    __KPIS__
  </div>
  <canvas id="eq"></canvas>
  <h2 style="font-size:1rem;margin:1.25rem 0 .5rem;">Structure x attach matrix (max_open=10 mid/mid)</h2>
  <div style="overflow:auto;max-height:420px;">
  <table>
    <thead>
      <tr>
        <th>Key</th><th class="num">LEAPS n</th><th class="num">LEAPS PnL</th>
        <th class="num">LEAPS MDD</th><th class="num">Hedge n</th><th class="num">Hedge PnL</th>
        <th class="num">Combined PnL</th><th class="num">Combined PF</th>
        <th class="num">Combined MDD</th><th class="num">MDD lift</th>
      </tr>
    </thead>
    <tbody>__ROWS__</tbody>
  </table>
  </div>
  <p class="note">__NOTE__</p>
</main>
<script>
const PAYLOAD = __PAYLOAD__;
(function(){
  const eq = PAYLOAD.equity || [];
  const labels = eq.map(r => r.date);
  const leaps = eq.map(r => r.leaps_equity);
  const hedged = eq.map(r => r.hedged_equity);
  const spy = eq.map(r => r.spy_equity);
  new Chart(document.getElementById('eq'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'LEAPS-only', data: leaps, borderColor: '#5b9fd4', tension: 0.1, pointRadius: 0, borderWidth: 1.5 },
        { label: 'LEAPS+hedge', data: hedged, borderColor: '#3ecf8e', tension: 0.1, pointRadius: 0, borderWidth: 1.5 },
        { label: 'SPY B&H', data: spy, borderColor: '#e6a23c', tension: 0.1, pointRadius: 0, borderWidth: 1.5 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#e7ecf3' } } },
      scales: {
        x: { ticks: { color: '#8b9bb4', maxTicksLimit: 10 }, grid: { color: '#2a3548' } },
        y: { ticks: { color: '#8b9bb4' }, grid: { color: '#2a3548' } }
      }
    }
  });
})();
</script>
</body>
</html>
"""


def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"${v:,.0f}"


def _cls(v: Optional[float]) -> str:
    if v is None:
        return ""
    return "good" if v >= 0 else "bad"


def load_spy_daily(start: date, end: date) -> pd.Series:
    frames = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="ALPACA",
        start=datetime(start.year, start.month, start.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    raw = frames.get("SPY")
    if raw is None or raw.empty:
        raise RuntimeError("No SPY")
    closes = pd.to_numeric(raw["close"], errors="coerce").dropna()
    closes.index = pd.to_datetime(closes.index).date
    return closes.sort_index()


def build_equity(
    leaps_rows: List[Dict[str, Any]],
    hedge_rows: List[Dict[str, Any]],
    spy: pd.Series,
    capital: float,
) -> List[Dict[str, float]]:
    events_l: List[tuple] = []
    events_h: List[tuple] = []
    for r in leaps_rows:
        events_l.append((str(r["exit_date"])[:10], float(r["pnl_cash"])))
    for r in hedge_rows:
        if r.get("skipped") in (True, "True", "true", 1):
            continue
        if r.get("pnl_cash") is None or (isinstance(r.get("pnl_cash"), float) and math.isnan(r["pnl_cash"])):
            continue
        if not r.get("exit_date"):
            continue
        events_h.append((str(r["exit_date"])[:10], float(r["pnl_cash"])))
    events_l.sort()
    events_h.sort()
    all_dates = sorted(set([d for d, _ in events_l] + [d for d, _ in events_h] + [str(d) for d in spy.index]))
    all_dates = [d for d in all_dates if WINDOW_START.isoformat() <= d <= WINDOW_END.isoformat()]
    i_l = i_h = 0
    cum_l = cum_h = 0.0
    spy0 = float(spy.iloc[0]) if len(spy) else 1.0
    out = []
    for d in all_dates:
        while i_l < len(events_l) and events_l[i_l][0] <= d:
            cum_l += events_l[i_l][1]
            i_l += 1
        while i_h < len(events_h) and events_h[i_h][0] <= d:
            cum_h += events_h[i_h][1]
            i_h += 1
        dd = date.fromisoformat(d)
        spy_px = float(spy.get(dd, spy0)) if dd in spy.index else spy0
        # forward-fill spy
        if dd not in spy.index:
            prior = spy.loc[:dd]
            spy_px = float(prior.iloc[-1]) if len(prior) else spy0
        out.append(
            {
                "date": d,
                "leaps_equity": round(capital + cum_l, 2),
                "hedged_equity": round(capital + cum_l + cum_h, 2),
                "spy_equity": round(capital * (spy_px / spy0), 2),
            }
        )
    # downsample monthly + last
    if len(out) > 80:
        keep = []
        last_m = None
        for row in out:
            m = row["date"][:7]
            if m != last_m:
                keep.append(row)
                last_m = m
        if keep[-1]["date"] != out[-1]["date"]:
            keep.append(out[-1])
        return keep
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hedge-dir", type=Path, default=DEFAULT_HEDGE_DIR)
    ap.add_argument("--leaps-csv", type=Path, default=DEFAULT_LEAPS_CSV)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--max-open", type=int, default=10)
    ap.add_argument("--focus-key", default="", help="matrix key to chart; default best MDD lift")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    matrix_path = args.hedge_dir / "matrix_summary.json"
    if not matrix_path.exists():
        LOG.error("Missing %s — run claimcheck_qqq_leaps_gap_hedge.py first", matrix_path)
        return 1
    summary = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix = summary.get("matrix") or []
    focus_rows = [
        r
        for r in matrix
        if r.get("max_open") == args.max_open and r.get("mark_mode") == "mid_mid"
    ]
    if not focus_rows:
        focus_rows = list(matrix)
    if args.focus_key:
        focus = next((r for r in matrix if r["key"] == args.focus_key), None)
        if focus is None:
            LOG.error("focus key not found: %s", args.focus_key)
            return 1
    else:
        focus = max(
            focus_rows,
            key=lambda r: (r.get("mdd_lift") is not None, r.get("mdd_lift") or -1e18),
        )

    leaps = apply_max_open(leaps_from_csv(args.leaps_csv), args.max_open)
    leaps_rows = [
        {
            "exit_date": t.exit_date.isoformat(),
            "pnl_cash": t.pnl_cash,
        }
        for t in leaps
        if t.exit_date and t.pnl_cash is not None
    ]
    hedge_csv = args.hedge_dir / ("%s_hedges.csv" % focus["key"])
    if hedge_csv.exists():
        hdf = pd.read_csv(hedge_csv)
        hedge_rows = hdf.to_dict(orient="records")
    else:
        hedge_rows = []
        LOG.warning("No hedge CSV for %s", focus["key"])

    spy = load_spy_daily(WINDOW_START, WINDOW_END)
    equity = build_equity(leaps_rows, hedge_rows, spy, args.capital)

    rows_html = []
    for r in matrix:
        lift = r.get("mdd_lift")
        rows_html.append(
            "<tr>"
            f"<td>{r.get('key')}</td>"
            f"<td class='num'>{r.get('leaps_n')}</td>"
            f"<td class='num'>{_fmt_money(r.get('leaps_pnl'))}</td>"
            f"<td class='num'>{_fmt_money(r.get('leaps_mdd'))}</td>"
            f"<td class='num'>{r.get('hedge_n')}</td>"
            f"<td class='num {_cls(r.get('hedge_pnl'))}'>{_fmt_money(r.get('hedge_pnl'))}</td>"
            f"<td class='num {_cls(r.get('combined_pnl'))}'>{_fmt_money(r.get('combined_pnl'))}</td>"
            f"<td class='num'>{r.get('combined_pf')}</td>"
            f"<td class='num'>{_fmt_money(r.get('combined_mdd'))}</td>"
            f"<td class='num {_cls(lift)}'>{lift if lift is not None else '—'}</td>"
            "</tr>"
        )

    kpis = [
        ("Focus", focus["key"]),
        ("LEAPS PnL", _fmt_money(focus.get("leaps_pnl"))),
        ("Hedge PnL", _fmt_money(focus.get("hedge_pnl"))),
        ("Combined PnL", _fmt_money(focus.get("combined_pnl"))),
        ("Combined MDD", _fmt_money(focus.get("combined_mdd"))),
        ("MDD lift", str(focus.get("mdd_lift"))),
        ("Combined PF", str(focus.get("combined_pf"))),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='l'>{lab}</div><div class='v'>{val}</div></div>"
        for lab, val in kpis
    )
    payload = {
        "focus": focus,
        "matrix": matrix,
        "equity": equity,
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capital": args.capital,
            "max_open": args.max_open,
            "verdict_gate": summary.get("verdict_gate"),
        },
    }
    html = (
        TEMPLATE.replace("__TITLE__", "QQQ LEAPS Gap — Low-VIX Hedge Matrix")
        .replace(
            "__SUB__",
            f"Focus {focus['key']} | capital ${_fmt_money(args.capital)[1:]} | "
            f"window {WINDOW_START} → {WINDOW_END}",
        )
        .replace("__KPIS__", kpi_html)
        .replace("__ROWS__", "\n".join(rows_html))
        .replace(
            "__NOTE__",
            "Gate: improve MDD vs LEAPS-only at max_open=10 without destroying PF/CAGR; "
            "collar that kills LEAPS upside = no promote. Theta VIX options available.",
        )
        .replace("__PAYLOAD__", json.dumps(payload, default=str))
    )
    outdir = args.outdir or args.hedge_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "qqq_leaps_gap_hedge_matrix.html"
    out.write_text(html, encoding="utf-8")
    LOG.info("Wrote %s focus=%s", out, focus["key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
