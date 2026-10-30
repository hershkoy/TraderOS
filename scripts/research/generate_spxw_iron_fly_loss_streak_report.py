#!/usr/bin/env python3
"""HTML report for SPXW iron-fly L5/S3/P2 loss-streak overlay.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_spxw_iron_fly_loss_streak_report.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = (
    ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "2026-10-27_from_20220511"
)
START_EQUITY = 25_000.0
RULE = "count_L5_S3_P2"


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _equity(rows: List[Dict[str, Any]], start: float = START_EQUITY) -> Dict[str, Any]:
    eq = float(start)
    labels: List[str] = []
    data: List[float] = []
    for r in rows:
        eq += float(r["pnl"])
        labels.append(r["date"])
        data.append(round(eq, 2))
    peak = start
    max_dd = 0.0
    for v in data:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, v / peak - 1.0)
    return {
        "labels": labels,
        "data": data,
        "end": round(eq, 2),
        "max_dd_path": max_dd,
    }


def _book_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [float(r["pnl"]) for r in rows]
    if not pnls:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw = sum(wins)
    gl = -sum(losses)
    return {
        "n": len(pnls),
        "wr": 100.0 * len(wins) / len(pnls),
        "avg_win": (sum(wins) / len(wins)) if wins else None,
        "avg_loss": (sum((-x) for x in losses) / len(losses)) if losses else None,
        "expectancy": sum(pnls) / len(pnls),
        "total": sum(pnls),
        "pf": (gw / gl) if gl > 0 else None,
    }


def _load_natural_trades(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    df = df.loc[df["fill"].astype(str) == "natural"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        pnl = _num(r.get("pnl"))
        if pnl is None:
            continue
        rows.append(
            {
                "date": r["date"].isoformat(),
                "vix": _num(r.get("vix")),
                "spot": _num(r.get("spot")),
                "settle": _num(r.get("settle")),
                "short_put": _num(r.get("short_put")),
                "short_call": _num(r.get("short_call")),
                "credit": _num(r.get("credit")),
                "pnl": round(pnl, 2),
                "won": bool(pnl > 0),
            }
        )
    return rows


def build_payload(run_root: Path, rule: str = RULE) -> Dict[str, Any]:
    streak_dir = run_root / "loss_streak"
    summary = json.loads(
        (streak_dir / "summary_loss_streak.json").read_text(encoding="utf-8")
    )
    kept_path = streak_dir / f"kept_{rule}.csv"
    kept_dates: Set[str] = set()
    if kept_path.exists():
        kd = pd.read_csv(kept_path)
        kept_dates = {str(x)[:10] for x in kd["date"].astype(str).tolist()}

    all_rows = _load_natural_trades(run_root / "trades.csv")
    for r in all_rows:
        r["kept"] = r["date"] in kept_dates
    taken = [r for r in all_rows if r["kept"]]
    skipped = [r for r in all_rows if not r["kept"]]

    rule_row = next((r for r in summary.get("locked") or [] if r.get("label") == rule), {})
    baseline = summary.get("baseline") or {}
    phase5 = rule_row.get("phase5") or {}
    base_p5 = baseline.get("phase5") or {}

    # Combined equity chart: baseline every trade date; overlay only on kept dates
    # (step flat when skipped so x-axis matches).
    base_eq = START_EQUITY
    ov_eq = START_EQUITY
    labels: List[str] = []
    base_series: List[float] = []
    ov_series: List[float] = []
    for r in all_rows:
        base_eq += float(r["pnl"])
        if r["kept"]:
            ov_eq += float(r["pnl"])
        labels.append(r["date"])
        base_series.append(round(base_eq, 2))
        ov_series.append(round(ov_eq, 2))

    window = f"{all_rows[0]['date']} to {all_rows[-1]['date']}" if all_rows else ""
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "window": window,
        "rule": rule,
        "rule_desc": "Active until 5 consecutive taken losses, then Sit 3 signals (paper/skip), then Probe 2; any probe win -> Active",
        "start_equity": START_EQUITY,
        "verdict": summary.get("verdict") or {},
        "hostile_longer": summary.get("hostile_longer_vs_2024_2026")
        or summary.get("hostile_longer")
        or {},
        "streak_diag": summary.get("streak_diag") or [],
        "baseline": {
            "stats": _book_stats(all_rows),
            "years": baseline.get("years") or {},
            "phase5": base_p5,
        },
        "overlay": {
            "stats": _book_stats(taken),
            "years": rule_row.get("years") or {},
            "phase5": phase5,
            "diag": rule_row.get("diag") or {},
            "n_skipped": len(skipped),
        },
        "equity": {
            "labels": labels,
            "baseline": base_series,
            "overlay": ov_series,
            "baseline_end": round(base_eq, 2),
            "overlay_end": round(ov_eq, 2),
        },
        "trades": all_rows,
        "taken": taken,
        "skipped": skipped,
    }


def render_html(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, default=str)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SPXW iron fly — L5/S3/P2 loss-streak overlay</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#0f1218; --panel:#171b24; --line:#2a3140; --text:#e7ecf3; --muted:#8b93a7;
  --pos:#3dcf8e; --neg:#f07178; --accent:#6aa6ff; --warn:#e6b35a; --skip:#7a6bb5;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.45 "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:24px 18px 48px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-0.02em; }}
h2 {{ font-size:18px; margin:0 0 12px; }}
h3 {{ font-size:15px; margin:16px 0 8px; color:var(--accent); }}
.sub {{ color:var(--muted); margin-bottom:18px; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }}
.badge.soft {{ background:#2a3a28; color:var(--pos); }}
.badge.nowire {{ background:#3a2e1f; color:var(--warn); }}
.badge.fail {{ background:#3a2226; color:var(--neg); }}
.badge.near {{ background:#1f2f3a; color:var(--accent); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:14px 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }}
.card {{ background:#12161e; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.card .l {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.04em; }}
.card .v {{ font-size:18px; font-weight:650; margin-top:4px; }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.chart {{ height:340px; }}
.table-wrap {{ overflow:auto; max-height:520px; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child, td.left, th.left {{ text-align:left; }}
tr.loss {{ background:rgba(240,113,120,0.06); }}
tr.skip {{ background:rgba(122,107,181,0.12); color:var(--muted); }}
.row {{ display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:10px; }}
label {{ display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }}
select, input {{ background:#12161e; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:7px 10px; }}
.grid2 {{ display:grid; grid-template-columns:1.15fr 1fr; gap:14px; }}
@media (max-width:900px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
ul.notes {{ margin:0; padding-left:18px; color:var(--muted); }}
ul.notes li {{ margin:0 0 6px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SPXW 0DTE iron fly — L5/S3/P2 sit-out</h1>
  <div class="sub">Consecutive-loss circuit breaker on the natural settle book. Window <span id="window"></span>. Generated <span id="gen"></span>.</div>
  <div id="badges"></div>
  <p class="sub" id="ruledesc"></p>

  <div class="panel">
    <h2>Headline (overlay vs baseline)</h2>
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Equity ($25k start, 1 lot)</h2>
      <div class="chart"><canvas id="eq"></canvas></div>
    </div>
    <div class="panel">
      <h2>Phase-5 vs SPY</h2>
      <div class="table-wrap"><table id="phase5"></table></div>
      <h3>Year buckets (taken trades)</h3>
      <div class="table-wrap"><table id="years"></table></div>
    </div>
  </div>

  <div class="panel">
    <h2>Streak diagnostic</h2>
    <div class="table-wrap"><table id="diag"></table></div>
    <ul class="notes" id="notes"></ul>
  </div>

  <div class="panel">
    <h2>Trades</h2>
    <div class="row">
      <div><label>Show</label>
        <select id="filter">
          <option value="all">All (skipped greyed)</option>
          <option value="taken">Taken only</option>
          <option value="skip">Skipped only</option>
          <option value="loss">Taken losses</option>
        </select>
      </div>
      <div><label>Find date</label><input id="q" placeholder="2023-07"/></div>
    </div>
    <div class="table-wrap"><table id="trades"></table></div>
  </div>
</div>
<script>
const PAYLOAD = {blob};
const money = (n) => {{
  if (n == null || Number.isNaN(n)) return "";
  const s = Math.abs(n).toLocaleString(undefined, {{maximumFractionDigits:0}});
  return (n < 0 ? "-$" : "$") + s;
}};
const pct = (n, d=1) => (n == null || Number.isNaN(n)) ? "" : (100*n).toFixed(d) + "%";
const num = (n, d=2) => (n == null || Number.isNaN(n)) ? "" : Number(n).toLocaleString(undefined, {{maximumFractionDigits:d, minimumFractionDigits:d}});
const cls = (n) => (n == null || n === 0) ? "" : (n < 0 ? "neg" : "pos");

document.getElementById("window").textContent = PAYLOAD.window || "";
document.getElementById("gen").textContent = PAYLOAD.generated || "";
document.getElementById("ruledesc").textContent = PAYLOAD.rule_desc || "";

(function badges() {{
  const v = PAYLOAD.verdict || {{}};
  const p5 = (PAYLOAD.overlay.phase5 || {{}});
  const bits = [];
  bits.push('<span class="badge near">Near-miss MDD</span>');
  bits.push('<span class="badge nowire">Not wired</span>');
  if (p5.gate_pass) bits.push('<span class="badge soft">Phase-5 PASS</span>');
  else bits.push('<span class="badge fail">Phase-5 FAIL</span>');
  if (!v.soft_promote) bits.push('<span class="badge nowire">No soft-promote</span>');
  document.getElementById("badges").innerHTML = bits.join(" ") +
    `<div class="sub" style="margin-top:8px">${{v.reason || p5.gate || ""}}</div>`;
}})();

(function kpis() {{
  const o = PAYLOAD.overlay.stats || {{}};
  const b = PAYLOAD.baseline.stats || {{}};
  const os = (PAYLOAD.overlay.phase5 || {{}}).strat || {{}};
  const spy = (PAYLOAD.overlay.phase5 || {{}}).spy || {{}};
  const d = PAYLOAD.overlay.diag || {{}};
  const items = [
    ["Taken n", o.n, ""],
    ["Skipped", PAYLOAD.overlay.n_skipped, ""],
    ["Sit episodes", d.n_sit_episodes, ""],
    ["WR", num(o.wr,1) + "%", ""],
    ["E / trade", money(o.expectancy), cls(o.expectancy)],
    ["PF", o.pf == null ? "" : num(o.pf,2), ""],
    ["Total 1-lot", money(o.total), cls(o.total)],
    ["End equity", money(PAYLOAD.equity.overlay_end), "pos"],
    ["Baseline end", money(PAYLOAD.equity.baseline_end), ""],
    ["Sharpe", num(os.sharpe,2), ""],
    ["SPY Sharpe", num(spy.sharpe,2), ""],
    ["MDD", pct(Math.abs(os.max_drawdown||0),1), "neg"],
    ["SPY MDD", pct(Math.abs(spy.max_drawdown||0),1), "neg"],
    ["Invested %", pct(os.pct_time_invested||0,0), ""],
    ["Baseline n", b.n, ""],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c]) =>
    `<div class="card"><div class="l">${{l}}</div><div class="v ${{c}}">${{v}}</div></div>`
  ).join("");
}})();

(function phase5() {{
  const o = (PAYLOAD.overlay.phase5 || {{}}).strat || {{}};
  const spy = (PAYLOAD.overlay.phase5 || {{}}).spy || {{}};
  const b = (PAYLOAD.baseline.phase5 || {{}}).strat || {{}};
  const head = `<tr><th class="left"></th><th>L5/S3/P2</th><th>Baseline</th><th>SPY</th></tr>`;
  const rows = [
    ["Gate", PAYLOAD.overlay.phase5.gate || "", PAYLOAD.baseline.phase5.gate || "", ""],
    ["CAGR", pct(o.cagr,1), pct(b.cagr,1), pct(spy.cagr,1)],
    ["Sharpe", num(o.sharpe,2), num(b.sharpe,2), num(spy.sharpe,2)],
    ["Max DD", pct(Math.abs(o.max_drawdown||0),1), pct(Math.abs(b.max_drawdown||0),1), pct(Math.abs(spy.max_drawdown||0),1)],
    ["Calmar", num(o.calmar,2), num(b.calmar,2), num(spy.calmar,2)],
    ["Invested %", pct(o.pct_time_invested||0,0), pct(b.pct_time_invested||0,0), pct(spy.pct_time_invested||0,0)],
    ["n trades", o.n_trades, b.n_trades, ""],
  ];
  document.getElementById("phase5").innerHTML = head + rows.map(r =>
    `<tr><td class="left">${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td><td>${{r[3]}}</td></tr>`
  ).join("");
}})();

(function years() {{
  const y = PAYLOAD.overlay.years || {{}};
  const head = `<tr><th class="left">Bucket</th><th>n</th><th>WR</th><th>E</th><th>PF</th><th>Total</th></tr>`;
  const keys = ["2022","2023","2022-2023","2024-2026"];
  const body = keys.filter(k => y[k]).map(k => {{
    const s = y[k];
    return `<tr><td class="left">${{k}}</td><td>${{s.n||""}}</td><td>${{s.win_rate==null?"":num(100*s.win_rate,1)+"%"}}</td><td class="${{cls(s.expectancy)}}">${{money(s.expectancy)}}</td><td>${{s.pf==null?"":num(s.pf,2)}}</td><td class="${{cls(s.total)}}">${{money(s.total)}}</td></tr>`;
  }}).join("");
  document.getElementById("years").innerHTML = head + body;
}})();

(function diag() {{
  const rows = PAYLOAD.streak_diag || [];
  const head = `<tr><th class="left">Bucket</th><th>n</th><th>Losses</th><th>Max consec</th><th>Mean run</th><th>Runs≥5</th><th>% in ≥5</th></tr>`;
  const body = rows.map(r =>
    `<tr><td class="left">${{r.bucket}}</td><td>${{r.n_trades}}</td><td>${{r.n_losses}}</td><td>${{r.max_consec_loss}}</td><td>${{num(r.mean_run_len,2)}}</td><td>${{r.n_runs_ge5}}</td><td>${{num(r.pct_trades_in_run_ge5,1)}}%</td></tr>`
  ).join("");
  document.getElementById("diag").innerHTML = head + body;
  const h = PAYLOAD.hostile_longer || {{}};
  const notes = [
    `Hostile 2022-23 longer-tailed vs 2024-26: ${{h.longer_tailed === false ? "no" : (h.longer_tailed ? "yes" : "n/a")}} (max ${{h.hostile_max}} vs ${{h.compare_max}}).`,
    "Phase-5 FAIL is MDD only: overlay 20.9% vs SPY 19.0% (~1.9pp). Sharpe 1.86 vs SPY 0.96; invested ~56%.",
    "Locked L5/S3/P2 — no soft-promote (gate miss). Do not L/S/P-fish further.",
  ];
  document.getElementById("notes").innerHTML = notes.map(x => `<li>${{x}}</li>`).join("");
}})();

(function chart() {{
  const eq = PAYLOAD.equity;
  new Chart(document.getElementById("eq"), {{
    type: "line",
    data: {{
      labels: eq.labels,
      datasets: [
        {{
          label: "L5/S3/P2 (taken)",
          data: eq.overlay,
          borderColor: "#6aa6ff",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.05,
        }},
        {{
          label: "Baseline natural",
          data: eq.baseline,
          borderColor: "#8b93a7",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 1.5,
          borderDash: [4,3],
          tension: 0.05,
        }},
      ],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{ legend: {{ labels: {{ color: "#c5cad6" }} }} }},
      scales: {{
        x: {{ ticks: {{ color: "#8b93a7", maxTicksLimit: 10 }}, grid: {{ color: "#2a3140" }} }},
        y: {{ ticks: {{ color: "#8b93a7" }}, grid: {{ color: "#2a3140" }} }},
      }},
    }},
  }});
}})();

function renderTrades() {{
  const f = document.getElementById("filter").value;
  const q = (document.getElementById("q").value || "").trim();
  let rows = PAYLOAD.trades || [];
  if (f === "taken") rows = rows.filter(r => r.kept);
  if (f === "skip") rows = rows.filter(r => !r.kept);
  if (f === "loss") rows = rows.filter(r => r.kept && r.pnl <= 0);
  if (q) rows = rows.filter(r => r.date.includes(q));
  const head = `<tr><th class="left">Date</th><th>Status</th><th>VIX</th><th>Spot</th><th>Settle</th><th>Put</th><th>Call</th><th>Credit</th><th>P&amp;L</th></tr>`;
  const body = rows.map(r => {{
    const st = r.kept ? "taken" : "skip";
    const trc = r.kept ? (r.pnl <= 0 ? "loss" : "") : "skip";
    return `<tr class="${{trc}}"><td class="left">${{r.date}}</td><td>${{st}}</td><td>${{num(r.vix,2)}}</td><td>${{num(r.spot,1)}}</td><td>${{num(r.settle,1)}}</td><td>${{num(r.short_put,0)}}</td><td>${{num(r.short_call,0)}}</td><td>${{num(r.credit,2)}}</td><td class="${{cls(r.pnl)}}">${{money(r.pnl)}}</td></tr>`;
  }}).join("");
  document.getElementById("trades").innerHTML = head + body;
}}
renderTrades();
document.getElementById("filter").addEventListener("change", renderTrades);
document.getElementById("q").addEventListener("input", renderTrades);
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rule", default=RULE)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <root>/loss_streak/spxw_iron_fly_L5_S3_P2_report.html",
    )
    args = ap.parse_args()
    out = args.out or (
        args.root / "loss_streak" / "spxw_iron_fly_L5_S3_P2_report.html"
    )
    payload = build_payload(args.root, rule=args.rule)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
