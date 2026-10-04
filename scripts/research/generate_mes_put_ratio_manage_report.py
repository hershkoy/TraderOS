#!/usr/bin/env python3
"""HTML report for take-profit and credit-stop overlays on the 56 DTE put ratio.

Reads the sweep JSON files and rebuilds daily P&amp;L from the cached chains.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_mes_put_ratio_manage_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TP = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-02_tp" / "tp_sweep.json"
DEFAULT_STOP = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop" / "stop_sweep.json"
DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop" / "mes_put_ratio_manage.html"

TP_ORDER = [
    ("hold", "Hold to expiry"),
    ("tp25", "Bank 25% of credit"),
    ("tp50", "Bank 50% of credit"),
    ("tp75", "Bank 75% of credit"),
]
STOP_ORDER = [
    ("tp50", "50% only"),
    ("tp50_sl2", "50% + 2x credit stop"),
    ("tp50_sl3", "50% + 3x credit stop"),
    ("tp50_sl4", "50% + 4x credit stop"),
    ("hold_sl3", "Hold + 3x credit stop"),
    ("hold_sl4", "Hold + 4x credit stop"),
]

# key, label, color, tp_frac, stop multiple. tp_frac 0.75 banks 25% of the credit.
CURVE_BOOKS: List[Tuple[str, str, str, Optional[float], Optional[float]]] = [
    ("hold", "Hold to expiry", "#b0b8c4", None, None),
    ("tp25", "Bank 25%", "#5b9fd4", 0.75, None),
    ("tp50", "Bank 50%", "#26a69a", 0.50, None),
    ("tp75", "Bank 75%", "#ce93d8", 0.25, None),
    ("tp50_sl2", "50% + 2x stop", "#ffb74d", 0.50, 2.0),
    ("tp50_sl3", "50% + 3x stop", "#ef5350", 0.50, 3.0),
    ("tp50_sl4", "50% + 4x stop", "#ab47bc", 0.50, 4.0),
]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _level(block: Dict[str, Any], label: str) -> Dict[str, Any]:
    stats = block.get("stats") or {}
    book = block.get("book") or {}
    return {
        "label": label,
        "n": stats.get("n"),
        "losses": stats.get("losses"),
        "wr": stats.get("win_rate_pct"),
        "mean": stats.get("mean_pnl"),
        "median": stats.get("median_pnl"),
        "total": stats.get("total_pnl"),
        "worst": stats.get("worst"),
        "mae": block.get("mae_worst"),
        "n_mae_10k": block.get("n_mae_lt_10k"),
        "n_mae_18k": block.get("n_mae_lt_18k"),
        "stops": block.get("n_stop") or 0,
        "dd": book.get("max_dd"),
        "open": book.get("worst_open"),
        "open_day": book.get("worst_day"),
        "open_n": book.get("open_on_worst"),
    }


def _stops(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    recovery = (block.get("recovery") or {}).get("rows") or []
    if recovery:
        return recovery
    return block.get("stops") or []


def _align_curves(raw: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    dates = sorted({point["date"] for points in raw.values() for point in points})
    series: Dict[str, Any] = {}
    for key, _label, color, _frac, _stop in CURVE_BOOKS:
        by_day = {point["date"]: point for point in raw.get(key) or []}
        realized: Optional[float] = None
        equity: Optional[float] = None
        realized_data: List[Optional[float]] = []
        equity_data: List[Optional[float]] = []
        for day in dates:
            point = by_day.get(day)
            if point is not None:
                realized = point["realized"]
                equity = point["equity"]
            realized_data.append(realized)
            equity_data.append(equity)
        series[key] = {
            "label": _label,
            "color": color,
            "realized": realized_data,
            "equity": equity_data,
        }
    return {"labels": dates, "series": series}


def load_curves() -> Dict[str, Any]:
    """Daily closed P&L and marked book from the cached SPXW chains."""
    from scripts.research.claimcheck_mes_put_ratio import prepare_setups, run_variant
    from scripts.research.sweep_mes_put_ratio_stop import _book_path
    from utils.api.thetadata.thetadata_client import get_thetadata_client
    from utils.research.forward_factor import list_expirations
    from utils.research.mes_put_ratio import friday_sessions, index_closes, load_spx_eod

    start = date(2021, 10, 1)
    end = date(2026, 10, 1)
    client = get_thetadata_client()
    if not client.is_ready():
        raise RuntimeError("Theta Terminal is not up on :25503")
    spot = load_spx_eod(client, start, end + timedelta(days=80))
    spot_by_day = index_closes(spot)
    expirations = list_expirations(client, "SPXW")
    fridays = friday_sessions(sorted(spot_by_day), start, end)
    cache = ROOT / "reports" / "research" / "mes_put_ratio" / "_cache"
    setups = prepare_setups(
        client,
        fridays,
        expirations,
        symbol="SPXW",
        cache=cache,
        workers=4,
        long_delta=0.14,
        short_delta=0.10,
        target_dte=56,
    )
    raw: Dict[str, List[Dict[str, Any]]] = {}
    for key, _label, _color, frac, stop in CURVE_BOOKS:
        trades = run_variant(
            setups,
            spot_by_day,
            {
                "mode": "mid",
                "leg_slip": 0.0,
                "tp_frac": frac,
                "exit_dte": None,
                "stop_credit_mult": stop,
            },
        )
        book = _book_path(setups, trades)
        raw[key] = [
            {"date": row["date"], "realized": row["realized"], "equity": row["equity"]}
            for row in book.get("curve") or []
        ]
    return _align_curves(raw)


def build_payload(tp: Dict[str, Any], stop: Dict[str, Any], curves: Dict[str, Any]) -> Dict[str, Any]:
    tp_levels = tp.get("levels") or {}
    stop_levels = stop.get("levels") or {}
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "curves": curves,
        "tp": [_level(tp_levels[key], label) for key, label in TP_ORDER if key in tp_levels],
        "stop": [_level(stop_levels[key], label) for key, label in STOP_ORDER if key in stop_levels],
        "focus": {
            key: (tp_levels.get(key) or {}).get("focus") or []
            for key, _label in TP_ORDER
            if key in tp_levels
        },
        "stops": {
            key: _stops(stop_levels[key])
            for key, _label in STOP_ORDER
            if key in stop_levels and key != "tp50"
        },
        "recovery": {
            key: (stop_levels.get(key) or {}).get("recovery") or {}
            for key, _label in STOP_ORDER
            if key in stop_levels
        },
    }


def render(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return HTML.replace("__PAYLOAD__", data)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>56 DTE put ratio — take-profit and credit stop</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:#0d1421; --panel:#131722; --panel2:#1e222d; --border:#2a2e39;
    --text:#d1d4dc; --muted:#787b86; --pos:#26a69a; --neg:#ef5350;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
  .wrap { max-width:1200px; margin:0 auto; padding:20px; }
  .header, .panel { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:16px 18px; margin-bottom:12px; }
  h1 { font-size:20px; color:#fff; font-weight:650; }
  h2 { font-size:15px; color:#fff; margin-bottom:10px; }
  .sub { color:var(--muted); font-size:13px; margin-top:6px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-top:12px; }
  .card { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
  .card .l { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:16px; font-weight:650; margin-top:3px; font-variant-numeric:tabular-nums; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  .chart-box { height:320px; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:900px) { .charts { grid-template-columns:1fr; } }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { padding:7px 8px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0; background:#1a1f2b; }
  tr.loss td { background:rgba(239,83,80,.08); }
  tr.hl td { background:rgba(41,98,255,.12); }
  tr:hover td { background:rgba(41,98,255,.08); }
  .table-wrap { max-height:560px; overflow:auto; border:1px solid var(--border); border-radius:6px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:10px; }
  label { display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
  select { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:7px 10px; font-size:13px; }
  .muted { color:var(--muted); font-size:12px; margin-top:8px; }
  ul { margin:8px 0 0 18px; font-size:13px; }
  li { margin:4px 0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>56 DTE put ratio — early take-profit and a credit stop</h1>
    <div class="sub">Same 233 Friday SPXW packages, mid marks, 2021-10-01 through 2026-10-01. Dollars are one SPX package (x100). A 3x stop means the first daily close whose loss is at least three times the credit received. The fill is that close.</div>
    <div class="cards" id="kpis"></div>
  </div>

  <div class="panel">
    <h2>Closed P&amp;L over time — take-profit</h2>
    <div class="chart-box"><canvas id="closedTp"></canvas></div>
    <p class="muted">Sum of closed trades, carried forward between exits. Open packages are not marked here.</p>
  </div>

  <div class="panel">
    <h2>Closed P&amp;L over time — credit stop</h2>
    <div class="chart-box"><canvas id="closedStop"></canvas></div>
    <p class="muted">Same closed-trade sum. The stop books step down when a daily close crosses the credit multiple.</p>
  </div>

  <div class="panel">
    <h2>Marked book over time</h2>
    <div class="chart-box"><canvas id="marked"></canvas></div>
    <p class="muted">Closed P&amp;L plus the end-of-day mark of packages still open. This is the line that shows the April 2025 hole.</p>
  </div>

  <div class="panel">
    <h2>Take-profit</h2>
    <div class="table-wrap"><table id="tp"></table></div>
    <p class="muted">25% means close once a quarter of the credit is banked. 50% means half. 75% means three quarters. Every target filled on all 233 trades.</p>
  </div>

  <div class="panel">
    <h2>Credit stop on top of 50%, and on the hold book</h2>
    <div class="table-wrap"><table id="stop"></table></div>
  </div>

  <div class="panel">
    <h2>Trades the stop closed</h2>
    <div class="row">
      <div><label>Book</label><select id="which"></select></div>
    </div>
    <div id="recap" class="muted"></div>
    <div class="table-wrap" style="margin-top:8px"><table id="stops"></table></div>
    <p class="muted">Highlighted rows are 2025-03-21 and 2025-03-28. "50% P&amp;L" is the same entry with the take-profit and no stop. "Hold P&amp;L" is the expiry result.</p>
  </div>

  <div class="panel">
    <h2>How the path behaves</h2>
    <ul>
      <li>Banking 50% retires the February packages before April and removes both expiry losses. Total P&amp;L falls from $341,855 to $193,591. Peak-to-trough falls from $229,776 to $43,625.</li>
      <li>2025-03-21 and 2025-03-28 had no 50% profit yet, so they still printed −$21,725 and −$17,945 before the target filled, then closed green.</li>
      <li>A 3x stop on that 50% book locks 9 trades for −$78,530. Those same nine closed for +$7,635 with the target alone. All nine were green under 50%.</li>
      <li>The March stop fills are the 2025-04-04 close: −$17,239 and −$13,659, against lines near −$4,425 and −$4,635. The prior close was still inside the line.</li>
    </ul>
  </div>
</div>
<script>
const PAYLOAD = __PAYLOAD__;
const money = (n) => {
  if (n == null || Number.isNaN(Number(n))) return "";
  const v = Number(n);
  return (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits:0});
};
const cls = (n) => Number(n) < 0 ? "neg" : "pos";
const cell = (n) => `<td class="${cls(n)}">${money(n)}</td>`;

function kpis() {
  const tp = PAYLOAD.tp.find(r => r.label.startsWith("Bank 50"));
  const sl = PAYLOAD.stop.find(r => r.label === "50% + 3x credit stop");
  const hold = PAYLOAD.tp.find(r => r.label.startsWith("Hold"));
  const items = [
    ["Hold total", money(hold.total), cls(hold.total)],
    ["50% total", money(tp.total), cls(tp.total)],
    ["50% + 3x total", money(sl.total), cls(sl.total)],
    ["Hold peak-to-trough", money(hold.dd), "neg"],
    ["50% peak-to-trough", money(tp.dd), "neg"],
    ["50% + 3x peak-to-trough", money(sl.dd), "neg"],
    ["3x stops", String(sl.stops), "neg"],
    ["3x worst close", money(sl.worst), "neg"],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c]) =>
    `<div class="card"><div class="l">${l}</div><div class="v ${c}">${v}</div></div>`
  ).join("");
}

function tpTable() {
  const head = `<tr><th>Book</th><th>Losses</th><th>Total</th><th>Median</th><th>Worst close</th><th>Worst mark</th><th>Marks &lt; −$10k</th><th>Peak-to-trough</th><th>Worst open book</th><th>That day</th><th>Open</th></tr>`;
  const body = PAYLOAD.tp.map(r => `<tr>
    <td>${r.label}</td><td class="${r.losses?"neg":""}">${r.losses}</td>
    ${cell(r.total)}${cell(r.median)}${cell(r.worst)}${cell(r.mae)}
    <td>${r.n_mae_10k}</td>${cell(-Math.abs(r.dd))}<td class="neg">${money(r.open)}</td>
    <td>${r.open_day || ""}</td><td>${r.open_n}</td></tr>`).join("");
  document.getElementById("tp").innerHTML = head + body;
}

function stopTable() {
  const head = `<tr><th>Book</th><th>Stops</th><th>Losses</th><th>Total</th><th>Worst close</th><th>Worst mark</th><th>Peak-to-trough</th><th>Worst open book</th></tr>`;
  const body = PAYLOAD.stop.map(r => `<tr>
    <td>${r.label}</td><td class="${r.stops?"neg":""}">${r.stops}</td>
    <td class="${r.losses?"neg":""}">${r.losses}</td>
    ${cell(r.total)}${cell(r.worst)}${cell(r.mae)}${cell(-Math.abs(r.dd))}<td class="neg">${money(r.open)}</td></tr>`).join("");
  document.getElementById("stop").innerHTML = head + body;
}

function fillSelect() {
  const sel = document.getElementById("which");
  const labels = {
    tp50_sl2: "50% + 2x",
    tp50_sl3: "50% + 3x",
    tp50_sl4: "50% + 4x",
    hold_sl3: "Hold + 3x",
    hold_sl4: "Hold + 4x",
  };
  Object.keys(PAYLOAD.stops).forEach(key => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = labels[key] || key;
    sel.appendChild(opt);
  });
  sel.value = "tp50_sl3";
  sel.addEventListener("change", renderStops);
}

function renderStops() {
  const key = document.getElementById("which").value;
  const rows = PAYLOAD.stops[key] || [];
  const rec = PAYLOAD.recovery[key] || {};
  const recap = document.getElementById("recap");
  if (rec.n_stop != null) {
    recap.textContent = `${rec.n_stop} stops. ${rec.n_tp50_green} of them finished green under 50% with no stop (those entries ${money(rec.same_entries_tp50_pnl)} vs the stop book ${money(rec.stop_pnl)}). ${rec.n_hold_green} expired green. Hold P&L of the same entries: ${money(rec.same_entries_hold_pnl)}.`;
  } else {
    recap.textContent = "";
  }
  const head = `<tr><th>Entry</th><th>Stop exit</th><th>Credit</th><th>Stop P&amp;L</th><th>50% P&amp;L</th><th>Hold P&amp;L</th><th>50% worst mark</th></tr>`;
  const body = rows.map(r => {
    const hl = (r.entry === "2025-03-21" || r.entry === "2025-03-28") ? "hl" : "";
    const loss = Number(r.stop_pnl) < 0 ? "loss" : "";
    return `<tr class="${hl || loss}"><td>${r.entry}</td><td>${r.stop_exit || r.exit || ""}</td>
      ${cell(r.credit)}${cell(r.stop_pnl)}${cell(r.tp50_pnl)}${cell(r.hold_pnl)}${cell(r.tp50_mae)}</tr>`;
  }).join("");
  document.getElementById("stops").innerHTML = head + body;
}

const axis = {
  ticks: { color:"#787b86" },
  grid: { color:"#2a2e39" }
};
function moneyTick(v) { return money(v); }

function lineChart(canvasId, keys, field) {
  const curves = PAYLOAD.curves;
  const sets = keys.map(key => {
    const series = curves.series[key];
    return {
      label: series.label,
      data: series[field],
      borderColor: series.color,
      backgroundColor: "transparent",
      borderWidth: 1.6,
      pointRadius: 0,
      spanGaps: true,
      tension: 0
    };
  });
  new Chart(document.getElementById(canvasId), {
    type: "line",
    data: { labels: curves.labels, datasets: sets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#d1d4dc" } } },
      scales: {
        x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 8 } },
        y: { ...axis, ticks: { ...axis.ticks, callback: moneyTick } }
      }
    }
  });
}

function charts() {
  lineChart("closedTp", ["hold", "tp25", "tp50", "tp75"], "realized");
  lineChart("closedStop", ["tp50", "tp50_sl2", "tp50_sl3", "tp50_sl4"], "realized");
  lineChart("marked", ["hold", "tp50", "tp50_sl3"], "equity");
}

kpis();
tpTable();
stopTable();
fillSelect();
renderStops();
charts();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HTML for put-ratio take-profit and credit stop")
    parser.add_argument("--tp", default=str(DEFAULT_TP))
    parser.add_argument("--stop", default=str(DEFAULT_STOP))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    tp_path = Path(args.tp)
    stop_path = Path(args.stop)
    if not tp_path.exists() or not stop_path.exists():
        print("missing sweep json", tp_path, stop_path)
        return 1
    payload = build_payload(_load(tp_path), _load(stop_path), load_curves())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
