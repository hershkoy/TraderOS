#!/usr/bin/env python3
"""HTML account simulator for the MES-sized 56 DTE put ratio.

Reads mes_daily.json from export_mes_put_ratio_daily.py.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_mes_put_ratio_account_report.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop" / "mes_daily.json"
DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop" / "mes_put_ratio_account.html"


def render(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return HTML.replace("__PAYLOAD__", data)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MES put ratio — $50k account</title>
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
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin-top:12px; }
  .card { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
  .card .l { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:16px; font-weight:650; margin-top:3px; font-variant-numeric:tabular-nums; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  .chart-box { height:380px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { padding:7px 8px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0; background:#1a1f2b; }
  tr.loss td { background:rgba(239,83,80,.08); }
  .table-wrap { max-height:360px; overflow:auto; border:1px solid var(--border); border-radius:6px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:end; }
  label { display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
  input, select { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:7px 10px; font-size:13px; }
  input[type=range] { width:180px; padding:0; }
  .muted { color:var(--muted); font-size:12px; margin-top:8px; }
  ul { margin:8px 0 0 18px; font-size:13px; }
  li { margin:4px 0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>MES 2x1 put ratio in a cash account</h1>
    <div class="sub">Buys use the end-of-day bid/ask mid. There is no separate 5pm candle in the cache; the Theta daily quote is the close bid and ask. A 50% order fills at that limit once the close mid is through it. The yellow line still books the better close. Natural bid/ask buys the long at the ask and sells the shorts at the bid, then fills the same 50% limit.</div>
    <div class="row" style="margin-top:14px">
      <div>
        <label>MES packages each Friday</label>
        <input id="contracts" type="range" min="1" max="10" value="1"/>
        <span id="contractsLabel">1</span>
      </div>
      <div>
        <label>Starting cash</label>
        <input id="cash" type="number" min="1000" step="1000" value="50000"/>
      </div>
      <div>
        <label>Margin chart book</label>
        <select id="book"></select>
      </div>
    </div>
    <div class="cards" id="kpis"></div>
  </div>

  <div class="panel">
    <h2>Account equity over time</h2>
    <div class="chart-box"><canvas id="equity"></canvas></div>
    <p class="muted">Each line is starting cash plus that day's marked P&amp;L, including packages that have not exited. It is not the sum of expiry results only.</p>
  </div>

  <div class="panel">
    <h2 id="marginTitle">Equity and modeled buying power</h2>
    <div class="chart-box"><canvas id="margin"></canvas></div>
    <p class="muted">Buying power is a model, not CME SPAN: $500 per short put while the package loss is within 1x the credit, rising to $2,000 per short at a 4x-credit loss. Two shorts per package. The long put is not given a margin offset in this sketch. Buffer is equity minus that requirement.</p>
  </div>

  <div class="panel">
    <h2>Tightest buffer days</h2>
    <div class="table-wrap"><table id="worst"></table></div>
  </div>

  <div class="panel">
    <h2>What is scaled</h2>
    <ul>
      <li>Option P&amp;L scales by 5/100 per MES package. Commission stays $0.65 per leg ($3.90 to open and close), not 1/20 of the SPX fee.</li>
      <li>The green 50% line fills the buy-to-close at half the credit. The yellow line fills that same signal at the close mid, which is often a better price than the limit. Natural bid/ask is the orange line.</li>
      <li>Margin expands only while a package is open and underwater. It drops back to the calm rate after the 50% order fills.</li>
    </ul>
  </div>
</div>
<script>
const RAW = __PAYLOAD__;
const BOOKS = ["hold", "tp25", "tp50", "tp50_close", "tp75", "natural_tp50", "tp50_sl3"];
const COLORS = {
  hold: "#787b86",
  tp25: "#5b9fd4",
  tp50: "#26a69a",
  tp50_close: "#ffee58",
  tp75: "#ab47bc",
  natural_tp50: "#ffa726",
  tp50_sl3: "#ef5350"
};
const money = (n) => {
  const v = Number(n);
  return (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits:0});
};
const cls = (n) => Number(n) < 0 ? "neg" : "pos";

function inputs() {
  return {
    n: Math.max(1, Number(document.getElementById("contracts").value) || 1),
    cash: Number(document.getElementById("cash").value) || 0,
    book: document.getElementById("book").value || "tp50"
  };
}

function series(key, n, cash) {
  const book = RAW.books[key];
  const feeScale = n * RAW.fee_per_package * (1 - RAW.mes_mult / RAW.spx_mult);
  const px = n * RAW.mes_mult / RAW.spx_mult;
  const equity = [];
  const margin = [];
  const open = [];
  const daily = [];
  let prev = cash;
  for (let i = 0; i < book.dates.length; i++) {
    const eq = cash + book.equity[i] * px - book.closed[i] * feeScale;
    equity.push(eq);
    margin.push(book.margin[i] * n);
    open.push(book.open[i] * px);
    daily.push(eq - prev);
    prev = eq;
  }
  return { dates: book.dates, equity, margin, open, daily, nOpen: book.n, label: book.label };
}

function sharedDates() {
  const set = new Set();
  BOOKS.forEach(key => RAW.books[key].dates.forEach(d => set.add(d)));
  return [...set].sort();
}

function project(s, dates) {
  const at = {};
  s.dates.forEach((d, i) => { at[d] = i; });
  const equity = [];
  const margin = [];
  const open = [];
  const daily = [];
  const nOpen = [];
  let lastEq = null;
  let lastMg = 0;
  let lastOpen = 0;
  let lastN = 0;
  dates.forEach(d => {
    const i = at[d];
    if (i == null) {
      equity.push(lastEq);
      margin.push(lastMg);
      open.push(lastOpen);
      nOpen.push(lastN);
      daily.push(0);
      return;
    }
    lastEq = s.equity[i];
    lastMg = s.margin[i];
    lastOpen = s.open[i];
    lastN = s.nOpen[i];
    equity.push(lastEq);
    margin.push(lastMg);
    open.push(lastOpen);
    nOpen.push(lastN);
    daily.push(s.daily[i]);
  });
  return { dates, equity, margin, open, daily, nOpen, label: s.label };
}

function summarize(s) {
  let peak = s.equity[0];
  let maxDd = 0;
  let minEq = s.equity[0];
  let minEqI = 0;
  let maxMargin = s.margin[0];
  let maxMarginI = 0;
  let minBuffer = s.equity[0] - s.margin[0];
  let minBufferI = 0;
  let breach = 0;
  let maxOpen = 0;
  let worstOpen = 0;
  const rows = [];
  for (let i = 0; i < s.equity.length; i++) {
    const eq = s.equity[i];
    const mg = s.margin[i];
    const buffer = eq - mg;
    if (eq > peak) peak = eq;
    const dd = peak - eq;
    if (dd > maxDd) maxDd = dd;
    if (eq < minEq) { minEq = eq; minEqI = i; }
    if (mg > maxMargin) { maxMargin = mg; maxMarginI = i; }
    if (buffer < minBuffer) { minBuffer = buffer; minBufferI = i; }
    if (buffer < 0) breach += 1;
    if (s.nOpen[i] > maxOpen) maxOpen = s.nOpen[i];
    if (s.open[i] < worstOpen) worstOpen = s.open[i];
    rows.push({ i, date: s.dates[i], eq, mg, buffer, open: s.open[i], n: s.nOpen[i], daily: s.daily[i] });
  }
  rows.sort((a, b) => a.buffer - b.buffer);
  return {
    end: s.equity[s.equity.length - 1],
    minEq, minEqDay: s.dates[minEqI],
    maxDd, maxMargin, maxMarginDay: s.dates[maxMarginI],
    minBuffer, minBufferDay: s.dates[minBufferI],
    breach, maxOpen, worstOpen,
    worst: rows.slice(0, 8)
  };
}

function renderKpis(stats) {
  const items = [
    ["Ending equity", money(stats.end), cls(stats.end)],
    ["Lowest equity", money(stats.minEq), cls(stats.minEq - 1)],
    ["Peak-to-trough", money(stats.maxDd), "neg"],
    ["Worst open mark", money(stats.worstOpen), "neg"],
    ["Max buying power", money(stats.maxMargin), ""],
    ["Tightest buffer", money(stats.minBuffer), cls(stats.minBuffer)],
    ["Days buffer < 0", String(stats.breach), stats.breach ? "neg" : ""],
    ["Max packages open", String(stats.maxOpen), ""]
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c]) =>
    `<div class="card"><div class="l">${l}</div><div class="v ${c}">${v}</div></div>`
  ).join("");
}

function renderWorst(stats) {
  const head = `<tr><th>Date</th><th>Packages</th><th>Open mark</th><th>Day change</th><th>Equity</th><th>Buying power</th><th>Buffer</th></tr>`;
  const body = stats.worst.map(r => `<tr class="${r.buffer<0?"loss":""}">
    <td>${r.date}</td><td>${r.n}</td><td class="${cls(r.open)}">${money(r.open)}</td>
    <td class="${cls(r.daily)}">${money(r.daily)}</td>
    <td class="${cls(r.eq)}">${money(r.eq)}</td><td>${money(r.mg)}</td>
    <td class="${cls(r.buffer)}">${money(r.buffer)}</td></tr>`).join("");
  document.getElementById("worst").innerHTML = head + body;
}

const axis = { ticks: { color:"#787b86", maxTicksLimit: 8 }, grid: { color:"#2a2e39" } };
const yAxis = { ...axis, ticks: { color:"#787b86", callback: (v) => money(v) } };
let equityChart;
let marginChart;

function draw() {
  const { n, cash, book } = inputs();
  document.getElementById("contractsLabel").textContent = String(n);
  const chosenRaw = series(book, n, cash);
  const stats = summarize(chosenRaw);
  renderKpis(stats);
  renderWorst(stats);
  document.getElementById("marginTitle").textContent = "Equity and modeled buying power — " + chosenRaw.label;
  const labels = sharedDates();
  const chosen = project(chosenRaw, labels);

  const sets = BOOKS.map(key => {
    const s = project(key === book ? chosenRaw : series(key, n, cash), labels);
    return {
      label: s.label,
      data: s.equity,
      borderColor: COLORS[key],
      backgroundColor: "transparent",
      borderWidth: key === book ? 2.2 : 1.2,
      pointRadius: 0,
      tension: 0
    };
  });
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(document.getElementById("equity"), {
    type: "line",
    data: { labels, datasets: sets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#d1d4dc" } } },
      scales: { x: axis, y: yAxis }
    }
  });
  if (marginChart) marginChart.destroy();
  marginChart = new Chart(document.getElementById("margin"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Equity", data: chosen.equity, borderColor: COLORS[book] || "#26a69a", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, tension: 0 },
        { label: "Buying power", data: chosen.margin, borderColor: "#ffa726", backgroundColor: "transparent", borderWidth: 1.6, pointRadius: 0, tension: 0 },
        { label: "Buffer", data: chosen.equity.map((v, i) => v - chosen.margin[i]), borderColor: "#5b9fd4", backgroundColor: "transparent", borderWidth: 1.2, pointRadius: 0, tension: 0 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#d1d4dc" } } },
      scales: { x: axis, y: yAxis }
    }
  });
}

const sel = document.getElementById("book");
BOOKS.forEach(key => {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = RAW.books[key].label;
  sel.appendChild(opt);
});
sel.value = "tp50";
document.getElementById("contracts").addEventListener("input", draw);
document.getElementById("cash").addEventListener("input", draw);
sel.addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MES account HTML for the 56 DTE put ratio")
    parser.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    src = Path(args.src)
    if not src.exists():
        print("missing", src)
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
