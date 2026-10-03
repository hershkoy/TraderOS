#!/usr/bin/env python3
"""HTML report for the 56 DTE Friday 2x1 put-ratio claim-check.

Reads the CSVs written by claimcheck_mes_put_ratio.py.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_mes_put_ratio_report.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.mes_put_ratio import realized_equity  # noqa: E402

DEFAULT_IN = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-02"

BOOKS = [
    ("mid_expiry", "Mid, hold to expiry"),
    ("mid_expiry_slip05", "Mid, $0.05/leg"),
    ("natural_expiry", "Natural bid/ask, hold"),
    ("mid_tp50", "Mid, 50% credit target"),
    ("mid_tp50_dte21", "Mid, 50% or 21 DTE"),
]

CHART_KEYS = ["mid_expiry", "natural_expiry", "mid_tp50"]


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def load_book(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    if "skipped" in df.columns:
        flag = df["skipped"].astype(str).str.lower().isin(("true", "1"))
        df = df.loc[~flag]
    df = df.dropna(subset=["pnl_dollars", "entry_date"])
    rows: List[Dict[str, Any]] = []
    for _, raw in df.iterrows():
        exit_day = _date(raw.get("exit_date")) or _date(raw.get("expiration"))
        entry_day = _date(raw.get("entry_date"))
        if entry_day is None or exit_day is None:
            continue
        pnl = _num(raw.get("pnl_dollars"))
        if pnl is None:
            continue
        mae_pts = _num(raw.get("mae_pts"))
        rows.append(
            {
                "entry": entry_day.isoformat(),
                "exit": exit_day.isoformat(),
                "dte": int(_num(raw.get("dte_entry")) or 0),
                "spot": _num(raw.get("spot")),
                "long_k": _num(raw.get("long_k")),
                "short_k": _num(raw.get("short_k")),
                "long_d": _num(raw.get("long_delta")),
                "short_d": _num(raw.get("short_delta")),
                "width": _num(raw.get("width")),
                "credit": _num(raw.get("credit_pts")),
                "pnl": round(pnl, 2),
                "mae": None if mae_pts is None else round(mae_pts * 100.0, 2),
                "reason": str(raw.get("exit_reason") or ""),
                "hold": int(_num(raw.get("hold_days")) or 0),
            }
        )
    rows.sort(key=lambda row: (row["exit"], row["entry"]))
    return rows


def book_payload(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    points = [
        (date.fromisoformat(row["exit"]), float(row["pnl"]))
        for row in rows
    ]
    return {
        "label": label,
        "trades": rows,
        "equity": realized_equity(points),
    }


def align_curves(books: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Shared exit-date axis. Each series forward-fills its last cumulative dollars."""
    by_key: Dict[str, Dict[str, float]] = {}
    dates = set()
    for key in keys:
        last: Dict[str, float] = {}
        for point in books[key]["equity"]:
            last[point["date"]] = float(point["equity"])
            dates.add(point["date"])
        by_key[key] = last
    labels = sorted(dates)
    series = []
    for key in keys:
        running: Optional[float] = None
        data: List[Optional[float]] = []
        for day in labels:
            if day in by_key[key]:
                running = by_key[key][day]
            data.append(running)
        series.append({"key": key, "label": books[key]["label"], "data": data})
    return {"labels": labels, "series": series}


def render(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return HTML.replace("__PAYLOAD__", data)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>56 DTE Friday 2x1 put ratio</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:#0d1421; --panel:#131722; --panel2:#1e222d; --border:#2a2e39;
    --text:#d1d4dc; --muted:#787b86; --accent:#2962ff; --pos:#26a69a; --neg:#ef5350;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
  .wrap { max-width:1200px; margin:0 auto; padding:20px; }
  .header, .panel { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:16px 18px; margin-bottom:12px; }
  h1 { font-size:20px; color:#fff; font-weight:650; }
  h2 { font-size:15px; color:#fff; margin-bottom:10px; }
  .sub { color:var(--muted); font-size:13px; margin-top:6px; }
  .pill { display:inline-block; margin-left:8px; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:650; background:#1b3a2a; color:var(--pos); vertical-align:middle; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px; margin-top:12px; }
  .card { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
  .card .l { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:16px; font-weight:650; margin-top:3px; font-variant-numeric:tabular-nums; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  .chart-box { height:340px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { padding:7px 8px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align:left; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0; background:#1a1f2b; }
  tr.loss td { background:rgba(239,83,80,.08); }
  tr:hover td { background:rgba(41,98,255,.08); }
  .table-wrap { max-height:520px; overflow:auto; border:1px solid var(--border); border-radius:6px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:10px; }
  label { display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
  select, input { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:7px 10px; font-size:13px; }
  .muted { color:var(--muted); font-size:12px; }
  ul { margin:8px 0 0 18px; font-size:13px; }
  li { margin:4px 0; }
  a { color:#7aa2ff; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>56 DTE Friday 2x1 put ratio <span class="pill">win-rate claim replicates</span></h1>
    <div class="sub">dtoptions, <a href="https://youtu.be/_Vd7WKRSiV4">youtu.be/_Vd7WKRSiV4</a>. Listed SPXW proxy for /MES. Dollars are one SPX package (x100). One MES 2x1 is about 1/20 of these dollars. Futures margin return is not in this report.</div>
    <div class="sub">Claim: 242 trades, 2 losses, 99% wins, no 2022 loss. Hold-to-expiry mid is the matching book.</div>
    <div class="cards" id="kpis"></div>
  </div>

  <div class="panel">
    <h2>Cumulative realized P&amp;L</h2>
    <div class="chart-box"><canvas id="eq"></canvas></div>
    <p class="muted">Sum of closed-trade P&amp;L on each exit date. About eight packages overlap, and this line does not mark them to market while they are open.</p>
  </div>

  <div class="panel">
    <h2>Books</h2>
    <div class="table-wrap"><table id="books"></table></div>
  </div>

  <div class="panel">
    <h2 id="yearTitle">By entry year</h2>
    <div class="table-wrap"><table id="years"></table></div>
  </div>

  <div class="panel">
    <h2>Trades</h2>
    <div class="row">
      <div><label>Book</label><select id="book"></select></div>
      <div><label>Show</label>
        <select id="filter">
          <option value="all">All</option>
          <option value="loss">Losses</option>
          <option value="win">Winners</option>
          <option value="2022">2022 entries</option>
          <option value="2025">2025 entries</option>
        </select>
      </div>
      <div><label>Find date</label><input id="q" placeholder="2025-04"/></div>
    </div>
    <div class="table-wrap"><table id="trades"></table></div>
  </div>

  <div class="panel">
    <h2>How to read it</h2>
    <ul>
      <li>Every Friday: buy the ~14-delta put, sell two ~10-delta puts, nearest listed Friday expiry to 56 days.</li>
      <li>Hold to expiration is the book with two losses. Both are the April 2025 expiry.</li>
      <li>A 50% credit target has no losing trade and about half the dollars, because it exits before the profit tent.</li>
      <li>The closest 2022 call expired about $107 green after an end-of-day mark near -$6,100. Natural bid/ask turns that trade into an $8 loss.</li>
    </ul>
  </div>
</div>
<script>
const PAYLOAD = __PAYLOAD__;
const money = (n) => (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString(undefined, {maximumFractionDigits:0});
const num = (n, d=1) => (n == null || Number.isNaN(n)) ? "" : Number(n).toLocaleString(undefined, {maximumFractionDigits:d, minimumFractionDigits:d});
const cls = (n) => n < 0 ? "neg" : "pos";

function stats(rows) {
  const pnls = rows.map(r => r.pnl);
  const wins = pnls.filter(p => p > 0);
  const losses = pnls.filter(p => p < 0);
  const grossW = wins.reduce((a,b)=>a+b, 0);
  const grossL = -losses.reduce((a,b)=>a+b, 0);
  const pf = grossL > 1e-9 ? grossW / grossL : null;
  const mean = pnls.length ? pnls.reduce((a,b)=>a+b,0) / pnls.length : 0;
  const sorted = pnls.slice().sort((a,b)=>a-b);
  const med = sorted.length ? sorted[Math.floor(sorted.length/2)] : 0;
  const y2022 = rows.filter(r => r.entry.startsWith("2022") && r.pnl < 0).length;
  return { n: rows.length, wins: wins.length, losses: losses.length,
    wr: rows.length ? 100 * wins.length / rows.length : 0,
    mean, med, pf, total: pnls.reduce((a,b)=>a+b,0),
    worst: sorted.length ? sorted[0] : 0, y2022 };
}

function renderKpis(book) {
  const s = stats(book.trades);
  const items = [
    ["Trades", s.n.toLocaleString(), ""],
    ["Win rate", s.wr.toFixed(1) + "%", ""],
    ["Losses", String(s.losses), s.losses ? "neg" : ""],
    ["2022 losses", String(s.y2022), s.y2022 ? "neg" : ""],
    ["Profit factor", s.pf == null ? "—" : s.pf.toFixed(2), ""],
    ["Mean", money(s.mean), cls(s.mean)],
    ["Median", money(s.med), cls(s.med)],
    ["Total", money(s.total), cls(s.total)],
    ["Worst", money(s.worst), "neg"],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c]) =>
    `<div class="card"><div class="l">${l}</div><div class="v ${c}">${v}</div></div>`
  ).join("");
}

function renderBooks() {
  const head = `<tr><th>Book</th><th>n</th><th>WR%</th><th>Losses</th><th>2022 losses</th><th>PF</th><th>Mean</th><th>Total</th><th>Worst</th></tr>`;
  const body = Object.entries(PAYLOAD.books).map(([key, book]) => {
    const s = stats(book.trades);
    return `<tr><td>${book.label}</td><td>${s.n}</td><td>${s.wr.toFixed(1)}</td><td class="${s.losses? "neg":""}">${s.losses}</td><td class="${s.y2022? "neg":""}">${s.y2022}</td><td>${s.pf==null?"—":s.pf.toFixed(2)}</td><td class="${cls(s.mean)}">${money(s.mean)}</td><td class="${cls(s.total)}">${money(s.total)}</td><td class="neg">${money(s.worst)}</td></tr>`;
  }).join("");
  document.getElementById("books").innerHTML = head + body;
}

function renderYears(book) {
  document.getElementById("yearTitle").textContent = "By entry year — " + book.label;
  const years = {};
  book.trades.forEach(r => {
    const y = r.entry.slice(0,4);
    (years[y] = years[y] || []).push(r);
  });
  const head = `<tr><th>Year</th><th>n</th><th>Losses</th><th>WR%</th><th>Mean</th><th>Total</th><th>Worst</th></tr>`;
  const body = Object.keys(years).sort().map(y => {
    const s = stats(years[y]);
    return `<tr><td>${y}</td><td>${s.n}</td><td class="${s.losses?"neg":""}">${s.losses}</td><td>${s.wr.toFixed(1)}</td><td class="${cls(s.mean)}">${money(s.mean)}</td><td class="${cls(s.total)}">${money(s.total)}</td><td class="${cls(s.worst)}">${money(s.worst)}</td></tr>`;
  }).join("");
  document.getElementById("years").innerHTML = head + body;
}

function renderTrades() {
  const key = document.getElementById("book").value;
  const book = PAYLOAD.books[key];
  const mode = document.getElementById("filter").value;
  const q = document.getElementById("q").value.trim();
  let rows = book.trades.slice();
  if (mode === "loss") rows = rows.filter(r => r.pnl < 0);
  if (mode === "win") rows = rows.filter(r => r.pnl > 0);
  if (mode === "2022" || mode === "2025") rows = rows.filter(r => r.entry.startsWith(mode));
  if (q) rows = rows.filter(r => r.entry.includes(q) || r.exit.includes(q));
  const head = `<tr><th>Entry</th><th>Exit</th><th>Why</th><th>DTE</th><th>Spot</th><th>Long</th><th>Short</th><th>Width</th><th>Credit</th><th>P&amp;L</th><th>Worst mark</th><th>Hold</th></tr>`;
  const body = rows.map(r => `<tr class="${r.pnl<0?"loss":""}"><td>${r.entry}</td><td>${r.exit}</td><td>${r.reason}</td><td>${r.dte}</td><td>${num(r.spot,0)}</td><td>${num(r.long_k,0)}</td><td>${num(r.short_k,0)}</td><td>${num(r.width,0)}</td><td>${num(r.credit,2)}</td><td class="${cls(r.pnl)}">${money(r.pnl)}</td><td class="${cls(r.mae||0)}">${r.mae==null?"":money(r.mae)}</td><td>${r.hold}</td></tr>`).join("");
  document.getElementById("trades").innerHTML = head + body;
  renderKpis(book);
  renderYears(book);
}

let chart;
function renderChart() {
  const colors = { mid_expiry:"#26a69a", natural_expiry:"#ef5350", mid_tp50:"#5b9fd4" };
  const sets = PAYLOAD.chart.series.map(s => ({
    label: s.label,
    data: s.data,
    borderColor: colors[s.key] || "#ccc",
    backgroundColor: "transparent",
    borderWidth: 1.6,
    pointRadius: 0,
    spanGaps: true,
    tension: 0
  }));
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("eq"), {
    type: "line",
    data: { labels: PAYLOAD.chart.labels, datasets: sets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#d1d4dc" } } },
      scales: {
        x: { ticks: { color:"#787b86", maxTicksLimit: 8 }, grid: { color:"#2a2e39" } },
        y: { ticks: { color:"#787b86", callback: (v) => money(v) }, grid: { color:"#2a2e39" } }
      }
    }
  });
}

const sel = document.getElementById("book");
Object.entries(PAYLOAD.books).forEach(([key, book]) => {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = book.label;
  sel.appendChild(opt);
});
sel.value = "mid_expiry";
sel.addEventListener("change", renderTrades);
document.getElementById("filter").addEventListener("change", renderTrades);
document.getElementById("q").addEventListener("input", renderTrades);
renderBooks();
renderTrades();
renderChart();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HTML report for the 56 DTE put ratio")
    parser.add_argument("--indir", default=str(DEFAULT_IN))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    indir = Path(args.indir)
    books: Dict[str, Any] = {}
    for key, label in BOOKS:
        path = indir / f"{key}_trades.csv"
        if not path.exists():
            print("missing", path)
            return 1
        books[key] = book_payload(label, load_book(path))
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "books": books,
        "chart": align_curves(books, CHART_KEYS),
    }
    out = Path(args.out) if args.out else indir / "mes_put_ratio.html"
    out.write_text(render(payload), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
