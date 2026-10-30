#!/usr/bin/env python3
"""HTML report: SPXW 0DTE short iron fly research + real-account how-to.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_spxw_0dte_iron_fly_report.py
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

DEFAULT_IN = ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "2026-10-27"
DEFAULT_OUT = DEFAULT_IN / "spxw_0dte_iron_fly_report.html"
START_EQUITY = 25_000.0


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _load_trades(path: Path, fill: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    df = df.loc[df["fill"].astype(str) == fill].copy()
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
                "wing": _num(r.get("wing")),
                "short_put": _num(r.get("short_put")),
                "short_call": _num(r.get("short_call")),
                "long_put": _num(r.get("long_put")),
                "long_call": _num(r.get("long_call")),
                "put_delta": _num(r.get("put_delta")),
                "call_delta": _num(r.get("call_delta")),
                "credit": _num(r.get("credit")),
                "credit_mid": _num(r.get("credit_mid")),
                "credit_natural": _num(r.get("credit_natural")),
                "slip": None
                if _num(r.get("credit_mid")) is None or _num(r.get("credit_natural")) is None
                else round(
                    (_num(r.get("credit_mid")) - _num(r.get("credit_natural"))) * 100.0, 2
                ),
                "pnl": round(pnl, 2),
                "won": bool(pnl > 0),
            }
        )
    return rows


def _equity_curve(rows: List[Dict[str, Any]], start: float = START_EQUITY) -> Dict[str, Any]:
    eq = float(start)
    labels: List[str] = []
    data: List[float] = []
    for r in rows:
        eq += float(r["pnl"])
        labels.append(r["date"])
        data.append(round(eq, 2))
    return {"labels": labels, "data": data, "end": round(eq, 2)}


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


def build_payload(indir: Path) -> Dict[str, Any]:
    trades_path = indir / "trades.csv"
    summary_path = indir / "path_phase5" / "summary_phase5.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    books: Dict[str, Any] = {}
    for fill, label in (("natural", "Natural bid/ask (ranking)"), ("mid", "Mid (optimistic)")):
        rows = _load_trades(trades_path, fill)
        books[fill] = {
            "label": label,
            "stats": _book_stats(rows),
            "equity": _equity_curve(rows),
            "trades": rows,
        }

    path_nat = (summary.get("path") or {}).get("natural") or {}
    phase5 = ((summary.get("phase5") or {}).get("natural") or {}).get("settle") or {}
    verdict = summary.get("verdict") or {}

    window = "unknown"
    nat_rows = _load_trades(trades_path, "natural")
    if nat_rows:
        window = f"{nat_rows[0]['date']} to {nat_rows[-1]['date']}"
    elif phase5.get("strat"):
        window = f"{phase5['strat'].get('start')} to {phase5['strat'].get('end')}"

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "window": window,
        "start_equity": START_EQUITY,
        "verdict": verdict,
        "claimed": {
            "n": 299,
            "wr": 57.1,
            "avg_win": 513,
            "avg_loss": 492,
            "cagr": 44.0,
        },
        "path": {
            "mae_median": path_nat.get("mae_median"),
            "mfe_median": path_nat.get("mfe_median"),
            "underwater_half_frac": path_nat.get("underwater_half_frac"),
            "winner_underwater_half_frac": path_nat.get("winner_underwater_half_frac"),
            "tp_hit_frac": path_nat.get("tp_hit_frac"),
            "entry_slip_dollars_median": path_nat.get("entry_slip_dollars_median"),
            "entry_slip_dollars_p90": path_nat.get("entry_slip_dollars_p90"),
            "settle": path_nat.get("settle"),
            "tp50": path_nat.get("tp50"),
        },
        "phase5": {
            "gate": phase5.get("gate"),
            "gate_pass": phase5.get("gate_pass"),
            "strat": phase5.get("strat") or {},
            "spy": phase5.get("spy") or {},
        },
        "books": books,
        "howto": {
            "title": "How to run this in a real account",
            "product": "SPX / SPXW weekly PM-settled 0DTE options (cash-settled; no stock assignment)",
            "steps": [
                {
                    "t": "1. Check the day filter before 13:45 ET",
                    "b": [
                        "Skip Wednesday (Fed speak / midweek). Trade Mon, Tue, Thu, Fri only.",
                        "VIX session open must be under 20 (research used open; prefer ~15-16).",
                        "Same-day SPXW expiry must exist (daily 0DTE since mid-2022).",
                        "Know the day's events (CPI, FOMC, OPEX) — the backtest does not skip them beyond Wed/VIX.",
                    ],
                },
                {
                    "t": "2. At 13:45 ET, pick the strikes",
                    "b": [
                        "Sell the put nearest 50 delta and the call nearest 40 delta on the 0DTE SPXW chain.",
                        "Buy the put 15 points below the short put; buy the call 15 points above the short call (10-pt wings for smaller accounts).",
                        "This is a narrow short iron condor / iron fly — slightly bullish because the put short is closer than the call short.",
                        "Target a credit near half the wing width (e.g. ~$7.50 on a 15-wide) so risk/reward is about 1:1. Research median natural credit was roughly that order.",
                    ],
                },
                {
                    "t": "3. Enter as one 4-leg combo (BAG), not four tickets",
                    "b": [
                        "In TWS/Gateway: Combo / BAG order, CBOE, SPX (SPXW trading class), SELL the package for a credit.",
                        "Limit at the mid credit first; if it does not fill in 1-2 minutes, work toward the natural (sell shorts at bid, buy wings at ask). Research mid-to-natural haircut median was about $20/lot.",
                        "Size: 1 lot on ~$25k risk capital in the study. Max loss per lot = (wing width - credit) x $100. Example: 15-wide, $7.50 credit -> $750 max loss.",
                        "Paper: python scripts\\trading\\spxw_iron_fly_combo.py --place --wait-sec 120 (client 8831). Live needs --live-en --account U*.",
                        "Live /hot Detector IF + Telegram: CronRunner spxw_iron_fly_live (13:20-14:05 ET); L5/S3/P2 sit-out included.",
                    ],
                },
                {
                    "t": "4. Manage to the close (default)",
                    "b": [
                        "Default research book: hold to cash settle at the SPX close. SPXW is PM-settled — no share delivery.",
                        "Optional discretionary: if the package reaches ~50% of credit with ~1 hour left, you may take profit. The automated settle book beat the 50% TP overlay (E +$93 vs +$56) — do not make early TP the default.",
                        "Do not adjust strikes intraday on this recipe. It is a ride-or-die final ~2 hour trade.",
                        "Expect median adverse MTM around -$180/lot; about 12% of eventual winners were underwater by more than half the credit at some point.",
                    ],
                },
                {
                    "t": "5. Risk, account, and compliance notes",
                    "b": [
                        "Defined risk: wings cap loss. Still 0DTE gamma — P&L can move hundreds of dollars in minutes near the shorts.",
                        "Buying power is much smaller than a naked short strangle; still confirm your broker's SPX combo margin.",
                        "Use a paper or small live size until you have logged real BAG fills vs mid (research soft-promote; not wired as an automated sleeve yet).",
                        "This is not tax or regulatory advice. Index options have specific settlement and tax treatment (Section 1256 in the US for many broad-based indexes) — confirm with your broker/tax advisor.",
                    ],
                },
            ],
            "tws_checklist": [
                "Symbol: SPX, Trading class SPXW, Expiry = today, Exchange CBOE",
                "Legs: SELL put (50d), BUY put (-15), SELL call (40d), BUY call (+15)",
                "Order: BAG / COMB, Action SELL, LMT = mid credit (negative price = credit in IB)",
                "TIF: DAY; quantity 1; no stock hedge",
                "After fill: leave alone until cash settle unless taking a discretionary 50% credit cover",
            ],
        },
    }


def render_html(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SPXW 0DTE short iron fly — research + live how-to</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#0f1218; --panel:#171b24; --line:#2a3140; --text:#e7ecf3; --muted:#8b93a7;
  --pos:#3dcf8e; --neg:#f07178; --accent:#6aa6ff; --warn:#e6b35a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.45 "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:24px 18px 48px; }}
h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-0.02em; }}
h2 {{ font-size:18px; margin:0 0 12px; }}
h3 {{ font-size:15px; margin:16px 0 8px; color:var(--accent); }}
.sub {{ color:var(--muted); margin-bottom:18px; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }}
.badge.soft {{ background:#2a3a28; color:var(--pos); }}
.badge.nowire {{ background:#3a2e1f; color:var(--warn); }}
.badge.fail {{ background:#3a2226; color:var(--neg); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:14px 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; }}
.card {{ background:#12161e; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.card .l {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.04em; }}
.card .v {{ font-size:20px; font-weight:650; margin-top:4px; }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.row {{ display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:10px; }}
label {{ display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }}
select, input {{ background:#12161e; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:7px 10px; }}
.chart {{ height:320px; }}
.table-wrap {{ overflow:auto; max-height:480px; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child, td.left, th.left {{ text-align:left; }}
tr.loss {{ background:rgba(240,113,120,0.06); }}
ol.howto, ul.checklist {{ margin:0; padding-left:18px; }}
ol.howto li, ul.checklist li {{ margin:0 0 8px; }}
.step {{ margin:0 0 14px; padding-bottom:10px; border-bottom:1px dashed var(--line); }}
.step:last-child {{ border-bottom:0; }}
code {{ background:#12161e; padding:1px 6px; border-radius:4px; font-size:12px; }}
a {{ color:var(--accent); }}
.grid2 {{ display:grid; grid-template-columns:1.2fr 1fr; gap:14px; }}
@media (max-width:900px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SPXW 0DTE short iron fly</h1>
  <div class="sub">Jason Rutschel / Theta Profits recipe — claim-check, path risk, Phase-5 vs SPY, and how to trade it live. Window {payload.get("window")}.</div>
  <div id="badges"></div>

  <div class="panel">
    <h2>Live how-to (real account)</h2>
    <p class="sub" style="margin-top:0">Product: <strong id="product"></strong>. Research ranking fill is <strong>natural</strong> (sell shorts at bid / buy wings at ask). Soft-promote only — log paper BAG fills before scaling.</p>
    <div id="howto"></div>
    <h3>TWS checklist</h3>
    <ul class="checklist" id="checklist"></ul>
  </div>

  <div class="panel">
    <h2>Headline results</h2>
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Equity ($25k start, 1 lot)</h2>
      <div class="row">
        <div><label>Book</label>
          <select id="eqBook">
            <option value="natural">Natural</option>
            <option value="mid">Mid</option>
          </select>
        </div>
      </div>
      <div class="chart"><canvas id="eq"></canvas></div>
    </div>
    <div class="panel">
      <h2>Phase-5 vs SPY (natural settle)</h2>
      <div class="table-wrap"><table id="phase5"></table></div>
      <h3>Path risk</h3>
      <div class="table-wrap"><table id="path"></table></div>
    </div>
  </div>

  <div class="panel">
    <h2>Claim vs replay</h2>
    <div class="table-wrap"><table id="claim"></table></div>
  </div>

  <div class="panel">
    <h2>Trades</h2>
    <div class="row">
      <div><label>Book</label>
        <select id="book">
          <option value="natural">Natural</option>
          <option value="mid">Mid</option>
        </select>
      </div>
      <div><label>Show</label>
        <select id="filter">
          <option value="all">All</option>
          <option value="win">Winners</option>
          <option value="loss">Losses</option>
        </select>
      </div>
      <div><label>Find date</label><input id="q" placeholder="2025-04"/></div>
    </div>
    <div class="table-wrap"><table id="trades"></table></div>
  </div>

  <div class="panel">
    <h2>Read this before wiring</h2>
    <ul id="caveats"></ul>
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

function renderBadges() {{
  const v = PAYLOAD.verdict || {{}};
  const bits = [];
  if (v.soft_promote) bits.push('<span class="badge soft">Soft-promote</span>');
  else bits.push('<span class="badge nowire">No soft-promote</span>');
  bits.push('<span class="badge nowire">Not wired</span>');
  if (v.gate_pass_natural_settle) bits.push('<span class="badge soft">Phase-5 PASS</span>');
  else bits.push('<span class="badge nowire">Phase-5 FAIL</span>');
  document.getElementById("badges").innerHTML = bits.join(" ") +
    `<div class="sub" style="margin-top:8px">${{v.reason || ""}}</div>`;
}}

function renderCaveats() {{
  const v = PAYLOAD.verdict || {{}};
  const win = PAYLOAD.window || "";
  const items = [];
  if (v.gate_pass_natural_settle) {{
    items.push(`Natural settle clears Phase-5 vs SPY on ${{win}} (Sharpe and MDD).`);
  }} else {{
    items.push(`Natural settle fails Phase-5 on ${{win}}: ${{v.gate || v.reason || "gate fail"}} (2y sample can still PASS — compare windows).`);
  }}
  items.push("Hold to cash settle beat the discretionary 50% take-profit overlay — prefer expire.");
  items.push("Median entry slip mid→natural is about $20/lot; size for max loss = (wing − credit) × $100.");
  items.push("Not an automated CronRunner sleeve yet. Paper BAG fills at 13:45 ET are the missing gate for wiring.");
  document.getElementById("caveats").innerHTML = items.map(x => `<li>${{x}}</li>`).join("");
}}

function renderHowto() {{
  const h = PAYLOAD.howto;
  document.getElementById("product").textContent = h.product;
  document.getElementById("howto").innerHTML = h.steps.map(s =>
    `<div class="step"><strong>${{s.t}}</strong><ul>` +
    s.b.map(x => `<li>${{x}}</li>`).join("") + `</ul></div>`
  ).join("");
  document.getElementById("checklist").innerHTML =
    h.tws_checklist.map(x => `<li>${{x}}</li>`).join("");
}}

function renderKpis() {{
  const s = PAYLOAD.books.natural.stats;
  const p5 = PAYLOAD.phase5.strat || {{}};
  const spy = PAYLOAD.phase5.spy || {{}};
  const items = [
    ["Trades (natural)", s.n, ""],
    ["Win rate", num(s.wr,1) + "%", ""],
    ["Expectancy", money(s.expectancy), cls(s.expectancy)],
    ["PF", s.pf == null ? "" : num(s.pf,2), ""],
    ["Total 1-lot", money(s.total), cls(s.total)],
    ["Sharpe", num(p5.sharpe,2), ""],
    ["SPY Sharpe", num(spy.sharpe,2), ""],
    ["MDD", pct(Math.abs(p5.max_drawdown||0),1), "neg"],
    ["SPY MDD", pct(Math.abs(spy.max_drawdown||0),1), "neg"],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c]) =>
    `<div class="card"><div class="l">${{l}}</div><div class="v ${{c}}">${{v}}</div></div>`
  ).join("");
}}

function renderClaim() {{
  const c = PAYLOAD.claimed;
  const n = PAYLOAD.books.natural.stats;
  const m = PAYLOAD.books.mid.stats;
  const head = `<tr><th class="left"></th><th>Claimed</th><th>Natural</th><th>Mid</th></tr>`;
  const rows = [
    ["Trades", c.n, n.n, m.n],
    ["Win rate %", num(c.wr,1), num(n.wr,1), num(m.wr,1)],
    ["Avg win", money(c.avg_win), money(n.avg_win), money(m.avg_win)],
    ["Avg loss", money(c.avg_loss), money(n.avg_loss), money(m.avg_loss)],
    ["Expectancy", "", money(n.expectancy), money(m.expectancy)],
    ["PF", "", n.pf==null?"":num(n.pf,2), m.pf==null?"":num(m.pf,2)],
  ].map(r => `<tr><td class="left">${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td><td>${{r[3]}}</td></tr>`).join("");
  document.getElementById("claim").innerHTML = head + rows;
}}

function renderPhase5() {{
  const s = PAYLOAD.phase5.strat || {{}};
  const spy = PAYLOAD.phase5.spy || {{}};
  const head = `<tr><th class="left"></th><th>Iron fly</th><th>SPY</th></tr>`;
  const rows = [
    ["Gate", PAYLOAD.phase5.gate || "", "benchmark"],
    ["CAGR", pct(s.cagr,1), pct(spy.cagr,1)],
    ["Sharpe", num(s.sharpe,2), num(spy.sharpe,2)],
    ["Max DD", pct(Math.abs(s.max_drawdown||0),1), pct(Math.abs(spy.max_drawdown||0),1)],
    ["Invested %", pct(s.pct_time_invested,0), "100%"],
  ].map(r => `<tr><td class="left">${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td></tr>`).join("");
  document.getElementById("phase5").innerHTML = head + rows;

  const p = PAYLOAD.path || {{}};
  const ph = `<tr><th class="left">Path (natural entry)</th><th></th></tr>`;
  const pr = [
    ["MAE median", money(p.mae_median)],
    ["MFE median", money(p.mfe_median)],
    ["Underwater >1/2 credit", pct(p.underwater_half_frac,1)],
    ["Winners that were underwater", pct(p.winner_underwater_half_frac,1)],
    ["Hit 50% TP", pct(p.tp_hit_frac,1)],
    ["Entry slip mid-nat median", money(p.entry_slip_dollars_median)],
    ["Entry slip p90", money(p.entry_slip_dollars_p90)],
    ["Settle E", money((p.settle||{{}}).expectancy)],
    ["50% TP E", money((p.tp50||{{}}).expectancy)],
  ].map(r => `<tr><td class="left">${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join("");
  document.getElementById("path").innerHTML = ph + pr;
}}

let chart;
function renderChart() {{
  const key = document.getElementById("eqBook").value;
  const eq = PAYLOAD.books[key].equity;
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("eq"), {{
    type: "line",
    data: {{
      labels: eq.labels,
      datasets: [{{
        label: PAYLOAD.books[key].label,
        data: eq.data,
        borderColor: "#6aa6ff",
        backgroundColor: "transparent",
        borderWidth: 1.6,
        pointRadius: 0,
        tension: 0
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: "#c5cad6" }} }} }},
      scales: {{
        x: {{ ticks: {{ color:"#8b93a7", maxTicksLimit: 8 }}, grid: {{ color:"#2a3140" }} }},
        y: {{ ticks: {{ color:"#8b93a7", callback: (v) => money(v) }}, grid: {{ color:"#2a3140" }} }}
      }}
    }}
  }});
}}

function renderTrades() {{
  const key = document.getElementById("book").value;
  const mode = document.getElementById("filter").value;
  const q = document.getElementById("q").value.trim();
  let rows = PAYLOAD.books[key].trades.slice();
  if (mode === "win") rows = rows.filter(r => r.pnl > 0);
  if (mode === "loss") rows = rows.filter(r => r.pnl <= 0);
  if (q) rows = rows.filter(r => r.date.includes(q));
  const head = `<tr><th class="left">Date</th><th>VIX</th><th>Spot</th><th>Settle</th><th>Put short</th><th>Call short</th><th>Wings</th><th>Credit</th><th>Slip $</th><th>P&amp;L</th></tr>`;
  const body = rows.map(r => `<tr class="${{r.pnl<=0?"loss":""}}"><td class="left">${{r.date}}</td><td>${{num(r.vix,2)}}</td><td>${{num(r.spot,1)}}</td><td>${{num(r.settle,1)}}</td><td>${{num(r.short_put,0)}} (d ${{num(r.put_delta,2)}})</td><td>${{num(r.short_call,0)}} (d ${{num(r.call_delta,2)}})</td><td>${{num(r.long_put,0)}} / ${{num(r.long_call,0)}}</td><td>${{num(r.credit,2)}}</td><td>${{num(r.slip,0)}}</td><td class="${{cls(r.pnl)}}">${{money(r.pnl)}}</td></tr>`).join("");
  document.getElementById("trades").innerHTML = head + body;
}}

renderBadges();
renderCaveats();
renderHowto();
renderKpis();
renderClaim();
renderPhase5();
renderChart();
renderTrades();
document.getElementById("eqBook").addEventListener("change", renderChart);
document.getElementById("book").addEventListener("change", renderTrades);
document.getElementById("filter").addEventListener("change", renderTrades);
document.getElementById("q").addEventListener("input", renderTrades);
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    payload = build_payload(args.indir)
    html = render_html(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
