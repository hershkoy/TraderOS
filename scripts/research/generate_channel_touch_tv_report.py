#!/usr/bin/env python3
"""
Generate a TradingView Strategy Tester-style HTML report from channel-touch trades CSV.

Interactive controls in the HTML (client-side recalc):
  - Portfolio size (initial capital)
  - Position sizing: fixed $, %% of initial, or %% of equity
  - Round-trip friction %%
  - Max entries per day (0=all; 1+=rank by RS vs SPY when available)

Portfolio model:
  - Equity marks on trade exit dates
  - SPY buy-and-hold overlay for comparison

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_channel_touch_tv_report.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_channel_touch_tv_report.py --trades reports\\ascending_channels\\channel_touch_trades_XXXX.csv --rs-top1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_channel_touch_tv_report")


def _latest_trades_csv(outdir: Path) -> Path:
    files = sorted(outdir.glob("channel_touch_trades_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No channel_touch_trades_*.csv in {outdir}")
    return files[-1]


def _apply_rs_topn(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or "rs_spy_126d" not in df.columns:
        return df
    ranked = df.sort_values(["buy_date", "rs_spy_126d"], ascending=[True, False], na_position="last")
    return ranked.groupby("buy_date", sort=False).head(int(n)).reset_index(drop=True)


def load_spy_close(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    provider: str = "ALPACA",
) -> pd.Series:
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider=provider,
        start=datetime(start.year, start.month, start.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty or "close" not in spy.columns:
        raise RuntimeError("SPY OHLCV unavailable for S&P 500 comparison")
    close = spy["close"].astype(float).copy()
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.DatetimeIndex(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    return close.sort_index()


def trades_to_raw(df: pd.DataFrame) -> List[dict]:
    """Compact trade rows for client-side recalculation."""
    out: List[dict] = []
    t = df.copy()
    t["buy_date"] = pd.to_datetime(t["buy_date"])
    t["sell_date"] = pd.to_datetime(t["sell_date"])
    t = t.sort_values(["sell_date", "buy_date", "stock"]).reset_index(drop=True)
    for _, row in t.iterrows():
        item = {
            "symbol": str(row["stock"]),
            "buy": row["buy_date"].strftime("%Y-%m-%d"),
            "sell": row["sell_date"].strftime("%Y-%m-%d"),
            "entry": float(row["buy_price"]),
            "exit": float(row["sell_price"]),
            "gain": float(row["gain_pct"]),
            "hold": int(row["hold_days"]) if "hold_days" in row and pd.notna(row["hold_days"]) else None,
            "reason": str(row.get("exit_reason", "") or ""),
            "touch": int(row["touch_num"]) if "touch_num" in row and pd.notna(row["touch_num"]) else None,
        }
        if "rs_spy_126d" in row and pd.notna(row["rs_spy_126d"]):
            item["rs"] = float(row["rs_spy_126d"])
        out.append(item)
    return out


def spy_to_raw(spy_close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> List[dict]:
    hist = spy_close.loc[start:end].dropna()
    if hist.empty:
        hist = spy_close[(spy_close.index >= start) & (spy_close.index <= end)].dropna()
    return [{"x": d.strftime("%Y-%m-%d"), "c": float(v)} for d, v in hist.items()]


def render_html(
    *,
    raw_trades: List[dict],
    spy_closes: List[dict],
    defaults: Dict[str, Any],
    title: str,
    source: str,
) -> str:
    payload = json.dumps(
        {
            "trades": raw_trades,
            "spy": spy_closes,
            "defaults": defaults,
            "source": source,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        separators=(",", ":"),
    )
    # Escape </script> in JSON
    payload = payload.replace("</", "<\\/")

    has_rs = any("rs" in t for t in raw_trades)
    rs_note = "RS vs SPY (126d) available for same-day ranking." if has_rs else "No RS column in trades CSV; max/day keeps first N by exit order."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0d1421; --panel:#131722; --panel2:#1e222d; --border:#2a2e39;
    --text:#d1d4dc; --muted:#787b86; --accent:#2962ff; --pos:#26a69a; --neg:#ef5350; --spy:#f7931a;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Trebuchet MS",Roboto,Ubuntu,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:20px; }}
  .header {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:18px 20px; margin-bottom:12px; }}
  .header h1 {{ font-size:20px; font-weight:600; color:#fff; }}
  .header .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .controls {{
    background:var(--panel2); border:1px solid var(--border); border-radius:6px;
    padding:14px 16px; margin-bottom:14px; display:grid;
    grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px 14px; align-items:end;
  }}
  .controls label {{ display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
  .controls input, .controls select {{
    width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text);
    border-radius:4px; padding:8px 10px; font-size:13px;
  }}
  .controls .actions {{ display:flex; gap:8px; align-items:end; }}
  .btn {{
    background:var(--accent); color:#fff; border:none; border-radius:4px; padding:9px 14px;
    font-size:13px; font-weight:600; cursor:pointer;
  }}
  .btn.secondary {{ background:#2a2e39; color:var(--text); }}
  .btn:hover {{ filter:brightness(1.08); }}
  .hint {{ grid-column:1/-1; color:var(--muted); font-size:12px; }}
  .tabs {{ display:flex; gap:4px; margin:12px 0; border-bottom:1px solid var(--border); }}
  .tab {{ background:transparent; border:none; color:var(--muted); padding:10px 14px; cursor:pointer; font-size:13px; font-weight:500; border-bottom:2px solid transparent; margin-bottom:-1px; }}
  .tab.active {{ color:#fff; border-bottom-color:var(--accent); }}
  .panel {{ display:none; background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:16px; }}
  .panel.active {{ display:block; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }}
  .card {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:12px 14px; }}
  .card .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .card .value {{ font-size:18px; font-weight:600; margin-top:4px; }}
  .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
  .muted {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  .chart-box {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px; padding:12px; margin-bottom:12px; height:320px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ color:var(--muted); font-weight:500; position:sticky; top:0; background:var(--panel); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .table-scroll {{ max-height:560px; overflow:auto; border:1px solid var(--border); border-radius:6px; }}
  .metric-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; border:1px solid var(--border); border-radius:6px; overflow:hidden; margin-bottom:14px; }}
  .metric-grid .mh,.metric-grid .mc {{ padding:9px 12px; border-bottom:1px solid var(--border); border-right:1px solid var(--border); font-size:12px; }}
  .metric-grid .mh {{ background:var(--panel2); color:var(--muted); }}
  .metric-grid .mc:nth-child(4n) {{ border-right:none; }}
  h2 {{ font-size:15px; margin:8px 0 12px; font-weight:600; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>{title}</h1>
    <div class="sub" id="subtitle">Loading…</div>
    <div class="sub">Source: {source} · Generated <span id="genAt"></span></div>
  </div>

  <div class="controls">
    <div>
      <label for="capital">Portfolio size ($)</label>
      <input id="capital" type="number" min="1000" step="1000" />
    </div>
    <div>
      <label for="sizeMode">Position sizing</label>
      <select id="sizeMode">
        <option value="fixed">Fixed $ per trade</option>
        <option value="pct_initial">% of initial capital</option>
        <option value="pct_equity">% of equity (compound)</option>
      </select>
    </div>
    <div>
      <label for="sizeVal" id="sizeValLabel">Size ($)</label>
      <input id="sizeVal" type="number" min="0" step="100" />
    </div>
    <div>
      <label for="friction">Friction / round-trip (%)</label>
      <input id="friction" type="number" min="0" max="5" step="0.05" />
    </div>
    <div>
      <label for="maxPerDay">Max entries / day (0 = all)</label>
      <input id="maxPerDay" type="number" min="0" max="50" step="1" />
    </div>
    <div class="actions">
      <button class="btn" id="btnApply" type="button">Apply</button>
      <button class="btn secondary" id="btnReset" type="button">Reset</button>
    </div>
    <div class="hint">{rs_note} Change parameters and click Apply (or press Enter). Charts and metrics update in-browser.</div>
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="compare">Compare to S&amp;P 500</button>
    <button class="tab" data-tab="performance">Performance</button>
    <button class="tab" data-tab="trades">List of trades</button>
    <button class="tab" data-tab="monthly">Monthly</button>
  </div>

  <div id="overview" class="panel active">
    <div class="cards" id="overviewCards"></div>
    <h2>Equity curve vs S&amp;P 500</h2>
    <div class="chart-box"><canvas id="eqChart"></canvas></div>
    <h2>Drawdown</h2>
    <div class="chart-box"><canvas id="ddChart"></canvas></div>
    <p class="note" id="modelNote"></p>
  </div>

  <div id="compare" class="panel">
    <h2>Compare to S&amp;P 500 (SPY buy &amp; hold)</h2>
    <div class="cards" id="compareCards"></div>
    <h2>Normalized equity (start = 100)</h2>
    <div class="chart-box"><canvas id="cmpChart"></canvas></div>
    <div class="metric-grid" id="compareGrid"></div>
    <p class="note">SPY buy &amp; hold uses the same portfolio size over the strategy window. Strategy equity is forward-filled onto trading days for the overlay.</p>
  </div>

  <div id="performance" class="panel">
    <h2>Performance summary</h2>
    <div class="metric-grid" id="perfGrid"></div>
    <p class="muted" id="exitMix"></p>
  </div>

  <div id="trades" class="panel">
    <h2>List of trades</h2>
    <p class="muted" id="tradesNote"></p>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Symbol</th><th>Signal</th><th>Entry</th><th>Exit</th>
            <th>Entry px</th><th>Exit px</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Cum. P&amp;L</th><th>Bars</th><th>Exit</th>
          </tr>
        </thead>
        <tbody id="tradesBody"></tbody>
      </table>
    </div>
  </div>

  <div id="monthly" class="panel">
    <h2>Monthly net profit (by exit month)</h2>
    <div class="table-scroll" style="max-height:640px">
      <table>
        <thead><tr><th>Month</th><th>Net P&amp;L</th></tr></thead>
        <tbody id="monthBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const RAW = {payload};

function money(x, signed) {{
  const a = Math.abs(x);
  const s = a.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}});
  if (!signed) return '$' + s;
  return (x >= 0 ? '+$' : '-$') + s;
}}
function pct(x, signed) {{
  const s = x.toFixed(2) + '%';
  if (!signed) return s;
  return (x >= 0 ? '+' : '') + s;
}}
function cls(x) {{ return x >= 0 ? 'pos' : 'neg'; }}
function setHTML(id, html) {{ const el = document.getElementById(id); if (el) el.innerHTML = html; }}

function readParams() {{
  return {{
    capital: Math.max(1000, Number(document.getElementById('capital').value) || 100000),
    sizeMode: document.getElementById('sizeMode').value,
    sizeVal: Math.max(0, Number(document.getElementById('sizeVal').value) || 0),
    friction: Math.max(0, Number(document.getElementById('friction').value) || 0),
    maxPerDay: Math.max(0, Math.floor(Number(document.getElementById('maxPerDay').value) || 0)),
  }};
}}

function syncSizeLabel() {{
  const mode = document.getElementById('sizeMode').value;
  document.getElementById('sizeValLabel').textContent =
    mode === 'fixed' ? 'Size ($)' : 'Size (%)';
}}

function filterMaxPerDay(trades, maxPerDay) {{
  if (!maxPerDay || maxPerDay <= 0) return trades.slice();
  const byDay = {{}};
  for (const t of trades) {{
    if (!byDay[t.buy]) byDay[t.buy] = [];
    byDay[t.buy].push(t);
  }}
  const out = [];
  Object.keys(byDay).sort().forEach(day => {{
    const arr = byDay[day].slice();
    arr.sort((a, b) => {{
      const ra = (a.rs === undefined || a.rs === null) ? -Infinity : a.rs;
      const rb = (b.rs === undefined || b.rs === null) ? -Infinity : b.rs;
      return rb - ra;
    }});
    out.push(...arr.slice(0, maxPerDay));
  }});
  out.sort((a, b) => (a.sell < b.sell ? -1 : a.sell > b.sell ? 1 : a.buy < b.buy ? -1 : a.symbol.localeCompare(b.symbol)));
  return out;
}}

function notionalFor(equity, capital, p) {{
  if (p.sizeMode === 'pct_equity') return equity * (p.sizeVal / 100);
  if (p.sizeMode === 'pct_initial') return capital * (p.sizeVal / 100);
  return p.sizeVal;
}}

function simulate(trades, p) {{
  let equity = p.capital;
  let peak = p.capital;
  let cum = 0;
  const rows = [];
  const eqCurve = [];
  const ddCurve = [];
  const monthly = {{}};
  const exits = {{}};
  if (trades.length) {{
    eqCurve.push({{x: trades[0].buy, y: p.capital}});
    ddCurve.push({{x: trades[0].buy, y: 0}});
  }}
  let wins = 0, losses = 0, gp = 0, gl = 0;
  let sumHold = 0, nHold = 0;
  let largestWin = 0, largestLoss = 0;
  let avgWinSum = 0, avgLossSum = 0;

  for (let i = 0; i < trades.length; i++) {{
    const t = trades[i];
    const notion = Math.max(0, notionalFor(equity, p.capital, p));
    const gainPct = t.gain - p.friction;
    const pnl = notion * gainPct / 100;
    cum += pnl;
    equity = p.capital + cum;
    if (equity > peak) peak = equity;
    const dd = peak > 0 ? (peak - equity) / peak * 100 : 0;
    eqCurve.push({{x: t.sell, y: equity}});
    ddCurve.push({{x: t.sell, y: -dd}});
    const month = t.sell.slice(0, 7);
    monthly[month] = (monthly[month] || 0) + pnl;
    exits[t.reason || 'unknown'] = (exits[t.reason || 'unknown'] || 0) + 1;
    if (pnl > 0) {{ wins++; gp += pnl; avgWinSum += pnl; largestWin = Math.max(largestWin, pnl); }}
    else {{ losses++; gl += -pnl; avgLossSum += pnl; largestLoss = Math.min(largestLoss, pnl); }}
    if (t.hold != null) {{ sumHold += t.hold; nHold++; }}
    rows.push({{
      n: i + 1, symbol: t.symbol, signal: t.touch != null ? ('Touch ' + t.touch) : 'Long',
      entry_date: t.buy, exit_date: t.sell, entry_price: t.entry, exit_price: t.exit,
      pnl, pnl_pct: gainPct, cum_pnl: cum, hold_days: t.hold, exit_reason: t.reason
    }});
  }}

  const n = trades.length;
  const net = equity - p.capital;
  const ret = p.capital > 0 ? net / p.capital * 100 : 0;
  const wr = n ? wins / n * 100 : 0;
  const pf = gl > 1e-12 ? gp / gl : (gp > 0 ? Infinity : 0);
  let maxDd = 0;
  for (const pnt of ddCurve) maxDd = Math.max(maxDd, -pnt.y);

  return {{
    metrics: {{
      capital: p.capital, final: equity, net, ret, n, wins, losses, wr, pf,
      gp, gl,
      avgTrade: n ? cum / n : 0,
      avgWin: wins ? avgWinSum / wins : 0,
      avgLoss: losses ? avgLossSum / losses : 0,
      avgBars: nHold ? sumHold / nHold : 0,
      maxDd, largestWin, largestLoss,
      symbols: new Set(trades.map(t => t.symbol)).size,
      from: trades.length ? trades.reduce((a,t)=>t.buy<a?t.buy:a, trades[0].buy) : '',
      to: trades.length ? trades.reduce((a,t)=>t.sell>a?t.sell:a, trades[0].sell) : '',
      exits
    }},
    rows, eqCurve, ddCurve, monthly, params: p
  }};
}}

function simulateSpy(spy, capital, from, to) {{
  if (!spy || !spy.length || !from || !to) return null;
  const bars = spy.filter(p => p.x >= from && p.x <= to);
  if (!bars.length) return null;
  const start = bars[0].c;
  if (!(start > 0)) return null;
  const eq = bars.map(p => ({{x: p.x, y: capital * p.c / start}}));
  let peak = eq[0].y, maxDd = 0;
  for (const p of eq) {{
    if (p.y > peak) peak = p.y;
    const dd = peak > 0 ? (peak - p.y) / peak * 100 : 0;
    if (dd > maxDd) maxDd = dd;
  }}
  const final = eq[eq.length - 1].y;
  const net = final - capital;
  const ret = capital > 0 ? net / capital * 100 : 0;
  const days = (Date.parse(eq[eq.length-1].x) - Date.parse(eq[0].x)) / 86400000;
  const years = Math.max(days / 365.25, 1e-9);
  const cagr = (Math.pow(final / capital, 1 / years) - 1) * 100;
  return {{ eq, ret, net, final, maxDd, cagr, from: eq[0].x, to: eq[eq.length-1].x }};
}}

function buildOverlay(stratEq, spyEq, capital) {{
  if (!spyEq || !spyEq.length) return [];
  const map = {{}};
  for (const p of stratEq) map[p.x] = p.y;
  let last = capital;
  const out = [];
  for (const p of spyEq) {{
    if (map[p.x] !== undefined) last = map[p.x];
    out.push({{
      x: p.x,
      strategy: 100 * last / capital,
      spy: 100 * p.y / capital
    }});
  }}
  return out;
}}

let eqChart, ddChart, cmpChart;

function ensureCharts() {{
  const common = {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#d1d4dc' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#787b86', maxTicksLimit:10 }}, grid: {{ color:'#2a2e39' }} }},
      y: {{ ticks: {{ color:'#787b86' }}, grid: {{ color:'#2a2e39' }} }}
    }}
  }};
  if (!eqChart) {{
    eqChart = new Chart(document.getElementById('eqChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Strategy', data: [], borderColor:'#2962ff', pointRadius:0, borderWidth:1.5, tension:0.05 }},
        {{ label:'S&P 500 (SPY)', data: [], borderColor:'#f7931a', pointRadius:0, borderWidth:1.5, tension:0.05 }}
      ]}}, options: common
    }});
  }}
  if (!ddChart) {{
    ddChart = new Chart(document.getElementById('ddChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Drawdown %', data: [], borderColor:'#ef5350', backgroundColor:'rgba(239,83,80,0.15)', fill:true, pointRadius:0, borderWidth:1.5, tension:0.05 }}
      ]}}, options: {{ ...common, plugins: {{ legend: {{ display:false }} }} }}
    }});
  }}
  if (!cmpChart) {{
    cmpChart = new Chart(document.getElementById('cmpChart'), {{
      type: 'line', data: {{ labels: [], datasets: [
        {{ label:'Strategy (norm)', data: [], borderColor:'#2962ff', pointRadius:0, borderWidth:1.5, tension:0.05 }},
        {{ label:'S&P 500 (norm)', data: [], borderColor:'#f7931a', pointRadius:0, borderWidth:1.5, tension:0.05 }}
      ]}}, options: common
    }});
  }}
}}

function alignSeries(dates, points) {{
  const m = Object.fromEntries(points.map(p => [p.x, p.y]));
  let last = null;
  return dates.map(d => {{ if (m[d] !== undefined) last = m[d]; return last; }});
}}

function render(sim, spy) {{
  const m = sim.metrics;
  const p = sim.params;
  const excess = spy ? m.ret - spy.ret : null;
  document.getElementById('subtitle').textContent =
    m.from + ' to ' + m.to + ' · ' + m.n + ' trades · sizing ' + p.sizeMode + ' ' +
    (p.sizeMode === 'fixed' ? money(p.sizeVal,false) : (p.sizeVal.toFixed(2) + '%')) +
    ' · friction ' + p.friction.toFixed(2) + '% · max/day ' + (p.maxPerDay || 'all');

  setHTML('overviewCards', `
    <div class="card"><div class="label">Net profit</div><div class="value ${{cls(m.net)}}">${{money(m.net,true)}}</div></div>
    <div class="card"><div class="label">Total return</div><div class="value ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div></div>
    <div class="card"><div class="label">Max equity DD</div><div class="value neg">${{pct(m.maxDd,false)}}</div></div>
    <div class="card"><div class="label">Total closed trades</div><div class="value">${{m.n}}</div></div>
    <div class="card"><div class="label">Percent profitable</div><div class="value">${{pct(m.wr,false)}}</div></div>
    <div class="card"><div class="label">Profit factor</div><div class="value">${{Number.isFinite(m.pf) ? m.pf.toFixed(3) : 'inf'}}</div></div>
    <div class="card"><div class="label">Avg trade</div><div class="value">${{money(m.avgTrade,true)}}</div></div>
    <div class="card"><div class="label">Avg bars in trade</div><div class="value">${{m.avgBars.toFixed(1)}}</div></div>
    ${{spy ? `<div class="card"><div class="label">S&P 500 (SPY) return</div><div class="value">${{pct(spy.ret,true)}}</div></div>
    <div class="card"><div class="label">Excess vs SPY</div><div class="value ${{cls(excess)}}">${{pct(excess,true)}}</div></div>` : ''}}
  `);

  setHTML('modelNote',
    'Model: ' + (p.sizeMode === 'fixed' ? ('fixed ' + money(p.sizeVal,false) + '/trade') :
      (p.sizeMode === 'pct_initial' ? (p.sizeVal + '% of initial') : (p.sizeVal + '% of equity'))) +
    '; equity marks on exit dates. Friction ' + pct(p.friction,false) +
    ' round-trip. Orange = SPY buy & hold with same portfolio size.' +
    (p.sizeMode !== 'pct_equity' ? ' Overlapping multi-symbol fills are not capital-constrained.' : '')
  );

  // Compare
  if (spy) {{
    setHTML('compareCards', `
      <div class="card"><div class="label">Strategy return</div><div class="value ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div></div>
      <div class="card"><div class="label">SPY buy & hold</div><div class="value">${{pct(spy.ret,true)}}</div></div>
      <div class="card"><div class="label">Excess return</div><div class="value ${{cls(excess)}}">${{pct(excess,true)}}</div></div>
      <div class="card"><div class="label">Beats SPY?</div><div class="value ${{m.ret > spy.ret ? 'pos' : 'neg'}}">${{m.ret > spy.ret ? 'Yes' : 'No'}}</div></div>
      <div class="card"><div class="label">Strategy max DD</div><div class="value neg">${{pct(m.maxDd,false)}}</div></div>
      <div class="card"><div class="label">SPY max DD</div><div class="value neg">${{pct(spy.maxDd,false)}}</div></div>
      <div class="card"><div class="label">Lower DD than SPY?</div><div class="value ${{m.maxDd < spy.maxDd ? 'pos' : 'neg'}}">${{m.maxDd < spy.maxDd ? 'Yes' : 'No'}}</div></div>
      <div class="card"><div class="label">SPY CAGR (approx)</div><div class="value">${{pct(spy.cagr,true)}}</div></div>
    `);
    setHTML('compareGrid', `
      <div class="mh">Metric</div><div class="mh">Strategy</div><div class="mh">S&P 500 (SPY)</div><div class="mh">Diff</div>
      <div class="mc">Total return</div><div class="mc ${{cls(m.ret)}}">${{pct(m.ret,true)}}</div><div class="mc">${{pct(spy.ret,true)}}</div><div class="mc ${{cls(excess)}}">${{pct(excess,true)}}</div>
      <div class="mc">Net profit</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc">${{money(spy.net,true)}}</div><div class="mc ${{cls(m.net-spy.net)}}">${{money(m.net-spy.net,true)}}</div>
      <div class="mc">Final equity</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">${{money(spy.final,false)}}</div><div class="mc ${{cls(m.final-spy.final)}}">${{money(m.final-spy.final,true)}}</div>
      <div class="mc">Max equity drawdown</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc neg">${{pct(spy.maxDd,false)}}</div><div class="mc">${{pct(m.maxDd-spy.maxDd,true)}}</div>
    `);
  }} else {{
    setHTML('compareCards', '<p class="muted">SPY data unavailable.</p>');
    setHTML('compareGrid', '');
  }}

  const pfS = Number.isFinite(m.pf) ? m.pf.toFixed(3) : 'inf';
  setHTML('perfGrid', `
    <div class="mh">Metric</div><div class="mh">All</div><div class="mh">Long</div><div class="mh">Short</div>
    <div class="mc">Net profit</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc ${{cls(m.net)}}">${{money(m.net,true)}}</div><div class="mc">—</div>
    <div class="mc">Gross profit</div><div class="mc">${{money(m.gp,false)}}</div><div class="mc">${{money(m.gp,false)}}</div><div class="mc">—</div>
    <div class="mc">Gross loss</div><div class="mc">${{money(m.gl,false)}}</div><div class="mc">${{money(m.gl,false)}}</div><div class="mc">—</div>
    <div class="mc">Max equity drawdown</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc neg">${{pct(m.maxDd,false)}}</div><div class="mc">—</div>
    <div class="mc">Total closed trades</div><div class="mc">${{m.n}}</div><div class="mc">${{m.n}}</div><div class="mc">—</div>
    <div class="mc">Winning / losing</div><div class="mc">${{m.wins}} / ${{m.losses}}</div><div class="mc">${{m.wins}} / ${{m.losses}}</div><div class="mc">—</div>
    <div class="mc">Percent profitable</div><div class="mc">${{pct(m.wr,false)}}</div><div class="mc">${{pct(m.wr,false)}}</div><div class="mc">—</div>
    <div class="mc">Profit factor</div><div class="mc">${{pfS}}</div><div class="mc">${{pfS}}</div><div class="mc">—</div>
    <div class="mc">Avg winning trade</div><div class="mc pos">${{money(m.avgWin,true)}}</div><div class="mc pos">${{money(m.avgWin,true)}}</div><div class="mc">—</div>
    <div class="mc">Avg losing trade</div><div class="mc neg">${{money(m.avgLoss,true)}}</div><div class="mc neg">${{money(m.avgLoss,true)}}</div><div class="mc">—</div>
    <div class="mc">Largest winning trade</div><div class="mc pos">${{money(m.largestWin,true)}}</div><div class="mc pos">${{money(m.largestWin,true)}}</div><div class="mc">—</div>
    <div class="mc">Largest losing trade</div><div class="mc neg">${{money(m.largestLoss,true)}}</div><div class="mc neg">${{money(m.largestLoss,true)}}</div><div class="mc">—</div>
    <div class="mc">Avg bars in trade</div><div class="mc">${{m.avgBars.toFixed(1)}}</div><div class="mc">${{m.avgBars.toFixed(1)}}</div><div class="mc">—</div>
    <div class="mc">Symbols traded</div><div class="mc">${{m.symbols}}</div><div class="mc">${{m.symbols}}</div><div class="mc">—</div>
    <div class="mc">Initial capital</div><div class="mc">${{money(m.capital,false)}}</div><div class="mc">${{money(m.capital,false)}}</div><div class="mc">—</div>
    <div class="mc">Final equity</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">${{money(m.final,false)}}</div><div class="mc">—</div>
  `);
  setHTML('exitMix', 'Exit mix: ' + Object.entries(m.exits).map(([k,v]) => k + ': ' + v).join(' · '));

  const maxRows = 500;
  const slice = sim.rows.slice(0, maxRows);
  setHTML('tradesNote', sim.rows.length > maxRows
    ? ('Showing first ' + maxRows + ' of ' + sim.rows.length + ' trades (sorted by exit date).')
    : (sim.rows.length + ' trades.'));
  setHTML('tradesBody', slice.map(t => `
    <tr>
      <td>${{t.n}}</td><td>${{t.symbol}}</td><td>${{t.signal}}</td>
      <td>${{t.entry_date}}</td><td>${{t.exit_date}}</td>
      <td class="num">${{t.entry_price.toFixed(4)}}</td><td class="num">${{t.exit_price.toFixed(4)}}</td>
      <td class="num ${{cls(t.pnl)}}">${{money(t.pnl,true)}}</td>
      <td class="num ${{cls(t.pnl_pct)}}">${{pct(t.pnl_pct,true)}}</td>
      <td class="num">${{money(t.cum_pnl,true)}}</td>
      <td class="num">${{t.hold_days == null ? '' : t.hold_days}}</td>
      <td>${{t.exit_reason}}</td>
    </tr>`).join(''));

  const months = Object.keys(sim.monthly).sort();
  setHTML('monthBody', months.map(mo => {{
    const v = sim.monthly[mo];
    return `<tr><td>${{mo}}</td><td class="num ${{cls(v)}}">${{money(v,true)}}</td></tr>`;
  }}).join(''));

  ensureCharts();
  const spyEq = spy ? spy.eq : [];
  const dates = Array.from(new Set([...sim.eqCurve.map(p=>p.x), ...spyEq.map(p=>p.x)])).sort();
  eqChart.data.labels = dates;
  eqChart.data.datasets[0].data = alignSeries(dates, sim.eqCurve);
  eqChart.data.datasets[1].data = alignSeries(dates, spyEq);
  eqChart.update();
  ddChart.data.labels = sim.ddCurve.map(p => p.x);
  ddChart.data.datasets[0].data = sim.ddCurve.map(p => p.y);
  ddChart.update();
  const overlay = spy ? buildOverlay(sim.eqCurve, spy.eq, p.capital) : [];
  cmpChart.data.labels = overlay.map(p => p.x);
  cmpChart.data.datasets[0].data = overlay.map(p => p.strategy);
  cmpChart.data.datasets[1].data = overlay.map(p => p.spy);
  cmpChart.update();
}}

function apply() {{
  const p = readParams();
  const filtered = filterMaxPerDay(RAW.trades, p.maxPerDay);
  const sim = simulate(filtered, p);
  const spy = simulateSpy(RAW.spy, p.capital, sim.metrics.from, sim.metrics.to);
  render(sim, spy);
}}

function resetDefaults() {{
  const d = RAW.defaults;
  document.getElementById('capital').value = d.capital;
  document.getElementById('sizeMode').value = d.sizeMode;
  document.getElementById('sizeVal').value = d.sizeVal;
  document.getElementById('friction').value = d.friction;
  document.getElementById('maxPerDay').value = d.maxPerDay;
  syncSizeLabel();
  apply();
}}

document.querySelectorAll('.tab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});
document.getElementById('sizeMode').addEventListener('change', syncSizeLabel);
document.getElementById('btnApply').addEventListener('click', apply);
document.getElementById('btnReset').addEventListener('click', resetDefaults);
document.querySelector('.controls').addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{ e.preventDefault(); apply(); }}
}});

document.getElementById('genAt').textContent = RAW.generated;
resetDefaults();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="TradingView-style strategy report for channel-touch trades")
    ap.add_argument("--trades", type=Path, default=None, help="Trades CSV path")
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--notional", type=float, default=10_000.0, help="Default fixed $ per trade")
    ap.add_argument("--size-mode", choices=("fixed", "pct_initial", "pct_equity"), default="fixed")
    ap.add_argument("--size-pct", type=float, default=10.0, help="Default %% when size-mode is pct_*")
    ap.add_argument("--friction-pct", type=float, default=0.0)
    ap.add_argument("--rs-top1", action="store_true", help="Default max entries/day = 1")
    ap.add_argument("--max-per-day", type=int, default=None, help="Default max entries/day (overrides --rs-top1)")
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    trades_path = args.trades or _latest_trades_csv(args.outdir)
    logger.info("Loading trades: %s", trades_path)
    df = pd.read_csv(trades_path)
    if df.empty:
        logger.error("Empty trades CSV")
        return 1

    # Always embed FULL trade list; UI can filter max/day
    raw_trades = trades_to_raw(df)
    max_per_day = args.max_per_day if args.max_per_day is not None else (1 if args.rs_top1 else 0)

    buy_min = pd.to_datetime(df["buy_date"]).min()
    sell_max = pd.to_datetime(df["sell_date"]).max()
    spy_closes: List[dict] = []
    try:
        spy_close = load_spy_close(
            buy_min - pd.Timedelta(days=5),
            sell_max + pd.Timedelta(days=5),
            provider=args.provider,
        )
        spy_closes = spy_to_raw(spy_close, buy_min, sell_max)
        logger.info("SPY bars for compare: %d", len(spy_closes))
    except Exception as exc:
        logger.warning("S&P 500 data skipped: %s", exc)

    if args.size_mode == "fixed":
        size_val = float(args.notional)
    else:
        size_val = float(args.size_pct)

    defaults = {
        "capital": float(args.initial_capital),
        "sizeMode": args.size_mode,
        "sizeVal": size_val,
        "friction": float(args.friction_pct),
        "maxPerDay": int(max_per_day),
    }

    label_bits = ["interactive"]
    if max_per_day == 1:
        label_bits.append("rs_top1_default")
    elif max_per_day > 1:
        label_bits.append(f"max{max_per_day}")
    if args.friction_pct:
        label_bits.append(f"fric{args.friction_pct:.2f}")
    if args.tag:
        label_bits.append(args.tag)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(label_bits)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_html = args.outdir / f"channel_touch_tv_report_{tag}_{stamp}.html"
    out_json = args.outdir / f"channel_touch_tv_report_{tag}_{stamp}.json"

    html = render_html(
        raw_trades=raw_trades,
        spy_closes=spy_closes,
        defaults=defaults,
        title="Ascending Channel Touch Long — Strategy Report",
        source=str(trades_path.name),
    )
    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "defaults": defaults,
                "n_trades_embedded": len(raw_trades),
                "n_spy_bars": len(spy_closes),
                "source": str(trades_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Wrote %s (%d trades embedded)", out_html, len(raw_trades))
    print(f"\nInteractive TradingView-style report")
    print(f"  Trades embedded: {len(raw_trades)}")
    print(f"  Default capital: ${args.initial_capital:,.0f}")
    print(f"  Default sizing:  {args.size_mode} = {size_val}")
    print(f"  Default max/day: {max_per_day or 'all'}")
    print(f"  Friction:        {args.friction_pct:.2f}%")
    print(f"\nOpen: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
