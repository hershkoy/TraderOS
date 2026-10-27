"""HTML report for last-15m one-cycle wheel claimcheck + occupancy + MTM.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_last_15m_wheel_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_wheel import (  # noqa: E402
    ThetaEquitySource,
    filter_wheel_by_liquidity,
    filled_trades_to_mtm_rows,
    filled_trades_to_occ_rows,
    load_option_mark_paths,
    occupancy_fcfs_wheel,
    spy_buyhold_equity,
    wheel_slot_equity,
    wheel_slot_equity_mtm,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("last_15m_wheel_report")

DIR = ROOT / "reports" / "ascending_channels" / "2026-10-09"
DEFAULT_MID = DIR / "last_15m_wheel_mid_repaired_20261010_112646.csv"
DEFAULT_ASKBID = DIR / "last_15m_wheel_ask_bid_20261010_112516.csv"
DEFAULT_OCC = DIR / "last_15m_wheel_occupancy_liq_20261010_120853.csv"
DEFAULT_MTM = DIR / "last_15m_wheel_occupancy_mtm_20261010_122146.csv"
DEFAULT_CURVES = DIR / "last_15m_wheel_equity_curves.json"
DEFAULT_MTM_CACHE = DIR / "theta_wheel_mtm_cache"
MAX_OPEN = 8
MIN_ADV = 25e6
MAX_HALF_PCT = 15.0


def _f(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _day(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    ts = pd.Timestamp(x)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _pf_e_wr(gains: pd.Series) -> Dict[str, Any]:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    if g.empty:
        return {"n": 0, "E": None, "PF": None, "WR": None}
    wins = float(g[g > 0].sum())
    losses = float((-g[g < 0]).sum())
    pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else None)
    return {
        "n": int(len(g)),
        "E": round(float(g.mean()), 4),
        "PF": None if pf is None or not math.isfinite(pf) else round(pf, 4),
        "WR": round(float((g > 0).mean()) * 100.0, 1),
    }


def _year_slices(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    d = df.copy()
    d["_y"] = pd.to_datetime(d["buy_date"], errors="coerce").dt.year
    out = {"FULL": _pf_e_wr(d[col])}
    for label, years in (
        ("2022", [2022]),
        ("2023", [2023]),
        ("2022-23", [2022, 2023]),
        ("2024-26", [2024, 2025, 2026]),
    ):
        out[label] = _pf_e_wr(d.loc[d["_y"].isin(years), col])
    return out


def _series_to_points(eq: pd.Series) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if eq is None or eq.empty:
        return out
    for ts, v in eq.items():
        fv = _f(v)
        if fv is None:
            continue
        out.append({"d": pd.Timestamp(ts).strftime("%Y-%m-%d"), "v": round(fv, 6)})
    return out


def build_equity_curves(
    mid: pd.DataFrame,
    ab: pd.DataFrame,
    *,
    cache_dir: Path,
    include_mtm: bool = True,
) -> Dict[str, Any]:
    """Liquid ADV25+half15 occ8 equity: mid cost, ask/bid cost, optional MTM, SPY."""
    mid_f, ab_f, filt = filter_wheel_by_liquidity(
        mid, ab, min_adv=MIN_ADV, max_half_pct_of_mid=MAX_HALF_PCT
    )
    mid_rows = filled_trades_to_occ_rows(mid_f)
    ab_rows = filled_trades_to_mtm_rows(ab_f)
    mid_kept, mid_meta = occupancy_fcfs_wheel(
        mid_rows, max_open=MAX_OPEN, unique_symbol=True
    )
    ab_kept, ab_meta = occupancy_fcfs_wheel(
        ab_rows, max_open=MAX_OPEN, unique_symbol=True
    )
    all_kept: Sequence[Dict[str, Any]] = list(mid_kept) + list(ab_kept)
    if not all_kept:
        return {"error": "no kept trades", "filter": filt}

    start = min(pd.Timestamp(r["entry_date"]) for r in all_kept) - pd.Timedelta(days=5)
    end = max(pd.Timestamp(r["exit_date"]) for r in all_kept) + pd.Timedelta(days=5)
    syms = sorted({r["symbol"] for r in ab_kept} | {"SPY"})
    LOG.info("equity panels n=%s %s..%s", len(syms), start.date(), end.date())
    panels = load_ohlcv_many(
        syms,
        timeframe="1d",
        start=start.date(),
        end=end.date(),
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        return {"error": "SPY missing", "filter": filt}
    close = spy["close"].astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    calendar = close.index[
        (close.index >= start.normalize()) & (close.index <= end.normalize())
    ]
    spy_eq = spy_buyhold_equity(close, calendar)
    eq_mid, _ = wheel_slot_equity(mid_kept, calendar, max_open=MAX_OPEN)
    eq_ab, _ = wheel_slot_equity(ab_kept, calendar, max_open=MAX_OPEN)

    curves: Dict[str, Any] = {
        "max_open": MAX_OPEN,
        "filter": {
            "min_adv": MIN_ADV,
            "max_half_pct_of_mid": MAX_HALF_PCT,
            **{k: v for k, v in filt.items() if k != "filter"},
        },
        "occupancy": {
            "mid": mid_meta,
            "ask_bid": ab_meta,
        },
        "series": {
            "spy": _series_to_points(spy_eq),
            "mid_cost": _series_to_points(eq_mid),
            "askbid_cost": _series_to_points(eq_ab),
        },
    }

    if include_mtm and ab_kept:
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = ThetaEquitySource(cache_dir=cache_dir)
        t0 = time.time()
        mark_paths = load_option_mark_paths(ab_kept, source, mark_mode="ask_bid")
        LOG.info("MTM mark paths wall=%.1fs", time.time() - t0)
        eq_mtm, _, mtm_meta = wheel_slot_equity_mtm(
            ab_kept,
            calendar,
            max_open=MAX_OPEN,
            mark_mode="ask_bid",
            spot_panels=panels,
            mark_paths=mark_paths,
        )
        curves["series"]["askbid_mtm"] = _series_to_points(eq_mtm)
        curves["mtm_meta"] = mtm_meta
    return curves


def build_trade_rows(mid: pd.DataFrame, ab: pd.DataFrame) -> List[Dict[str, Any]]:
    mid_ok = mid[mid["wheel_ok"] == True].copy()  # noqa: E712
    ab_ok = ab[ab["wheel_ok"] == True].copy()  # noqa: E712
    for df in (mid_ok, ab_ok):
        df["_k"] = (
            df["stock"].astype(str).str.upper()
            + "|"
            + df["put_entry_date"].astype(str)
            + "|"
            + df["put_expiry"].astype(str)
            + "|"
            + pd.to_numeric(df["put_strike"], errors="coerce").round(4).astype(str)
        )
    m = mid_ok.drop_duplicates("_k").set_index("_k")
    a = ab_ok.drop_duplicates("_k").set_index("_k")
    common = m.index.intersection(a.index)
    rows: List[Dict[str, Any]] = []
    for k in common:
        mr = m.loc[k]
        ar = a.loc[k]
        if isinstance(mr, pd.DataFrame):
            mr = mr.iloc[0]
        if isinstance(ar, pd.DataFrame):
            ar = ar.iloc[0]
        put_mid = _f(mr.get("put_credit"))
        put_ab = _f(ar.get("put_credit"))
        half = None
        half_pct = None
        if put_mid is not None and put_ab is not None and put_mid != 0:
            half = put_mid - put_ab
            half_pct = 100.0 * half / put_mid
        g_mid = _f(mr.get("gain_pct"))
        g_ab = _f(ar.get("gain_pct"))
        g_stk = _f(mr.get("gain_pct_stock"))
        adv = _f(mr.get("adv_20"))
        rows.append(
            {
                "symbol": str(mr.get("stock") or "").upper(),
                "buy": _day(mr.get("buy_date")),
                "put_entry": _day(mr.get("put_entry_date")),
                "put_expiry": _day(mr.get("put_expiry")),
                "put_strike": _f(mr.get("put_strike")),
                "put_dte": _f(mr.get("put_dte")),
                "put_delta": _f(mr.get("put_delta")),
                "exit": str(mr.get("wheel_exit_reason") or ""),
                "assigned": bool(mr.get("wheel_assigned")),
                "adv_20": adv,
                "put_credit_mid": put_mid,
                "put_credit_askbid": put_ab,
                "half_spread": half,
                "half_pct_of_mid": half_pct,
                "call_credit_mid": _f(mr.get("call_credit")),
                "call_credit_askbid": _f(ar.get("call_credit")),
                "gain_mid": g_mid,
                "gain_askbid": g_ab,
                "gain_diff": (
                    None
                    if g_mid is None or g_ab is None
                    else round(g_mid - g_ab, 4)
                ),
                "gain_stock": g_stk,
                "liquid_adv25": bool(adv is not None and adv >= 25e6),
                "liquid_half15": bool(half_pct is not None and half_pct <= 15.0),
            }
        )
    rows.sort(key=lambda r: (r.get("buy") or "", r.get("symbol") or ""))
    return rows


def render_html(
    *,
    trades: List[Dict[str, Any]],
    pooled: Dict[str, Any],
    occ_rows: List[Dict[str, Any]],
    mtm_rows: List[Dict[str, Any]],
    curves: Dict[str, Any],
    title: str,
) -> str:
    payload = json.dumps(trades, default=str)
    pooled_j = json.dumps(pooled, default=str)
    occ_j = json.dumps(occ_rows, default=str)
    mtm_j = json.dumps(mtm_rows, default=str)
    curves_j = json.dumps(curves, default=str)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#0b0f14; --panel:#121821; --border:#243041; --text:#e7eef7;
  --muted:#8b9bb0; --green:#3ecf8e; --red:#f07178; --accent:#5b9fd4;
  --warn:#e6b35a;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; padding:24px; font-family:ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--text); line-height:1.45;
}}
h1 {{ font-size:1.45rem; margin:0 0 6px; font-weight:650; }}
h2 {{ font-size:1.05rem; margin:22px 0 10px; color:var(--accent); font-weight:600; }}
.sub {{ color:var(--muted); max-width:960px; margin:0 0 18px; font-size:0.92rem; }}
.banner {{
  border:1px solid var(--warn); background:rgba(230,179,90,0.08);
  border-radius:10px; padding:12px 14px; margin:0 0 18px; max-width:1100px;
}}
.banner strong {{ color:var(--warn); }}
.cards {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:10px; margin:0 0 18px; max-width:1100px;
}}
.card {{
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:12px 14px;
}}
.card .k {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; }}
.card .v {{ font-size:1.25rem; font-weight:650; margin-top:4px; font-variant-numeric:tabular-nums; }}
.card .s {{ color:var(--muted); font-size:0.8rem; margin-top:2px; }}
.pos {{ color:var(--green); }}
.neg {{ color:var(--red); }}
.pass {{ color:var(--green); font-weight:600; }}
.fail {{ color:var(--red); font-weight:600; }}
table.score {{
  width:100%; max-width:1100px; border-collapse:collapse; font-size:0.85rem;
  font-variant-numeric:tabular-nums; margin-bottom:8px;
}}
table.score th, table.score td {{
  padding:7px 8px; border-bottom:1px solid var(--border); text-align:right;
}}
table.score th.left, table.score td.left {{ text-align:left; }}
table.score th {{ color:var(--muted); font-weight:600; background:#15202b; }}
.controls {{
  display:flex; flex-wrap:wrap; gap:12px 16px; align-items:end;
  margin:10px 0 12px; max-width:1100px;
}}
.controls label {{ display:flex; flex-direction:column; gap:4px; font-size:0.78rem; color:var(--muted); }}
.controls input, .controls select {{
  background:#0f1419; color:var(--text); border:1px solid var(--border);
  border-radius:6px; padding:7px 9px; min-width:120px; font-size:0.9rem;
}}
.controls .chk {{ flex-direction:row; align-items:center; gap:8px; padding-bottom:8px; }}
.wrap {{
  overflow:auto; max-height:68vh; border:1px solid var(--border);
  border-radius:8px; max-width:1400px;
}}
table.trades {{
  width:100%; border-collapse:collapse; font-size:0.8rem;
  font-variant-numeric:tabular-nums;
}}
table.trades th, table.trades td {{
  padding:6px 7px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap;
}}
table.trades th {{
  position:sticky; top:0; background:#15202b; color:var(--muted);
  font-weight:600; cursor:pointer; user-select:none;
}}
table.trades th.left, table.trades td.left {{ text-align:left; }}
table.trades tr:hover td {{ background:rgba(91,159,212,0.08); }}
.meta {{ color:var(--muted); font-size:0.8rem; margin:6px 0; }}
code {{ background:#1a2330; padding:1px 5px; border-radius:4px; font-size:0.85em; }}
.chart-box {{
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:14px 16px 8px; max-width:1100px; margin:0 0 8px; height:360px;
}}
.chart-box canvas {{ width:100% !important; height:100% !important; }}
.legend-note {{ color:var(--muted); font-size:0.8rem; max-width:1100px; margin:0 0 18px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">
One-cycle wheel (CSP → assign → CC) on <code>current_best</code> last-15m H2 fills.
Marks: Theta EOD mid vs sell-at-bid. Liquid gate: ADV≥$25M and put half-spread ≤15% of mid.
Occupancy: FCFS unique-symbol equal-slot. Option MTM marks short ask daily.
<strong>No promote</strong> — option-path MTM fails Sharpe&gt;1.
</p>
<div class="banner">
  <strong>Verdict:</strong> mark-at-cost liquid ask/bid occ8 PASSes Phase-5 FULL + 2022 (waived Sharpe&gt;1);
  option-path MTM FAIL (Sharpe 0.69, MDD −10.2%). Do not wire <code>/hot</code>.
</div>
<div class="cards" id="cards"></div>

<h2>Occupancy equity (liquid ADV≥$25M &amp; half≤15%, max_open=8)</h2>
<div class="chart-box"><canvas id="equityChart"></canvas></div>
<p class="legend-note">
Rebased to 1.0. Mid/ask-bid cost mark collateral flat until exit; option MTM marks short ask daily.
SPY is buy-and-hold on the same calendar.
</p>

<h2>Pooled wheel vs stock (matched fills)</h2>
<table class="score" id="pooled"></table>

<h2>Liquid occupancy gates (ADV≥$25M &amp; half≤15%)</h2>
<p class="meta">FULL needs Sharpe&gt;1 + ≥SPY + better MDD. Calendar 2022 waives Sharpe&gt;1 (still ≥SPY + MDD).</p>
<table class="score" id="occ"></table>

<h2>Option-path MTM vs mark-at-cost (ask/bid occ8)</h2>
<table class="score" id="mtm"></table>

<h2>Trades (mid vs ask/bid)</h2>
<div class="controls">
  <label>Symbol<input id="sym" type="text" placeholder="e.g. META"/></label>
  <label>Exit
    <select id="exit">
      <option value="all">All</option>
      <option value="put_expire_otm">put_expire_otm</option>
      <option value="call_assign">call_assign</option>
      <option value="share_liquidate">share_liquidate</option>
    </select>
  </label>
  <label class="chk"><input id="liqOnly" type="checkbox" checked/> Liquid ADV25+half15</label>
  <label class="chk"><input id="assignedOnly" type="checkbox"/> Assigned only</label>
  <span class="meta" id="count"></span>
</div>
<div class="wrap">
<table class="trades">
<thead>
<tr>
  <th class="left" data-k="symbol">Symbol</th>
  <th data-k="buy">Buy</th>
  <th class="left" data-k="exit">Exit</th>
  <th data-k="put_strike">Strike</th>
  <th data-k="put_dte">DTE</th>
  <th data-k="put_credit_mid">Put mid</th>
  <th data-k="put_credit_askbid">Put bid</th>
  <th data-k="half_pct_of_mid">Half%</th>
  <th data-k="gain_mid">Gain mid</th>
  <th data-k="gain_askbid">Gain ask/bid</th>
  <th data-k="gain_diff">Δ</th>
  <th data-k="gain_stock">Stock %</th>
  <th data-k="adv_20">ADV</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<script>
const TRADES = {payload};
const POOLED = {pooled_j};
const OCC = {occ_j};
const MTM = {mtm_j};
const CURVES = {curves_j};
let sortKey = 'buy';
let sortDir = -1;

function fmt(v, dig=2) {{
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(dig);
}}
function fmtAdv(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (v >= 1e9) return (v/1e9).toFixed(1) + 'B';
  if (v >= 1e6) return (v/1e6).toFixed(1) + 'M';
  return (v/1e3).toFixed(0) + 'K';
}}
function pctCls(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return v > 0 ? 'pos' : (v < 0 ? 'neg' : '');
}}
function gateCls(ok) {{ return ok ? 'pass' : 'fail'; }}

function renderCards() {{
  const fullMid = POOLED.mid.FULL || {{}};
  const fullAb = POOLED.ask_bid.FULL || {{}};
  const mtmFull = (MTM.find(r => r.book === 'option_mtm' && r.window === 'FULL') || {{}});
  const costFull = (MTM.find(r => r.book === 'mark_at_cost' && r.window === 'FULL') || {{}});
  const cards = [
    {{k:'Wheel mid FULL', v:(fullMid.E!=null?((fullMid.E>0?'+':'')+fmt(fullMid.E)):'—'), s:'n='+fullMid.n+' PF '+(fullMid.PF??'—')}},
    {{k:'Wheel ask/bid FULL', v:(fullAb.E!=null?((fullAb.E>0?'+':'')+fmt(fullAb.E)):'—'), s:'n='+fullAb.n+' PF '+(fullAb.PF??'—')}},
    {{k:'Occ8 mark-at-cost', v:fmt(costFull.sharpe), s:'Sharpe · MDD '+(costFull.mdd!=null?(100*costFull.mdd).toFixed(1)+'%':'—')}},
    {{k:'Occ8 option MTM', v:fmt(mtmFull.sharpe), s:(mtmFull.pass?'PASS':'FAIL')+' · MDD '+(mtmFull.mdd!=null?(100*mtmFull.mdd).toFixed(1)+'%':'—')}},
  ];
  document.getElementById('cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="k">${{c.k}}</div><div class="v">${{c.v}}</div><div class="s">${{c.s}}</div></div>`
  ).join('');
}}

function renderPooled() {{
  const slices = ['FULL','2022','2023','2022-23','2024-26'];
  let html = '<tr><th class="left">Slice</th><th>Mid n</th><th>Mid E</th><th>Mid PF</th><th>Ask/bid E</th><th>Ask/bid PF</th><th>Stock E</th></tr>';
  for (const s of slices) {{
    const m = POOLED.mid[s] || {{}};
    const a = POOLED.ask_bid[s] || {{}};
    const st = POOLED.stock[s] || {{}};
    html += `<tr>
      <td class="left">${{s}}</td>
      <td>${{m.n??0}}</td>
      <td class="${{pctCls(m.E)}}">${{fmt(m.E)}}</td>
      <td>${{fmt(m.PF)}}</td>
      <td class="${{pctCls(a.E)}}">${{fmt(a.E)}}</td>
      <td>${{fmt(a.PF)}}</td>
      <td class="${{pctCls(st.E)}}">${{fmt(st.E)}}</td>
    </tr>`;
  }}
  document.getElementById('pooled').innerHTML = html;
}}

function renderOcc() {{
  const focus = OCC.filter(r => String(r.book||'').includes('adv25_half15'));
  let html = '<tr><th class="left">Book</th><th>max_open</th><th>kept</th><th>FULL Sharpe</th><th>FULL</th><th>2022 Sharpe</th><th>2022</th><th>CAGR</th><th>MDD</th></tr>';
  for (const r of focus) {{
    html += `<tr>
      <td class="left">${{r.book}}</td>
      <td>${{r.max_open}}</td>
      <td>${{r.n_kept}}</td>
      <td>${{fmt(r.sharpe)}}</td>
      <td class="${{gateCls(r.pass)}}">${{r.pass?'PASS':'FAIL'}}</td>
      <td>${{fmt(r.sharpe_2022)}}</td>
      <td class="${{gateCls(r.pass_2022)}}">${{r.pass_2022?'PASS':'FAIL'}}</td>
      <td>${{r.cagr!=null?(100*r.cagr).toFixed(1)+'%':'—'}}</td>
      <td>${{r.mdd!=null?(100*r.mdd).toFixed(1)+'%':'—'}}</td>
    </tr>`;
  }}
  document.getElementById('occ').innerHTML = html;
}}

function renderMtm() {{
  let html = '<tr><th class="left">Book</th><th>Window</th><th>Sharpe</th><th>CAGR</th><th>MDD</th><th>SPY Sharpe</th><th>Gate</th></tr>';
  for (const r of MTM) {{
    html += `<tr>
      <td class="left">${{r.book}}</td>
      <td>${{r.window}}</td>
      <td>${{fmt(r.sharpe)}}</td>
      <td>${{r.cagr!=null?(100*r.cagr).toFixed(1)+'%':'—'}}</td>
      <td>${{r.mdd!=null?(100*r.mdd).toFixed(1)+'%':'—'}}</td>
      <td>${{fmt(r.spy_sharpe)}}</td>
      <td class="${{gateCls(r.pass)}}">${{r.pass?'PASS':'FAIL'}}</td>
    </tr>`;
  }}
  document.getElementById('mtm').innerHTML = html;
}}

function renderEquity() {{
  const series = (CURVES && CURVES.series) || {{}};
  const keys = [
    {{k:'askbid_cost', label:'Ask/bid mark-at-cost', color:'#3ecf8e'}},
    {{k:'askbid_mtm', label:'Ask/bid option MTM', color:'#e6b35a'}},
    {{k:'mid_cost', label:'Mid mark-at-cost', color:'#5b9fd4'}},
    {{k:'spy', label:'SPY buy&hold', color:'#8b9bb0'}},
  ];
  const allDates = new Set();
  for (const {{k}} of keys) {{
    for (const p of (series[k] || [])) allDates.add(p.d);
  }}
  const labels = Array.from(allDates).sort();
  if (!labels.length) return;
  const byDate = {{}};
  for (const {{k}} of keys) {{
    byDate[k] = {{}};
    for (const p of (series[k] || [])) byDate[k][p.d] = p.v;
  }}
  const datasets = keys.filter((row) => (series[row.k]||[]).length).map((row) => ({{
    label: row.label,
    data: labels.map(d => byDate[row.k][d] ?? null),
    borderColor: row.color,
    backgroundColor: 'transparent',
    borderWidth: row.k === 'spy' ? 1.5 : 2,
    borderDash: row.k === 'spy' ? [4,3] : [],
    pointRadius: 0,
    spanGaps: true,
    tension: 0,
  }}));
  new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          labels: {{ color: '#8b9bb0', boxWidth: 12 }},
        }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const v = ctx.parsed.y;
              if (v == null) return ctx.dataset.label + ': —';
              return ctx.dataset.label + ': ' + v.toFixed(3);
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{
            color: '#8b9bb0',
            maxTicksLimit: 10,
            maxRotation: 0,
          }},
          grid: {{ color: 'rgba(36,48,65,0.6)' }},
        }},
        y: {{
          ticks: {{ color: '#8b9bb0' }},
          grid: {{ color: 'rgba(36,48,65,0.6)' }},
          title: {{ display: true, text: 'Equity (rebased)', color: '#8b9bb0' }},
        }},
      }},
    }},
  }});
}}

function filtered() {{
  const sym = (document.getElementById('sym').value || '').trim().toUpperCase();
  const exit = document.getElementById('exit').value;
  const liqOnly = document.getElementById('liqOnly').checked;
  const assignedOnly = document.getElementById('assignedOnly').checked;
  let out = TRADES.slice();
  if (sym) out = out.filter(r => r.symbol.includes(sym));
  if (exit !== 'all') out = out.filter(r => r.exit === exit);
  if (liqOnly) out = out.filter(r => r.liquid_adv25 && r.liquid_half15);
  if (assignedOnly) out = out.filter(r => r.assigned);
  out.sort((a,b) => {{
    const av = a[sortKey], bv = b[sortKey];
    if (av === bv) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (av < bv) return -1 * sortDir;
    if (av > bv) return 1 * sortDir;
    return 0;
  }});
  return out;
}}

function renderTrades() {{
  const rows = filtered();
  document.getElementById('count').textContent = rows.length + ' / ' + TRADES.length + ' matched fills';
  document.getElementById('tbody').innerHTML = rows.map(r => `<tr>
    <td class="left">${{r.symbol}}</td>
    <td>${{r.buy || '—'}}</td>
    <td class="left">${{r.exit || '—'}}</td>
    <td>${{fmt(r.put_strike,1)}}</td>
    <td>${{fmt(r.put_dte,0)}}</td>
    <td>${{fmt(r.put_credit_mid)}}</td>
    <td>${{fmt(r.put_credit_askbid)}}</td>
    <td>${{fmt(r.half_pct_of_mid,1)}}</td>
    <td class="${{pctCls(r.gain_mid)}}">${{fmt(r.gain_mid)}}</td>
    <td class="${{pctCls(r.gain_askbid)}}">${{fmt(r.gain_askbid)}}</td>
    <td class="${{pctCls(r.gain_diff)}}">${{fmt(r.gain_diff)}}</td>
    <td class="${{pctCls(r.gain_stock)}}">${{fmt(r.gain_stock)}}</td>
    <td>${{fmtAdv(r.adv_20)}}</td>
  </tr>`).join('');
}}

document.querySelectorAll('table.trades th[data-k]').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.getAttribute('data-k');
    if (sortKey === k) sortDir *= -1;
    else {{
      sortKey = k;
      sortDir = (k === 'symbol' || k === 'buy' || k === 'exit') ? 1 : -1;
    }}
    renderTrades();
  }});
}});
['sym','exit','liqOnly','assignedOnly'].forEach(id => {{
  document.getElementById(id).addEventListener('input', renderTrades);
  document.getElementById(id).addEventListener('change', renderTrades);
}});

renderCards();
renderPooled();
renderEquity();
renderOcc();
renderMtm();
renderTrades();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mid", type=Path, default=DEFAULT_MID)
    ap.add_argument("--askbid", type=Path, default=DEFAULT_ASKBID)
    ap.add_argument("--occ", type=Path, default=DEFAULT_OCC)
    ap.add_argument("--mtm", type=Path, default=DEFAULT_MTM)
    ap.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    ap.add_argument("--mtm-cache", type=Path, default=DEFAULT_MTM_CACHE)
    ap.add_argument(
        "--rebuild-curves",
        action="store_true",
        help="Force rebuild equity curves (ignore --curves cache)",
    )
    ap.add_argument(
        "--skip-mtm-curve",
        action="store_true",
        help="Skip option-path MTM series (faster; cost+SPY only)",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    mid = pd.read_csv(args.mid)
    ab = pd.read_csv(args.askbid)
    trades = build_trade_rows(mid, ab)
    LOG.info("Matched mid/ask_bid fills: %d", len(trades))

    mid_ok = mid[mid["wheel_ok"] == True]  # noqa: E712
    ab_ok = ab[ab["wheel_ok"] == True]  # noqa: E712
    pooled = {
        "mid": _year_slices(mid_ok, "gain_pct"),
        "ask_bid": _year_slices(ab_ok, "gain_pct"),
        "stock": _year_slices(mid_ok, "gain_pct_stock"),
    }

    _, _, meta = filter_wheel_by_liquidity(
        mid, ab, min_adv=MIN_ADV, max_half_pct_of_mid=MAX_HALF_PCT
    )
    LOG.info("Liquid ADV25+half15 kept=%s", meta.get("n_kept"))

    occ_rows: List[Dict[str, Any]] = []
    if args.occ.exists():
        occ = pd.read_csv(args.occ)
        occ_rows = occ.to_dict(orient="records")
    mtm_rows: List[Dict[str, Any]] = []
    if args.mtm.exists():
        mtm = pd.read_csv(args.mtm)
        mtm_rows = mtm.to_dict(orient="records")

    curves: Dict[str, Any] = {}
    curves_path = Path(args.curves)
    if not args.rebuild_curves and curves_path.exists():
        try:
            cached = json.loads(curves_path.read_text(encoding="utf-8"))
            has_series = bool(cached.get("series"))
            has_mtm = bool((cached.get("series") or {}).get("askbid_mtm"))
            if has_series and (args.skip_mtm_curve or has_mtm):
                curves = cached
                LOG.info("Loaded equity curves from %s", curves_path)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not load curves cache: %s", exc)

    if not curves.get("series"):
        LOG.info("Building equity curves (include_mtm=%s)...", not args.skip_mtm_curve)
        curves = build_equity_curves(
            mid,
            ab,
            cache_dir=Path(args.mtm_cache),
            include_mtm=not args.skip_mtm_curve,
        )
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves_path.write_text(
            json.dumps(curves, indent=2, default=str), encoding="utf-8"
        )
        LOG.info("Wrote %s", curves_path)

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    html_path = outdir / "last_15m_wheel_report.html"
    html_path.write_text(
        render_html(
            trades=trades,
            pooled=pooled,
            occ_rows=occ_rows,
            mtm_rows=mtm_rows,
            curves=curves,
            title="Last-15m wheel — claimcheck / occupancy / MTM",
        ),
        encoding="utf-8",
    )
    csv_path = outdir / "last_15m_wheel_report_trades.csv"
    pd.DataFrame(trades).to_csv(csv_path, index=False)
    LOG.info("Wrote %s", html_path)
    LOG.info("Wrote %s", csv_path)
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
