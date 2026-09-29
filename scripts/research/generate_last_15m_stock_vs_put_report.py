#!/usr/bin/env python3
"""Per-trade stock vs short-put comparison HTML report.

Shows each last-15m H2 signal as:
  - stock trade P&L (original book)
  - short-put cash-secured P&L (BS ATM 45d)
  - when available: live Polygon mid / half-spread (honest entry-on-signal)

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_last_15m_stock_vs_put_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.short_put_live_realism import summarize_gains  # noqa: E402

LOG = logging.getLogger("stock_vs_put_report")

DEFAULT_BS = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-26"
    / "last_15m_short_put_bs_dte45_atm_trades_20260926_213902.csv"
)
DEFAULT_LIVE = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-27"
    / "last_15m_short_put_live_realism_honest_filter.csv"
)


def _pct(x: Any) -> Optional[float]:
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


def build_rows(bs: pd.DataFrame, live: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    live_map: Dict[tuple, dict] = {}
    if live is not None and not live.empty:
        for _, r in live.iterrows():
            key = (str(r["stock"]).upper(), str(r["buy_date"])[:10])
            live_map[key] = r.to_dict()

    rows: List[Dict[str, Any]] = []
    for _, r in bs.iterrows():
        if "put_ok" in r.index and not bool(r["put_ok"]):
            continue
        sym = str(r["stock"]).upper()
        buy = _day(r.get("buy_date"))
        if not buy:
            continue
        stock_g = _pct(r.get("gain_pct_stock"))
        if stock_g is None:
            stock_g = _pct(r.get("stock_gain_pct"))
        put_g = _pct(r.get("gain_pct"))
        live_r = live_map.get((sym, buy))
        row: Dict[str, Any] = {
            "symbol": sym,
            "buy": buy,
            "stock_sell": _day(r.get("sell_date")),
            "put_exit": _day(r.get("put_exit_date")),
            "put_reason": str(r.get("put_exit_reason") or ""),
            "stock_entry": _pct(r.get("buy_price")),
            "stock_exit": _pct(r.get("sell_price")),
            "stock_pct": stock_g,
            "put_strike": _pct(r.get("put_strike")),
            "put_credit": _pct(r.get("put_credit")),
            "put_exit_px": _pct(r.get("put_exit_premium")),
            "put_pct": put_g,
            "put_vs_stock": (
                None
                if put_g is None or stock_g is None
                else put_g - stock_g
            ),
            "hold_stock": _pct(r.get("hold_days")),
            "hold_put": _pct(r.get("put_hold_days")),
            "has_live": live_r is not None,
            "live_mid_pct": None,
            "live_spread_pct": None,
            "live_exit_vs_bs": None,
            "live_ticker": None,
        }
        if live_r is not None:
            row["live_mid_pct"] = _pct(live_r.get("live_gain_cs_mid"))
            row["live_spread_pct"] = _pct(live_r.get("live_gain_cs_spread"))
            row["live_exit_vs_bs"] = _pct(live_r.get("exit_vs_bs"))
            row["live_ticker"] = (
                None
                if live_r.get("live_ticker") is None or pd.isna(live_r.get("live_ticker"))
                else str(live_r.get("live_ticker"))
            )
        rows.append(row)
    return rows


def _slice_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return summarize_gains(vals)


def render_html(rows: List[Dict[str, Any]], *, title: str) -> str:
    all_rows = rows
    live_rows = [r for r in rows if r.get("has_live")]
    hs_rows = [r for r in rows if r.get("put_reason") == "hard_stop"]

    cards = [
        ("Stock (all)", _slice_stats(all_rows, "stock_pct")),
        ("Put BS (all)", _slice_stats(all_rows, "put_pct")),
        ("Stock (hard-stop)", _slice_stats(hs_rows, "stock_pct")),
        ("Put BS (hard-stop)", _slice_stats(hs_rows, "put_pct")),
        ("Put live mid (honest)", _slice_stats(live_rows, "live_mid_pct")),
        ("Put live spread (honest)", _slice_stats(live_rows, "live_spread_pct")),
    ]

    def fmt_card(label: str, s: Dict[str, float]) -> str:
        if not s or s.get("n", 0) == 0:
            return (
                f'<div class="card"><div class="k">{label}</div>'
                f'<div class="v muted">n=0</div></div>'
            )
        return (
            f'<div class="card"><div class="k">{label}</div>'
            f'<div class="v">n={int(s["n"])} &nbsp; E={s["E"]:+.2f}% &nbsp; '
            f'PF={s["PF"]:.2f} &nbsp; WR={s["WR"]:.0f}%</div></div>'
        )

    cards_html = "".join(fmt_card(a, b) for a, b in cards)
    payload = json.dumps(all_rows, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
:root {{
  --bg:#0f1419; --panel:#1a2332; --border:#2d3a4d; --text:#e7ecf3;
  --muted:#8b9bb4; --green:#3dd68c; --red:#f07178; --accent:#5b9fd4;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font-family:Segoe UI,system-ui,sans-serif; background:var(--bg);
  color:var(--text); padding:20px 22px 48px;
}}
h1 {{ font-size:1.35rem; font-weight:600; margin:0 0 6px; }}
.sub {{ color:var(--muted); font-size:0.9rem; margin-bottom:18px; max-width:900px; line-height:1.45; }}
.cards {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:10px; margin-bottom:16px;
}}
.card {{
  background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px;
}}
.card .k {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:.04em; }}
.card .v {{ margin-top:6px; font-size:0.92rem; font-variant-numeric:tabular-nums; }}
.muted {{ color:var(--muted); }}
.controls {{
  display:flex; flex-wrap:wrap; gap:10px 14px; align-items:end;
  background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; margin-bottom:12px;
}}
.controls label {{ display:flex; flex-direction:column; gap:4px; font-size:0.75rem; color:var(--muted); }}
.controls input, .controls select {{
  background:#0f1419; color:var(--text); border:1px solid var(--border);
  border-radius:6px; padding:7px 9px; min-width:120px; font-size:0.9rem;
}}
.controls .chk {{ flex-direction:row; align-items:center; gap:8px; padding-bottom:8px; }}
table {{
  width:100%; border-collapse:collapse; font-size:0.82rem;
  font-variant-numeric:tabular-nums;
}}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }}
th {{
  position:sticky; top:0; background:#15202b; color:var(--muted);
  font-weight:600; text-align:right; cursor:pointer; user-select:none;
}}
th.left, td.left {{ text-align:left; }}
tr:hover td {{ background:rgba(91,159,212,0.08); }}
.pos {{ color:var(--green); }}
.neg {{ color:var(--red); }}
.meta {{ color:var(--muted); font-size:0.8rem; margin:8px 0 4px; }}
.wrap {{ overflow:auto; max-height:70vh; border:1px solid var(--border); border-radius:8px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">
Same last-15m H2 signal shown two ways: buy stock (original book) vs sell ATM cash-secured put
(DTE~45, TP 50% credit, stop on underlying ATR k=2). Put % is cash-secured
<code>(credit − exit) / strike</code>. Live columns only on honest Polygon matches
(entry print on signal day). No promote.
</p>
<div class="cards">{cards_html}</div>
<div class="controls">
  <label>Symbol filter<input id="sym" type="text" placeholder="e.g. GOOG"/></label>
  <label>Put exit
    <select id="reason">
      <option value="all">All</option>
      <option value="hard_stop">hard_stop</option>
      <option value="tp50">tp50</option>
      <option value="expiry">expiry</option>
    </select>
  </label>
  <label class="chk"><input id="liveOnly" type="checkbox"/> Live honest only</label>
  <label class="chk"><input id="putBetter" type="checkbox"/> Put beat stock</label>
  <span class="meta" id="count"></span>
</div>
<div class="wrap">
<table>
<thead>
<tr>
  <th class="left" data-k="symbol">Symbol</th>
  <th data-k="buy">Buy</th>
  <th data-k="stock_sell">Stock sell</th>
  <th data-k="put_exit">Put exit</th>
  <th class="left" data-k="put_reason">Put reason</th>
  <th data-k="stock_pct">Stock %</th>
  <th data-k="put_pct">Put BS %</th>
  <th data-k="put_vs_stock">Put−Stock</th>
  <th data-k="live_mid_pct">Live mid %</th>
  <th data-k="live_spread_pct">Live spread %</th>
  <th data-k="stock_entry">Stock entry</th>
  <th data-k="put_credit">Put credit</th>
  <th data-k="put_exit_px">Put exit px</th>
  <th data-k="put_strike">Strike</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>
<script>
const ROWS = {payload};
let sortKey = 'buy';
let sortDir = -1;

function fmt(v, dig=2) {{
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(dig);
}}
function pctCls(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return v > 0 ? 'pos' : (v < 0 ? 'neg' : '');
}}
function filtered() {{
  const sym = (document.getElementById('sym').value || '').trim().toUpperCase();
  const reason = document.getElementById('reason').value;
  const liveOnly = document.getElementById('liveOnly').checked;
  const putBetter = document.getElementById('putBetter').checked;
  let out = ROWS.slice();
  if (sym) out = out.filter(r => r.symbol.includes(sym));
  if (reason !== 'all') out = out.filter(r => r.put_reason === reason);
  if (liveOnly) out = out.filter(r => r.has_live);
  if (putBetter) out = out.filter(r => r.put_vs_stock !== null && r.put_vs_stock > 0);
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
function render() {{
  const rows = filtered();
  document.getElementById('count').textContent = rows.length + ' / ' + ROWS.length + ' trades';
  const tb = document.getElementById('tbody');
  tb.innerHTML = rows.map(r => `<tr>
    <td class="left">${{r.symbol}}</td>
    <td>${{r.buy || '—'}}</td>
    <td>${{r.stock_sell || '—'}}</td>
    <td>${{r.put_exit || '—'}}</td>
    <td class="left">${{r.put_reason || '—'}}</td>
    <td class="${{pctCls(r.stock_pct)}}">${{fmt(r.stock_pct)}}</td>
    <td class="${{pctCls(r.put_pct)}}">${{fmt(r.put_pct)}}</td>
    <td class="${{pctCls(r.put_vs_stock)}}">${{fmt(r.put_vs_stock)}}</td>
    <td class="${{pctCls(r.live_mid_pct)}}">${{fmt(r.live_mid_pct)}}</td>
    <td class="${{pctCls(r.live_spread_pct)}}">${{fmt(r.live_spread_pct)}}</td>
    <td>${{fmt(r.stock_entry)}}</td>
    <td>${{fmt(r.put_credit)}}</td>
    <td>${{fmt(r.put_exit_px)}}</td>
    <td>${{fmt(r.put_strike, 1)}}</td>
  </tr>`).join('');
}}
document.querySelectorAll('th[data-k]').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.getAttribute('data-k');
    if (sortKey === k) sortDir *= -1;
    else {{ sortKey = k; sortDir = (k === 'symbol' || k === 'buy' || k === 'put_reason') ? 1 : -1; }}
    render();
  }});
}});
['sym','reason','liveOnly','putBetter'].forEach(id => {{
  document.getElementById(id).addEventListener('input', render);
  document.getElementById(id).addEventListener('change', render);
}});
render();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_BS)
    ap.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bs = pd.read_csv(args.trades)
    live = None
    if not args.no_live and args.live and Path(args.live).exists():
        live = pd.read_csv(args.live)
        LOG.info("Live honest rows: %d", len(live))
    else:
        LOG.info("No live honest join")

    rows = build_rows(bs, live)
    LOG.info("Built %d trade rows", len(rows))

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    html_path = outdir / "1d_last_15m_stock_vs_put.html"
    html_path.write_text(
        render_html(rows, title="Last-15m H2: stock vs short put"),
        encoding="utf-8",
    )

    # CSV for spreadsheet use
    csv_path = outdir / "1d_last_15m_stock_vs_put_trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    s_stock = _slice_stats(rows, "stock_pct")
    s_put = _slice_stats(rows, "put_pct")
    live_rows = [r for r in rows if r.get("has_live")]
    s_live = _slice_stats(live_rows, "live_spread_pct")
    LOG.info(
        "Stock n=%d E=%+.2f PF=%.2f | Put BS n=%d E=%+.2f PF=%.2f | "
        "Live honest spread n=%d E=%+.2f",
        s_stock["n"],
        s_stock["E"],
        s_stock["PF"],
        s_put["n"],
        s_put["E"],
        s_put["PF"],
        s_live["n"],
        s_live.get("E", float("nan")),
    )
    LOG.info("Wrote %s", html_path)
    LOG.info("Wrote %s", csv_path)
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
