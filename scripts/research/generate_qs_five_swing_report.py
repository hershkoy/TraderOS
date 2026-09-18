#!/usr/bin/env python3
"""QS Five Swing Strategies — SPY claim-check HTML report.

Source: Quantified Strategies article + video KuXV0LRfJx8.
Shared exit: sell at close when close > prior day's high.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_qs_five_swing_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.generate_deep_pullback_report import (  # noqa: E402
    YAHOO_CACHE,
    load_yahoo,
)
from utils.research.qs_five_swing import (  # noqa: E402
    STRATEGY_IDS,
    STRATEGY_NAMES,
    run_all,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_qs_five_swing_report")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "qs_five_swing"

# Claimed metrics from QS article / video (approx; article S5 block looks copy-pasted).
CLAIMED: Dict[str, Dict[str, Any]] = {
    "S1": {
        "cagr_pct": 8.0,
        "exposure_pct": 18.0,
        "mdd_pct": 23.0,
        "final_equity_100k": 1_200_000,
        "note": "article ~8% / 18% / 23% MDD; $100k -> ~$1.2M",
    },
    "S2": {
        "cagr_pct": 7.2,
        "exposure_pct": 11.0,
        "mdd_pct": 16.0,
        "n": 271,
        "avg_hold_days": 4.0,
        "final_equity_100k": 1_000_000,
        "risk_adj_return_pct": 65.0,
        "note": "article 271 trades, ~4d hold, 7.2% / 11% / 16% MDD",
    },
    "S3": {
        "cagr_pct": 8.0,
        "exposure_pct": 21.0,
        "final_equity_100k": 1_250_000,
        "risk_adj_return_pct": 37.0,
        "note": "article ~8% / 21%; $100k -> ~$1.25M",
    },
    "S4": {
        "note": "article: weaker than S1-S3; complements the set (no hard numbers)",
    },
    "S5": {
        "cagr_pct": 3.3,
        "mdd_pct": 8.0,
        "final_equity_100k": 300_000,
        "note": "video ~3.3% CAGR / 8% MDD; article $100k->$300k (ignore article 7.2% paste)",
    },
    "COMBINED": {
        "n": 975,
        "avg_gain_pct": 0.38,
        "cagr_pct": 11.1,
        "exposure_pct": 41.0,
        "mdd_pct": 23.0,
        "final_equity_100k": 3_200_000,
        "friction_note": "article includes ~0.03% cost per trade",
        "losing_years": {"2008": -1.0, "2018": -13.0},
        "note": "single slot; ignore buys while in position",
    },
}


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _slim(result: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"trades", "equity_monthly", "bh_monthly"}
    return {k: v for k, v in result.items() if k not in skip}


def _metric_row(label: str, claimed: Any, ours: Any) -> str:
    c = "-" if claimed is None else claimed
    o = "-" if ours is None else ours
    return "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (_esc(str(label)), c, o)


def render_html(
    *,
    results_by_friction: Dict[str, Dict[str, Dict[str, Any]]],
    claimed: Dict[str, Dict[str, Any]],
    title: str,
    generated: str,
    bars: int,
    first_bar: str,
    last_bar: str,
    wall_s: float,
) -> str:
    frictions = sorted(results_by_friction.keys(), key=lambda x: float(x))
    # Prefer friction 0 / 0.0 as primary claim-check book when present.
    zero_keys = [k for k in frictions if abs(float(k)) < 1e-12]
    primary = results_by_friction[zero_keys[0] if zero_keys else frictions[0]]

    # Combined equity chart data (friction 0)
    combo = primary["COMBINED"]
    eq_pts = combo.get("equity_monthly") or []
    bh_pts = combo.get("bh_monthly") or []

    sections: List[str] = []
    order = list(STRATEGY_IDS) + ["COMBINED"]
    for sid in order:
        r = primary[sid]
        cl = claimed.get(sid, {})
        name = r.get("strategy_name") or STRATEGY_NAMES.get(sid, sid)
        rows = [
            _metric_row("trades (n)", cl.get("n"), r.get("n")),
            _metric_row("avg gain %", cl.get("avg_gain_pct"), r.get("avg_gain_pct")),
            _metric_row("win rate %", cl.get("win_rate_pct"), r.get("win_rate_pct")),
            _metric_row("profit factor", cl.get("profit_factor"), r.get("profit_factor")),
            _metric_row("CAGR %", cl.get("cagr_pct"), r.get("cagr_full_window_pct")),
            _metric_row("exposure %", cl.get("exposure_pct"), r.get("exposure_pct")),
            _metric_row("MDD MTM %", cl.get("mdd_pct"), r.get("mdd_mtm_pct")),
            _metric_row("avg hold days", cl.get("avg_hold_days"), r.get("avg_hold_days")),
            _metric_row(
                "$100k final",
                cl.get("final_equity_100k"),
                r.get("final_equity_100k"),
            ),
            _metric_row(
                "risk-adj %",
                cl.get("risk_adj_return_pct"),
                r.get("risk_adj_return_pct"),
            ),
        ]
        note = cl.get("note") or ""
        fills = r.get("fills_by_strategy") or {}
        fills_html = ""
        if fills:
            fills_html = (
                "<p class='muted'>Fills by sleeve: "
                + ", ".join("%s=%s" % (k, v) for k, v in sorted(fills.items()))
                + "</p>"
            )
        losing = r.get("losing_years") or []
        lose_html = ""
        if losing:
            lose_html = (
                "<p class='muted'>Losing calendar years (sum trade %%): "
                + ", ".join("%s (%s)" % (x["year"], x["sum_ret"]) for x in losing)
                + "</p>"
            )
        sections.append(
            """
<section class="card">
  <h2>%s <span class="tag">%s</span></h2>
  <p class="muted">%s</p>
  <table>
    <thead><tr><th>Metric</th><th>Claimed</th><th>Ours (fric 0)</th></tr></thead>
    <tbody>%s</tbody>
  </table>
  %s%s
</section>
"""
            % (
                _esc(str(name)),
                _esc(sid),
                _esc(str(note)),
                "\n".join(rows),
                fills_html,
                lose_html,
            )
        )

    # Friction comparison for combined
    fric_rows: List[str] = []
    for fk in frictions:
        r = results_by_friction[fk]["COMBINED"]
        fric_rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                fk,
                r.get("n"),
                r.get("avg_gain_pct"),
                r.get("profit_factor"),
                r.get("cagr_full_window_pct"),
                r.get("mdd_mtm_pct"),
                r.get("exposure_pct"),
            )
        )

    eq_json = json.dumps(eq_pts, separators=(",", ":"))
    bh_json = json.dumps(bh_pts, separators=(",", ":"))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>%s</title>
<style>
  :root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#9aa7b8; --accent:#5b9fd4; --line:#2a3548; }
  body { margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:1100px; margin:0 auto; padding:28px 18px 60px; }
  h1 { font-size:1.6rem; margin:0 0 8px; }
  h2 { font-size:1.15rem; margin:0 0 10px; }
  .muted { color:var(--muted); font-size:0.92rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:16px 0; }
  table { width:100%%; border-collapse:collapse; font-size:0.92rem; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  .tag { font-size:0.75rem; color:var(--accent); margin-left:6px; }
  #eqChart { width:100%%; height:320px; background:#121821; border-radius:8px; }
  a { color:var(--accent); }
  ul.rules { margin:8px 0 0 18px; color:var(--muted); }
</style>
</head>
<body>
<main>
  <h1>%s</h1>
  <p class="muted">Yahoo SPY daily %s -> %s (%s bars). Generated %s. Wall %.1fs.
  Claim-check sleeve (not Sharpe&gt;1 promote). Source:
  <a href="https://www.quantifiedstrategies.com/swing-trading-strategies-for-beginners/">QS article</a>,
  <a href="https://www.youtube.com/watch?v=KuXV0LRfJx8">video</a>.</p>

  <section class="card">
    <h2>Rules</h2>
    <ul class="rules">
      <li>S1: close &lt; (10d high - 2.5 * 25d avg range) and IBS &lt; 0.3</li>
      <li>S2: Monday close down two sessions in a row</li>
      <li>S3: close &lt; min(low of prior 5 days)</li>
      <li>S4: range &lt; min(range of prior 6) and ADX(5) &gt; 40</li>
      <li>S5: high &gt; max(high of prior 10) and IBS &lt; 0.15</li>
      <li>Exit (all): close &gt; prior day high</li>
      <li>Combined: one slot; priority S1-&gt;S5 when flat</li>
    </ul>
  </section>

  <section class="card">
    <h2>Combined equity (friction 0)</h2>
    <canvas id="eqChart" width="1000" height="320"></canvas>
    <p class="muted">Blue = strategy MTM (cash when flat). Gray = SPY buy&amp;hold from warmup.</p>
  </section>

  <section class="card">
    <h2>Combined friction sensitivity</h2>
    <table>
      <thead><tr><th>fric %%</th><th>n</th><th>E %%</th><th>PF</th><th>CAGR %%</th><th>MDD %%</th><th>exp %%</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </section>

  %s
</main>
<script>
(function(){
  const eq = %s;
  const bh = %s;
  const c = document.getElementById('eqChart');
  if (!c || !eq.length) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {l:48,r:12,t:16,b:28};
  const all = eq.concat(bh);
  let ymin = Math.min.apply(null, all.map(p => p.equity));
  let ymax = Math.max.apply(null, all.map(p => p.equity));
  if (ymin === ymax) { ymin *= 0.9; ymax *= 1.1; }
  const x0 = pad.l, y0 = pad.t, x1 = W-pad.r, y1 = H-pad.b;
  function X(i, n){ return x0 + (x1-x0)*(i/Math.max(n-1,1)); }
  function Y(v){ return y1 - (y1-y0)*((v-ymin)/(ymax-ymin)); }
  function draw(pts, color){
    ctx.beginPath();
    pts.forEach((p,i) => {
      const x = X(i, pts.length), y = Y(p.equity);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
  }
  ctx.fillStyle = '#121821'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle = '#2a3548'; ctx.beginPath();
  ctx.moveTo(x0,y0); ctx.lineTo(x0,y1); ctx.lineTo(x1,y1); ctx.stroke();
  draw(bh, '#6b7785');
  draw(eq, '#5b9fd4');
  ctx.fillStyle = '#9aa7b8'; ctx.font = '12px sans-serif';
  ctx.fillText(ymin.toFixed(2), 4, y1); ctx.fillText(ymax.toFixed(2), 4, y0+10);
})();
</script>
</body>
</html>
""" % (
        _esc(title),
        _esc(title),
        _esc(first_bar),
        _esc(last_bar),
        bars,
        _esc(generated),
        wall_s,
        "\n".join(fric_rows),
        "\n".join(sections),
        eq_json,
        bh_json,
    )
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="QS Five Swing SPY claim-check report")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--start", default="1993-01-01")
    ap.add_argument(
        "--friction",
        default="0,0.03,0.25",
        help="Comma-separated round-turn friction percent values",
    )
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    yahoo = load_yahoo(YAHOO_CACHE, refresh=args.refresh_yahoo)
    df = yahoo[yahoo["ts"] >= args.start].reset_index(drop=True)
    frictions = [float(x.strip()) for x in str(args.friction).split(",") if x.strip()]

    results_by_friction: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in frictions:
        logger.info("Running all strategies friction=%.3f", f)
        results_by_friction[str(f)] = run_all(df, symbol="SPY", friction_pct=f)

    wall_s = time.perf_counter() - t0
    primary = results_by_friction[str(frictions[0])]
    combo0 = primary["COMBINED"]

    title = "QS Five Swing Strategies — SPY claim check"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = render_html(
        results_by_friction=results_by_friction,
        claimed=CLAIMED,
        title=title,
        generated=generated,
        bars=len(df),
        first_bar=str(df["ts"].iloc[0].date()),
        last_bar=str(df["ts"].iloc[-1].date()),
        wall_s=wall_s,
    )
    html_path = outdir / "qs_five_swing_spy.html"
    html_path.write_text(html, encoding="utf-8")

    # Trades CSVs
    for sid, res in primary.items():
        stem = "qs_five_swing_%s_trades.csv" % sid.lower()
        pd.DataFrame(res.get("trades") or []).to_csv(outdir / stem, index=False)

    summary = {
        "claimed": CLAIMED,
        "wall_clock_s": round(wall_s, 2),
        "bars": len(df),
        "first_bar": str(df["ts"].iloc[0].date()),
        "last_bar": str(df["ts"].iloc[-1].date()),
        "friction_keys": list(results_by_friction.keys()),
        "by_friction": {
            fk: {sid: _slim(res) for sid, res in books.items()}
            for fk, books in results_by_friction.items()
        },
        "html": str(html_path),
    }
    json_path = outdir / "qs_five_swing_spy_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    full_path = outdir / "qs_five_swing_spy_full.json"
    # Keep trades in full dump for friction 0 only to limit size.
    full_payload = {
        "claimed": CLAIMED,
        "wall_clock_s": round(wall_s, 2),
        "friction_0": primary,
        "friction_tables": {
            fk: {sid: _slim(res) for sid, res in books.items()}
            for fk, books in results_by_friction.items()
        },
    }
    full_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    logger.info(
        "COMBINED n=%s E=%.3f PF=%s CAGR=%.2f MDD=%.2f exp=%.2f%% final100k=%.0f wall=%.1fs",
        combo0.get("n"),
        combo0.get("avg_gain_pct") or 0.0,
        combo0.get("profit_factor"),
        combo0.get("cagr_full_window_pct") or 0.0,
        combo0.get("mdd_mtm_pct") or 0.0,
        combo0.get("exposure_pct") or 0.0,
        combo0.get("final_equity_100k") or 0.0,
        wall_s,
    )
    for sid in list(STRATEGY_IDS) + ["COMBINED"]:
        r = primary[sid]
        logger.info(
            "  %s n=%s CAGR=%.2f exp=%.2f MDD=%.2f E=%.3f",
            sid,
            r.get("n"),
            r.get("cagr_full_window_pct") or 0.0,
            r.get("exposure_pct") or 0.0,
            r.get("mdd_mtm_pct") or 0.0,
            r.get("avg_gain_pct") or 0.0,
        )
    logger.info("Wrote %s", html_path)
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
