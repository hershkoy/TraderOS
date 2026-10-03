"""Build a readable HTML report from SEC EDGAR companyfacts for one ticker."""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.api.sec.companyfacts_client import (
    USER_AGENT,
    CompanyfactsClient,
    dedupe_by_end,
    extract_series,
)

DEFAULT_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "CommonStockSharesOutstanding",
    "ResearchAndDevelopmentExpense",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
]


def _fmt_val(val: Any, unit: str) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        if unit in ("USD/shares", "pure") or abs(val) < 1000:
            return f"{val:,.4f}".rstrip("0").rstrip(".")
        return f"{val:,.0f}"
    return html.escape(str(val))


def render_html(
    ticker: str,
    cik: str,
    payload: Dict[str, Any],
    nbytes: int,
    annual: List[Dict[str, Any]],
    quarterly_ni: Optional[Dict[str, Any]],
) -> str:
    entity = html.escape(str(payload.get("entityName") or ""))
    facts = payload.get("facts") or {}
    tax_counts = {k: len(v) if isinstance(v, dict) else 0 for k, v in facts.items()}
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    concept_sections: List[str] = []
    for series in annual:
        rows_html = "\n".join(
            f"<tr>"
            f"<td>{html.escape(str(r.get('end') or ''))}</td>"
            f"<td>{html.escape(str(r.get('fy') or ''))}</td>"
            f"<td>{html.escape(str(r.get('fp') or ''))}</td>"
            f"<td>{html.escape(str(r.get('form') or ''))}</td>"
            f"<td>{html.escape(str(r.get('filed') or ''))}</td>"
            f"<td class='num'>{_fmt_val(r.get('val'), series['unit'])}</td>"
            f"</tr>"
            for r in series["rows"]
        )
        desc = html.escape(series["description"][:240])
        concept_sections.append(
            f"""
<section class="card">
  <h3>{html.escape(series['label'])}</h3>
  <p class="meta"><code>{html.escape(series['concept'])}</code> · unit <code>{html.escape(series['unit'])}</code>
     · {series['n_raw']} raw obs · {series['n_unique_ends']} unique period ends</p>
  <p class="desc">{desc}</p>
  <table>
    <thead><tr><th>Period end</th><th>FY</th><th>FP</th><th>Form</th><th>Filed</th><th>Value</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</section>
"""
        )

    q_section = ""
    if quarterly_ni and quarterly_ni["rows"]:
        rows_html = "\n".join(
            f"<tr>"
            f"<td>{html.escape(str(r.get('end') or ''))}</td>"
            f"<td>{html.escape(str(r.get('fy') or ''))}</td>"
            f"<td>{html.escape(str(r.get('fp') or ''))}</td>"
            f"<td>{html.escape(str(r.get('form') or ''))}</td>"
            f"<td>{html.escape(str(r.get('filed') or ''))}</td>"
            f"<td class='num'>{_fmt_val(r.get('val'), quarterly_ni['unit'])}</td>"
            f"</tr>"
            for r in quarterly_ni["rows"]
        )
        q_section = f"""
<section class="card">
  <h2>Quarterly sample — Net Income</h2>
  <p class="meta"><code>NetIncomeLoss</code> · unit <code>{html.escape(quarterly_ni['unit'])}</code></p>
  <table>
    <thead><tr><th>Period end</th><th>FY</th><th>FP</th><th>Form</th><th>Filed</th><th>Value</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</section>
"""

    tax_li = "".join(f"<li><code>{html.escape(k)}</code>: {v} concepts</li>" for k, v in tax_counts.items())
    missing = [c for c in DEFAULT_CONCEPTS if not any(s["concept"] == c for s in annual)]
    missing_html = (
        "<p class='meta'>All requested concepts present.</p>"
        if not missing
        else "<ul>" + "".join(f"<li><code>{html.escape(c)}</code></li>" for c in missing) + "</ul>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SEC Company Facts — {html.escape(ticker)}</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --card: #ffffff;
    --ink: #1a1f24;
    --muted: #5b6570;
    --line: #d7dde3;
    --accent: #0b6e4f;
    --num: #0b3d5c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", system-ui, sans-serif;
    color: var(--ink);
    background:
      radial-gradient(1200px 500px at 10% -10%, #d9efe6 0%, transparent 55%),
      radial-gradient(900px 400px at 100% 0%, #d6e6f5 0%, transparent 50%),
      var(--bg);
    line-height: 1.45;
  }}
  header {{
    padding: 2rem 1.5rem 1rem;
    max-width: 1100px;
    margin: 0 auto;
  }}
  header h1 {{
    margin: 0 0 0.35rem;
    font-size: 1.85rem;
    letter-spacing: -0.02em;
  }}
  header .sub {{ color: var(--muted); margin: 0; }}
  main {{
    max-width: 1100px;
    margin: 0 auto 3rem;
    padding: 0 1.5rem;
    display: grid;
    gap: 1rem;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.15rem 1.2rem;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03);
  }}
  h2 {{ margin: 0 0 0.75rem; font-size: 1.2rem; color: var(--accent); }}
  h3 {{ margin: 0 0 0.35rem; font-size: 1.05rem; }}
  .meta {{ color: var(--muted); font-size: 0.9rem; margin: 0.2rem 0 0.6rem; }}
  .desc {{ color: var(--muted); font-size: 0.88rem; margin: 0 0 0.75rem; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 0.4rem 0.5rem;
    text-align: left;
    vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; color: var(--num); font-family: ui-monospace, Consolas, monospace; }}
  code {{ font-size: 0.85em; background: #eef2f5; padding: 0.05rem 0.3rem; border-radius: 4px; }}
  ul {{ margin: 0.3rem 0 0; padding-left: 1.2rem; }}
  footer {{
    max-width: 1100px;
    margin: 0 auto 2rem;
    padding: 0 1.5rem;
    color: var(--muted);
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(ticker)} — SEC Company Facts</h1>
  <p class="sub">{entity} · CIK {html.escape(cik)} · payload {nbytes/1024/1024:.2f} MiB · generated {gen_at}</p>
</header>
<main>
  <section class="card">
    <h2>Overview</h2>
    <p class="meta">Source: <code>data.sec.gov/api/xbrl/companyfacts</code> (free, no API key). Annual tables below keep the latest filing per period-end.</p>
    <ul>{tax_li}</ul>
    <h3 style="margin-top:1rem">Requested concepts missing</h3>
    {missing_html}
  </section>
  <section class="card">
    <h2>Annual / FY series</h2>
  </section>
  {''.join(concept_sections)}
  {q_section}
</main>
<footer>
  User-Agent used for SEC fair-access: {html.escape(USER_AGENT)}. Values are as reported in XBRL; units are typically USD unless noted.
</footer>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument(
        "--outdir",
        default="reports/research/sec_companyfacts",
        help="Directory for HTML output",
    )
    ap.add_argument("--annual-limit", type=int, default=12)
    ap.add_argument("--quarterly-limit", type=int, default=12)
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()
    client = CompanyfactsClient()
    print(f"Resolving CIK for {ticker}...")
    cik = client.resolve_cik(ticker)
    print(f"Fetching companyfacts CIK={cik}...")
    payload = client.fetch_companyfacts(cik, use_resume=False)
    if payload is None:
        print(f"Failed to fetch companyfacts for {ticker}")
        return 1
    path = client.facts_dir / f"CIK{cik}.json"
    nbytes = path.stat().st_size if path.exists() else 0

    annual: List[Dict[str, Any]] = []
    for concept in DEFAULT_CONCEPTS:
        series = extract_series(payload, concept, annual_only=True, limit=args.annual_limit)
        if series and series["rows"]:
            annual.append(series)

    quarterly_ni = None
    ni_node = ((payload.get("facts") or {}).get("us-gaap") or {}).get("NetIncomeLoss") or {}
    ni_usd = (ni_node.get("units") or {}).get("USD") or []
    q_only = dedupe_by_end(
        [x for x in ni_usd if x.get("fp") in ("Q1", "Q2", "Q3", "Q4")]
    )[: args.quarterly_limit]
    if q_only:
        quarterly_ni = {
            "concept": "NetIncomeLoss",
            "label": "Net Income (Loss)",
            "unit": "USD",
            "rows": q_only,
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_html = outdir / f"{ticker.lower()}_companyfacts.html"
    out_html.write_text(
        render_html(ticker, cik, payload, nbytes, annual, quarterly_ni),
        encoding="utf-8",
    )

    sidecar = {
        "ticker": ticker,
        "cik": cik,
        "entity": payload.get("entityName"),
        "bytes": nbytes,
        "annual": [
            {
                "concept": s["concept"],
                "label": s["label"],
                "unit": s["unit"],
                "rows": s["rows"],
            }
            for s in annual
        ],
    }
    out_json = outdir / f"{ticker.lower()}_companyfacts_summary.json"
    out_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"Wrote {out_html}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
