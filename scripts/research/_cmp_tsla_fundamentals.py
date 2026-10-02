"""Compare Tesla fundamentals: SEC EDGAR companyfacts vs TradingView MCP TTM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

USER_AGENT = "backTraderTest research hershkoy@gmail.com"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
SEC_SUMMARY = Path("reports/research/sec_companyfacts/tsla_companyfacts_summary.json")

# TradingView mcp-tv-get-financials from this session (live retry was 429).
TV_TTM = {
    "close": 373.38,
    "market_cap_basic": 1_474_682_024_655.65,
    "total_revenue_ttm": 103_619_000_000,
    "gross_profit_ttm": 19_534_000_000,
    "oper_income_ttm": 4_772_000_000,
    "net_income_ttm": 3_806_000_000,
    "earnings_per_share_diluted_ttm": 1.0763,
    "earnings_per_share_basic_ttm": 1.1777,
    "free_cash_flow_ttm": 5_762_000_000,
    "ebitda_ttm": 11_249_000_000,
    "price_earnings_ttm": 346.9107126265911,
    "gross_margin": 18.8517549870198,
    "operating_margin": 4.60533299877436,
    "net_margin": 3.6730715409336105,
}


def _fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    ax = abs(float(x))
    if ax >= 1e9:
        return f"${float(x) / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"${float(x) / 1e6:,.1f}M"
    return f"${float(x):,.2f}"


def _dedupe_by_end(rows: List[dict]) -> List[dict]:
    by_end: Dict[str, dict] = {}
    for x in rows:
        end = x.get("end")
        if not end:
            continue
        prev = by_end.get(end)
        if prev is None or str(x.get("filed") or "") >= str(prev.get("filed") or ""):
            by_end[end] = x
    return sorted(by_end.values(), key=lambda x: x.get("end") or "", reverse=True)


def _latest_fy(summary: dict, concept: str) -> Tuple[Optional[str], Optional[float]]:
    for s in summary.get("annual") or []:
        if s.get("concept") != concept:
            continue
        rows = s.get("rows") or []
        if not rows:
            return None, None
        return rows[0].get("end"), float(rows[0]["val"])
    return None, None


def _quarterly_series(facts: dict, concept: str, unit: str = "USD") -> List[dict]:
    node = ((facts.get("facts") or {}).get("us-gaap") or {}).get(concept) or {}
    rows = list((node.get("units") or {}).get(unit) or [])
    q = [x for x in rows if x.get("fp") in ("Q1", "Q2", "Q3", "Q4")]
    return _dedupe_by_end(q)


def _ttm_sum(rows: List[dict], n: int = 4) -> Tuple[Optional[float], List[str]]:
    take = rows[:n]
    if len(take) < n:
        return None, [r.get("end") or "" for r in take]
    return sum(float(r["val"]) for r in take), [r.get("end") or "" for r in take]


def main() -> int:
    summary = json.loads(SEC_SUMMARY.read_text(encoding="utf-8"))
    print(f"SEC: {summary.get('entity')} CIK={summary.get('cik')}")
    print("TV:  NASDAQ:TSLA via mcp-tv-get-financials (session snapshot; live retry 429)")
    print()

    print("Fetching live companyfacts for trailing-quarter TTM...")
    resp = requests.get(FACTS_URL, headers=HEADERS, timeout=90)
    resp.raise_for_status()
    facts = resp.json()

    rev_concept = "RevenueFromContractWithCustomerExcludingAssessedTax"
    pairs = [
        ("Revenue", rev_concept, "total_revenue_ttm", "USD"),
        ("Gross profit", "GrossProfit", "gross_profit_ttm", "USD"),
        ("Op. income", "OperatingIncomeLoss", "oper_income_ttm", "USD"),
        ("Net income", "NetIncomeLoss", "net_income_ttm", "USD"),
        ("EPS diluted", "EarningsPerShareDiluted", "earnings_per_share_diluted_ttm", "USD/shares"),
    ]

    print()
    print(
        f"{'Metric':<12} {'SEC FY2025':>14} {'SEC TTM(4Q)':>14} {'TV TTM':>14} "
        f"{'FY vs TV':>9} {'4Q vs TV':>9}"
    )
    print("-" * 78)

    for label, concept, tv_key, unit in pairs:
        fy_end, fy_val = _latest_fy(summary, concept)
        if fy_val is None and concept == rev_concept:
            fy_end, fy_val = _latest_fy(summary, "Revenues")

        qrows = _quarterly_series(facts, concept, unit=unit)
        ttm_val, ttm_ends = _ttm_sum(qrows, 4)
        tv_val = float(TV_TTM[tv_key])

        if unit == "USD/shares":
            fy_s = f"{fy_val:.4f}" if fy_val is not None else "n/a"
            ttm_s = f"{ttm_val:.4f}" if ttm_val is not None else "n/a"
            tv_s = f"{tv_val:.4f}"
        else:
            fy_s = _fmt_usd(fy_val)
            ttm_s = _fmt_usd(ttm_val)
            tv_s = _fmt_usd(tv_val)

        d_fy = (
            f"{(tv_val - fy_val) / abs(fy_val) * 100:+.1f}%"
            if fy_val not in (None, 0)
            else "n/a"
        )
        d_ttm = (
            f"{(tv_val - ttm_val) / abs(ttm_val) * 100:+.1f}%"
            if ttm_val not in (None, 0)
            else "n/a"
        )
        print(f"{label:<12} {fy_s:>14} {ttm_s:>14} {tv_s:>14} {d_fy:>9} {d_ttm:>9}")
        if ttm_ends:
            print(f"{'':12} SEC 4Q ends: {', '.join(ttm_ends)}")

    # Sanity: TV margins vs TV income statement
    print()
    print("TV-derived checks:")
    rev = TV_TTM["total_revenue_ttm"]
    print(
        f"  gross_margin reported {TV_TTM['gross_margin']:.2f}% vs "
        f"gross/rev {TV_TTM['gross_profit_ttm']/rev*100:.2f}%"
    )
    print(
        f"  op_margin reported {TV_TTM['operating_margin']:.2f}% vs "
        f"op/rev {TV_TTM['oper_income_ttm']/rev*100:.2f}%"
    )
    print(
        f"  net_margin reported {TV_TTM['net_margin']:.2f}% vs "
        f"ni/rev {TV_TTM['net_income_ttm']/rev*100:.2f}%"
    )
    print(
        f"  P/E implied close/EPS {TV_TTM['close']/TV_TTM['earnings_per_share_diluted_ttm']:.1f} "
        f"vs reported {TV_TTM['price_earnings_ttm']:.1f}"
    )
    print()
    print("Notes:")
    print("  - SEC FY = calendar FY from latest 10-K (deduped by period-end).")
    print("  - SEC TTM = sum of last 4 distinct quarter-end XBRL points.")
    print("  - TV TTM = vendor-normalized trailing twelve months (can differ on tags/adj).")
    print("  - EPS TTM should NOT usually equal sum of quarterly EPS (share count changes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
