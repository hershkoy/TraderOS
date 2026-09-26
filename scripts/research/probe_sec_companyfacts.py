"""Probe SEC EDGAR companyfacts for a few tickers — size + field coverage.

No API key. Requires a descriptive User-Agent (SEC fair-access policy).
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

USER_AGENT = "backTraderTest research hershkoy@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS = ["AAPL", "MSFT", "LAUR"]
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}


def _get(url: str) -> Tuple[int, bytes, float]:
    t0 = time.perf_counter()
    resp = requests.get(url, headers=HEADERS, timeout=60)
    return resp.status_code, resp.content, time.perf_counter() - t0


def _load_ticker_map() -> Dict[str, str]:
    code, body, elapsed = _get(TICKERS_URL)
    print(f"company_tickers.json HTTP {code} bytes={len(body):,} time={elapsed:.2f}s")
    if code != 200:
        raise RuntimeError(f"ticker map failed: {body[:200]!r}")
    raw = json.loads(body)
    # shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for row in raw.values():
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker") or "").upper()
            cik = row.get("cik_str")
            if t and cik is not None:
                out[t] = f"{int(cik):010d}"
    return out


def _summarize_facts(payload: Dict[str, Any]) -> Dict[str, Any]:
    facts = payload.get("facts") or {}
    taxonomies = list(facts.keys()) if isinstance(facts, dict) else []
    concepts = 0
    sample_keys: List[str] = []
    interesting = (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "EarningsPerShareDiluted",
        "OperatingIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
    )
    found: Dict[str, int] = {}
    for tax, concepts_map in (facts.items() if isinstance(facts, dict) else []):
        if not isinstance(concepts_map, dict):
            continue
        concepts += len(concepts_map)
        for name, node in concepts_map.items():
            if name in interesting and name not in found:
                units = (node or {}).get("units") or {}
                n = 0
                for arr in units.values():
                    if isinstance(arr, list):
                        n += len(arr)
                found[name] = n
            if len(sample_keys) < 8:
                sample_keys.append(f"{tax}:{name}")
    return {
        "entity": payload.get("entityName"),
        "cik": payload.get("cik"),
        "taxonomies": taxonomies,
        "n_concepts": concepts,
        "interesting_obs": found,
        "sample_concepts": sample_keys,
    }


def main() -> int:
    print(f"User-Agent: {USER_AGENT}")
    print(f"Tickers: {TICKERS}")
    print("---")

    ticker_map = _load_ticker_map()
    print(f"ticker map size: {len(ticker_map):,}")
    time.sleep(0.15)

    total = 0
    rows: List[Dict[str, Any]] = []
    for t in TICKERS:
        cik = ticker_map.get(t)
        if not cik:
            print(f"[{t}] CIK not found in company_tickers.json")
            rows.append({"ticker": t, "error": "no_cik"})
            continue
        url = FACTS_TMPL.format(cik=cik)
        code, body, elapsed = _get(url)
        total += len(body)
        print(f"[{t}] CIK={cik} HTTP {code} bytes={len(body):,} time={elapsed:.2f}s")
        if code != 200:
            print(f"  preview: {body[:240]!r}")
            rows.append({"ticker": t, "cik": cik, "http": code, "bytes": len(body)})
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            print("  non-JSON body")
            continue
        summary = _summarize_facts(payload)
        print(f"  entity={summary['entity']!r} concepts={summary['n_concepts']}")
        print(f"  taxonomies={summary['taxonomies']}")
        print(f"  interesting obs counts: {summary['interesting_obs']}")
        rows.append(
            {
                "ticker": t,
                "cik": cik,
                "http": code,
                "bytes": len(body),
                **summary,
            }
        )
        time.sleep(0.15)  # stay well under 10 req/s

    ok = [r for r in rows if r.get("http") == 200]
    print("---")
    print(f"Probe total bytes: {total:,} ({total / (1024**2):.2f} MiB)")
    if ok:
        avg = sum(r["bytes"] for r in ok) / len(ok)
        print(f"Avg companyfacts size ({len(ok)}): {avg:,.0f} bytes ({avg/1024**2:.2f} MiB)")
        for n, label in ((2200, "daily panel"), (6070, "full universe")):
            est = n * avg
            print(
                f"  Extrapolate {label} @ 1 call/symbol: "
                f"{est/(1024**3):.2f} GiB download; "
                f"~{n/8:.0f}s wall @ ~8 req/s"
            )

    out_path = "logs/research/sec_companyfacts_probe.json"
    try:
        import os

        os.makedirs("logs/research", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"user_agent": USER_AGENT, "total_bytes": total, "rows": rows}, f, indent=2)
        print(f"Wrote {out_path}")
    except OSError as e:
        print(f"Could not write summary: {e}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
