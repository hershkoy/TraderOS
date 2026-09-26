"""Small FinFeedAPI SEC probe: measure filings + XBRL sizes for a few tickers.

Uses FINFEEDAPI_API_KEY from .env. Does not print the key.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.config.env_loader import get_env_var

BASE = "https://api.sec.finfeedapi.com"
TICKERS = ["AAPL", "MSFT", "LAUR"]  # mega-cap + one mid name from our research book
FORMS = "10-K,10-Q"
MAX_XBRL_PER_TICKER = 2  # latest few only — stay under free credits


def _auth_headers(api_key: str) -> Dict[str, str]:
    # Docs: Authorization header is the raw API key (not Bearer).
    return {
        "Accept": "application/json",
        "Authorization": api_key,
        "X-APIKey": api_key,
    }


def _get(
    path: str,
    api_key: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[int, bytes, float, Dict[str, str]]:
    url = f"{BASE}{path}"
    q = dict(params or {})
    # FinFeedAPI / CoinAPI style often accepts apikey as query too
    q.setdefault("apikey", api_key)
    t0 = time.perf_counter()
    resp = requests.get(url, headers=_auth_headers(api_key), params=q, timeout=60)
    elapsed = time.perf_counter() - t0
    return resp.status_code, resp.content, elapsed, dict(resp.headers)


def _pick_accession(payload: Any) -> List[str]:
    """Best-effort extract accession numbers from filings response."""
    items: List[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("data", "filings", "result", "items", "results"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        if not items and "accession_number" in payload:
            items = [payload]

    out: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in (
            "accession_number",
            "accessionNumber",
            "accession_no",
            "accession-no",
            "AccessionNumber",
        ):
            v = it.get(k)
            if v:
                out.append(str(v))
                break
    return out


def main() -> int:
    api_key = (get_env_var("FINFEEDAPI_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: FINFEEDAPI_API_KEY not set in environment / .env")
        return 1

    print(f"Key loaded: yes (len={len(api_key)})")
    print(f"Base: {BASE}")
    print(f"Tickers: {TICKERS}")
    print("---")

    total_bytes = 0
    rows: List[Dict[str, Any]] = []

    for ticker in TICKERS:
        code, body, elapsed, headers = _get(
            "/v1/filings",
            api_key,
            {
                "ticker": ticker,
                "form_type": FORMS,
                "page_size": 10,
                "page_number": 1,
                "sort_by": "FilingDate",
                "sort_order": "desc",
            },
        )
        total_bytes += len(body)
        preview = body[:400].decode("utf-8", errors="replace")
        print(f"[{ticker}] filings HTTP {code} bytes={len(body)} time={elapsed:.2f}s")
        if code != 200:
            print(f"  body preview: {preview}")
            rows.append(
                {
                    "ticker": ticker,
                    "step": "filings",
                    "http": code,
                    "bytes": len(body),
                    "error": preview[:200],
                }
            )
            continue

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            print(f"  non-JSON filings response: {preview}")
            continue

        accessions = _pick_accession(payload)
        print(f"  accessions found: {len(accessions)} sample={accessions[:3]}")
        rows.append(
            {
                "ticker": ticker,
                "step": "filings",
                "http": code,
                "bytes": len(body),
                "n_accessions": len(accessions),
            }
        )

        for acc in accessions[:MAX_XBRL_PER_TICKER]:
            # Try both param spellings used in docs
            for param_name in ("accession-no", "accession_number", "accession_no"):
                code2, body2, elapsed2, _ = _get(
                    "/v1/xbrl-converter",
                    api_key,
                    {param_name: acc},
                )
                if code2 == 200 and len(body2) > 100:
                    total_bytes += len(body2)
                    print(
                        f"  xbrl [{param_name}] {acc} HTTP {code2} "
                        f"bytes={len(body2)} time={elapsed2:.2f}s"
                    )
                    rows.append(
                        {
                            "ticker": ticker,
                            "step": "xbrl",
                            "accession": acc,
                            "param": param_name,
                            "http": code2,
                            "bytes": len(body2),
                        }
                    )
                    break
                if code2 != 200:
                    err = body2[:180].decode("utf-8", errors="replace")
                    print(
                        f"  xbrl [{param_name}] {acc} HTTP {code2} "
                        f"bytes={len(body2)} err={err}"
                    )
            time.sleep(0.3)

    gib = total_bytes / (1024**3)
    payg_cost = gib * 10.0
    print("---")
    print(f"Total downloaded this probe: {total_bytes:,} bytes ({gib:.6f} GiB)")
    print(f"Implied PAYG cost @ $10/GiB: ${payg_cost:.4f}")
    if rows:
        xbrl = [r for r in rows if r.get("step") == "xbrl" and r.get("http") == 200]
        if xbrl:
            avg = sum(r["bytes"] for r in xbrl) / len(xbrl)
            print(f"XBRL avg size ({len(xbrl)} ok): {avg:,.0f} bytes")
            for n_sym, n_filings, label in (
                (2200, 1, "daily panel latest 10-K only"),
                (2200, 5, "daily panel 10-K + 4x 10-Q"),
                (6070, 1, "full universe latest 10-K"),
                (6070, 5, "full universe 5 filings"),
            ):
                est = n_sym * n_filings * avg
                print(
                    f"  Extrapolate {label}: "
                    f"{est/(1024**3):.2f} GiB -> ~${est/(1024**3)*10:.0f}"
                )

    # Dump machine-readable summary for later
    out_path = "logs/research/finfeedapi_sec_probe.json"
    try:
        import os

        os.makedirs("logs/research", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_bytes": total_bytes, "rows": rows},
                f,
                indent=2,
            )
        print(f"Wrote {out_path}")
    except OSError as e:
        print(f"Could not write summary: {e}")

    return 0 if any(r.get("http") == 200 for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
