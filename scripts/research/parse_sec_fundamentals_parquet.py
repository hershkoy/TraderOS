"""Offline parse of cached SEC companyfacts -> parquet (no TimescaleDB).

Skips mkt_cap join. Use when Docker/DB is down; later re-run backfill
--parse-only to upsert into fundamentals_quarterly with mkt_cap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.api.sec.companyfacts_client import CompanyfactsClient, DEFAULT_CACHE_ROOT
from utils.research.sec_fundamentals import build_fundamental_rows, passes_four_filters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--outdir", default="reports/research/sec_fundamentals")
    ap.add_argument("--max-symbols", type=int, default=0)
    args = ap.parse_args()

    sec = CompanyfactsClient(cache_root=args.cache_root, sleep_s=0)
    ticker_map = sec.load_ticker_map()
    # Invert: prefer symbols that have cached facts
    facts_dir = Path(args.cache_root) / "companyfacts"
    ciks = sorted(p.stem.replace("CIK", "") for p in facts_dir.glob("CIK*.json"))
    cik_to_ticker = {v: k for k, v in ticker_map.items()}

    pairs = []
    for cik in ciks:
        sym = cik_to_ticker.get(cik)
        if not sym:
            # try int form
            try:
                sym = cik_to_ticker.get(f"{int(cik):010d}")
            except ValueError:
                sym = None
        if not sym:
            continue
        pairs.append((sym, f"{int(cik):010d}"))
    if args.max_symbols:
        pairs = pairs[: args.max_symbols]

    print(f"Parsing {len(pairs)} cached companyfacts -> parquet", flush=True)
    t0 = time.perf_counter()
    all_rows = []
    metas = []
    for i, (sym, cik) in enumerate(pairs, 1):
        facts = sec.load_cached_companyfacts(cik)
        if not facts:
            metas.append({"symbol": sym, "cik": cik, "error": "no_facts"})
            continue
        subs = sec.load_cached_submissions(cik)
        sic, sic_desc = sec.sic_from_submissions(subs)
        rows = build_fundamental_rows(sym, cik, facts, sic=sic, sic_desc=sic_desc)
        n_pass = 0
        for r in rows:
            d = r.to_db_dict()
            # drop inf for parquet
            for k, v in list(d.items()):
                if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                    d[k] = None
            all_rows.append(d)
            if not r.exclude_sector:
                flags = passes_four_filters(r)
                # f3 needs mkt_cap — will be None offline
                if all(
                    flags.get(k) is True
                    for k in ("f1_ocf_debt", "f2_capex_ocf", "f4_fcf_roa")
                ):
                    n_pass += 1
        metas.append(
            {
                "symbol": sym,
                "cik": cik,
                "sic": sic,
                "exclude_sector": bool(rows and rows[0].exclude_sector),
                "n_rows": len(rows),
                "n_ocf": sum(1 for r in rows if r.ocf is not None),
                "n_pass_f124": n_pass,
                "tag_ocf": rows[0].tag_ocf if rows else None,
            }
        )
        if i % 100 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} rows_so_far={len(all_rows)}", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    pq = outdir / f"fundamentals_quarterly_{stamp}.parquet"
    summary_path = outdir / f"fundamentals_quarterly_{stamp}_summary.json"

    df = pd.DataFrame(all_rows)
    df.to_parquet(pq, index=False)

    with_ocf = sum(1 for m in metas if (m.get("n_ocf") or 0) > 0)
    excluded = sum(1 for m in metas if m.get("exclude_sector"))
    summary = {
        "n_symbols": len(pairs),
        "n_rows": len(all_rows),
        "symbols_with_ocf": with_ocf,
        "symbols_exclude_sector": excluded,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "parquet": str(pq),
        "note": "mkt_cap null offline; re-run scripts/data/backfill_sec_fundamentals.py --parse-only when DB up",
        "per_symbol": metas,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {pq} rows={len(df)}")
    print(f"Wrote {summary_path}")
    print(f"with_ocf={with_ocf}/{len(pairs)} excluded={excluded} elapsed={summary['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
