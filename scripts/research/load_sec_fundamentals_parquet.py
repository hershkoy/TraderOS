"""Load fundamentals parquet into TimescaleDB (and optionally attach mkt_cap).

Run after Docker/TimescaleDB is up::

    venv\\Scripts\\activate && python scripts/research/load_sec_fundamentals_parquet.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

# Reuse upsert from backfill
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from backfill_sec_fundamentals import (  # type: ignore
    load_closes,
    row_to_params,
    upsert_rows,
)
from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.sec_fundamentals import FundamentalRow, attach_mkt_caps, compute_ratios


def df_to_rows(df: pd.DataFrame) -> list:
    rows = []
    for rec in df.to_dict(orient="records"):
        # rebuild FundamentalRow for mkt_cap attach
        row = FundamentalRow(
            symbol=str(rec["symbol"]),
            cik=str(rec["cik"]),
            period_end=pd.Timestamp(rec["period_end"]).date(),
            filed=pd.Timestamp(rec["filed"]).date(),
            fy=int(rec["fy"]) if pd.notna(rec.get("fy")) else None,
            fp=str(rec["fp"]),
            ocf=float(rec["ocf"]) if pd.notna(rec.get("ocf")) else None,
            capex=float(rec["capex"]) if pd.notna(rec.get("capex")) else None,
            fcf=float(rec["fcf"]) if pd.notna(rec.get("fcf")) else None,
            total_debt=float(rec["total_debt"]) if pd.notna(rec.get("total_debt")) else None,
            assets=float(rec["assets"]) if pd.notna(rec.get("assets")) else None,
            shares_out=float(rec["shares_out"]) if pd.notna(rec.get("shares_out")) else None,
            buybacks_ttm=float(rec["buybacks_ttm"]) if pd.notna(rec.get("buybacks_ttm")) else None,
            issuance_ttm=float(rec["issuance_ttm"]) if pd.notna(rec.get("issuance_ttm")) else None,
            net_buybacks_ttm=float(rec["net_buybacks_ttm"]) if pd.notna(rec.get("net_buybacks_ttm")) else None,
            mkt_cap=float(rec["mkt_cap"]) if pd.notna(rec.get("mkt_cap")) else None,
            ocf_to_debt=float(rec["ocf_to_debt"]) if pd.notna(rec.get("ocf_to_debt")) else None,
            capex_to_ocf=float(rec["capex_to_ocf"]) if pd.notna(rec.get("capex_to_ocf")) else None,
            net_buyback_pct=float(rec["net_buyback_pct"]) if pd.notna(rec.get("net_buyback_pct")) else None,
            fcf_roa=float(rec["fcf_roa"]) if pd.notna(rec.get("fcf_roa")) else None,
            sic=str(rec["sic"]) if pd.notna(rec.get("sic")) else None,
            sic_desc=str(rec["sic_desc"]) if pd.notna(rec.get("sic_desc")) else None,
            exclude_sector=bool(rec.get("exclude_sector")),
            debt_zero=bool(rec.get("debt_zero")),
            tag_ocf=str(rec["tag_ocf"]) if pd.notna(rec.get("tag_ocf")) else None,
            tag_capex=str(rec["tag_capex"]) if pd.notna(rec.get("tag_capex")) else None,
            tag_debt=str(rec["tag_debt"]) if pd.notna(rec.get("tag_debt")) else None,
        )
        compute_ratios(row)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        default="reports/research/sec_fundamentals/fundamentals_quarterly_20261002.parquet",
    )
    ap.add_argument("--skip-mkt-cap", action="store_true")
    ap.add_argument("--batch-symbols", type=int, default=50)
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        print(f"Missing {path}")
        return 1

    db = get_timescaledb_client()
    if not db.connect():
        print("Cannot connect to TimescaleDB")
        return 2

    df = pd.read_parquet(path)
    symbols = sorted(df["symbol"].unique())
    print(f"Loading {len(df)} rows / {len(symbols)} symbols from {path}")

    t0 = time.perf_counter()
    n_up = 0
    for i in range(0, len(symbols), args.batch_symbols):
        batch_syms = symbols[i : i + args.batch_symbols]
        part = df[df["symbol"].isin(batch_syms)]
        rows = df_to_rows(part)
        if not args.skip_mkt_cap:
            for sym in batch_syms:
                closes = load_closes(db, sym)
                attach_mkt_caps([r for r in rows if r.symbol == sym], closes)
        n_up += upsert_rows(db, rows)
        print(
            f"  upserted through {batch_syms[-1]} "
            f"({min(i + args.batch_symbols, len(symbols))}/{len(symbols)}) "
            f"rows={n_up}"
        )

    print(f"Done rows={n_up} elapsed={time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
