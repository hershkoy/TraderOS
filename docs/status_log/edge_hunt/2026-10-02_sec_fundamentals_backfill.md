# SEC companyfacts fundamentals backfill (2026-10-02)

## Verdict

**Backfill pipeline is live.** Full SEC download + offline panel parse completed. TimescaleDB upsert stalled when Docker died mid-run; resume with the loader below once `backtrader_timescaledb` is healthy.

## Artifacts

| Piece | Path |
|-------|------|
| Schema | [`init-scripts/19-fundamentals-quarterly.sql`](../../init-scripts/19-fundamentals-quarterly.sql) |
| Apply | [`scripts/db/apply_fundamentals_schema.py`](../../../scripts/db/apply_fundamentals_schema.py) |
| SEC client | [`utils/api/sec/companyfacts_client.py`](../../../utils/api/sec/companyfacts_client.py) |
| Parser | [`utils/research/sec_fundamentals.py`](../../../utils/research/sec_fundamentals.py) |
| Backfill CLI | [`scripts/data/backfill_sec_fundamentals.py`](../../../scripts/data/backfill_sec_fundamentals.py) |
| Offline parquet parse | [`scripts/research/parse_sec_fundamentals_parquet.py`](../../../scripts/research/parse_sec_fundamentals_parquet.py) |
| Parquet -> DB loader | [`scripts/research/load_sec_fundamentals_parquet.py`](../../../scripts/research/load_sec_fundamentals_parquet.py) |
| Auto-resume when Docker up | [`scripts/research/_resume_sec_parse_when_docker.py`](../../../scripts/research/_resume_sec_parse_when_docker.py) |
| Tests | [`tests/unit/test_sec_fundamentals.py`](../../../tests/unit/test_sec_fundamentals.py) (14 passed) |
| Report uses shared client | [`scripts/research/report_sec_companyfacts.py`](../../../scripts/research/report_sec_companyfacts.py) |
| Panel parquet | `reports/research/sec_fundamentals/fundamentals_quarterly_20261002.parquet` |
| Raw SEC cache | `data/sec/companyfacts/`, `data/sec/submissions/` (gitignored) |

## Timings / coverage

### Pilot (20 names)

- Wall ~188s; upserted **1426** rows into `fundamentals_quarterly`
- SIC excludes fired: JPM, XOM, NEE, AMT, UNH

### Full download (`--universe daily --workers 4`)

- Daily `1d` symbols: 2224; with SEC CIK: **2090** (134 missing)
- Download: **2086 OK / 4 HTTP 404**
- Wall ~30 min (4 workers, 0.1s global throttle)

### Offline parquet parse (no DB)

- Symbols with cache mapped to ticker: **2017** -> **1981** unique in panel
- Rows: **113,168** (~82k quarterly)
- Wall **~63s**
- Quarterly non-null rates: OCF **90%**, CapEx **80%**, Assets **82%**, shares **61%**, buybacks TTM **65%**, total debt **49%**
- Symbols with any OCF: **1838/2017**; SIC-excluded symbols: **642**
- OCF tag almost always `NetCashProvidedByUsedInOperatingActivities`
- OCF coverage by year (quarterly rows, 2018+): ~93–99%

### TimescaleDB upsert (2026-10-03 resume)

- Docker back up; loaded parquet with `--skip-mkt-cap` in **~21s** → ~115k rows
- **mkt_cap fill** via [`scripts/research/fill_fundamentals_mkt_cap.py`](../../../scripts/research/fill_fundamentals_mkt_cap.py) (batched ALPACA 1d closes): **~25k rows updated** in ~6 min; remaining gaps are no-price / no-shares names

```bat
venv\Scripts\activate
python scripts/research/fill_fundamentals_mkt_cap.py --batch-symbols 100
```

## Filter field mapping (implemented)

| Filter | Compute |
|--------|---------|
| 1 | TTM OCF / composed debt; debt=0 + OCF>0 passes |
| 2 | TTM CapEx (abs) / TTM OCF |
| 3 | (TTM buybacks - issuance) / (shares * close as-of `filed`) — needs DB prices |
| 4 | (OCF - CapEx) / Assets |
| Exclude | SIC: energy / materials / utilities / financials / real estate |

Strategy as-of date = **`filed`** (not `period_end`).

## Notes / caveats

- Debt composition covers ~half of quarterly rows — many non-financials report debt under uncommon tags; extend tag list as needed
- Shares often under `dei:EntityCommonStockSharesOutstanding` (parser checks `us-gaap` + `dei`)
- A few XBRL period ends land in 2027–2030 (junk ends); filter `period_end <= today` in claim-check
- Follow-on claim-check harness still TODO (`cashflow_four_filters`)
