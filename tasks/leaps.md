Following is a detail explanation on how to implement a new strategy.

This LEAPS strategy focuses on building long-term exposure to major ETFs like QQQ by buying deep-in-the-money call options with expirations 1–2 years out. The approach reduces time-decay risk, allows for leveraged upside participation, and provides a more capital-efficient alternative to holding the underlying shares.

1. **Create TimescaleDB tables for options (`option_contracts`, `option_quotes`)** ✅

   * [x] Write a SQL migration file `options_schema.sql` that creates:

     * [x] `option_contracts(option_id TEXT PK, underlying TEXT NOT NULL, expiration DATE NOT NULL, strike_cents INT NOT NULL, option_right CHAR(1) NOT NULL, multiplier INT NOT NULL DEFAULT 100, first_seen TIMESTAMPTZ, last_seen TIMESTAMPTZ)`.
     * [x] `option_quotes(ts TIMESTAMPTZ NOT NULL, option_id TEXT NOT NULL, bid NUMERIC(18,6), ask NUMERIC(18,6), last NUMERIC(18,6), volume BIGINT, open_interest BIGINT, iv NUMERIC(12,6), delta NUMERIC(12,6), gamma NUMERIC(12,6), theta NUMERIC(12,6), vega NUMERIC(12,6), snapshot_type TEXT DEFAULT 'eod', PRIMARY KEY (option_id, ts))`.
     * [x] Hypertable: `SELECT create_hypertable('option_quotes','ts', chunk_time_interval => INTERVAL '7 days');`
     * [x] Indexes: `CREATE INDEX ON option_contracts (underlying, expiration);` and `CREATE INDEX ON option_quotes (option_id, ts DESC);`

2. **Define deterministic `option_id` builder function** ✅

   * [x] Implement a pure function `build_option_id(underlying:str, expiration:date, strike:float, option_right:str) -> str` that returns e.g. `QQQ_2025-06-20_000350C`.
     * [x] Convert `strike` to `strike_cents = int(round(strike * 100))` and zero-pad to 6 digits in the ID portion.

3. **Add environment config for Polygon** ✅

   * [x] Extend `.env` with `POLYGON_API_KEY=...` (documented in POLYGON_SETUP.md).
   * [x] In a new `utils/polygon_client.py`, load the key and expose helpers `get_json(url, params)` with retry/backoff and rate-limit sleep (free plan \~5 req/min).

4. **Discover QQQ option contracts (daily)** ✅

   * [x] Implement `scripts/polygon_discover_contracts.py` that:

     * [x] Iterates near expirations for QQQ (e.g., next 365 days window) via Polygon options reference/snapshot endpoints.
     * [x] Extracts fields: `underlying='QQQ'`, `expiration (YYYY-MM-DD)`, `strike (float)`, `option_right ('C'/'P')`, `multiplier (int or default 100)`.
     * [x] Builds `option_id` and upserts into `option_contracts`, updating `first_seen` (if null) and `last_seen=now()`.

5. **Backfill historical contract list for the last 2 years** ✅

   * [x] For each calendar week in the past 730 days, pull the QQQ chain (pacing-safe).
   * [x] Upsert contracts, ensuring no duplicates by `option_id`.

6. **Ingest daily EOD quotes for discovered contracts** ✅

   * [x] Implement `scripts/polygon_ingest_eod_quotes.py` that:

     * [x] For each `option_id` active on a given date, fetch EOD (daily close) prices.
     * [x] Map to columns: `ts = <EOD timestamp UTC>`, `bid`, `ask`, `last`, `volume`, `open_interest (nullable)`, Greeks `iv/delta/gamma/theta/vega (nullable if not returned)`, `snapshot_type='eod'`.
     * [x] Batch UPSERT into `option_quotes` using COPY for performance.

7. **Write COPY-based bulk loader** ✅

   * [x] Create `utils/pg_copy.py` with a `copy_rows(conn, table, columns, rows_iter)` that streams CSV to `COPY ... FROM STDIN`.
   * [x] Use in both `discover_contracts` (for initial inserts) and `ingest_eod_quotes`.

8. **Implement idempotent UPSERT SQL helpers** ✅

   * [x] `upsert_option_contracts(rows)` using `ON CONFLICT (option_id) DO UPDATE SET last_seen=EXCLUDED.last_seen`.
   * [x] `upsert_option_quotes(rows)` using `ON CONFLICT (option_id, ts) DO UPDATE SET bid=EXCLUDED.bid, ask=EXCLUDED.ask, last=EXCLUDED.last, volume=EXCLUDED.volume, open_interest=EXCLUDED.open_interest, iv=EXCLUDED.iv, delta=EXCLUDED.delta, gamma=EXCLUDED.gamma, theta=EXCLUDED.theta, vega=EXCLUDED.vega, snapshot_type=EXCLUDED.snapshot_type`.

9. **Create helper view: “current chain at close”**

   * [x] SQL view `option_chain_eod AS SELECT q.ts, c.underlying, c.expiration, c.strike_cents, c.option_right, q.bid, q.ask, q.last, q.volume, q.open_interest, q.iv, q.delta, q.gamma, q.theta, q.vega, c.option_id FROM option_quotes q JOIN option_contracts c USING (option_id);`
   * [x] Add index on `option_quotes(option_id, ts DESC)` (already in Task 1) for fast latest-by-ts queries.

10. **Continuous aggregate for intraday (future-proofing)** ✅

* [x] Create materialized view `cq_option_eod` (continuous) that buckets `option_quotes` to 1 day and aggregates `max(open_interest)` and `sum(volume)`, carrying last `bid/ask/last`.
* [x] Refresh policy daily at 22:00 UTC.

11. **Retention policies** ✅

* [x] Add TimescaleDB retention policy to keep `option_quotes` raw (if you later add intraday) for 120 days, keep daily EOD indefinitely.
* [x] Use `add_retention_policy('option_quotes', INTERVAL '120 days')` guarded behind a feature flag.

12. **Join helpers: option ↔ underlying** ✅

* [x] Add SQL function `get_underlying_close(symbol TEXT, d TIMESTAMPTZ) RETURNS NUMERIC` that pulls `close` from existing `market_data` at daily resolution (nearest prior bar).
* [x] Create view `option_chain_with_underlying` joining `option_chain_eod` to `market_data` (`symbol=underlying`, same day) with columns: `underlying_close`, `moneyness = (underlying_close / (strike_cents/100.0))`.

13. **Delta fallback (if Greeks absent)** ✅

* [x] Implement a Python utility `utils/greeks.py` with Black–Scholes delta/iv solver:

  * [x] Inputs: `S=underlying_close`, `K=strike`, `T=days_to_exp/365`, `r=0.00`, `q=dividend_yield (0 for QQQ unless you feed it)`, `option mid = (bid+ask)/2`.
  * [x] If `iv` missing, solve IV; compute `delta/gamma/theta/vega`; update `option_quotes` for that `ts, option_id`.

14. **Daily scheduler** ✅

* [x] Add a `cron`/Taskfile entry to run:

  * [x] `polygon_discover_contracts.py` (after market close).
  * [x] `polygon_ingest_eod_quotes.py` (after market settle).
  * [x] Optional: `greeks_fill.py` to backfill missing Greeks for the last N days.

15. **Backtest selector SQL for PMCC/LEAPS** ✅

* [x] SQL view `pmcc_candidates AS`:

  * [x] For each `ts` (trading day):

    * [x] **LEAPS**: `expiration - date(ts) >= 365` and `right='C'` and `(delta BETWEEN 0.6 AND 0.85 OR moneyness BETWEEN 0.9 AND 1.1 if delta null)`.
    * [x] **Short Call**: `25 <= (expiration - date(ts)) <= 45` and `right='C'` and `(delta BETWEEN 0.15 AND 0.35 OR moneyness BETWEEN 1.02 AND 1.08 if delta null)`.
  * [x] Include: `option_id`, `expiration`, `strike_cents`, `delta`, `moneyness`, `bid/ask/last`.

16. **API module to fetch chain slices for a given date** ✅

* [x] Implement `data/options_repo.py` with:

  * [x] `get_chain_at(ts: datetime, underlying='QQQ') -> pd.DataFrame` using `option_chain_with_underlying`.
  * [x] `select_leaps(ts, delta_band=(0.6,0.85))` and `select_short_calls(ts, dte_band=(25,45), delta_band=(0.15,0.35))`.

17. **Assignment risk helper** ✅

* [x] Implement `utils/assignment.py`:

  * [x] Function `should_flag_assignment(short_delta, moneyness, days_to_exp, ex_div_calendar)` returning bool.
  * [x] For QQQ, accept an empty dividend calendar initially; return `True` if ITM and DTE ≤ 3.

18. **ETL tests (unit)** ✅

* [x] Create pytest cases:

  * [x] `test_build_option_id()` with multiple strikes/rights.
  * [ ] `test_contract_upsert_idempotent()` ensures repeat loads don’t duplicate rows.
  * [ ] `test_quotes_upsert()` checks PK conflict handling and updates.

19. **Data integrity checks** ✅

* [x] Daily job `scripts/options_data_checks.py`:

  * [x] Verify `count(distinct option_id)` growth reasonable vs yesterday (±20%).
  * [x] Spot-check random `option_id` for monotonic `last_seen`.
  * [x] Ensure `option_quotes` has the same (or within tolerance) number of rows as active contracts for the day.
  * [x] Unit tests for integrity checker

20. **Backtest integration glue** ✅

* [x] Implement a provider in your strategy code that:

  * [x] Calls `options_repo.select_leaps()` and `select_short_calls()` for each trading day.
  * [x] Prices fills at `mid = (bid+ask)/2` with a configurable haircut `%spread`.
  * [x] Emits a struct with the exact legs: `{'date': ts, 'leaps': {...}, 'short_call': {...}}`.

21. **Documentation (README additions)**

* [ ] Add `docs/options_pipeline.md` with:

  * [ ] Table schemas + indexes.
  * [ ] Command snippets to run discover/ingest.
  * [ ] Notes on rate-limits on Polygon free vs \$29 plan.
  * [ ] Backtest selector examples (delta vs moneyness fallback).

22. **Optional: retention & compression tuning**

* [ ] Enable TimescaleDB compression on `option_quotes` older than 90 days:

  * [ ] `SELECT add_compression_policy('option_quotes', INTERVAL '90 days');`
  * [ ] `ALTER TABLE option_quotes SET (timescaledb.compress, timescaledb.compress_segmentby = 'option_id');`

23. **Optional: performance index for selector**

* [ ] Add partial index on `option_contracts` for calls only:

  * [ ] `CREATE INDEX ON option_contracts (expiration, strike_cents) WHERE right='C' AND underlying='QQQ';`

24. **Optional: sanity stats dashboard**

* [ ] Create a simple SQL view `options_summary_daily` with per-day counts (`contracts`, `quotes`, avg bid-ask spread, median OI) for monitoring and plotting.

25. **Dry-run script for 2 years**

* [ ] Write `scripts/run_polygon_pipeline_2y.py` that:

  * [ ] Iterates dates from `today - 730d` to `today`, but only executes a **sample** (e.g., every 5th trading day) to respect the free plan.
  * [ ] Logs totals of contracts discovered and quotes inserted.



 don’t put options into your current `market_data` table. Make a **separate hypertable (or two)** for options and link them to the underlying by a clean key. It keeps things fast, sane, and avoids polluting your OHLCV with contract-level data.

Here’s a battle-tested layout that works well with Polygon (free or \$29 plan):

# 1) Core tables

```sql
-- One row per option contract (static metadata)
CREATE TABLE option_contracts (
  option_id       TEXT PRIMARY KEY,              -- e.g. OCC-style: QQQ_2025-06-20_000350C
  underlying      TEXT NOT NULL,                 -- 'QQQ'
  expiration      DATE NOT NULL,
  strike_cents    INT  NOT NULL,                 -- store as int to avoid fp rounding
  right           CHAR(1) NOT NULL,              -- 'C' or 'P'
  multiplier      INT  NOT NULL DEFAULT 100,
  first_seen      TIMESTAMPTZ,
  last_seen       TIMESTAMPTZ
);

-- Quotes/“EOD”/mid/greeks across time (hypertable)
CREATE TABLE option_quotes (
  ts              TIMESTAMPTZ NOT NULL,
  option_id       TEXT NOT NULL,
  bid             NUMERIC(18,6),
  ask             NUMERIC(18,6),
  last            NUMERIC(18,6),
  volume          BIGINT,
  open_interest   BIGINT,
  -- optional Greeks if you have them (Polygon paid add-ons or your own calc)
  iv              NUMERIC(12,6),
  delta           NUMERIC(12,6),
  gamma           NUMERIC(12,6),
  theta           NUMERIC(12,6),
  vega            NUMERIC(12,6),
  -- snapshot type helps when mixing “EOD close” vs “15:45” etc.
  snapshot_type   TEXT DEFAULT 'eod',            -- 'eod','1545','intraday_agg'...
  PRIMARY KEY (option_id, ts)
);
SELECT create_hypertable('option_quotes','ts', chunk_time_interval => INTERVAL '7 days');

-- Underlying link (optional, but handy for joins)
CREATE INDEX ON option_contracts (underlying, expiration);
CREATE INDEX ON option_quotes (option_id, ts DESC);
```

Why separate?

* **Cardinality/explosiveness:** one underlying → thousands of contracts/day. Keeping this out of `market_data` prevents bloating queries that expect one row/day/symbol.
* **Different semantics:** options have bid/ask/IV/Greeks/open interest, while `market_data` is OHLCV for the underlying. Different columns, different retention.
* **Performance:** typical backtests filter by expiration/strike/right or by `option_id` and time. The composite PK + hypertable chunks will be much faster than shoehorning into the stock table.

# 2) How to build `option_id`

Pick a deterministic key so you never double-insert:

```
option_id = UNDERLYING + '_' + YYYY-MM-DD (expiration) + '_' + strike_cents(6) + right
example: QQQ_2025-06-20_000350C
```

(If you prefer OCC’s full symbology, store it as a separate column, but still keep a compact key for joins.)

# 3) What to ingest from Polygon (free tier)

* **Chains per expiration** → discover contracts, upsert into `option_contracts`.
* **Daily/EOD quotes per contract** → upsert into `option_quotes` at the market close (or whatever snapshot Polygon returns consistently).
* **Open interest & volume** if exposed by your plan; many EOD endpoints include OI.
* **Greeks/IV**: if you don’t have them on your plan, leave columns nullable. You can fill later by:

  * computing **IV** from mid price (Bid/Ask mid) using a solver,
  * or running **delta** off whichever IV proxy you choose.

# 4) Timescale goodies (optional but recommended)

* **Continuous aggregate** to keep an “EOD” view if you ever mix intraday snapshots:

  ```sql
  CREATE MATERIALIZED VIEW cq_option_eod
  WITH (timescaledb.continuous) AS
  SELECT
    time_bucket('1 day', ts) AS day,
    option_id,
    last, bid, ask,
    max(open_interest) AS open_interest,
    sum(volume)        AS volume
  FROM option_quotes
  GROUP BY 1,2;
  ```
* **Retention**: you can keep intraday snapshots shorter (e.g., 90 days) and EOD forever.

# 5) ETL sketch (Polygon → TimescaleDB)

* **Discover contracts** (per expiration): create/merge rows in `option_contracts` (set `first_seen`/`last_seen`).
* **Pull quotes** (daily close or your chosen snapshot): batch UPSERT into `option_quotes`.
* Use `COPY FROM STDIN` or batched INSERTs (like your existing loader) for speed.
* Backfills: iterate expirations backward, respect free-tier rate limits.

# 6) How you’ll use it in backtests

* Join the options table to the underlying `market_data` by date (close vs close) and filter contracts by **DTE**/**right**/**moneyness** or **delta**.
* For PMCC/LEAPS logic:

  * Pick LEAPS: `WHERE expiration - day >= 365` and (delta or moneyness filter).
  * Pick short call: `WHERE 25–45 DTE` and target delta band.
* Price fills with `mid` ± spread haircut using `bid/ask` columns; apply commissions and assignment rules.

# 7) Why not reuse `market_data`?

* It’s designed for OHLCV of symbols, not option contracts. Merging will force lots of NULLs, slow scans, and ugly keys. Keeping options separate avoids that and keeps your current code for stocks untouched.
