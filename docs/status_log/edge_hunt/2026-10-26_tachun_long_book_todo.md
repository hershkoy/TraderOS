# Tachun long book — 20-year daily window and backtest todo

Date: 2026-10-27. Not started as a backtest. The daily prefix main pass finished 2026-10-27 06:47 Israel (1,575 symbols, 3,925,333 bars). Seven names are still short of 2006; the hourly job keeps retrying them. Roll-up: `docs/status_log/current_status.md`.

The course book is the long-only screen plus a 20-year discounted-cash-flow band (no derivatives). Two later runs are listed below. Both stay survivor-biased and estimate-free until a delisted price feed and point-in-time EPS estimates exist.

## Extend 1d history to 20 years

The 5-minute universe fill is paused (`backfill_ib_5m_universe` disabled in `crons/crontab.yaml`) until this prefix finishes. The overnight slot is now the 1d prefix: `after_rth_ib_backfill` at 16:30 ET runs 15m catch-up then `crons/backfill_ib_1d_prefix.bat` (client 8827). The hourly watchdog `backfill_ib_1d_prefix` uses the old 5m window (Mon–Fri 17:00–08:00 ET and all weekend hours) and restarts the prefix if the process died. `stop_ib_5m_backfill` at 09:15 ET stops it so live 15m owns Gateway. Re-enable the 5m job when a dry-run reports no symbols left.

## Extend 1d history to 20 years (how)

Alpaca daily bars for this universe often start around 2020–2022. The 2026-08-26 IB prefix run (`--start 2018-11-01`) filled the gap back to about November 2018 for names IB still lists. That run will not go further: a symbol was skipped once any IB bar existed before the first Alpaca bar.

`scripts/data/backfill_ib_daily_prefix.py` selects a name when the **earliest stored 1d bar (Alpaca or IB)** is after `--start`. Default start is **2006-01-01** (20 years before 2026, and the same clip `--complete-history` uses). Each name is one request ending now (`durationStr` covers the gap, timeout 180s). Bars already stored are dropped. A second request with a dated end is used only when that reply does not reach 2006. An empty or timed-out reply is retried next run. A name is written to `logs/data/ib_prefix_no_gap_symbols.txt` only when the oldest bar IB returned is the history already stored. Client id default **8827**.

IB will not return history for names that are no longer trading. This window is still the current listings.

Do not run a second IB historical client beside this job. Live 15m (8826) owns Gateway on weekdays 09:15–16:30 ET; the prefix stops itself at 09:15 and the hourly watchdog only fires outside that window.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\data\backfill_ib_daily_prefix.py --start 2006-01-01 --dry-run
python scripts\data\backfill_ib_daily_prefix.py --start 2006-01-01 --limit 5 --sleep 1
python scripts\data\backfill_ib_daily_prefix.py --start 2006-01-01 --sleep 1
```

`--dry-run` only lists symbols. `--limit 5` is the Gateway smoke test. A full pass is about one request per name that still starts after 2006, plus a 1 second pause (a second dated request only when the first reply stops short of 2006). A name whose earliest bar is within 10 days of `--start` is done, because 2006-01-01 was a Sunday and the first session was 2006-01-03. `logs/data/ib_prefix_no_gap_symbols.txt` skips names IB cannot extend, including 130 symbols with no IB contract (added 2026-10-27).

**2026-10-27 result.** Main log `logs/data/ib_1d_prefix_backfill_20261026_234449.log`: saved 1,575, failed 549, 3,925,333 bars. SPY and A are stored from 2006-01-03. The 07:40 pass found 757 still short: 620 no-gap, 130 no contract. As of the 10:42 pass, seven remain: SCCO 2008-01-03 (dated request missed), BSAC 2008-12-10, OLED 2010-05-18, TILE stuck at 2012-05-15, SATS stuck at 2022-07-25, CLBK Error 162 (no history), UVXY unadjusted prices overflow the column (not stored; IB min still 2018-11-01). Gateway was down at 10:00 Israel and up again by 10:35 (port 4001).

Research loads must pass `start=2006-01-01` (or `--complete-history`) with `fallback_provider="IB"` and `merge_mode="prefix"`. `start=None` does not stitch a prefix onto names that already have Alpaca bars.

## Todo — free backtest on this repo

- [x] Extend the SEC parser (`utils/research/sec_fundamentals.py`) so each filing row also has revenue, gross profit, operating income, net income, pretax income, interest expense and interest income, current assets, current liabilities, equity, diluted shares, diluted EPS, and depreciation. EBITDA, margins, ROE, current ratio, and debt/EBITDA are computed from those. As-of date stays `filed`. Quarterly income that is year-to-date has to be differenced. Quarterly flow fields are trailing-twelve-month sums of those single quarters; FY rows keep the reported year. Schema columns were applied 2026-10-27 (`fundamentals_quarterly` is 51 columns). The same-day `--parse-only` reload wrote 247,572 rows in 9271s; 1,792 symbols have revenue. IBN, OZK, PFBC, and SPY had no cached companyfacts.
- [ ] Each quarter, keep names that pass every **trailing** rule: market cap over $2B, five-year sales and EPS growth positive, EPS growth this year over 5%, ROE over 15%, current ratio over 1, gross margin over 40%, operating margin over 20%, net margin over 10%, five-year uptrend in revenue, net income, and operating cash flow, debt/EBITDA under about 3–4, net interest / operating cash flow under 40%. One bad year is allowed; two consecutive bad filings exit.
- [ ] Value passers with a 20-year DCF. Discount rate uses FRED `DGS10`, beta computed from our daily closes versus SPY (cap or shrink a spike beta; do not leave the haircut discretionary), and trailing growth capped then faded. No analyst estimates.
- [ ] Equal-weight up to 15 names. Hold the rest in cash. Add when price falls through the value band. Trim when price clears a fixed multiple of the optimistic value. No stop.
- [ ] Price window can start in 2006 for the **994** names whose first bar is 2006 or earlier. The other **621** still begin in 2019 or later, so the panel widens as listings appear. SEC XBRL is thinner before about 2009. Loads must use `start=2006-01-01` / `--complete-history` with IB prefix merge (`start=None` does not stitch). Label the result **survivor-biased and estimate-free**.

## Todo — still free, tighter universe

- [ ] Same rules on a SEC CIK crawl, not only the current daily-bar universe, so dead filers at least fail the screen on their last reports.
- [ ] Prices for those names stay missing. Do not book a loss you cannot mark. Say that in the write-up.
