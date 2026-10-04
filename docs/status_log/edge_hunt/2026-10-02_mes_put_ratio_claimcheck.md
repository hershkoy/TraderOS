# dtoptions 56 DTE MES put ratio — claim-check (2026-10-02)

**Source:** [The MES Put Ratio Strategy (Huge Returns Plus 99% Win Rate!)](https://youtu.be/_Vd7WKRSiV4) — dtoptions, published 2026-10-02.

**Verdict: the win-rate claim replicates on listed SPX weeklies held to expiration. The futures return-on-buying-power claim is untested.** Mid hold-to-expiry is **n=233, 2 losses, WR 99.1%, PF 9.63**. Both losses are the April 2025 expiry. 2022 entries: **51 trades, 0 losses**. One of those 2022 winners settled about **$107** green, a point or two above the lower breakeven, after being about **$6.1k** underwater on an EOD mark. A **$0.05/leg** slip does not add losses. Selling the natural bid and buying the natural ask adds one 2022 scratch (**−$8**).

This is an SPX-sized package (multiplier **$100**). One MES 2×1 is about **1/20** of these dollars. SPAN margin is not in the test, so the video's percent-return claim is open.

## Mechanical rules

| Rule | What we ran |
|------|-------------|
| Structure | Long 1 put near **14 delta**, short **2** puts near **10 delta** (listed Theta delta) |
| Cadence | Every Friday SPX session |
| DTE | Nearest listed Friday expiry to **56** calendar days (band 35–63; exact 56 on **158/239**) |
| Hold | To SPX expiration settlement |
| Also tested | 50% of credit take-profit; exit at 21 DTE; **$0.05 / $0.10** per leg; natural bid/ask |
| Stop | None (matches the author's published 90 DTE version) |
| Claim | 242 trades, 2 losses, 99% wins, no 2022 loss |

The video does not state a profit target. The 2-loss count matches **hold to expiration**. A 50% credit target on the same entries has **0 losses** and about half the dollars, because it exits before the profit tent.

## Data

| Piece | Source |
|-------|--------|
| Options | Theta Terminal `:25503` — SPXW EOD greeks (entry delta) and EOD bid/ask |
| Spot | Yahoo `^GSPC` (Theta index EOD was empty on this window) |
| Product | **SPXW**, because Theta has no MES option root. ES exists and is mostly monthly, so it cannot host an every-Friday 56 DTE book |
| Window | Entries 2021-10-01 → 2026-10-01. 252 Fridays, 239 priced, **233** expired inside the spot history |
| Costs | $0.65 × 3 legs × open and close |
| Wall clock | **616 s** |

Harness: `utils/research/mes_put_ratio.py`, `scripts/research/claimcheck_mes_put_ratio.py`  
Artifacts: `reports/research/mes_put_ratio/2026-10-02/`  
HTML: `reports/research/mes_put_ratio/2026-10-02/mes_put_ratio.html`

Average entry: credit **14.5 pts** (~$1,450), wing **121 pts**, median DTE **56**.

## Hold to expiration (the 2-loss book)

Dollars are one SPX 2×1.

| Book | n | Wins | Losses | WR% | mean $ | med $ | PF | Total $ | Worst $ |
|------|--:|-----:|-------:|----:|-------:|------:|---:|--------:|--------:|
| mid | 233 | 231 | **2** | **99.1** | +1,467 | +1,426 | **9.63** | +341,855 | −29,881 |
| mid ± $0.05/leg | 233 | 231 | 2 | 99.1 | +1,452 | +1,411 | 9.54 | +338,360 | −29,896 |
| mid ± $0.10/leg | 233 | 231 | 2 | 99.1 | +1,437 | +1,396 | 9.44 | +334,865 | −29,911 |
| natural bid/ask | 233 | 230 | 3 | 98.7 | +1,282 | +1,236 | 8.48 | +298,700 | −30,036 |

| Entry year | n | Losses | mean $ | PF | Worst $ |
|------------|--:|-------:|-------:|---:|--------:|
| 2022 | 51 | **0** | +1,814 | — | **+107** |
| 2023 | 51 | 0 | +1,069 | — | +771 |
| 2024 | 51 | 0 | +1,233 | — | +796 |
| 2025 | 49 | **2** | +1,336 | **2.65** | −29,881 |
| 2026 (closed) | 30 | 0 | +2,164 | — | +1,546 |

Median trade is about the credit itself (the "skinny" winner). The best trade is **+$9,257** (the tent). Mean win / mean loss is **0.08**: the win rate is high because the loss, when it hits, is large.

### The two losses (April 2025)

| Entry | Expiry | Spot in | Short K | Long K | Credit | P&L |
|-------|--------|--------:|--------:|-------:|-------:|----:|
| 2025-02-07 | 2025-04-04 | 6,026 | 5,500 | 5,610 | 17.2 | **−$29,881** |
| 2025-02-14 | 2025-04-11 | 6,115 | 5,600 | 5,725 | 14.6 | **−$9,713** |

Nine packages were still alive on 2025-04-04. The two February shorts that had been sold with SPX near 6,000 expired through the strike. Entries from 2025-02-21 onward had lower shorts and expired in the tent (**+$6.9k** and **+$8.8k** on the next two). Combined P&L of those nine overlapping packages is about **−$14k**. Several of the eventual winners were **$18k–$48k** underwater at their worst EOD mark.

### 2022

Closed P&L never went red on mid. The closest call is **2022-08-19 → 2022-10-07**: short 3,750 / long 3,850, settled a little through the short strike and about **one point** above the lower breakeven (**+$107**). Worst EOD mark on that trade was about **−$6.1k**. Natural bid/ask turns that same trade into **−$8**. The 2022-08-26 package expired 2022-10-21, after the October low, with the short down at 3,495, and finished **+$1,416**.

## Other exits (same entries)

| Book | Losses | 2022 losses | WR% | PF | Total $ |
|------|-------:|------------:|----:|---:|--------:|
| 50% of credit, else expiry | **0** | 0 | 100 | — | +193,591 |
| 50% of credit, else 21 DTE | 5 | **1** | 97.9 | 34.2 | +181,266 |
| natural + 50% + 21 DTE | 12 | 4 | 94.8 | 13.9 | +143,301 |

The 21 DTE exit is what creates a 2022 loser: **2022-09-09**, flattened 2022-10-14 for about **−$2.4k**, instead of being held to the November expiry (that same structure, held, finished green).

## Take-profit early (2026-10-03)

Same 233 entries, mid marks. The target is a fraction of the **entry credit** kept. 50% means close once half the credit is banked. Every target below filled on all 233 trades (no expiry leftovers).

| Target kept | Closed losses | Total $ | Median $ | Worst closed $ | Worst single EOD mark | Trades worse than −$10k mark | Worst open book |
|-------------|----------------:|--------:|---------:|---------------:|----------------------:|-----------------------------:|----------------:|
| Hold | 2 | +341,855 | +1,426 | −29,881 | **−$48,165** | 9 | **−$190,105** on 2025-04-08 (8 open) |
| 25% | 0 | +119,336 | +456 | +201 | −$17,945 | 2 | −$19,215 on 2025-04-08 (2 open) |
| **50%** | **0** | **+193,591** | +801 | +376 | **−$21,725** | 3 | −$42,825 on 2024-08-05 (5 open) |
| 75% | 0 | +270,089 | +1,121 | +481 | −$21,725 | 4 | −$58,855 on 2025-04-08 (4 open) |

Hold-to-expiry peak-to-trough of the overlapping book is **$229,776**. The 50% target cuts that to **$43,625**. Banking only 25% of the credit cuts it to **$19,495**.

The 50% rule does close the two expiry disasters before the hole:

| Entry | Hold result | 50% exit | 50% P&L | Worst mark before that exit |
|-------|-------------|----------|--------:|----------------------------:|
| 2025-02-07 | −$29,881 on 2025-04-04 | 2025-03-19 | +881 | −$4,715 |
| 2025-02-14 | −$9,713 on 2025-04-11 (mark −$48,165) | 2025-03-24 | +1,386 | −$7,960 |

Packages opened in the two weeks before the selloff had not reached 50% yet, so they still sat through the drop:

| Entry | 50% exit | 50% P&L | Worst mark |
|-------|----------|--------:|-----------:|
| 2025-03-21 | 2025-04-28 | +821 | **−$21,725** |
| 2025-03-28 | 2025-04-25 | +801 | **−$17,945** |

Banking 25% gets the 2025-03-21 trade out on 2025-03-24 (+$581, no drawdown). The 2025-03-28 entry is still −$17,945, because the selloff started before that package had a profit to take. Early take-profit retires inventory that is already a winner. It does not give a new Friday entry a stop.

Sweep: `scripts/research/sweep_mes_put_ratio_tp.py`. JSON: `reports/research/mes_put_ratio/2026-10-02_tp/tp_sweep.json`.

## Credit-multiple stop (2026-10-03)

Stop means the first **daily close** whose loss is at least N times the entry credit (3.0 = a mark of −300% of the credit). The fill is that close, so a one-day gap exits worse than the line. Same 233 mid entries. The 50% take-profit is still on, and the stop is checked first.

| Book | Stops | Closed losses | Total $ | Worst closed $ | Peak-to-trough |
|------|------:|--------------:|--------:|---------------:|---------------:|
| 50% only | 0 | 0 | +193,591 | +376 | $43,625 |
| 50% + 2× credit | 19 | 19 | +95,886 | −13,659 | $46,182 |
| 50% + 3× credit | 9 | 9 | +107,426 | −17,239 | $43,257 |
| 50% + 4× credit | 8 | 8 | +112,526 | −17,239 | $43,257 |

Every stop on the 50% book is a trade the 50% rule alone closed green. At 3×, those 9 entries locked **−$78,530** instead of **+$7,635**. Eight of the nine also expired green. The one hold-to-expiry loser in the set is 2025-02-14, and the 50% rule had already banked that one at +$1,386 on 2025-03-24.

The March entries are not capped at the line. The first close through it is the crash close:

| Entry | Credit $ | 3× line | Stop fill (2025-04-04) | Later 50% result |
|-------|----------:|--------:|-----------------------:|-----------------:|
| 2025-03-21 | 1,475 | −4,425 | **−17,239** | +821 after a −$21,725 mark |
| 2025-03-28 | 1,545 | −4,635 | **−13,659** | +801 after a −$17,945 mark |

The day before each of those fills was still inside the line, which is why the stop had not fired yet. The same gap shows up on 2024-08-05 (July packages fill −$8.0k to −$9.8k against lines near −$3.2k to −$3.8k). That cluster is why peak-to-trough stays about **$43k**.

A 3× stop on the unmanaged hold book locks 19 trades for **−$208,054**. The same 19 entries, held, summed to **−$3,556**, because 17 of them expired green. 2025-02-07 is stopped on the expiry print itself at −$29,669 versus a hold of −$29,881. 2025-02-21 and 2025-02-28, which expired +$6,887 and +$8,767, are locked at −$25,404 and −$24,099.

Sweep: `scripts/research/sweep_mes_put_ratio_stop.py`. JSON: `reports/research/mes_put_ratio/2026-10-03_stop/stop_sweep.json`. HTML: `reports/research/mes_put_ratio/2026-10-03_stop/mes_put_ratio_manage.html`. Daily MES account (session marks, contract slider, $500-to-$2,000 short-put buying-power model, not SPAN): `reports/research/mes_put_ratio/2026-10-03_stop/mes_put_ratio_account.html`. The 50% buy-to-close fills at the limit once the close bid/ask mid is through it. That cuts the SPX book from +$193,591 (fill at the better close) to +$167,634. Natural bid/ask entry plus the same limit is +$146,171. The open-book marks before the exit day are unchanged, so the August 2024 buying-power spike is the same.

## What this does not measure

- MES or ES fills, and SPAN buying power. Percent return on a futures account can be much larger than these SPX dollars divided by a Reg-T naked-put margin, and it can also blow up when margin expands in a crash.
- Intraday marks. The credit stop above is the first daily close through the line, not a live stop order during the session.
- A single-slot account. This book opens a new ratio every Friday, so about eight packages overlap.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_mes_put_ratio.py --start 2021-10-01 --end 2026-10-01 --outdir reports\research\mes_put_ratio\2026-10-02
python scripts\research\generate_mes_put_ratio_report.py --indir reports\research\mes_put_ratio\2026-10-02
python scripts\research\sweep_mes_put_ratio_tp.py
python scripts\research\sweep_mes_put_ratio_stop.py
python scripts\research\generate_mes_put_ratio_manage_report.py
python scripts\research\export_mes_put_ratio_daily.py
python scripts\research\generate_mes_put_ratio_account_report.py
```
