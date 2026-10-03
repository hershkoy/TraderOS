# Silent Flip (ProRealAlgos) — claim-check (2026-10-03)

**Verdict: no promote.** The 15-minute open-range "touch, silent candle, stop at the silent extreme" book is a coin flip on IB RTH 15m. Uncapped 300-ADV expectancy is about **0.00%** with PF **1.01**. A dime of round-turn friction turns it into **E −0.10 PF 0.79**. Occupancy 8/15 is worse than uncapped. Calendar 2022 is not a hole in isolation; it is just more of the same ~1.0 PF. Do not treat the Uber live clip as a 1:9 edge.

YouTube [Q2QjB5jBTro](https://www.youtube.com/watch?v=Q2QjB5jBTro) (ProRealAlgos, 2026-09-29). No transcript was available; rules follow the pasted summary plus the public description.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Range High / Low | Prior completed RTH session high / low |
| Flip High / Low | High (low) of the most recent earlier session whose high (low) is strictly beyond the prior-day range |
| Window | `first_60`: candle 1 is 09:30, 09:45, or 10:00 ET. Variant `open_only` is 09:30 only |
| Candle 1 | Tags the line (0.10% band), does not open or close through it, moves into the line (red at support, green at resistance), body/range >= 0.30 |
| Silent candle | Opposite color. Entry is a stop at that candle's high (long) or low (short), filled on candle 3 (gap uses the open) |
| Stop | Just beyond `min(level, candle-1 extreme)` by 0.05% (floor $0.01) |
| Target | Opposite **inner** range rail (Range High for longs, Range Low for shorts), matching the Uber Flip-High → Range-Low example |
| Exit | Same-session; same-bar stop+target is the stop; flatten 15:45 ET close |
| Trail variant | After the first later bar that closes in favor, ratchet stop to entry |
| Friction | 0.10% round trip |
| Data | IB 15m, 2018-11-01 → 2026-08-21 |

Harness: `utils/research/silent_flip.py`, `scripts/research/claimcheck_silent_flip.py`  
Artifacts: `reports/research/silent_flip/2026-10-03/` (video four-name CSVs) and `.../adv300/`  
Wall clock: video names **52 s** (load 48 s); 300 ADV **429 s** (list 158 s, load 24 s, fills 224 s)

NVDA has no IB 15m in TimescaleDB, so the video scan is BAC / PFE / UBER / XOM.

## Results — 300 ADV IB 15m

| Book | n | WR% | E% | med% | PF | geo | fric0.10 E | fric PF | max conc | med conc |
|------|--:|----:|---:|-----:|---:|----:|-----------:|--------:|---------:|---------:|
| **baseline first_60** | **39090** | **42.6** | **+0.003** | **−0.24** | **1.007** | **1.003** | **−0.097** | **0.791** | **48** | **9** |
| baseline occ-8 | 19286 | 43.0 | −0.014 | −0.25 | 0.968 | 0.962 | −0.114 | 0.769 | 8 | 4 |
| baseline occ-15 | 31523 | 42.7 | −0.004 | −0.25 | 0.991 | 0.982 | −0.104 | 0.781 | 15 | 7 |
| open_only 09:30 | 17754 | 45.7 | +0.000 | −0.17 | 1.001 | 0.987 | −0.100 | 0.797 | 34 | 4 |
| no_strong (any body) | 59628 | 44.2 | +0.005 | | 1.012 | 1.011 | −0.095 | 0.798 | 66 | 13 |
| breakeven trail | 39090 | 20.6 | +0.000 | 0.00 | 1.002 | 0.993 | −0.100 | 0.689 | 48 | 8 |

Baseline year buckets are all PF ~0.98–1.04. Calendar 2022 n=4781 E +0.012 PF 1.02; after friction E −0.088 PF 0.84.

Stops 17970 / EOD 14005 / target 7115. About **46%** of fills die at the stop. Planned Uber-style range targets print on about **18%**.

## Video names (no NVDA 15m)

Baseline first_60, four names, n=612 E +0.055 PF 1.17 geo 1.17; fric0.10 E −0.045 PF 0.88. 2022 n=77 E −0.06 PF 0.87. Max concurrent 3.

| Symbol | n | mean E% | median % |
|--------|--:|--------:|---------:|
| BAC | 180 | +0.13 | +0.09 |
| PFE | 172 | +0.05 | −0.18 |
| XOM | 126 | +0.08 | −0.00 |
| **UBER** | **134** | **−0.06** | **−0.32** |

The live example name is the worst of the four on mean and median.

## What this says about the sketch

1. **A 10-minute morning scan of a few names is not a 1:9 factory.** Across liquid 15m names the typical trade is slightly negative on the median. The target is the far side of yesterday's range; most paths never get there before the stop or the close.

2. **Friction is the whole edge.** Gross PF 1.01 is noise. 0.10% round trip (the same haircut used on the 15m VWAP reclaim book) leaves PF 0.79.

3. **Tightening the open to the 09:30 bar, dropping the body filter, or ratcheting to breakeven does not rescue it.** Open-only and breakeven stay at PF ~1.00 gross and lose after friction. Breakeven cuts win rate to 21% by turning winners into scratches.

4. **Do not occupancy-walk this onto last-15m H2 or any sleeve.** Uncapped max concurrent 48 / median 9; cap 8 already has PF 0.97 gross.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\claimcheck_silent_flip.py --video-names --variants all --outdir reports\research\silent_flip\2026-10-03
python -u scripts\research\claimcheck_silent_flip.py --n-symbols 300 --variants all --outdir reports\research\silent_flip\2026-10-03\adv300
```
