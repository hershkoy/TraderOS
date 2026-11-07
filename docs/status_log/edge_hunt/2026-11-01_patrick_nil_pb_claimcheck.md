# Patrick Nil P/B 1d claim-check (2026-11-01)

**Verdict: no promote.**

User-pasted Nil sketch mapped to daily bars + weekly SMA50 HTF. Three books on 99 names (33 small / 33 mid / 33 large SEC `mkt_cap`, ADV-ranked within bucket). Signal on completed daily close; fill next open; structure stops; friction 0.10 / 0.25.

Harness: `utils/research/patrick_nil_pb.py`, `scripts/research/claimcheck_patrick_nil_pb.py`  
Artifacts: `reports/research/patrick_nil_pb/2026-11-01/`  
Wall clock: **158.9s** (universe 13.2s, load 114.6s, fills 30.9s)

## Takeaway

- **Fake breakout** (Nil favorite) is flat/red: pooled E −0.06 PF 0.98; fric0.25 E −0.31 PF 0.89. Occupancy equity blows up (MDD > 100%).
- **Range fade** is thin: pooled E +0.23 PF 1.19; fric0.25 flips to E −0.02 PF 0.99. Occ8 Sharpe 0.49 < SPY.
- **Confirmed breakout** looks best on trade stats (E +0.96 PF 1.26 geo 1.32; fric0.25 still E +0.71 PF 1.19) but FCFS occupancy vs SPY fails hard (occ1 Sharpe −0.19 MDD −91%). Clustered books, not a Phase-5 sleeve.
- Do not parameter-fish ATR/touch knobs on this pass.

## Universe

| bucket | n | sec | adv-proxy |
|--------|--:|----:|----------:|
| small | 33 | 33 | 0 |
| mid | 33 | 33 | 0 |
| large | 33 | 33 | 0 |

## Results

### range_fade

Pooled **n=6464 E=0.2292 PF=1.19 geo=1.173** (drop-top-3 E=0.1884)
- fric 0.1: E=0.1292 PF=1.101
- fric 0.25: E=-0.0208 PF=0.985
- small: n=1565 E=0.4454 PF=1.275
- mid: n=2293 E=0.2568 PF=1.204
- large: n=2606 E=0.075 PF=1.083
- occ1: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=0.20 MDD=-0.50 n=2599)
- occ3: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=0.32 MDD=-0.36 n=5427)
- occ8: FAIL (Sharpe<=1.0, Sharpe<SPY) (Sharpe=0.49 MDD=-0.19 n=6440)

### confirmed_breakout

Pooled **n=2857 E=0.9615 PF=1.264 geo=1.316** (drop-top-3 E=0.8581)
- fric 0.1: E=0.8615 PF=1.234
- fric 0.25: E=0.7115 PF=1.19
- small: n=640 E=1.2921 PF=1.248
- mid: n=994 E=1.3865 PF=1.384
- large: n=1223 E=0.4431 PF=1.156
- occ1: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=-0.19 MDD=-0.91 n=446)
- occ3: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=-0.24 MDD=-0.87 n=1243)
- occ8: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=0.01 MDD=-0.62 n=2454)

### fake_breakout

Pooled **n=6352 E=-0.057 PF=0.979 geo=0.942** (drop-top-3 E=-0.1068)
- fric 0.1: E=-0.157 PF=0.944
- fric 0.25: E=-0.307 PF=0.893
- small: n=1437 E=0.5278 PF=1.148
- mid: n=2221 E=-0.3097 PF=0.896
- large: n=2694 E=-0.1605 PF=0.924
- occ1: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=nan MDD=-1.22 n=910)
- occ3: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=nan MDD=-1.10 n=2553)
- occ8: FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) (Sharpe=nan MDD=-1.02 n=5234)

