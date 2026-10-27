# L3 wait-12 WR filters (range-skip + next-bar hold)

Date: 2026-10-10  
Harness: `scripts/research/overlay_l3_wait12_wr_filters.py`  
Book: `reports/ascending_channels/channel_touch_15m_l3_wait12.csv` (300 ADV keeper, n=1744)  
Wall-clock: ~213s (268 IB 15m panels)

## Filters (causal)

1. **range_le_1.00** — skip if signal-bar `(high-low)/close > 1%`
2. **hold_1b / hold_2b** — require next 1 or 2 completed 15m bars to hold close ≥ support rail
3. **range+hold** — AND of range skip and hold

## Results

| variant | n | E | PF | WR | drop3_PF | win$keep | lose$keep | years+ |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1744 | +0.13 | 1.22 | 30.1% | 1.16 | 1.00 | 1.00 | Y |
| range_le_1.00 | 1190 | +0.21 | 1.36 | 30.9% | 1.28 | 0.71 | 0.64 | Y |
| hold_1b | 1175 | +0.36 | 1.70 | 35.6% | 1.60 | 0.80 | 0.58 | Y |
| range+hold | 771 | +0.42 | 1.85 | 35.8% | 1.70 | 0.55 | 0.36 | Y |
| hold_2b | 930 | +0.56 | 2.23 | 40.2% | 2.08 | 0.73 | 0.40 | Y |
| range+hold_2b | 594 | +0.63 | 2.45 | 40.6% | 2.23 | 0.49 | 0.24 | Y |

Annotate counts: range_ok=1190/1744, hold_1b_ok=1175/1744, both=771.

## Verdict

**No promote.** All year buckets stay positive and WR/E/PF rise, but every overlay fails the winner-$ gate (`win$keep` ≪ 1.0): the PF lift comes from dropping fat winners (and also cutting losers). Same family as rejected last-15m early-exits / occupancy WR overlays — do not wire into keeper or live 15m L3.

Artifacts:
- `reports/ascending_channels/2026-10-10/l3_wait12_wr_filters_summary_20261010_163819.csv`
- `reports/ascending_channels/2026-10-10/l3_wait12_wr_filters_annotated_20261010_163819.csv`
