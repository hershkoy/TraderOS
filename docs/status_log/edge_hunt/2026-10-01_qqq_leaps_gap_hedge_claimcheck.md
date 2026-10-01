# QQQ LEAPS gap — low-VIX hedge claim-check (2026-10-01)

**Verdict: no promote** — none of the three hedges lift MDD vs LEAPS-only at max_open=10; H1/H2 are small bleed or noise; H3 never fills under the $0.10 net-debit collar rule (mostly `no_nbbo` / rich debit).

Baseline LEAPS (Theta mid/mid, max_open=10): **n=89 PnL $164,758 PF 13.35 MDD $13,341** (matches the HTML capital walk).

## Setup

- LEAPS pool: `reports/research/qqq_leaps_gap/2026-09-30_theta/mid_mid_uncapped_combined.csv`
- Window: 2020-01-03 → 2025-06-05
- Cash VIX gate: prior-session close from TimescaleDB/IB
- Marks: Theta Standard NBBO entry + EOD path (VIX options **available** — probe `2026-10-01_hedge/probe_thetadata_vix_options.json`)
- Occupancy: **max_open=10** primary
- Wall: ~217s mid/mid 3×3 matrix; ~96s ask/bid stress subset

Harness:
- `utils/research/qqq_leaps_gap_hedge.py`
- `scripts/research/claimcheck_qqq_leaps_gap_hedge.py`
- `scripts/research/generate_qqq_leaps_gap_hedge_report.py`
- HTML: `reports/research/qqq_leaps_gap/2026-10-01_hedge/qqq_leaps_gap_hedge_matrix.html`

### Structures

| ID | Structure | Arm | Size / notes |
|----|-----------|-----|----------------|
| H1 | QQQ put debit ~45 DTE, long ~5% OTM / short ~10% OTM | prior VIX &lt; 15 | 1 vertical per open LEAPS lot |
| H2 | VIX call debit ~45 DTE, long ~18 / short ~25 | prior VIX &lt; 14 | 1 vertical per 2 open lots (floor 1) |
| H3 | QQQ collar long put ~5% OTM + short call ~10% OTM | prior VIX &lt; 15 | skip if net debit mid &gt; $0.10 |

Exit: LEAPS exit / hedge expiry / prior VIX ≥ 20 (close hedge only) / target shrink (calendar modes).

Attach: `while_open`, `continuous`, `entry_day`.

## Mid/mid matrix (max_open=10)

| Key | Hedge n | Hedge PnL | Combined PnL | Combined PF | Combined MDD | MDD lift |
|-----|--------:|----------:|-------------:|------------:|-------------:|---------:|
| H1_while_open | 8 | −810 | 163,948 | 12.59 | 13,341 | 0 |
| H1_continuous | 9 | −694 | 164,064 | 12.66 | 13,341 | 0 |
| H1_entry_day | 2 | −276 | 164,482 | 13.08 | 13,341 | 0 |
| H2_while_open | 33 | −122 | 164,636 | 13.08 | 13,341 | 0 |
| H2_continuous | 36 | −186 | 164,572 | 13.02 | 13,341 | 0 |
| H2_entry_day | 6 | **+140** | **164,898** | 13.12 | 13,341 | 0 |
| H3_while_open | 0 | 0 | 164,758 | 13.35 | 13,341 | 0 |
| H3_continuous | 0 | 0 | 164,758 | 13.35 | 13,341 | 0 |
| H3_entry_day | 0 | 0 | 164,758 | 13.35 | 13,341 | 0 |

H3 fill failure: calendar attempts mostly `no_nbbo`; entry_day mostly `skip_vix_gate` (gap days are rarely VIX&lt;15) plus `no_nbbo` / `collar_debit_too_rich`.

## Ask/bid stress (subset)

| Key | Hedge n | Hedge PnL | Combined MDD | MDD lift |
|-----|--------:|----------:|-------------:|---------:|
| H2_entry_day_ask_bid | 6 | −284 | 13,341 | 0 |
| H2_while_open_ask_bid | 33 | −794 | 13,341 | 0 |
| H1_entry_day_ask_bid | 2 | −291 | 13,341 | 0 |
| H1_while_open_ask_bid | 8 | −883 | 13,341 | 0 |

Marketable marks flip the lone mid/mid H2_entry_day +$140 into −$284 bleed. Still **no MDD lift**.

## Interpretation

1. **Low-VIX arming does not cover LEAPS left-tail events.** Gap-down LEAPS entries usually print with VIX already elevated (`entry_day` H1: 74/89 `skip_vix_gate`). Big LEAPS losers sit in high-vol regimes where the hedge is disarmed (arm off or VIX≥20 exit).
2. **At max_open=10 the book is nearly always long**, so `while_open` ≈ `continuous` for coverage days; both still only bleed a few hundred dollars of premium — not enough to move the $13.3k LEAPS MDD.
3. **H2 VIX call verticals** are the least harmful mid/mid (entry_day slight +$140), but ask/bid kills that and MDD is unchanged. Not a promote.
4. **H3 zero-cost collar** does not clear the locked $0.10 debit screen on listed Theta marks in this window — treat as unavailable under plan params, not as a pass.
5. **VIX≥20 hedge exit** closes insurance as vol expands (by design in the plan). That is coherent with “disarm after spike starts” but works against collecting crash payoff; even without promoting a change here, any follow-up should A/B hold-through-spike vs this exit.

## Gate

Improve MDD / left-tail vs LEAPS-only at max_open=10 without destroying PF/CAGR. **Failed** (MDD lift 0 on all books; PF flat-to-slightly-worse from hedge bleed).

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\probe_thetadata_vix_options.py
python scripts\research\claimcheck_qqq_leaps_gap_hedge.py --max-open 10
python scripts\research\claimcheck_qqq_leaps_gap_hedge.py --hedges H2,H1 --attach entry_day,while_open --mark-modes ask_bid --max-open 10 --outdir reports\research\qqq_leaps_gap\2026-10-01_hedge\ask_bid
python scripts\research\generate_qqq_leaps_gap_hedge_report.py
pytest tests\unit\test_qqq_leaps_gap_hedge.py -v
```
