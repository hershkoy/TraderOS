# Amy Meissner TimeZone — proxy v2 (1m / slip / adjust) (2026-10-01)

Follow-up to [2026-10-01_amy_timezone_claimcheck.md](2026-10-01_amy_timezone_claimcheck.md).

**Verdict: still no promote.** Closing the structural gaps (nickel/dime slip, 1m TP/SL, IV gate, basic +/-2% rolls) makes the book **more honest and weaker**, not closer to her 80% WR / PF~4 / +100% 2025 slide.

## What changed (v2)

| Gap | Change |
|-----|--------|
| Full ask/bid on 4 legs | `mid_slip` = mid +/- $0.05 or $0.10 per leg, **clamped to NBBO** |
| No-adjust hold | On +/-2% spot move (parity from ATM), roll puts up or roll calendar ATM (+puts) down; max 3 |
| Rigid entry | Skip if front IV - back IV > 0.08 (steep front backwardation) |
| EOD stops | Theta **1m NBBO** path; enter first common minute in 14:00–15:45 ET |

Harness: `utils/research/amy_timezone_proxy.py`, `scripts/research/claimcheck_amy_timezone.py`  
Artifacts: `reports/research/amy_timezone/2026-10-01_v2/`  
Wall: **~292s** for 2025 weekly x mid / mid_slip05 / mid_slip10.

## 2025 Jan–Aug weekly (n≈30, claim 80% WR / PF4 / +100%)

| Mark | n | WR% | PF | Ret% cap | Avg adj | vs v1 EOD mid |
|------|--:|----:|---:|---------:|--------:|---------------|
| **mid** (1m+adj) | 30 | 60.0 | 1.78 | **+46.9** | 0.23 | was EOD mid WR68 PF3.78 **+125%** |
| **mid_slip $0.05** | 30 | 43.3 | 0.91 | **−7.3** | 0.23 | near flat / slight lose |
| **mid_slip $0.10** | 30 | 30.0 | 0.60 | **−41.9** | 0.23 | fails |
| ask_bid (pilot Q1) | 12 | 0.0 | 0.00 | −60 | 0.0 | still dead |

Skips (same across marks): no_1m_entry 2, iv_backwardation 1, no_front_back 2.

## Interpretation

1. **1m exits are the big honesty cut.** Same mid fills but intraday TP/SL fire earlier → WR/PF/return drop vs EOD-clamped path. That is expected; her alerts are intraday.
2. **Nickel slip already kills the edge** on this proxy (PF 0.91). Dime is worse. Full ask/bid remains untradeable on a 4-leg package without mid improvement / combo orders.
3. **Simple +/-2% rolls are not her playbook.** Avg ~0.23 adjusts/trade; they do not recover the claimed PF4. Workshop adjustment tactics remain unknown.
4. **IV gate is mild** (1 skip in sample). Term-structure selection may still matter, but cannot be reverse-engineered from the roundtable.
5. **Do not promote.** Direction is not abandoned for research curiosity, but listed realism does not support her headline book under any fill model short of pure mid + EOD forgiveness — and that was already labeled optimistic.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_amy_timezone.py --window 2025 --marks mid,mid_slip,mid_slip10
python scripts\research\claimcheck_amy_timezone.py --window pilot --marks mid_slip --no-adjust
```
