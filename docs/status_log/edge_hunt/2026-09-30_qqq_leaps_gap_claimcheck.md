# QQQ LEAPS gap claim-check (2026-09-30)

**Source:** Options With Ravish — YouTube [dY2ralsWdQ8](https://www.youtube.com/watch?v=dY2ralsWdQ8) (2025-07-05).

**Verdict: no promote** (uncapped BS undershot the 96% WR; ATR-IV was rich vs listed mid on gaps). **Superseded for marks by Theta listed replay** — see [2026-09-30_qqq_leaps_gap_theta_claimcheck.md](2026-09-30_qqq_leaps_gap_theta_claimcheck.md) where mid/mid uncapped **does** reprint ~96% / ~$180k.

Claim is a **5y OptionsKit backtest** on QQQ only: buy ~60-delta / ~12mo calls on gap-downs; TP +50%; no stop. Video intro dashboard (~91% WR / $176k / 112 trades) is the **prior** gap-only video, not this SMA-filtered pair.

## Claimed

| Book | n | WR | PnL | PF | Avg days | MDD | Contracts |
|------|--:|---:|----:|---:|---------:|----:|----------:|
| Strat1 gap≥1% + SMA100 | 53 | 96% | $88,400 | 10.48 | 102 | $9,330 | 1 |
| Strat2 gap≥2% | 27 | 96% | $91,000 | 15 | — | $6,000 | 2 |
| Combined | — | 96% | ~$180k | — | — | — | — |

## Ours (mechanical)

- Window: **2020-01-03 → 2025-06-05** (OptionsKit shot in the thumbnail)
- Underlying: QQQ IB 1d
- Entry: **open** on gap day; SMA = prior close > SMA100 (causal at open)
- Option: BS call, δ≈0.60, DTE=365, r=5%, q=0.5%
- IV: prior ATR20 annualized (constant, or elev× + linear crush)
- Data: QQQ `option_contracts` exist but **0 quotes** in TimescaleDB; OptionsDX EOD is SPY/VIX only. **ThetaData Standard** has live QQQ LEAPS hist NBBO (see below); Polygon/Massive current plan does not unlock the claim window.

Harness: `utils/research/qqq_leaps_gap.py`, `scripts/research/claimcheck_qqq_leaps_gap.py`  
Probes: `scripts/research/probe_qqq_leaps_theta_poly_fast.py`, `scripts/research/probe_qqq_leaps_bs_vs_theta.py`  
Artifacts: `reports/research/qqq_leaps_gap/2026-09-30/`

## Listed marks: ThetaData vs Polygon/Massive (2026-09-30)

| Source | QQQ LEAPS hist? | What we saw |
|--------|-----------------|-------------|
| **ThetaData Standard** (Terminal :25503) | **Yes** (NBBO 1m + EOD) | 1730 QQQ expirations listed **2012-06 → 2029-01**. Nonzero bid/ask on Jan/Mar/Jun/Sep LEAPS-cycle contracts; many non-cycle listed exps return **472**. EOD endpoint also returns bid/ask. |
| **Polygon / Massive** (current key) | **No for this book** | Option **day/minute aggs 403** on 2020–2024 probe days; **quotes 403** (Advanced). 2025-04-04 day/min HTTP 200 but **0 results** on OCC guess; API connect timeouts intermittent. Contract reference often works. |
| TimescaleDB / OptionsDX | No | QQQ contracts leftover only; quotes empty. SPY/VIX EOD through 2023 only. |

### BS (ATR-IV) vs Theta first nonzero RTH mid (spot)

| Day | Gap% | Exp | K | BS | Theta mid | BS−mid% | Half-sprd% |
|-----|-----:|-----|--:|---:|----------:|--------:|-----------:|
| 2020-03-09 | −7.0 | 2021-03-19 | 202 | 38.96 | **19.00** | **+105%** | 13.2 |
| 2020-09-03 | −1.5 | 2021-09-17 | 300 | 35.34 | **35.28** | +0.2 | 6.3 |
| 2022-05-05 | −1.3 | 2023-06-16 | 340 | 64.69 | **31.69** | **+104%** | 1.3 |
| 2022-11-02 | ~0 | 2023-09-15 | 280 | 48.23 | **32.57** | +48 | 0.9 |
| 2024-07-24 | −1.4 | 2025-06-20 | 480 | 46.15 | **43.52** | +6 | 1.5 |
| 2025-04-04 | −2.8 | 2026-03-20 | 445 | 76.18 | **47.81** | **+59%** | 3.8 |

Calm days: BS ≈ mid. Stress/gap days: ATR-IV BS is **much richer than listed mid** (long-entry BS debit is pessimistic vs mid; marketable **ask** is still above mid). Half-spreads on panic opens can be double-digit %.

**Implication:** a full Theta mid/ask replay is doable on Standard and is the right next step before trusting dollar PnL; it is unlikely to invent the claimed 96% WR (BS already undershoots that). Do not expect Massive/Polygon on this key to replace Theta for 2020–2024 QQQ LEAPS.

### Signal counts (raw, no occupancy)

| Rule | Ours | Claim |
|------|-----:|------:|
| Gap ≤ −1% only | 125 | (prior video ~112 fills) |
| Strat1 gap1 + SMA100 prior | **66** | 53 |
| Strat1 gap1 + SMA100 open | 58 | 53 |
| Strat2 gap ≤ −2% | **31** | 27 |

Signal frequency is in the same neighborhood (slightly more signals on IB opens than OptionsKit).

### A) Const IV + max_open=3 (video risk note)

| Book | n | WR% | PnL | PF | Avg d | MDD | vs claim |
|------|--:|----:|----:|---:|------:|----:|----------|
| Strat1 | 32 | 84.4 | $35k | 3.7 | 142 | $11k | WR/n miss (34 occ skips) |
| Strat2 | 16 | 81.2 | $34k | 2.2 | 131 | $26k | WR/n miss |
| Combined | 48 | 83.3 | $68k | 2.7 | 138 | $38k | far from $180k |

Year buckets (combined): 2020-21 WR 92 / PF 4.4; **2022 WR 40 PF 0.38**; 2024-26 WR 75.

### B) Const IV, **no** occupancy cap (closest $PnL match)

| Book | n | WR% | PnL | PF | Avg d | MDD | vs claim |
|------|--:|----:|----:|---:|------:|----:|----------|
| Strat1 | 66 | **84.8** | **$92k** | 6.9 | 124 | $11k | $PnL ~claim; WR 85≠96; n 66>53 |
| Strat2 | 31 | **90.3** | **$112k** | 5.1 | 105 | $26k | $PnL ≥claim; WR 90≠96 |
| Combined | 97 | 86.6 | $204k | 5.8 | 118 | $33k | $PnL ≥$180k; WR still short |

Year buckets (combined): 2020-21 **WR 96.2 PF 11.9** (claim-like); 2022-23 WR 82 PF 2.05; **2022 alone WR 64 PF 1.12**; 2024-26 WR 68.

### C) Entry IV ×1.25, crush 20d, max_open=3

| Book | n | WR% | PnL | PF |
|------|--:|----:|----:|---:|
| Combined | 39 | **74.4** | $31k | 1.58 |

### D) Entry IV ×1.50, crush 10d, max_open=3

| Book | n | WR% | PnL | PF |
|------|--:|----:|----:|---:|
| Combined | 33 | **60.6** | **−$19k** | 0.74 |

## Interpretation

1. **Headline 96% WR does not reprint** under a mechanical open-gap + BS LEAPS book. Best uncapped WR is ~85% (strat1) / ~90% (strat2). The claim’s 2-loss / 1-loss tails are tighter than our BS path.
2. **Dollar PnL without a position cap can look claim-like** (~$92k / ~$112k) because one LEAPS contract × many QQQ dips in a bull window accumulates large cash P&L — that is not risk-adjusted edge. Cap-3 (the live rule he states) halves n and cuts combined PnL to ~$68k with WR ~83%.
3. **2020-21 carries the story.** Uncapped combined 2020-21 alone is WR 96 / PF 12 — almost the marketing slide. 2022 and 2024-26 are weaker; video itself admits the sleeve needs a bull market.
4. **Avg win << avg loss** in the claim (RR ~0.4) is the classic high-WR / fat-left-tail shape. No stop + full debit risk means a few extra losers destroy PF; n≈50 is small.
5. **Gap-down IV elevation kills the promote.** Buying 60δ LEAPS into a panic gap pays rich IV; elev 1.25 / 1.50 stress drops WR to mid-70s / ~60 and can flip combined PnL negative. Constant ATR-IV is a coarse stress — Theta spot-checks show ATR-IV BS can be **+50–100% above listed mid** on gap days (so the const-IV book is not “too cheap”).
6. **Listed realism path is Theta, not Massive/Polygon on this key.** Standard Terminal already returns QQQ LEAPS NBBO on LEAPS-cycle exps back through 2020; Polygon option aggs/quotes 403 for the claim window. Full Theta mid/ask book not walked yet — next step if we want fill realism, not a promote expectation.
7. **Promotion gate:** WR is not the gate. Need occupancy equity Sharpe>1 and MDD better than QQQ (`docs/assessing_strategies.md`). Uncapped MDD already ~$33k cash on this sizing vs claim’s $6–9k — inconsistent with “safer than buy&hold.”

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_qqq_leaps_gap.py
python scripts\research\claimcheck_qqq_leaps_gap.py --no-max-open --iv-elev 1.25 --iv-crush-days 20
python scripts\pipeline\thetadata_terminal.py
python scripts\research\probe_qqq_leaps_bs_vs_theta.py
pytest tests\unit\test_qqq_leaps_gap.py -v
```
