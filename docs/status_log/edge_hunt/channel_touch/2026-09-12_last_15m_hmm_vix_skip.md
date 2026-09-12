# Causal VIX backwardation + expanding HMM skip on last-15m (2026-09-12)

Same-list skip on the LAUR-fixed last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (geo 1.095). Prior-session `completed_asof` only (fill-day daily SPY/VIX closes are illegal at 15:45 ET). Occupancy not re-walked. **No promote.**

Pre-registered: `VIX_back`, `VIX_back_min20`, 2-state HMM P(bull)<0.50/0.65, 3-state same, 2-state + TLT/GLD/UUP P(bull)<0.50. Expanding refit every 21 sessions, Wasserstein template match, warmup keeps. Runner: `scripts/research/overlay_last_15m_hmm_vix_skip.py`.

## Gate

A last-15m overlay that "fixes 2022-23" has to make **2022 itself** non-red without clipping 2020-21 / 2024-26 winner dollars. SPY SMA200/20d already failed that (bear-rally chases). This run asks whether a causal vol-curve or HMM state is a better switch.

## Results

| Overlay | n | E | PF | geo | skipped | 2022 n / E / PF | 2023 E / PF | 2022-23 E / PF |
|---------|---|---|----|-----|---------|-----------------|-------------|----------------|
| Baseline | 2297 | +0.27 | 1.09 | 1.095 | — | 117 / **−2.37 / 0.41** | −0.33 / 0.90 | −0.90 / 0.73 |
| **VIX_back** | 2152 | +0.14 | 1.04 | 1.067 | 145 | 106 / **−2.69 / 0.35** | −0.33 / 0.90 | −0.95 / 0.72 |
| **VIX_back_min20** | 2158 | +0.13 | 1.04 | 1.063 | 139 | 106 / **−2.69 / 0.35** | −0.33 / 0.90 | −0.95 / 0.72 |
| **HMM2_p50** | 1398 | −0.07 | 0.98 | 0.96 | 899 | 24 / **−3.71 / 0.09** | −0.38 / 0.88 | −0.72 / 0.77 |
| **HMM2_p65** | 1373 | −0.08 | 0.97 | 0.97 | 924 | 19 / **−3.63 / 0.10** | −0.26 / 0.91 | −0.54 / 0.83 |
| **HMM3_p50** | 690 | −0.36 | 0.88 | 0.60 | 1607 | **0** | −1.73 / 0.50 | −1.73 / 0.50 |
| **HMM3_p65** | 676 | −0.29 | 0.90 | 0.61 | 1621 | **0** | −1.73 / 0.50 | −1.73 / 0.50 |
| **HMM2_macro_p50** | 1796 | +0.29 | 1.09 | 1.113 | 501 | 55 / **−2.14 / 0.42** | −0.50 / 0.84 | −0.78 / 0.76 |

2023 VIX gates skip **zero** fills (curve never >= 1 on those prior sessions). The 11 names VIX_back drops in 2022 were the *better* 2022 prints: leftover E falls from −2.37 to −2.69.

## Winner-$ (2020-21 / 2024-26)

| Overlay | 2020-21 win$ / lose$ | blunt | 2024-26 win$ / lose$ | blunt |
|---------|----------------------|-------|----------------------|-------|
| VIX_back | 240 / −114 | **yes** | 465 / −286 | no |
| HMM2_p50 | 1064 / −614 | **yes** | 1597 / −1282 | no |
| HMM3_p50 | 2367 / −1884 | no | 2195 / −1810 | no |
| HMM2_macro_p50 | 196 / −291 | no | 1289 / −961 | no |

HMM3 "fixes" 2022 by taking **no 2022 trades** and also wrecks 2020-21 (kept n=97 E **−1.72** PF 0.40 vs +0.46 / 1.15) and 2018-19 (n=1). That is a kill switch, not a regime overlay. HMM2 leftover 2022 is the SMA pattern: skip the obvious bear, keep bounce chases that hard-stop harder. Macro 2-state is the only pooled E/geo bump (+0.27→+0.29 / 1.095→1.113); 2022 stays −2.14, 2023 gets worse, 2024-26 E 0.48→0.24 while dropping more winner $ than loser $.

## Why (same as SPY SMA)

Last-15m H2 in 2022 is a failed-breakout / bear-rally book. Term-structure and HMM risk-off fire on the high-VIX trend legs and stay on through the squeezes that print above the daily rail. Prior-session as-of is honest and does not change that. Do not threshold-fish P(bull) or add Wasserstein K selection.

15m L3 wait-12 remains the 2022-23 playbook that is already green. Do not copy it onto last-15m via an HMM switch (oracle calendar splice already failed to make 2023 last-15m work).

No L1 / OFI / Almgren-Chriss until a causal skip actually clears this gate.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_hmm_vix_skip.py
```

Artifacts: `reports/ascending_channels/2026-09-12/last_15m_hmm_vix_skip_summary.csv`.
