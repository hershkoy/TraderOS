# Nightly switched to H2 resist-break — 2026-08-29

## Live trigger

Nightly EOD now fills a **close above resistance after H2** (span<=365, unique-symbol/day,
ATR k=2), not the L3 support-tag. Breakouts skip in-channel / RSI<=50 /
beyond-width. `--h2-resist-break-only` drops same-setup L3 tags so Telegram
is the new book, not a mix.

Rollback to the L3 keeper:

```bat
python scripts\scanners\channel_touch_nightly.py --no-h2-resist-break --require-in-channel --max-rsi 50 --max-beyond-width 0.25
```

## Daily book (already measured)

Resist-break span365, 0.25% friction, no RS cap: **n=2253 E +2.80 PF 2.21**.
HTML: `reports/ascending_channels/current_best/1d_channel_touch.html`
(same bytes as `1d_h2_resist_break.html`). Retired L3: `1d_l3_touch.html`.

## 15m A/B (300 IB names, wait-12, friction 0.10, 242s)

Same 300 names as the 15m L3 wait-12 keeper. Detector `--preset 15m`. Max wait
after H2 = 252 sessions in 15m bars. Do **not** copy daily span 365 or beyond 0.25.

```bat
python scripts\research\backtest_channel_touch_h2_break.py --preset 15m
```

| book | n | E | PF | notes |
|------|---|---|----|--------|
| 15m L3 wait-12 keeper (RS top1) | 1751 | +0.31 | 1.57 | frozen research |
| Resist-break, no span, no RS | 37277 | +0.40 | 1.80 | all years + |
| Resist-break, span<=10d, no RS | 13259 | +0.38 | 1.78 | all years + |
| **Resist-break span<=10 + RS top1** | **1664** | **+0.46** | **1.87** | all years +; beats L3 |
| Resist-break no-span + RS top1 | 1757 | +0.38 | 1.67 | 2022-23 E +0.08 (weak) |
| L3 quality + span10 break + RS | 1767 | +0.33 | 1.58 | mild lift vs L3 RS |

15m analog of the daily best is **span<=10 + RS top1**. It beats wait-12 L3 on
E and PF. Still **research only** — do not stream IB 15m; nightly stays daily EOD.

HTML: `reports/ascending_channels/current_best/15m_h2_resist_break.html`
(default max/day=1). CSV: `channel_touch_15m_h2_resist_break_20260829_212720.csv`.

## 15m full universe (1177 loaded, 3064s)

`--preset 15m --all-symbols`. Cold TimescaleDB load 1857s. After RS, **H2 does not beat L3**:

| book | n | E | PF |
|------|---|---|----|
| Resist-break span<=10 + RS top1 | 1766 | +0.30 | 1.48 |
| L3 quality + RS top1 | 1775 | +0.34 | 1.61 |

Do not promote 15m H2 over wait-12 L3 on the full IB 15m set. HTML: `current_best/15m_full_h2_resist_break.html`. See [15m full](2026-08-29_channel_touch_15m_full_h2_break.md).

## Code

- `utils/scanning/channel_touch.py` — `LIVE_DEFAULTS` h2_resist_break / only;
  5-tuple fill unpack; span365 only
- `scripts/scanners/channel_touch_nightly.py` — CLI `--h2-resist-break` /
  `--h2-resist-break-only` (default on)
- `scripts/research/backtest_channel_touch_h2_break.py` — `--preset 15m`
- Tests: `tests/unit/test_channel_touch.py`, `test_channel_touch_nightly.py`
