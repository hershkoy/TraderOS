# Last-15m listed buy-write (Theta Options VALUE) + earn flatten≤10 — 2026-11-03

**Verdict: listed buy-write is real now, but weaker than the BS fantasy and weaker than wheel mid; ask/bid + earn fails 2022. No promote.**

Spine: same full-pool wheel list (ADV≥$2M, buy≥2022, n=1118). Marks: Theta Terminal **Options VALUE** via `list/strikes` + `history/eod` (ATM strike; no `greeks/eod` — that needs Standard). Earn: flatten≤10.

## Headline

| Book | n | E | PF | geo | 2022 E / PF |
|------|--:|--:|---:|----:|------------:|
| BW listed mid | 802 | +1.10 | 1.48 | 1.45 | +0.47 / 1.15 |
| BW listed mid + earn | 794 | +1.02 | 1.51 | 1.49 | +0.57 / 1.23 |
| BW listed ask/bid | 802 | +0.45 | 1.18 | 1.15 | **−0.24 / 0.93** |
| BW listed ask/bid + earn | 794 | +0.26 | 1.11 | 1.09 | **−0.38 / 0.87** |

Hit rate **802/1118 (72%)**; 316 `no_call_chain`. Wall ~35 min mid (Value strike-by-strike EOD).

## vs prior books

| Book | notes |
|------|--------|
| BW BS mid + earn (prior) | n=1103 E +2.22 PF 2.20 — **fantasy**; all BS |
| Wheel mid + earn | n=686 E +1.46 PF 2.04 — still best mid research book |
| Wheel ask/bid + earn | n=686 E +0.15 PF 1.08 — listed stress |
| **BW listed ask/bid + earn** | n=794 E +0.26 PF 1.11 — similar stress fail; 2022 red |

Earn on listed BW: mid E **down** slightly (1.10→1.02), ask/bid **worse** (0.45→0.26). Not a promote lever here (unlike the BS run).

## Value-plan notes

- `greeks/eod` → 403 on VALUE (needs Standard).
- Working path: expirations + strikes + `history/eod` bid/ask; ATM pick (`strike_mode=atm`).
- 1m quote/OHLC also work on VALUE; unused for this hold-to-expiry book.

## Promote

**No promote.** Listed buy-write does not beat wheel mid, and natural fills fail the 2022 gate. Do not wire `/hot`. Keep Value only if you want more listed option research; cancel if unused.

## Artifacts

- `reports/ascending_channels/2026-11-03_cc_earnings_listed/`
- Runner: `scripts/research/claimcheck_last_15m_cc_earnings.py --listed-only --buy-write-only`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_last_15m_cc_earnings.py --buy-write-only --listed-only --outdir reports\ascending_channels\2026-11-03_cc_earnings_listed
```
