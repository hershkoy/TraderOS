# Last-15m 3x MOC DCA (pause below rail)

**Verdict: no promote.** Splitting the current_best 1d last-15m book into three equal MOC buys does not lift E/PF/geo vs baseline; 2022 stays red. Primary metric is return on **allocated** capital (idle cash from paused tranches = 0%).

## Rules (locked)

- Signal unchanged: last-RTH 15m open at/above daily resist (`current_best` 1d).
- Capital split into 3 equal notionals. Buy at session **close** (MOC) on signal day + next 2 RTH sessions.
- If that close is **below** the projected resist rail → **pause** (skip tranche; do not extend the window).
- Once any shares are held: ATR k=2 hard-stop from **first** fill + 10% trail. Daily low during scale-in can close the position (`scalein_stop` / `scalein_trail`); remaining unfilled tranches cancel.
- After the window (or early stop), remaining shares exit via **15m N+1 mid** (same walker as baseline), trail peak seeded from scale-in highs.
- Occupancy **not** re-walked.

## Results (wall ~22 min load + 90s walk)

| Book | n | E % | PF | geo | 2022 E / PF | 2022-23 E / PF |
|------|--:|----:|---:|----:|-------------:|---------------:|
| baseline last-15m | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.90 / 0.73 |
| DCA allocated | 2295 | +0.03 | 1.01 | 1.03 | −2.26 / 0.31 | −0.77 / 0.73 |
| DCA deployed | 2295 | −0.37 | 0.89 | 0.91 | −3.03 / 0.25 | −1.30 / 0.63 |
| DCA fric0.25 (alloc) | 2295 | −0.47 | 0.84 | 0.86 | −2.76 / 0.25 | −1.27 / 0.61 |

- Tranche fills mean **2.57** / paused mean **0.19** / capital deployed mean **0.86** of allocated.
- Scale-in exits: **264** hard-stop + **96** trail (cut before later MOCs).
- Paired delta (allocated − baseline) mean **−0.23** pp, median **−0.11** pp.
- Skips: 2 `exit_fail` only.

Deployed looks worse than allocated because pauses leave cash idle — measuring only invested dollars still loses to baseline. DCA also blunts avg winners (10.8% → 9.5% allocated) while the 2022 PF falls (0.41 → 0.31).

## Artifacts

- Runner: `scripts/research/overlay_last_15m_moc_dca.py`
- Logic: `utils/research/last_15m_moc_dca.py`
- Tests: `tests/unit/test_last_15m_moc_dca.py`
- CSV: `reports/ascending_channels/2026-09-23/last_15m_moc_dca3_*.csv`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_moc_dca.py --workers 8
```

Number to beat remains last-15m + 15m N+1 mid **n=2297 E +0.27 PF 1.09**. Do not wire MOC DCA into `/hot` or nightly.
