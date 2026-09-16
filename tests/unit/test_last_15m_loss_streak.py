"""Unit tests for causal last-15m loss-streak sit-out."""
from __future__ import annotations

import pandas as pd

from utils.research.last_15m_loss_streak import (
    apply_loss_streak_mask,
    baseline_loss_streak_diag,
    streak_longer_in_hostile,
)


def _book(gains, start="2021-01-01"):
    rows = []
    t0 = pd.Timestamp(start)
    for i, g in enumerate(gains):
        ts = t0 + pd.Timedelta(days=i)
        rows.append(
            {
                "stock": "AAA",
                "buy_time": ts.strftime("%Y-%m-%d 20:00"),
                "buy_date": ts.strftime("%Y-%m-%d"),
                "gain_pct": g,
            }
        )
    return pd.DataFrame(rows)


def test_five_losers_skip_three_then_probe_two_losers_sit_again():
    # 5 losses -> sit 3 -> probe 2 losses -> sit 3 -> then take rest
    gains = [-1] * 5 + [1, 1, 1] + [-1, -1] + [1, 1, 1] + [2, 2]
    # indices: 0-4 taken (trip), 5-7 skip, 8-9 probe take (lose->sit),
    # 10-12 skip, 13-14 take
    df = _book(gains)
    keep, diag = apply_loss_streak_mask(df, L=5, S=3, P=2, mode="count")
    assert list(keep.astype(int)) == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert diag["n_sit_episodes"] == 2
    assert diag["n_skipped"] == 6


def test_probe_win_resets_to_active():
    gains = [-1] * 5 + [9, 9, 9] + [5] + [-1, -1, -1]
    # 0-4 take trip, 5-7 skip, 8 probe win -> active, 9-11 take
    df = _book(gains)
    keep, diag = apply_loss_streak_mask(df, L=5, S=3, P=2, mode="count")
    assert list(keep.astype(int)) == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert diag["n_sit_episodes"] == 1
    assert diag["n_skipped"] == 3


def test_win_resets_consec_before_trip():
    gains = [-1, -1, -1, -1, 2, -1, -1, -1, -1, -1, 1, 1, 1]
    # four losses, win resets; then five losses trip; skip 3
    df = _book(gains)
    keep, _ = apply_loss_streak_mask(df, L=5, S=3, P=2, mode="count")
    # 0-9 taken (trip on idx 9), 10-12 skip
    assert list(keep.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_dollar_sum_trigger():
    # two large losers sum to -16 <= -15 -> trip; skip 3
    gains = [-8, -8, 1, 1, 1, 2]
    df = _book(gains)
    keep, diag = apply_loss_streak_mask(
        df, L=5, S=3, P=2, mode="dollar", dollar_sum=-15.0
    )
    assert list(keep.astype(int)) == [1, 1, 0, 0, 0, 1]
    assert diag["n_sit_episodes"] == 1
    assert diag["mode"] == "dollar"


def test_baseline_diag_max_consec():
    gains = [1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1]
    df = _book(gains, start="2022-01-01")
    diag = baseline_loss_streak_diag(df)
    full = diag.loc[diag["bucket"] == "FULL"].iloc[0]
    assert int(full["max_consec_loss"]) == 5
    assert int(full["n_runs_ge5"]) == 1


def test_streak_longer_hostile_flag():
    rows = [
        {
            "bucket": "2020-2021",
            "n_trades": 100,
            "n_losses": 50,
            "max_consec_loss": 3,
            "mean_run_len": 1.5,
            "n_runs_ge5": 0,
            "n_runs_ge4": 1,
            "pct_trades_in_run_ge5": 0.0,
        },
        {
            "bucket": "2022-2023",
            "n_trades": 100,
            "n_losses": 70,
            "max_consec_loss": 8,
            "mean_run_len": 2.5,
            "n_runs_ge5": 4,
            "n_runs_ge4": 6,
            "pct_trades_in_run_ge5": 20.0,
        },
    ]
    out = streak_longer_in_hostile(pd.DataFrame(rows))
    assert out["ok"] is True
    assert out["longer_tailed"] is True
