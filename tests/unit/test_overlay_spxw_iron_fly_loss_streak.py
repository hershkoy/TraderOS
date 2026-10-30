"""Smoke tests for iron-fly loss-streak overlay helpers."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from utils.research.last_15m_loss_streak import apply_loss_streak_mask


def _synthetic_natural(n_loss: int = 5, n_win: int = 3) -> pd.DataFrame:
    start = date(2023, 1, 3)
    rows = []
    day = start
    for i in range(n_loss):
        rows.append({"date": day, "pnl": -500.0, "fill": "natural"})
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    for i in range(n_win):
        rows.append({"date": day, "pnl": 400.0, "fill": "natural"})
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["buy_date"] = pd.to_datetime(df["date"])
    df["buy_time"] = df["buy_date"]
    df["gain_pct"] = df["pnl"]
    return df


def test_count_L5_sits_next_signals():
    df = _synthetic_natural(n_loss=5, n_win=3)
    keep, diag = apply_loss_streak_mask(df, L=5, S=3, P=2, mode="count")
    # After 5 losses, next 3 are skipped (sit).
    assert diag["n_sit_episodes"] >= 1
    assert int((~keep).sum()) >= 1


def test_dollar_trip_on_sum():
    df = _synthetic_natural(n_loss=3, n_win=2)
    keep, diag = apply_loss_streak_mask(
        df, L=5, S=2, P=1, mode="dollar", dollar_sum=-1400.0
    )
    # 3 * -500 = -1500 trips dollar rule.
    assert diag["n_sit_episodes"] >= 1
    assert int(diag["n_skipped"]) >= 1
