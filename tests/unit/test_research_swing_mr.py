"""Unit tests for swing mean-reversion simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.swing_mr import VARIANT_DUMP_STOCK, VARIANT_RSI_STOCK, simulate_swing_mr


def test_dump_variant_takes_trade_and_exits_at_sma() -> None:
    idx = pd.bdate_range("2020-01-02", periods=80)
    px = np.full(80, 100.0)
    # Gradual so SMA20 ~ 100, then 3-day dump, then recover
    px[40] = 96.0
    px[41] = 93.0
    px[42] = 90.0
    px[43:55] = np.linspace(91.0, 101.0, 12)
    close = pd.DataFrame({"AAA": px}, index=idx)
    open_px = close.shift(1).fillna(100.0)
    low = close * 0.995
    vol = pd.DataFrame(2_000_000.0, index=idx, columns=["AAA"])
    spy = pd.Series(np.linspace(300.0, 350.0, 80), index=idx)

    eq, inv, trades, exits, notes = simulate_swing_mr(
        open_px,
        close,
        low,
        vol,
        spy,
        variant=VARIANT_DUMP_STOCK,
        eval_start="2020-01-15",
        eval_end="2020-04-30",
        max_positions=1,
        hold_days=15,
        stop_loss_pct=0.25,
        cost_bps_rt=10.0,
        liquid_n=1,
        min_price=1.0,
        min_adv=1.0,
        dump_n=3,
        dump_thresh=-0.05,
    )
    assert "variant=dump3_stock" in notes
    assert len(trades) >= 1
    assert float(eq.iloc[-1]) > 0.9
    assert int(inv.sum()) >= 1


def test_rsi_variant_runs_without_error() -> None:
    idx = pd.bdate_range("2020-01-02", periods=60)
    px = np.linspace(80.0, 120.0, 60)
    px[40:43] = [118.0, 110.0, 102.0]
    close = pd.DataFrame({"AAA": px, "BBB": px * 1.01}, index=idx)
    open_px = close.shift(1).bfill()
    low = close * 0.99
    vol = pd.DataFrame(2_000_000.0, index=idx, columns=["AAA", "BBB"])
    spy = pd.Series(np.linspace(200.0, 220.0, 60), index=idx)
    eq, inv, trades, exits, notes = simulate_swing_mr(
        open_px,
        close,
        low,
        vol,
        spy,
        variant=VARIANT_RSI_STOCK,
        eval_start="2020-02-01",
        eval_end="2020-03-31",
        max_positions=2,
        hold_days=5,
        stop_loss_pct=0.10,
        liquid_n=2,
        min_price=1.0,
        min_adv=1.0,
        rsi_period=2,
        rsi_thresh=30.0,
    )
    assert "variant=rsi2_stock" in notes
    assert len(eq) > 10
    assert eq.iloc[0] == 1.0 or abs(float(eq.iloc[0]) - 1.0) < 1e-9
