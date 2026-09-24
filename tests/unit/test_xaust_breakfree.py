"""Unit tests for XAUSTBreakFree strategy helpers (synthetic 1h panel)."""
from __future__ import annotations

from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd

from strategies.xaust_breakfree import XAUSTBreakFree


def _synth_ohlcv(n: int = 500, start: float = 180.0) -> pd.DataFrame:
    rows = []
    t0 = datetime(2024, 1, 2, 14, 30)
    px = start
    for i in range(n):
        drift = 0.08 if i < n // 2 else -0.08
        o = px
        c = px + drift
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        rows.append(
            {
                "datetime": t0 + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0,
            }
        )
        px = c
    return pd.DataFrame(rows).set_index("datetime")


def test_strategy_runs_on_synthetic_1h_4h():
    df = _synth_ohlcv(600)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(100_000.0)
    data_1h = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_1h)
    cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)

    class _Capture(XAUSTBreakFree):
        def next(self):
            self.track_portfolio_value()
            # Record every bar after warmup so multi-TF clock sync still hits
            if len(self) > 100:
                self._last_regime = self._regime()
                self._last_qty = self._qty(
                    float(self.data.close[0]), float(self.data.close[0]) - 1.0
                )

    cerebro.addstrategy(_Capture, printlog=False, don_period=20, st_period=10, ema4=0)
    st = cerebro.run()[0]
    assert getattr(st, "_last_regime", None) in (-1, 0, 1)
    assert getattr(st, "_last_qty", 0) > 0


def test_qty_risk_math():
    """qty ~= risk_pct * equity / stop_distance (capped by max_lev / cash)."""
    df = _synth_ohlcv(200)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(100_000.0)
    data_1h = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_1h)
    cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)

    class _QtyOnly(XAUSTBreakFree):
        def next(self):
            if len(self) > 30:
                self._q = self._qty(200.0, 194.0)

    cerebro.addstrategy(_QtyOnly, risk_pct=3.0, max_lev=4.0)
    st = cerebro.run()[0]
    # risk wants 500 shares; cash hard-cap 0.95*equity/entry = 475
    assert st._q == 475


def test_data_requirements():
    dr = XAUSTBreakFree.get_data_requirements()
    assert dr["base_timeframe"] == "hourly"
    assert "4h" in dr["additional_timeframes"]


def test_discovery_key():
    from strategies import get_strategy

    assert get_strategy("xaust_breakfree") is XAUSTBreakFree
