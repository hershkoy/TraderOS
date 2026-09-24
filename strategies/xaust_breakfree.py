"""
XAU ST BreakFree — Jesse XAUSTBreakFree ported to Backtrader.

Original: 1h Donchian breakout + 4h SuperTrend regime (long and short),
ATR stops/trails, optional ADX gates, 3% risk / 4x notional cap.

Repo data reality:
  - No XAUUSD / GC futures in TimescaleDB.
  - GLD has daily only until 1h is fetched (Alpaca ETF = RTH proxy, not CME Globex).
  - CME session calendar from the Jesse strategy does not apply to GLD;
    use_session_filter stays off by default (Alpaca 1h is already RTH-scoped).
"""
from __future__ import annotations

import math
from datetime import time

import backtrader as bt

from indicators.supertrend import SuperTrend
from utils.backtesting.custom_tracking import CustomTrackingMixin


class XAUSTBreakFree(CustomTrackingMixin, bt.Strategy):
    """1h Donchian breakout filtered by 4h SuperTrend regime (long + short)."""

    params = (
        # Hyperparameters (Jesse defaults after their optimization)
        ("st_period", 24),
        ("st_factor", 4.0),
        ("ema4", 10),
        ("don_period", 85),
        ("atr_stop", 3.0),
        ("trail_atr", 6.5),
        ("exit_on_flip", 0),
        ("l_adx", 10),
        ("s_adx", 15),
        ("s_stop", 4.0),
        ("s_trail", 3.5),
        ("s_tp", 5.0),
        # Risk / broker
        ("risk_pct", 3.0),
        ("max_lev", 4.0),
        ("commission", 0.0002),  # 0.02% per side (Jesse fee)
        ("atr_period", 14),
        ("adx_period", 14),
        ("higher_tf_idx", 1),
        # Session: off for GLD ETF; enable only if running true metals futures
        ("use_session_filter", False),
        ("session_tz", "America/New_York"),
        # Runner compatibility
        ("size", 1),
        ("printlog", False),
        ("log_level", "INFO"),
    )

    def __init__(self):
        super().__init__()
        self.broker.setcommission(commission=self.p.commission)

        d0 = self.data
        d4 = self.datas[self.p.higher_tf_idx]

        # 4h regime stack — read completed bar via [-1] (matches Jesse get_candles[:-1])
        self.st = SuperTrend(d4, period=self.p.st_period, multiplier=self.p.st_factor)
        self.atr4 = bt.indicators.ATR(d4, period=self.p.atr_period)
        self.adx4 = bt.indicators.ADX(d4, period=self.p.adx_period)
        self.ema4 = (
            bt.indicators.EMA(d4.close, period=self.p.ema4) if self.p.ema4 > 0 else None
        )

        # 1h Donchian of PRIOR bars only (exclude current high/low)
        self.don_upper = bt.indicators.Highest(d0.high(-1), period=self.p.don_period)
        self.don_lower = bt.indicators.Lowest(d0.low(-1), period=self.p.don_period)
        self.atr1 = bt.indicators.ATR(d0, period=self.p.atr_period)

        self.order = None
        # Do not name attrs `_stop` / `_tp` — `_stop` shadows bt.Strategy._stop()
        self._stop_px = None
        self._tp_px = None
        self._best = None
        self._entry_side = None  # 'long' | 'short'

    # ----- helpers -----

    def _log(self, msg: str) -> None:
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"{dt.isoformat()} {msg}")

    def _atr4_value(self) -> float:
        """Completed 4h ATR; fall back to 1h ATR * 4 like Jesse."""
        a = float(self.atr4[-1]) if len(self.atr4) > 1 else 0.0
        if a > 0:
            return a
        a1 = float(self.atr1[0]) if len(self.atr1) else 0.0
        return a1 * 4.0 if a1 > 0 else 0.0

    def _st_line(self) -> float:
        if len(self.st) < 2:
            return 0.0
        return float(self.st.trend[-1])

    def _regime(self) -> int:
        """+1 bull, -1 bear, 0 neutral. Price = current 1h close vs completed 4h ST/EMA."""
        line = self._st_line()
        if line <= 0:
            return 0
        close = float(self.data.close[0])
        d = 1 if close > line else -1
        if self.ema4 is not None and len(self.ema4) > 1:
            e = float(self.ema4[-1])
            if d == 1 and close < e:
                return 0
            if d == -1 and close > e:
                return 0
        return d

    def _adx4(self) -> float:
        if len(self.adx4) < 2:
            return 0.0
        return float(self.adx4[-1])

    def _in_session(self) -> bool:
        if not self.p.use_session_filter:
            return True
        # Minimal CME metals-style weekday gate (no holiday calendar ported).
        # Sun 18:00+ ET, Mon-Thu all session hours, Fri until 17:00 — approximate.
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore

        dt = self.data.datetime.datetime(0)
        if dt.tzinfo is None:
            # Alpaca/IB bars in this repo are usually UTC-naive after loader strip
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        et = dt.astimezone(ZoneInfo(self.p.session_tz))
        wd = et.weekday()  # Mon=0 .. Sun=6
        t = et.time()
        if wd == 6:  # Sunday evening open
            return t >= time(18, 0)
        if wd in (0, 1, 2, 3):  # Mon-Thu: closed 17:00-18:00
            return not (time(17, 0) <= t < time(18, 0))
        if wd == 4:  # Friday: open until 17:00
            return t < time(17, 0)
        return False  # Saturday

    def _qty(self, entry: float, stop: float) -> int:
        equity = float(self.broker.getvalue())
        dist = abs(entry - stop)
        if dist <= 0 or entry <= 0 or equity <= 0:
            return 0
        risk_qty = (self.p.risk_pct / 100.0) * equity / dist
        lev_qty = (equity * self.p.max_lev) / entry
        hard_qty = (equity * 0.95) / entry
        q = math.floor(min(risk_qty, lev_qty, hard_qty))
        return max(int(q), 0)

    # ----- signals -----

    def _should_long(self) -> bool:
        if not self._in_session():
            return False
        if self._regime() != 1:
            return False
        if self.p.l_adx > 0 and self._adx4() < self.p.l_adx:
            return False
        if len(self.don_upper) < self.p.don_period:
            return False
        return float(self.data.close[0]) > float(self.don_upper[0])

    def _should_short(self) -> bool:
        if not self._in_session():
            return False
        if self._regime() != -1:
            return False
        if self.p.s_adx > 0 and self._adx4() < self.p.s_adx:
            return False
        if len(self.don_lower) < self.p.don_period:
            return False
        return float(self.data.close[0]) < float(self.don_lower[0])

    # ----- entries / exits -----

    def _go_long(self) -> None:
        entry = float(self.data.close[0])
        atr4 = self._atr4_value()
        if atr4 <= 0:
            return
        stop = entry - atr4 * self.p.atr_stop
        qty = self._qty(entry, stop)
        if qty <= 0:
            return
        self._stop_px = stop
        self._tp_px = None
        self._best = entry
        self._entry_side = "long"
        self.order = self.buy(size=qty)
        self.track_trade_entry(entry, qty)
        self._log(f"LONG qty={qty} entry~{entry:.4f} stop={stop:.4f}")

    def _go_short(self) -> None:
        entry = float(self.data.close[0])
        atr4 = self._atr4_value()
        if atr4 <= 0:
            return
        stop = entry + atr4 * self.p.s_stop
        qty = self._qty(entry, stop)
        if qty <= 0:
            return
        self._stop_px = stop
        self._tp_px = (
            entry - atr4 * self.p.s_tp if self.p.s_tp > 0 else None
        )
        self._best = entry
        self._entry_side = "short"
        self.order = self.sell(size=qty)
        self.track_trade_entry(entry, -qty)
        self._log(f"SHORT qty={qty} entry~{entry:.4f} stop={stop:.4f} tp={self._tp_px}")

    def _update_open(self) -> None:
        close = float(self.data.close[0])
        high = float(self.data.high[0])
        low = float(self.data.low[0])
        atr4 = self._atr4_value()

        if self.position.size > 0:
            if self.p.exit_on_flip == 1 and self._regime() != 1:
                self._log("LONG exit on regime flip")
                self.order = self.close()
                return
            self._best = max(self._best or close, close)
            if atr4 > 0:
                trail = self._best - atr4 * self.p.trail_atr
                if self._stop_px is None or (self._stop_px < trail < close):
                    self._stop_px = trail
            if self._stop_px is not None and low <= self._stop_px:
                self._log(f"LONG stop hit @ {self._stop_px:.4f}")
                self.order = self.close()
                return

        elif self.position.size < 0:
            if self.p.exit_on_flip == 1 and self._regime() != -1:
                self._log("SHORT exit on regime flip")
                self.order = self.close()
                return
            self._best = min(self._best or close, close)
            if atr4 > 0:
                trail = self._best + atr4 * self.p.s_trail
                if self._stop_px is None or (close < trail < self._stop_px):
                    self._stop_px = trail
            if self._tp_px is not None and low <= self._tp_px:
                self._log(f"SHORT TP hit @ {self._tp_px:.4f}")
                self.order = self.close()
                return
            if self._stop_px is not None and high >= self._stop_px:
                self._log(f"SHORT stop hit @ {self._stop_px:.4f}")
                self.order = self.close()
                return

    def next(self):
        self.track_portfolio_value()

        if self.order:
            return

        if self.position:
            self._update_open()
            return

        # Flat — look for entries
        self._stop_px = None
        self._tp_px = None
        self._best = None
        self._entry_side = None

        if self._should_long():
            self._go_long()
        elif self._should_short():
            self._go_short()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Rejected, order.Margin):
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            # Closing notify often has trade.size == 0; use tracked entry size.
            size = self._custom_entry_size
            if (
                not size
                and self._custom_entry_price
                and trade.price != self._custom_entry_price
            ):
                size = int(round(trade.pnl / (trade.price - self._custom_entry_price)))
            if self._custom_entry_price is not None:
                self.track_trade_exit(float(trade.price), size)
            self._stop_px = None
            self._tp_px = None
            self._best = None
            self._entry_side = None

    @staticmethod
    def get_data_requirements():
        return {
            "base_timeframe": "hourly",
            "additional_timeframes": ["4h"],
            "requires_resampling": False,
        }

    @staticmethod
    def get_description():
        return (
            "Jesse XAUSTBreakFree port: 1h Donchian breakout + 4h SuperTrend "
            "regime, long/short, ATR trail (GLD ETF proxy until XAU futures exist)"
        )
