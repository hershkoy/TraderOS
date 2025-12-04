# strategies/vcp_avwap_breakout.py
# Playbook #1: 52w-high + VCP base + AVWAP trigger (1h entries, daily/weekly context)
# This strategy expects the runner to provide 3 feeds in this order:
#   0: 1h (base), 1: Daily (resampled), 2: Weekly (resampled)
# It uses a local AVWAP implementation and a simple VCP base detector.

import backtrader as bt
from datetime import datetime, timedelta

# Import the custom tracking mixin from the utils package
from utils.backtesting.custom_tracking import CustomTrackingMixin


class VcpAvwapBreakoutStrategy(CustomTrackingMixin, bt.Strategy):
    params = dict(
        # Base/VCP detection (daily)
        vcp_min_weeks=6,
        vcp_max_weeks=12,
        vcp_max_depth_pct=25.0,
        vcp_min_contractions=2,
        vcp_volume_dryup_window=20,
        zigzag_pct=3.0,  # swing detection threshold (percent)
        # 52w
        near_52w_threshold=0.92,
        # AVWAP
        avwap_anchor='base_start',  # or 'last_swing_low' (future)
        # Intraday RVOL (1h)
        rvol_window_days=20,
        bars_per_day=7,       # approximate 1h bars per session for US equities
        breakout_rvol_min=1.5,
        # Stops & exits
        atr_period_days=14,
        atr_mult_stop=1.0,
        take_profit_ext_mult=0.25,  # 25% above AVWAP triggers partial
        weekly_ma_period=10,        # 10-week trend filter
        # Backtrader / misc
        size=1,
        commission=0.001,
        printlog=True,
        log_level="DEBUG",
    )

    def log(self, txt, level='INFO'):
        if self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{timestamp}] {self.datetime.datetime(0)} [{level}] {txt}')

    def debug_log(self, txt):
        if self.p.log_level.upper() == 'DEBUG' and self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{timestamp}] {self.datetime.datetime(0)} [DEBUG] {txt}')

    @staticmethod
    def get_data_requirements():
        """Runner uses this to attach/resample data feeds."""
        return {
            'base_timeframe': 'hourly',
            'additional_timeframes': ['daily', 'weekly'],
            'requires_resampling': True,
        }

    @staticmethod
    def get_description():
        return "52w-high + VCP base + AVWAP trigger (1h entries, daily/weekly context)"

    # -----------------------
    # Utility: Anchored VWAP
    # -----------------------
    class _AnchoredVWAP(bt.Indicator):
        lines = ('avwap', )
        params = dict(
            anchor_dt=None,  # python datetime
            use_typical=True,
        )

        def __init__(self):
            self.addminperiod(1)
            self._cum_pv = 0.0
            self._cum_v = 0.0
            self._started = False

        def next(self):
            dt = self.data.datetime.datetime(0)
            price = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0 if self.p.use_typical else self.data.close[0]
            vol = float(self.data.volume[0] or 0.0)

            if self.p.anchor_dt is None:
                self.lines.avwap[0] = float('nan')
                return

            # Start accumulation on/after the first bar at/after anchor_dt
            if not self._started:
                if dt >= self.p.anchor_dt:
                    self._started = True
                else:
                    self.lines.avwap[0] = float('nan')
                    return

            self._cum_pv += price * vol
            self._cum_v += vol
            self.lines.avwap[0] = (self._cum_pv / self._cum_v) if self._cum_v > 0 else float('nan')

        def reset_anchor(self, anchor_dt):
            """Re-anchor and reset accumulation."""
            self.p.anchor_dt = anchor_dt
            self._cum_pv = 0.0
            self._cum_v = 0.0
            self._started = False

    # -----------------------
    # Lifecycle
    # -----------------------
    def __init__(self):
        # init tracking mixin
        super().__init__()

        # Feeds (runner guarantees order)
        self.h1 = self.datas[0]  # 1h
        self.d1 = self.datas[1]  # daily (resampled)
        self.w1 = self.datas[2]  # weekly (resampled)

        # Indicators on daily/weekly
        self.highest_252 = bt.ind.Highest(self.d1.high, period=252)
        self.atr_d = bt.ind.ATR(self.d1, period=self.p.atr_period_days)
        self.weekly_ma = bt.ind.SMA(self.w1.close, period=self.p.weekly_ma_period)

        # Placeholders for VCP base
        self.active_base = None  # dict with base metadata
        self._last_vcp_calc_dt = None

        # Anchored VWAP on 1h (anchor will be set when a base is validated)
        self.avwap = self._AnchoredVWAP(self.h1, anchor_dt=None)

        # rVol window length in bars
        self._rvol_window_bars = int(self.p.rvol_window_days * self.p.bars_per_day)

        # Order/position state
        self.order = None
        self.partial_taken = False
        self.current_stop = None

        # Warmup flags
        self._ready = False

        self.debug_log("Initialized strategy")

    # -----------------------
    # Helpers
    # -----------------------
    def _near_52w(self):
        try:
            if self.highest_252[0] == 0:
                return False
            return (self.d1.close[0] / self.highest_252[0]) >= float(self.p.near_52w_threshold)
        except Exception:
            return False

    def _calc_rvol(self):
        """Relative volume on 1h vs trailing average of last N bars."""
        n = self._rvol_window_bars
        if len(self.h1) < max(n, 2):
            return float('nan')
        # compute simple mean of last n volumes excluding current
        total = 0.0
        count = 0
        for i in range(-1, -n-1, -1):
            try:
                v = float(self.h1.volume[i])
            except IndexError:
                break
            total += v
            count += 1
        avg = (total / count) if count > 0 else 0.0
        vnow = float(self.h1.volume[0] or 0.0)
        return (vnow / avg) if avg > 0 else float('nan')

    def _find_vcp_base(self):
        """Detect VCP base on daily data and update self.active_base when valid."""
        # Only compute once per daily bar
        ddt = self.d1.datetime.datetime(0)
        if self._last_vcp_calc_dt is not None and ddt <= self._last_vcp_calc_dt:
            return  # same day
        self._last_vcp_calc_dt = ddt

        # Need sufficient history
        min_days = self.p.vcp_min_weeks * 5
        max_days = self.p.vcp_max_weeks * 5
        if len(self.d1) < max(252, max_days + 10):
            return

        # Take the most recent window between min and max weeks ending yesterday (avoid using today's partial)
        end_idx = -1  # current daily bar index
        # translate to python range indexes
        def get_slice(days):
            try:
                return [float(self.d1.close[i]) for i in range(end_idx - days + 1, end_idx + 1)]
            except IndexError:
                return None

        best_base = None

        for days in range(max_days, min_days - 1, -1):
            closes = get_slice(days)
            highs = None
            lows = None
            vols = None
            if not closes or len(closes) < days:
                continue
            highs = [float(self.d1.high[i]) for i in range(end_idx - days + 1, end_idx + 1)]
            lows = [float(self.d1.low[i]) for i in range(end_idx - days + 1, end_idx + 1)]
            vols = [float(self.d1.volume[i]) for i in range(end_idx - days + 1, end_idx + 1)]

            # ZigZag swings on closes
            swings = self._zigzag_swings(closes, pct=self.p.zigzag_pct / 100.0)
            if len(swings) < 3:
                continue

            # Compute pullback depths (negative percentages)
            pullbacks = []
            for k in range(1, len(swings)):
                prev = swings[k-1]
                curr = swings[k]
                # we only care about pullbacks from swing highs to subsequent swing lows
                if prev['type'] == 'H' and curr['type'] == 'L':
                    depth = (prev['price'] - curr['price']) / prev['price'] * 100.0
                    pullbacks.append(-abs(depth))

            if len(pullbacks) < self.p.vcp_min_contractions:
                continue

            # Strict contraction check
            ok_contract = all(abs(pullbacks[i]) > abs(pullbacks[i+1]) for i in range(len(pullbacks)-1))
            if not ok_contract:
                continue

            peak = max(highs)
            trough = min(lows)
            depth_pct = (peak - trough) / peak * 100.0
            if depth_pct > self.p.vcp_max_depth_pct:
                continue

            # Volume dry-up across base window (20-day SMA volume decreasing)
            dry_ok = self._volume_dryup(vols, window=self.p.vcp_volume_dryup_window)
            if not dry_ok:
                continue

            pivot_price = peak
            start_dt = self.d1.datetime.datetime(end_idx - days + 1)
            end_dt = self.d1.datetime.datetime(end_idx)

            base = dict(
                start_idx=end_idx - days + 1,
                end_idx=end_idx,
                start_dt=start_dt,
                end_dt=end_dt,
                pivot_price=pivot_price,
                pullbacks=pullbacks,
                depth_pct=depth_pct,
                valid=True,
            )
            best_base = base
            break  # take the first (longest) valid base

        if best_base:
            self.active_base = best_base
            # Reset AVWAP from base start
            if self.p.avwap_anchor == 'base_start':
                anchor_dt = self._align_to_h1_start(best_base['start_dt'])
                self.avwap.reset_anchor(anchor_dt)
            self.partial_taken = False
            self.debug_log(f"VCP VALID | start={best_base['start_dt']} end={best_base['end_dt']} pivot={best_base['pivot_price']:.2f} depth={best_base['depth_pct']:.1f}%")
        else:
            # Keep previous base until invalidated by a big drawdown or time
            self.debug_log("No valid VCP base found today")

    def _align_to_h1_start(self, dt):
        """Align a daily date to the first 1h bar at/after that date."""
        # Assuming 1h data contains bars for market hours only; just return dt at 00:00 which will advance to first bar >= dt
        return datetime(dt.year, dt.month, dt.day)

    def _volume_dryup(self, vols, window=20):
        if len(vols) < window * 2:
            return True  # not enough to judge → don't block
        import statistics
        first = statistics.mean(vols[:window])
        last = statistics.mean(vols[-window:])
        return last < first

    def _zigzag_swings(self, closes, pct=0.03):
        """Return list of swings: [{'idx':i,'price':p,'type':'H'|'L'}...] using simple percent threshold."""
        swings = []
        if not closes:
            return swings
        last_pivot_idx = 0
        last_pivot_price = closes[0]
        trend = None  # 'up' or 'down'

        for i in range(1, len(closes)):
            change = (closes[i] - last_pivot_price) / last_pivot_price
            if trend in (None, 'down'):
                # look for upswing
                if change >= pct:
                    # pivot low at last_pivot_idx
                    swings.append({'idx': last_pivot_idx, 'price': last_pivot_price, 'type': 'L'})
                    trend = 'up'
                    # reset pivot
                    last_pivot_idx = i
                    last_pivot_price = closes[i]
                elif change <= -pct:
                    # continue down: update pivot
                    last_pivot_idx = i
                    last_pivot_price = closes[i]
            if trend in (None, 'up'):
                # look for downswing
                if change <= -pct:
                    swings.append({'idx': last_pivot_idx, 'price': last_pivot_price, 'type': 'H'})
                    trend = 'down'
                    last_pivot_idx = i
                    last_pivot_price = closes[i]
                elif change >= pct:
                    last_pivot_idx = i
                    last_pivot_price = closes[i]
        # Close with final swing of current trend
        swings.append({'idx': last_pivot_idx, 'price': last_pivot_price, 'type': 'H' if trend == 'up' else 'L'})
        return swings

    # -----------------------
    # Backtrader callbacks
    # -----------------------
    def notify_order(self, order):
        try:
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(f'BUY EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                    # track entry
                    try:
                        self.track_trade_entry(order.executed.price, order.executed.size)
                    except Exception:
                        pass
                elif order.issell():
                    self.log(f'SELL EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                    try:
                        self.track_trade_exit(order.executed.price, order.executed.size)
                    except Exception:
                        pass

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"ORDER {order.getstatusname()}")

            # Clear reference to allow new orders
            if self.order is order:
                self.order = None
        except Exception as e:
            self.debug_log(f'notify_order error: {e}')

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            pnl_pct = (pnl / (trade.price * abs(trade.size))) * 100 if trade.price else 0.0
            self.log(f'TRADE CLOSED, pnl={pnl:.2f} ({pnl_pct:.2f}%)')

    def start(self):
        # Apply commission if broker supports it
        try:
            self.broker.setcommission(commission=self.p.commission)
        except Exception:
            pass

    def next(self):
        # Always track portfolio value
        try:
            self.track_portfolio_value()
        except Exception:
            pass

        # Warmup readiness
        if not self._ready:
            if len(self.d1) >= max(252, self.p.atr_period_days + 5) and len(self.w1) >= self.p.weekly_ma_period:
                self._ready = True
                self.debug_log("Warmup complete")
            else:
                return  # not enough data yet

        # Recompute VCP base once per daily bar
        self._find_vcp_base()

        # Compute inputs
        near52 = self._near_52w()
        rvol = self._calc_rvol()
        avwap_val = float(self.avwap.avwap[0]) if len(self.avwap) > 0 else float('nan')
        close_1h = float(self.h1.close[0])

        # Exits first (if in position)
        if self.position:
            # Weekly trend exit
            try:
                if len(self.w1) >= self.p.weekly_ma_period and self.w1.close[0] < self.weekly_ma[0]:
                    self.log("Weekly close below 10-week MA → EXIT ALL")
                    self.order = self.close()  # market on next bar
                    self.current_stop = None
                    self.partial_taken = True  # avoid partials after full exit
                    return
            except Exception as e:
                self.debug_log(f"Weekly exit check error: {e}")

            # Profit-taking partial
            try:
                if not self.partial_taken and avwap_val == avwap_val and close_1h >= avwap_val * (1.0 + self.p.take_profit_ext_mult):
                    size_to_sell = min(self.position.size, self.p.size)
                    if size_to_sell > 0:
                        self.log(f"Partial take profit at {close_1h:.2f}")
                        self.sell(size=size_to_sell)  # market next bar
                        self.partial_taken = True
            except Exception as e:
                self.debug_log(f"Partial take error: {e}")

            # AVWAP loss of control → tighten stop
            try:
                if avwap_val == avwap_val and close_1h < avwap_val and rvol == rvol and rvol >= 1.0:
                    new_stop = max(self.current_stop or -1e9, avwap_val - 0.5 * float(self.atr_d[0] or 0.0))
                    if self.current_stop is None or new_stop > self.current_stop:
                        # Cancel previous stop and place a new one
                        if self.current_stop is not None and self.order:
                            try:
                                self.broker.cancel(self.order)
                            except Exception:
                                pass
                        self.current_stop = new_stop
                        self.log(f"Tighten stop to {self.current_stop:.2f} (AVWAP control lost)")
                        self.order = self.sell(exectype=bt.Order.Stop, price=self.current_stop, size=self.position.size)
            except Exception as e:
                self.debug_log(f"Stop tighten error: {e}")

        # Entries (only if flat)
        if not self.position and self.order is None:
            conds = []
            cond_near = near52
            conds.append(('near_52w', cond_near))

            cond_base = bool(self.active_base and self.active_base.get('valid', False))
            conds.append(('vcp_valid', cond_base))

            cond_rvol = bool(rvol == rvol and rvol >= self.p.breakout_rvol_min)
            conds.append(('rvol_ok', cond_rvol))

            cond_pivot = False
            if cond_base:
                pivot = float(self.active_base['pivot_price'])
                cond_pivot = close_1h > pivot
            conds.append(('pivot_break', cond_pivot))

            cond_avwap = (avwap_val == avwap_val) and (close_1h > avwap_val)
            conds.append(('above_avwap', cond_avwap))

            all_ok = all(flag for name, flag in conds)
            self.debug_log("ENTRY CHECK → " + ", ".join([f"{n}={v}" for n, v in conds]) + f" | rvol={rvol:.2f if rvol==rvol else float('nan')} avwap={avwap_val:.2f if avwap_val==avwap_val else float('nan')}")

            if all_ok:
                # Entry
                self.log(f"BUY SIGNAL → price={close_1h:.2f} pivot={self.active_base['pivot_price']:.2f} avwap={avwap_val:.2f} rvol={rvol:.2f}")
                self.order = self.buy(size=self.p.size)

                # Initial stop using daily ATR
                try:
                    atrd = float(self.atr_d[0] or 0.0)
                    self.current_stop = close_1h - self.p.atr_mult_stop * atrd
                    self.sell(exectype=bt.Order.Stop, price=self.current_stop, size=self.p.size)
                    self.log(f"Initial stop placed at {self.current_stop:.2f}")
                except Exception as e:
                    self.debug_log(f"Stop placement error: {e}")

    # Optional: stop() to print summary
    def stop(self):
        try:
            stats = self.get_trade_statistics()
            self.log(f"SUMMARY: total_trades={stats.get('total_trades')} win_rate={stats.get('win_rate'):.1f}% profit_factor={stats.get('profit_factor'):.2f}", level='INFO')
        except Exception:
            pass
