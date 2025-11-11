import argparse
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

# ---- IB API imports ----
from ib_insync import IB, Stock, RealTimeBar

# ──────────────────────────────────────────────────────────────────────────────
# Settings (edit to taste)
# ──────────────────────────────────────────────────────────────────────────────
SYMBOL = os.getenv("SPX_SCANNER_SYMBOL", "SPY")          # use SPY as SPX proxy
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")             # IB TWS/Gateway host
IB_PORT = int(os.getenv("IB_PORT", "4001"))              # 7496 for TWS paper, 4001 for Gateway
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))       # unique client ID
INITIAL_BARS_TO_PULL = 120                                # minutes of historical data to initialize
PRINT_MATCHED_BARS = True
DEBUG_MODE = False                                       # set via --debug command line flag

# Heikin Ashi + pattern thresholds (tuned for 1m; tweak freely)
BODY_VS_RANGE_DOJI = 0.20        # ≤ 20% body of total range = doji
WICK_MIN_FRAC_DOJI = 0.15        # each wick ≥ 15% of range for doji
WICK_EPS_STRONG = 0.05           # "no wick" side ≤ 5% of range
BODY_MIN_FRAC_STRONG = 0.55      # body ≥ 55% of range
REQUIRE_GROWING = True           # require the 2nd strong bar's body >= 1st
TZ = timezone.utc                # keep everything in UTC

# ──────────────────────────────────────────────────────────────────────────────
# Heikin Ashi helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HA candles from standard OHLC. df index must be datetime and columns open/high/low/close."""
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    # seed HA open with first bar's (O+C)/2, then recursive
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["ha_close"].iloc[i - 1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)

    ha["ha_high"] = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"]  = pd.concat([df["low"],  ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha

@dataclass
class HABar:
    ts: pd.Timestamp
    o: float
    h: float
    l: float
    c: float

    @property
    def range(self) -> float:
        return max(self.h - self.l, 1e-9)

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def upper_wick(self) -> float:
        return self.h - max(self.o, self.c)

    @property
    def lower_wick(self) -> float:
        return min(self.o, self.c) - self.l

    @property
    def is_green(self) -> bool:
        return self.c >= self.o

def is_doji(bar: HABar) -> bool:
    r = bar.range
    if r <= 0:
        return False
    small_body = (bar.body / r) <= BODY_VS_RANGE_DOJI
    wicks_both = (bar.upper_wick / r) >= WICK_MIN_FRAC_DOJI and (bar.lower_wick / r) >= WICK_MIN_FRAC_DOJI
    return small_body and wicks_both

def is_strong_bull(bar: HABar) -> bool:
    r = bar.range
    if r <= 0:
        return False
    no_lower = (bar.lower_wick / r) <= WICK_EPS_STRONG
    big_body = (bar.body / r) >= BODY_MIN_FRAC_STRONG
    return bar.is_green and no_lower and big_body

def is_strong_bear(bar: HABar) -> bool:
    r = bar.range
    if r <= 0:
        return False
    no_upper = (bar.upper_wick / r) <= WICK_EPS_STRONG
    big_body = (bar.body / r) >= BODY_MIN_FRAC_STRONG
    return (not bar.is_green) and no_upper and big_body

def bodies_growing(b1: HABar, b2: HABar) -> bool:
    return b2.body >= b1.body

# ──────────────────────────────────────────────────────────────────────────────
# Historical data for initialization
# ──────────────────────────────────────────────────────────────────────────────
def get_initial_bars(symbol: str, n_minutes: int, ib: IB) -> pd.DataFrame:
    """Fetch initial historical bars to populate the lookback window."""
    try:
        contract = Stock(symbol=symbol, exchange='SMART', currency='USD')
        qualified_contracts = ib.qualifyContracts(contract)
        if not qualified_contracts:
            print(f"Warning: Could not qualify contract for {symbol}")
            return pd.DataFrame(columns=["open","high","low","close"]).set_index(pd.DatetimeIndex([], tz=TZ))
        contract = qualified_contracts[0]
        
        if n_minutes < 1440:
            duration_str = "1 D"
        else:
            days = max(1, int(n_minutes / 1440))
            duration_str = f"{days} D"
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        
        if not bars:
            return pd.DataFrame(columns=["open","high","low","close"]).set_index(pd.DatetimeIndex([], tz=TZ))
        
        df = pd.DataFrame([{
            'timestamp': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close
        } for bar in bars])
        
        if df.empty:
            return pd.DataFrame(columns=["open","high","low","close"]).set_index(pd.DatetimeIndex([], tz=TZ))
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        
        df = df.set_index("timestamp").sort_index()
        df = df[["open","high","low","close"]]
        
        if DEBUG_MODE:
            print(f"DEBUG: Loaded {len(df)} initial bars from {df.index[0]} to {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching initial bars: {e}")
        return pd.DataFrame(columns=["open","high","low","close"]).set_index(pd.DatetimeIndex([], tz=TZ))

# ──────────────────────────────────────────────────────────────────────────────
# Pattern detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_reversal_signals(df_ohlc: pd.DataFrame, check_last_n: int = 5):
    """
    Returns a list of signals. Each signal is:
      {"direction": "bull"|"bear", "t_doji": ts, "t1": ts, "t2": ts}
    
    check_last_n: Check the last N triplets to catch patterns that might have been missed
    """
    if len(df_ohlc) < 3:
        return []

    ha = to_heikin_ashi(df_ohlc)
    merged = pd.concat([df_ohlc, ha], axis=1).dropna()
    signals = []

    def as_bar(row) -> HABar:
        return HABar(ts=row.name, o=row["ha_open"], h=row["ha_high"], l=row["ha_low"], c=row["ha_close"])

    # Check the last N triplets to catch patterns that might have been missed
    # Start from the end and work backwards
    start_idx = max(2, len(merged) - check_last_n)
    for i in range(start_idx, len(merged)):
        b0 = as_bar(merged.iloc[i - 2])
        b1 = as_bar(merged.iloc[i - 1])
        b2 = as_bar(merged.iloc[i])

        # Verify timestamps are sequential (not duplicates)
        if b0.ts >= b1.ts or b1.ts >= b2.ts:
            if DEBUG_MODE:
                print(f"DEBUG: Skipping invalid triplet: {b0.ts} -> {b1.ts} -> {b2.ts}")
            continue

        if not is_doji(b0):
            continue

        # bullish reversal: doji then two strong bulls
        if is_strong_bull(b1) and is_strong_bull(b2):
            if not REQUIRE_GROWING or bodies_growing(b1, b2):
                signals.append({"direction": "bull", "t_doji": b0.ts, "t1": b1.ts, "t2": b2.ts})
            elif DEBUG_MODE:
                print(f"DEBUG: Bull pattern found but bodies not growing: b1.body={b1.body:.4f}, b2.body={b2.body:.4f} @ {b2.ts}")

        # bearish reversal: doji then two strong bears
        if is_strong_bear(b1) and is_strong_bear(b2):
            if not REQUIRE_GROWING or bodies_growing(b1, b2):
                signals.append({"direction": "bear", "t_doji": b0.ts, "t1": b1.ts, "t2": b2.ts})
            elif DEBUG_MODE:
                print(f"DEBUG: Bear pattern found but bodies not growing: b1.body={b1.body:.4f}, b2.body={b2.body:.4f} @ {b2.ts}")

    return signals

# ──────────────────────────────────────────────────────────────────────────────
# Real-time bar handler
# ──────────────────────────────────────────────────────────────────────────────
class RealtimeScanner:
    def __init__(self, symbol: str, ib: IB, initial_bars: pd.DataFrame):
        self.symbol = symbol
        self.ib = ib
        self.bars_df = initial_bars.copy()
        self.last_alert_t2 = None
        self.current_bar = None  # the bar currently being formed
        self.last_completed_time = None  # track last completed bar time to avoid duplicates
        
        # Lock for thread-safe DataFrame updates
        self.lock = threading.Lock()
        
    def on_bar_update(self, bars: list[RealTimeBar], has_new_bar: bool):
        """Handle real-time bar updates."""
        if not bars:
            return
            
        # Get the latest bar
        latest = bars[-1]
        
        # Convert bar time to UTC
        # RealTimeBar.time is already a datetime, convert to pandas Timestamp
        bar_time = pd.to_datetime(latest.time)
        if bar_time.tz is None:
            # If no timezone, assume US/Eastern (IB default)
            bar_time = bar_time.tz_localize("US/Eastern").tz_convert("UTC")
        else:
            # Convert to UTC if not already UTC
            bar_time = bar_time.tz_convert("UTC")
        
        # Round to minute boundary (this is the bar's timestamp)
        bar_minute = bar_time.replace(second=0, microsecond=0)
        
        with self.lock:
            if has_new_bar:
                # New bar started - previous bar is complete
                if self.current_bar is not None:
                    # The completed bar's timestamp is the previous minute
                    completed_time = bar_minute - timedelta(minutes=1)
                    
                    # Check if this is a new completed bar (avoid duplicates)
                    if self.last_completed_time is None or completed_time > self.last_completed_time:
                        # Remove any existing bar at this timestamp (in case of duplicates)
                        if completed_time in self.bars_df.index:
                            self.bars_df = self.bars_df.drop(index=completed_time)
                        
                        # Add completed bar to DataFrame
                        new_row = pd.DataFrame({
                            'open': [self.current_bar['open']],
                            'high': [self.current_bar['high']],
                            'low': [self.current_bar['low']],
                            'close': [self.current_bar['close']]
                        }, index=[completed_time])
                        
                        # Append new bar
                        self.bars_df = pd.concat([self.bars_df, new_row]).sort_index()
                        
                        # Keep only last INITIAL_BARS_TO_PULL + 10 bars
                        if len(self.bars_df) > INITIAL_BARS_TO_PULL + 10:
                            self.bars_df = self.bars_df.iloc[-(INITIAL_BARS_TO_PULL + 10):]
                        
                        self.last_completed_time = completed_time
                        
                        if DEBUG_MODE:
                            print(f"DEBUG: Completed bar @ {completed_time}: O={self.current_bar['open']:.2f} H={self.current_bar['high']:.2f} L={self.current_bar['low']:.2f} C={self.current_bar['close']:.2f}")
                        
                        # Check for patterns immediately when bar completes
                        self.check_patterns()
                    elif DEBUG_MODE:
                        print(f"DEBUG: Skipping duplicate bar @ {completed_time} (last_completed={self.last_completed_time})")
                
                # Start new bar
                self.current_bar = {
                    'open': latest.open_,
                    'high': latest.high,
                    'low': latest.low,
                    'close': latest.close,
                    'time': bar_minute
                }
            else:
                # Same bar - update current bar with latest data
                if self.current_bar:
                    self.current_bar['high'] = max(self.current_bar['high'], latest.high)
                    self.current_bar['low'] = min(self.current_bar['low'], latest.low)
                    self.current_bar['close'] = latest.close
    
    def check_patterns(self):
        """Check for reversal patterns in the current bar set."""
        if len(self.bars_df) < 3:
            return
        
        signals = detect_reversal_signals(self.bars_df)
        
        if signals:
            for sig in signals:
                t2 = sig["t2"]
                # Only alert on new signals
                if self.last_alert_t2 is None or t2 > self.last_alert_t2:
                    self.last_alert_t2 = t2
                    direction = sig["direction"]
                    msg = f"[{self.symbol}] 1m HA reversal {direction.upper()} at {t2.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    print("ALERT", msg)
                    if PRINT_MATCHED_BARS:
                        print(f"  doji @ {sig['t_doji']}, strong#1 @ {sig['t1']}, strong#2 @ {sig['t2']}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser(description="Heikin Ashi Reversal Scanner (Real-Time)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    
    print(f"Caveman HA Reversal Scanner • 1m Real-Time • {SYMBOL} • IB API")
    print("Looking for: doji → two strong HA bars (bullish or bearish).")
    if DEBUG_MODE:
        print("DEBUG mode enabled.")
    
    # Connect to IB
    ib = IB()
    try:
        print(f"Connecting to IB at {IB_HOST}:{IB_PORT} (clientId={IB_CLIENT_ID})...")
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        ib.reqMarketDataType(1)  # request real-time data
        print("Connected to IB successfully.")
    except Exception as e:
        print(f"Failed to connect to IB: {e}")
        print("Make sure TWS or IB Gateway is running and API is enabled.")
        return
    
    try:
        # Get initial historical bars for lookback
        print(f"Loading initial {INITIAL_BARS_TO_PULL} minutes of historical data...")
        initial_bars = get_initial_bars(SYMBOL, INITIAL_BARS_TO_PULL, ib)
        
        if initial_bars.empty:
            print("ERROR: Could not load initial bars. Exiting.")
            return
        
        # Create and qualify contract
        contract = Stock(symbol=SYMBOL, exchange='SMART', currency='USD')
        qualified_contracts = ib.qualifyContracts(contract)
        if not qualified_contracts:
            print(f"ERROR: Could not qualify contract for {SYMBOL}")
            return
        contract = qualified_contracts[0]
        
        # Create scanner
        scanner = RealtimeScanner(SYMBOL, ib, initial_bars)
        
        # Subscribe to real-time bars (1 minute bars)
        print(f"Subscribing to real-time 1-minute bars for {SYMBOL}...")
        print("Scanner is now running. Press Ctrl+C to stop.\n")
        
        # Request real-time bars
        bars = ib.reqRealTimeBars(contract, 5, 'TRADES', False)  # 5 = 1 minute bars
        
        # Set up event handler
        def on_bar_update(bars_list, has_new_bar):
            scanner.on_bar_update(bars_list, has_new_bar)
        
        bars.updateEvent += on_bar_update
        
        # Run the event loop
        ib.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Scanner error: {e}")
        import traceback
        if DEBUG_MODE:
            traceback.print_exc()
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB.")

if __name__ == "__main__":
    main()
