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
# Historical pattern scan
# ──────────────────────────────────────────────────────────────────────────────
def scan_historical_bars(df_ohlc: pd.DataFrame, n_bars: int = 100):
    """
    Scan the last N bars for reversal patterns and report all signals found.
    Returns the most recent signal timestamp (if any) for state initialization.
    """
    if len(df_ohlc) < 3:
        print("INFO: Not enough bars for historical scan (need at least 3)")
        return None
    
    # Get the last N bars
    bars_to_scan = df_ohlc.tail(n_bars)
    
    print(f"\n{'='*70}")
    print(f"SCANNING LAST {len(bars_to_scan)} HISTORICAL BARS FOR PATTERNS")
    print(f"{'='*70}")
    print(f"Time range: {bars_to_scan.index[0]} to {bars_to_scan.index[-1]}")
    print()
    
    # Use the same detection function as live scanning, but scan ALL triplets for historical data
    signals = detect_reversal_signals(bars_to_scan, scan_all=True)
    
    if not signals:
        print("No reversal patterns detected in historical data.")
        print()
        return None
    
    print(f"Found {len(signals)} reversal pattern(s):\n")
    
    last_signal_t2 = None
    for i, sig in enumerate(signals, 1):
        direction = sig["direction"]
        t_doji = sig["t_doji"]
        t1 = sig["t1"]
        t2 = sig["t2"]
        
        print(f"Pattern #{i}: {direction.upper()} reversal")
        print(f"  Doji:     {t_doji.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Strong #1: {t1.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Strong #2: {t2.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()
        
        if last_signal_t2 is None or t2 > last_signal_t2:
            last_signal_t2 = t2
    
    print(f"{'='*70}\n")
    
    return last_signal_t2

# ──────────────────────────────────────────────────────────────────────────────
# Pattern detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_reversal_signals(df_ohlc: pd.DataFrame, check_last_n: int = 5, scan_all: bool = False):
    """
    Returns a list of signals. Each signal is:
      {"direction": "bull"|"bear", "t_doji": ts, "t1": ts, "t2": ts}
    
    check_last_n: Check the last N triplets to catch patterns that might have been missed (used when scan_all=False)
    scan_all: If True, check all triplets in the dataframe (ignores check_last_n)
    """
    if len(df_ohlc) < 3:
        return []

    ha = to_heikin_ashi(df_ohlc)
    merged = pd.concat([df_ohlc, ha], axis=1).dropna()
    signals = []
    
    if DEBUG_MODE:
        print(f"DEBUG: detect_reversal_signals: checking {len(merged)} bars, scan_all={scan_all}, check_last_n={check_last_n}")

    def as_bar(row) -> HABar:
        return HABar(ts=row.name, o=row["ha_open"], h=row["ha_high"], l=row["ha_low"], c=row["ha_close"])

    # Determine which triplets to check
    if scan_all:
        # Check all triplets from index 2 to the end
        start_idx = 2
        end_idx = len(merged)
    else:
        # Check the last N triplets
        start_idx = max(2, len(merged) - check_last_n)
        end_idx = len(merged)
    
    if DEBUG_MODE:
        print(f"DEBUG: Checking triplets from index {start_idx} to {end_idx-1} (total triplets: {end_idx - start_idx})")
    
    # First pass: find all dojis and check each one independently
    # This handles cases where multiple consecutive bars are dojis
    for doji_idx in range(start_idx - 2, end_idx - 2):
        if doji_idx < 0:
            continue
        doji_bar = as_bar(merged.iloc[doji_idx])
        
        if not is_doji(doji_bar):
            continue
        
        if scan_all:
            print(f"INFO: Doji detected @ {doji_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} (historical scan) - looking ahead for strong bars...")
        
        # Found a doji - look ahead for the next two strong bars (skipping any intermediate dojis)
        # Look for bar #1 (first strong bar after doji)
        bar1_idx = None
        bar1_direction = None
        for j in range(doji_idx + 1, len(merged)):
            candidate = as_bar(merged.iloc[j])
            if is_strong_bull(candidate):
                bar1_idx = j
                bar1_direction = "bull"
                if scan_all:
                    print(f"INFO: Bar #1 @ {candidate.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL")
                break
            elif is_strong_bear(candidate):
                bar1_idx = j
                bar1_direction = "bear"
                if scan_all:
                    print(f"INFO: Bar #1 @ {candidate.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR")
                break
            # If we hit another doji, continue looking
            # If we hit a weak bar that's not a doji, stop (pattern broken)
            if not is_doji(candidate):
                if scan_all:
                    r = candidate.range
                    body_pct = (candidate.body / r * 100) if r > 0 else 0
                    print(f"INFO: Bar @ {candidate.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK (body={body_pct:.1f}%, pattern reset)")
                break
        
        if bar1_idx is None:
            if scan_all:
                print(f"INFO: No strong bar found after doji @ {doji_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            continue
        
        # Found bar #1 - now look for bar #2 (second strong bar, same direction)
        bar2_idx = None
        for j in range(bar1_idx + 1, len(merged)):
            candidate = as_bar(merged.iloc[j])
            if bar1_direction == "bull" and is_strong_bull(candidate):
                bar2_idx = j
                break
            elif bar1_direction == "bear" and is_strong_bear(candidate):
                bar2_idx = j
                break
            # If we hit a doji, continue looking
            # If we hit a weak bar or opposite direction, stop (pattern broken)
            if not is_doji(candidate):
                if scan_all:
                    direction_str = "BULL" if bar1_direction == "bull" else "BEAR"
                    print(f"INFO: Bar #2 @ {candidate.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK (expected {direction_str}, pattern reset)")
                break
        
        if bar2_idx is None:
            if scan_all:
                print(f"INFO: No second strong bar found after bar #1 @ {merged.iloc[bar1_idx].name}")
            continue
        
        # Found both strong bars - check if bodies are growing (if required)
        bar1 = as_bar(merged.iloc[bar1_idx])
        bar2 = as_bar(merged.iloc[bar2_idx])
        
        if REQUIRE_GROWING and not bodies_growing(bar1, bar2):
            if scan_all:
                print(f"INFO: Bar #2 @ {bar2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {bar1_direction.upper()} but bodies not growing (pattern reset)")
            if DEBUG_MODE:
                print(f"DEBUG: Pattern found but bodies not growing: doji@{doji_bar.ts}, bar1@{bar1.ts}, bar2@{bar2.ts}")
            continue
        
        # Pattern detected!
        if scan_all:
            print(f"INFO: Bar #2 @ {bar2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {bar1_direction.upper()} - PATTERN DETECTED")
        signals.append({
            "direction": bar1_direction,
            "t_doji": doji_bar.ts,
            "t1": bar1.ts,
            "t2": bar2.ts
        })
        if DEBUG_MODE:
            print(f"DEBUG: {bar1_direction.upper()} pattern detected: doji@{doji_bar.ts}, strong#1@{bar1.ts}, strong#2@{bar2.ts}")
    
    # Second pass: standard triplet check for logging purposes (only if scan_all)
    if scan_all:
        for i in range(start_idx, end_idx):
            b0 = as_bar(merged.iloc[i - 2])
            b1 = as_bar(merged.iloc[i - 1])
            b2 = as_bar(merged.iloc[i])

            # Verify timestamps are sequential (not duplicates)
            if b0.ts >= b1.ts or b1.ts >= b2.ts:
                continue

            # Debug: Log bar characteristics
            if DEBUG_MODE:
                body_pct_b0 = (b0.body / b0.range * 100) if b0.range > 0 else 0
                body_pct_b1 = (b1.body / b1.range * 100) if b1.range > 0 else 0
                body_pct_b2 = (b2.body / b2.range * 100) if b2.range > 0 else 0
                print(f"DEBUG: Checking triplet @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                      f"b0(doji={is_doji(b0)}, body={body_pct_b0:.1f}%) "
                      f"b1(bull={is_strong_bull(b1)}, bear={is_strong_bear(b1)}, body={body_pct_b1:.1f}%) "
                      f"b2(bull={is_strong_bull(b2)}, bear={is_strong_bear(b2)}, body={body_pct_b2:.1f}%)")

            if not is_doji(b0):
                continue

            # Doji found - log it
            # During historical scan, log all dojis found
            print(f"INFO: Doji detected @ {b0.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} (historical scan)")
            # Check what happens with subsequent bars
            if is_strong_bull(b1):
                print(f"INFO: Bar #1 @ {b1.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL")
            elif is_strong_bear(b1):
                print(f"INFO: Bar #1 @ {b1.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR")
            else:
                # Log why bar #1 is not strong
                r1 = b1.range
                body_pct_b1 = (b1.body / r1 * 100) if r1 > 0 else 0
                lower_wick_pct_b1 = (b1.lower_wick / r1 * 100) if r1 > 0 else 0
                upper_wick_pct_b1 = (b1.upper_wick / r1 * 100) if r1 > 0 else 0
                is_green_b1 = b1.is_green
                reason = []
                if not is_green_b1:
                    reason.append("not green")
                if body_pct_b1 < BODY_MIN_FRAC_STRONG * 100:
                    reason.append(f"body too small ({body_pct_b1:.1f}% < {BODY_MIN_FRAC_STRONG*100:.0f}%)")
                if is_green_b1 and lower_wick_pct_b1 > WICK_EPS_STRONG * 100:
                    reason.append(f"has lower wick ({lower_wick_pct_b1:.1f}% > {WICK_EPS_STRONG*100:.0f}%)")
                if not is_green_b1 and upper_wick_pct_b1 > WICK_EPS_STRONG * 100:
                    reason.append(f"has upper wick ({upper_wick_pct_b1:.1f}% > {WICK_EPS_STRONG*100:.0f}%)")
                reason_str = ", ".join(reason) if reason else "unknown"
                print(f"INFO: Bar #1 @ {b1.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK ({reason_str})")
            
            if is_strong_bull(b1) or is_strong_bear(b1):
                # Bar #1 was strong, check bar #2
                if is_strong_bull(b1) and is_strong_bull(b2):
                    if not REQUIRE_GROWING or bodies_growing(b1, b2):
                        print(f"INFO: Bar #2 @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL - PATTERN DETECTED")
                    else:
                        print(f"INFO: Bar #2 @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL but bodies not growing (pattern reset)")
                elif is_strong_bear(b1) and is_strong_bear(b2):
                    if not REQUIRE_GROWING or bodies_growing(b1, b2):
                        print(f"INFO: Bar #2 @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR - PATTERN DETECTED")
                    else:
                        print(f"INFO: Bar #2 @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR but bodies not growing (pattern reset)")
                else:
                    direction_str = "BULL" if is_strong_bull(b1) else "BEAR"
                    print(f"INFO: Bar #2 @ {b2.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK (expected {direction_str}, pattern reset)")

        # bullish reversal: doji then two strong bulls
        if is_strong_bull(b1) and is_strong_bull(b2):
            if not REQUIRE_GROWING or bodies_growing(b1, b2):
                signals.append({"direction": "bull", "t_doji": b0.ts, "t1": b1.ts, "t2": b2.ts})
                if DEBUG_MODE:
                    print(f"DEBUG: BULL pattern detected: doji@{b0.ts}, strong#1@{b1.ts}, strong#2@{b2.ts}")
            elif DEBUG_MODE:
                print(f"DEBUG: Bull pattern found but bodies not growing: b1.body={b1.body:.4f}, b2.body={b2.body:.4f} @ {b2.ts}")

        # bearish reversal: doji then two strong bears
        if is_strong_bear(b1) and is_strong_bear(b2):
            if not REQUIRE_GROWING or bodies_growing(b1, b2):
                signals.append({"direction": "bear", "t_doji": b0.ts, "t1": b1.ts, "t2": b2.ts})
                if DEBUG_MODE:
                    print(f"DEBUG: BEAR pattern detected: doji@{b0.ts}, strong#1@{b1.ts}, strong#2@{b2.ts}")
            elif DEBUG_MODE:
                print(f"DEBUG: Bear pattern found but bodies not growing: b1.body={b1.body:.4f}, b2.body={b2.body:.4f} @ {b2.ts}")

    if DEBUG_MODE:
        print(f"DEBUG: detect_reversal_signals: found {len(signals)} pattern(s)")

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
        
        # Pattern tracking state
        self.pattern_state = None  # None, "armed", "bar1_strong", "waiting_bar3"
        self.pattern_doji_time = None
        self.pattern_bar1_time = None
        self.pattern_bar2_time = None
        self.pattern_bar1_direction = None  # "bull" or "bear" if strong
        
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
                        self.check_patterns(completed_time)
                    # Removed verbose duplicate bar logging - only log at critical level if needed
                
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
    
    def check_patterns(self, completed_time: pd.Timestamp):
        """Check for reversal patterns in the current bar set and log pattern progression."""
        if len(self.bars_df) < 3:
            return
        
        # Get the last few bars for pattern tracking
        ha = to_heikin_ashi(self.bars_df)
        merged = pd.concat([self.bars_df, ha], axis=1).dropna()
        
        if len(merged) < 1:
            return
        
        def as_bar(row) -> HABar:
            return HABar(ts=row.name, o=row["ha_open"], h=row["ha_high"], l=row["ha_low"], c=row["ha_close"])
        
        # Get the most recent completed bar
        last_bar_row = merged.iloc[-1]
        last_bar = as_bar(last_bar_row)
        
        # Debug: Log bar characteristics for diagnosis
        if DEBUG_MODE:
            body_pct = (last_bar.body / last_bar.range * 100) if last_bar.range > 0 else 0
            upper_wick_pct = (last_bar.upper_wick / last_bar.range * 100) if last_bar.range > 0 else 0
            lower_wick_pct = (last_bar.lower_wick / last_bar.range * 100) if last_bar.range > 0 else 0
            print(f"DEBUG: Bar @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                  f"O={last_bar.o:.2f} H={last_bar.h:.2f} L={last_bar.l:.2f} C={last_bar.c:.2f} | "
                  f"body={body_pct:.1f}% upper_wick={upper_wick_pct:.1f}% lower_wick={lower_wick_pct:.1f}% | "
                  f"is_doji={is_doji(last_bar)} is_strong_bull={is_strong_bull(last_bar)} is_strong_bear={is_strong_bear(last_bar)}")
        
        # State machine for pattern tracking
        if self.pattern_state is None:
            # Not tracking - check if this bar or recent bars contain a doji
            # Check the last 3 bars to catch dojis we might have missed
            check_bars = min(3, len(merged))
            for i in range(check_bars):
                check_idx = len(merged) - 1 - i
                if check_idx < 0:
                    break
                check_bar_row = merged.iloc[check_idx]
                check_bar = HABar(
                    ts=check_bar_row.name,
                    o=check_bar_row["ha_open"],
                    h=check_bar_row["ha_high"],
                    l=check_bar_row["ha_low"],
                    c=check_bar_row["ha_close"]
                )
                if is_doji(check_bar):
                    # Found a doji - check if we can still track the pattern
                    # We need at least 2 more bars after the doji to complete the pattern
                    if check_idx + 2 < len(merged):
                        # Pattern already complete, let detect_reversal_signals handle it
                        break
                    elif check_idx + 1 < len(merged):
                        # We have bar #1 after doji, check it
                        bar1_row = merged.iloc[check_idx + 1]
                        bar1 = HABar(
                            ts=bar1_row.name,
                            o=bar1_row["ha_open"],
                            h=bar1_row["ha_high"],
                            l=bar1_row["ha_low"],
                            c=bar1_row["ha_close"]
                        )
                        is_bull = is_strong_bull(bar1)
                        is_bear = is_strong_bear(bar1)
                        if is_bull or is_bear:
                            self.pattern_state = "bar1_strong"
                            self.pattern_doji_time = check_bar.ts
                            self.pattern_bar1_time = bar1.ts
                            self.pattern_bar1_direction = "bull" if is_bull else "bear"
                            print(f"INFO: Doji detected @ {check_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - detector armed (found in lookback)")
                            print(f"INFO: Bar #1 @ {bar1.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {'BULL' if is_bull else 'BEAR'}")
                            break
                        else:
                            # Doji found but bar #1 is weak, start fresh from this doji
                            self.pattern_state = "armed"
                            self.pattern_doji_time = check_bar.ts
                            print(f"INFO: Doji detected @ {check_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - detector armed (found in lookback)")
                            break
                    else:
                        # Doji found, waiting for bar #1
                        self.pattern_state = "armed"
                        self.pattern_doji_time = check_bar.ts
                        print(f"INFO: Doji detected @ {check_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - detector armed")
                        break
        
        elif self.pattern_state == "armed":
            # Waiting for bar #1 after doji - look ahead, skipping any intermediate dojis
            # Find the doji index in merged
            doji_idx = None
            for i in range(len(merged)):
                if merged.index[i] == self.pattern_doji_time:
                    doji_idx = i
                    break
            
            if doji_idx is None:
                self.pattern_state = None
                return
            
            # Look ahead for the first strong bar after the doji (skipping intermediate dojis)
            bar1_idx = None
            bar1_direction = None
            for j in range(doji_idx + 1, len(merged)):
                candidate = as_bar(merged.iloc[j])
                if is_strong_bull(candidate):
                    bar1_idx = j
                    bar1_direction = "bull"
                    break
                elif is_strong_bear(candidate):
                    bar1_idx = j
                    bar1_direction = "bear"
                    break
                # If we hit another doji, continue looking
                # If we hit a weak bar that's not a doji, stop (pattern broken)
                if not is_doji(candidate):
                    break
            
            if bar1_idx is None:
                # No strong bar found yet, check if current bar is the expected next bar
                expected_bar1_time = self.pattern_doji_time + timedelta(minutes=1)
                if last_bar.ts == expected_bar1_time:
                    # Current bar is the immediate next bar - check if it's strong
                    is_bull = is_strong_bull(last_bar)
                    is_bear = is_strong_bear(last_bar)
                    
                    if is_bull:
                        self.pattern_state = "bar1_strong"
                        self.pattern_bar1_time = last_bar.ts
                        self.pattern_bar1_direction = "bull"
                        print(f"INFO: Bar #1 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL")
                    elif is_bear:
                        self.pattern_state = "bar1_strong"
                        self.pattern_bar1_time = last_bar.ts
                        self.pattern_bar1_direction = "bear"
                        print(f"INFO: Bar #1 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR")
                    else:
                        # Check if it's a doji - if so, continue waiting
                        if is_doji(last_bar):
                            if DEBUG_MODE:
                                print(f"DEBUG: Bar @ {last_bar.ts} is also a doji, continuing to look ahead...")
                        else:
                            self.pattern_state = None
                            print(f"INFO: Bar #1 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK (pattern reset)")
                elif last_bar.ts > expected_bar1_time:
                    # We've passed the expected time - check if we skipped dojis
                    # Look back to see if there were dojis we should have skipped
                    found_doji_chain = False
                    for j in range(doji_idx + 1, len(merged)):
                        if merged.index[j] >= last_bar.ts:
                            break
                        candidate = as_bar(merged.iloc[j])
                        if is_doji(candidate):
                            found_doji_chain = True
                        elif not is_doji(candidate) and not is_strong_bull(candidate) and not is_strong_bear(candidate):
                            # Hit a weak bar, pattern broken
                            self.pattern_state = None
                            if DEBUG_MODE:
                                print(f"DEBUG: Skipped bar #1 - expected {expected_bar1_time}, got {last_bar.ts}, pattern broken")
                            return
                    
                    if found_doji_chain:
                        # There were dojis in between, continue looking ahead
                        if DEBUG_MODE:
                            print(f"DEBUG: Found doji chain, continuing to look ahead from {last_bar.ts}")
                    else:
                        # No dojis found, pattern broken
                        self.pattern_state = None
                        if DEBUG_MODE:
                            print(f"DEBUG: Skipped bar #1 - expected {expected_bar1_time}, got {last_bar.ts}, resetting pattern")
            else:
                # Found bar #1 in lookback - check if it's the current bar or we need to wait
                bar1_time = merged.index[bar1_idx]
                if bar1_time == last_bar.ts:
                    # Current bar is bar #1
                    self.pattern_state = "bar1_strong"
                    self.pattern_bar1_time = bar1_time
                    self.pattern_bar1_direction = bar1_direction
                    print(f"INFO: Bar #1 @ {bar1_time.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {bar1_direction.upper()}")
                elif bar1_time < last_bar.ts:
                    # Bar #1 already passed, we missed it - should have been caught earlier
                    # This shouldn't happen, but handle it gracefully
                    self.pattern_state = "bar1_strong"
                    self.pattern_bar1_time = bar1_time
                    self.pattern_bar1_direction = bar1_direction
                    if DEBUG_MODE:
                        print(f"DEBUG: Bar #1 @ {bar1_time} already passed, setting state")
                # else: bar #1 is in the future, continue waiting
        
        elif self.pattern_state == "bar1_strong":
            # Waiting for bar #2 after strong bar #1 - look ahead, skipping any intermediate dojis
            # Find bar #1 index in merged
            bar1_idx = None
            for i in range(len(merged)):
                if merged.index[i] == self.pattern_bar1_time:
                    bar1_idx = i
                    break
            
            if bar1_idx is None:
                self.pattern_state = None
                return
            
            # Look ahead for bar #2 (second strong bar, same direction, skipping intermediate dojis)
            bar2_idx = None
            for j in range(bar1_idx + 1, len(merged)):
                candidate = as_bar(merged.iloc[j])
                if self.pattern_bar1_direction == "bull" and is_strong_bull(candidate):
                    bar2_idx = j
                    break
                elif self.pattern_bar1_direction == "bear" and is_strong_bear(candidate):
                    bar2_idx = j
                    break
                # If we hit a doji, continue looking
                # If we hit a weak bar or opposite direction, stop (pattern broken)
                if not is_doji(candidate):
                    break
            
            if bar2_idx is None:
                # No bar #2 found yet, check if current bar is the expected next bar
                expected_bar2_time = self.pattern_bar1_time + timedelta(minutes=1)
                if last_bar.ts == expected_bar2_time:
                    # Current bar is the immediate next bar - check if it's strong
                    is_bull = is_strong_bull(last_bar)
                    is_bear = is_strong_bear(last_bar)
                    
                    if self.pattern_bar1_direction == "bull" and is_bull:
                        # Check if bodies are growing (if required)
                        if len(merged) >= 2:
                            bar1_row = merged.iloc[-2]
                            bar1 = as_bar(bar1_row)
                            if not REQUIRE_GROWING or bodies_growing(bar1, last_bar):
                                self.pattern_state = "waiting_bar3"
                                self.pattern_bar2_time = last_bar.ts
                                print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL - PATTERN DETECTED")
                            else:
                                self.pattern_state = None
                                print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL but bodies not growing (pattern reset)")
                        else:
                            self.pattern_state = "waiting_bar3"
                            self.pattern_bar2_time = last_bar.ts
                            print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BULL - PATTERN DETECTED")
                    elif self.pattern_bar1_direction == "bear" and is_bear:
                        # Check if bodies are growing (if required)
                        if len(merged) >= 2:
                            bar1_row = merged.iloc[-2]
                            bar1 = as_bar(bar1_row)
                            if not REQUIRE_GROWING or bodies_growing(bar1, last_bar):
                                self.pattern_state = "waiting_bar3"
                                self.pattern_bar2_time = last_bar.ts
                                print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR - PATTERN DETECTED")
                            else:
                                self.pattern_state = None
                                print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR but bodies not growing (pattern reset)")
                        else:
                            self.pattern_state = "waiting_bar3"
                            self.pattern_bar2_time = last_bar.ts
                            print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG BEAR - PATTERN DETECTED")
                    else:
                        # Check if it's a doji - if so, continue waiting
                        if is_doji(last_bar):
                            if DEBUG_MODE:
                                print(f"DEBUG: Bar @ {last_bar.ts} is also a doji, continuing to look ahead for bar #2...")
                        else:
                            self.pattern_state = None
                            direction_str = "BULL" if self.pattern_bar1_direction == "bull" else "BEAR"
                            print(f"INFO: Bar #2 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK (expected {direction_str}, pattern reset)")
                elif last_bar.ts > expected_bar2_time:
                    # We've passed the expected time - check if we skipped dojis
                    found_doji_chain = False
                    for j in range(bar1_idx + 1, len(merged)):
                        if merged.index[j] >= last_bar.ts:
                            break
                        candidate = as_bar(merged.iloc[j])
                        if is_doji(candidate):
                            found_doji_chain = True
                        elif not is_doji(candidate):
                            # Hit a weak bar, pattern broken
                            self.pattern_state = None
                            if DEBUG_MODE:
                                print(f"DEBUG: Skipped bar #2 - expected {expected_bar2_time}, got {last_bar.ts}, pattern broken")
                            return
                    
                    if found_doji_chain:
                        # There were dojis in between, continue looking ahead
                        if DEBUG_MODE:
                            print(f"DEBUG: Found doji chain, continuing to look ahead for bar #2 from {last_bar.ts}")
                    else:
                        # No dojis found, pattern broken
                        self.pattern_state = None
                        if DEBUG_MODE:
                            print(f"DEBUG: Skipped bar #2 - expected {expected_bar2_time}, got {last_bar.ts}, resetting pattern")
            else:
                # Found bar #2 in lookback - check if it's the current bar or we need to wait
                bar2_time = merged.index[bar2_idx]
                bar1_row = merged.iloc[bar1_idx]
                bar2_row = merged.iloc[bar2_idx]
                bar1 = as_bar(bar1_row)
                bar2 = as_bar(bar2_row)
                
                if bar2_time == last_bar.ts:
                    # Current bar is bar #2
                    if not REQUIRE_GROWING or bodies_growing(bar1, bar2):
                        self.pattern_state = "waiting_bar3"
                        self.pattern_bar2_time = bar2_time
                        direction_str = "BULL" if self.pattern_bar1_direction == "bull" else "BEAR"
                        print(f"INFO: Bar #2 @ {bar2_time.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {direction_str} - PATTERN DETECTED")
                    else:
                        self.pattern_state = None
                        direction_str = "BULL" if self.pattern_bar1_direction == "bull" else "BEAR"
                        print(f"INFO: Bar #2 @ {bar2_time.strftime('%Y-%m-%d %H:%M:%S %Z')} - STRONG {direction_str} but bodies not growing (pattern reset)")
                elif bar2_time < last_bar.ts:
                    # Bar #2 already passed, we missed it - should have been caught earlier
                    # This shouldn't happen, but handle it gracefully
                    if not REQUIRE_GROWING or bodies_growing(bar1, bar2):
                        self.pattern_state = "waiting_bar3"
                        self.pattern_bar2_time = bar2_time
                        if DEBUG_MODE:
                            print(f"DEBUG: Bar #2 @ {bar2_time} already passed, setting state")
                # else: bar #2 is in the future, continue waiting
        
        elif self.pattern_state == "waiting_bar3":
            # Waiting for bar #3 after pattern completion (should be exactly 1 minute later)
            expected_bar3_time = self.pattern_bar2_time + timedelta(minutes=1)
            if last_bar.ts == expected_bar3_time:
                is_bull = is_strong_bull(last_bar)
                is_bear = is_strong_bear(last_bar)
                
                if is_bull or is_bear:
                    bar_type = "STRONG BULL" if is_bull else "STRONG BEAR"
                    print(f"INFO: Bar #3 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - {bar_type} - pattern continues")
                else:
                    print(f"INFO: Bar #3 @ {last_bar.ts.strftime('%Y-%m-%d %H:%M:%S %Z')} - WEAK - pattern ends")
                
                # Reset after logging bar #3
                self.pattern_state = None
            elif last_bar.ts > expected_bar3_time:
                # We skipped bar #3, reset
                self.pattern_state = None
        
        # Also check for complete patterns using the existing detection logic
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
        
        # Scan historical bars for patterns (last 100 minutes)
        last_signal_time = scan_historical_bars(initial_bars, n_bars=100)
        
        # Create and qualify contract
        contract = Stock(symbol=SYMBOL, exchange='SMART', currency='USD')
        qualified_contracts = ib.qualifyContracts(contract)
        if not qualified_contracts:
            print(f"ERROR: Could not qualify contract for {SYMBOL}")
            return
        contract = qualified_contracts[0]
        
        # Create scanner
        scanner = RealtimeScanner(SYMBOL, ib, initial_bars)
        
        # Initialize last_alert_t2 with the most recent historical signal (if any)
        # This prevents re-alerting on patterns we've already seen
        if last_signal_time is not None:
            scanner.last_alert_t2 = last_signal_time
            print(f"Initialized scanner: will only alert on patterns after {last_signal_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        
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
