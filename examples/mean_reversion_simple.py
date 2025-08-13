import argparse
from pathlib import Path
import pandas as pd
import backtrader as bt
import numpy as np

# -----------------------------
# Simple Mean Reversion Strategy
# -----------------------------
class MeanReversionStrategy(bt.Strategy):
    params = dict(
        lookback=20,        # Period for moving average
        std_dev=2.0,        # Standard deviations for entry/exit
        size=1,             # Position size
        commission=0.001,   # 0.1%
        printlog=True,
    )

    def log(self, txt):
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"{dt} {txt}")

    def __init__(self):
        # Calculate moving average and standard deviation
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.lookback)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.lookback)
        
        # Upper and lower bands
        self.upper_band = self.sma + (self.std * self.p.std_dev)
        self.lower_band = self.sma - (self.std * self.p.std_dev)
        
        # Track trades
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0

    def next(self):
        if not self.position:  # No position
            # Buy when price is below lower band (oversold)
            if self.data.close[0] < self.lower_band[0]:
                self.buy(size=self.p.size)
                self.log(f"BUY {self.p.size} @ {self.data.close[0]:.2f}")
                self.trade_count += 1
                
        else:  # Have position
            # Sell when price is above upper band (overbought)
            if self.data.close[0] > self.upper_band[0]:
                self.close()
                self.log(f"SELL {self.p.size} @ {self.data.close[0]:.2f}")
                
                # Calculate P&L for this trade
                trade_pnl = (self.data.close[0] - self.broker.getposition(self.data).price) * self.p.size
                self.total_pnl += trade_pnl
                
                if trade_pnl > 0:
                    self.win_count += 1
                    self.log(f"TRADE WON: P&L = ${trade_pnl:.2f}")
                else:
                    self.log(f"TRADE LOST: P&L = ${trade_pnl:.2f}")

    def stop(self):
        # Print final results
        final_value = self.broker.getvalue()
        initial_value = self.broker.startingcash
        total_return = (final_value - initial_value) / initial_value * 100
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Net Profit: ${final_value - initial_value:,.2f}")
        print(f"Total Trades: {self.trade_count}")
        print(f"Winning Trades: {self.win_count}")
        print(f"Win Rate: {(self.win_count/self.trade_count*100):.1f}%" if self.trade_count > 0 else "Win Rate: 0%")
        print(f"Total P&L: ${self.total_pnl:.2f}")
        print("="*50)

# -----------------------------
# PandasData feed wrapper for Parquet
# -----------------------------
class Parquet1hPandas(bt.feeds.PandasData):
    params = (
        ('datetime', None),  # Use index as datetime
        ('open',     'open'),
        ('high',     'high'),
        ('low',      'low'),
        ('close',    'close'),
        ('volume',   'volume'),
        ('openinterest', None),
    )

def load_parquet_1h(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    # Convert to naive UTC datetimes for Backtrader
    ts = pd.to_datetime(df['ts_event'], unit='ns', utc=True).dt.tz_localize(None)
    out = pd.DataFrame({
        'datetime': ts,
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'volume': df['volume'].astype(float),
    }).sort_values('datetime').reset_index(drop=True)
    out.set_index('datetime', inplace=True)
    return out

# -----------------------------
# Main Runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=str, required=True, help='Path to 1h Parquet file')
    ap.add_argument('--cash', type=float, default=100000.0, help='Starting cash')
    ap.add_argument('--lookback', type=int, default=20, help='Moving average period')
    ap.add_argument('--std', type=float, default=2.0, help='Standard deviations for bands')
    ap.add_argument('--size', type=int, default=1, help='Position size')
    ap.add_argument('--fromdate', type=str, default='2018-01-01', help='Start date')
    ap.add_argument('--todate', type=str, default='2024-12-31', help='End date')
    args = ap.parse_args()

    # Load data
    df_1h = load_parquet_1h(Path(args.parquet))
    df_1h = df_1h.loc[(df_1h.index >= pd.to_datetime(args.fromdate)) & 
                       (df_1h.index <= pd.to_datetime(args.todate))]

    data_1h = Parquet1hPandas(dataname=df_1h)

    # Setup Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addtz('UTC')
    cerebro.adddata(data_1h)

    # Add strategy
    cerebro.addstrategy(
        MeanReversionStrategy,
        lookback=args.lookback,
        std_dev=args.std,
        size=args.size,
        printlog=True,
    )

    # Run backtest
    print(f"Running Mean Reversion Strategy...")
    print(f"Data: {args.parquet}")
    print(f"Period: {args.fromdate} to {args.todate}")
    print(f"Parameters: Lookback={args.lookback}, StdDev={args.std}, Size={args.size}")
    print("-" * 50)

    try:
        cerebro.run()
    except Exception as e:
        print(f"Error during backtest: {e}")
        return

if __name__ == '__main__':
    main()
