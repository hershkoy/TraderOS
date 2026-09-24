"""Summarize XAUSTBreakFree on TV COMEX GC1! 1h (futures proxy)."""
import math
from datetime import datetime

import backtrader as bt
import numpy as np
import pandas as pd

from strategies.xaust_breakfree import XAUSTBreakFree
from utils.db.timescaledb_loader import load_timescaledb_1h


def sharpe_from_equity(equity: pd.Series, bars_per_year: float = 252 * 18) -> float:
    """Rough ann. Sharpe; Globex ~18 hourly bars/day."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * math.sqrt(bars_per_year))


def main():
    df = load_timescaledb_1h(
        "GC1!", "TRADINGVIEW", start_date="2023-01-01", end_date="2026-09-25"
    )
    df = df[~df.index.duplicated(keep="first")]
    print(f"bars={len(df)} {df.index.min()} -> {df.index.max()}")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.set_checksubmit(False)
    data_1h = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_1h)
    cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)
    cerebro.addstrategy(
        XAUSTBreakFree,
        printlog=False,
        use_session_filter=True,  # CME metals hours
    )
    start = cerebro.broker.getvalue()
    st = cerebro.run()[0]
    end = cerebro.broker.getvalue()

    eq = pd.Series(
        [r["value"] for r in st._custom_portfolio_values],
        index=[r["date"] for r in st._custom_portfolio_values],
    )
    eq_2024 = eq[eq.index >= datetime(2024, 1, 1)]
    stats = st.get_trade_statistics()
    trades = st._custom_trades
    n_short = sum(1 for t in trades if t.get("size", 0) < 0)
    n_long = sum(1 for t in trades if t.get("size", 0) > 0)
    wins = sum(1 for t in trades if t["pnl"] > 0)

    print("=== XAUSTBreakFree / COMEX GC1! (TV) ===")
    print(f"start={start:.2f} end={end:.2f} ret={(end/start-1)*100:.2f}%")
    print(f"MDD={stats['max_drawdown_pct']:.2f}%")
    print(f"Sharpe_full={sharpe_from_equity(eq):.3f}")
    print(f"Sharpe_2024plus={sharpe_from_equity(eq_2024):.3f}")
    print(f"trades={len(trades)} long={n_long} short={n_short} wins={wins}")
    if trades:
        pnls = [t["pnl"] for t in trades]
        print(f"sum_closed_pnl={sum(pnls):.2f} mean={np.mean(pnls):.2f}")


if __name__ == "__main__":
    main()
