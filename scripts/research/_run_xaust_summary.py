"""Summarize XAUSTBreakFree GLD backtest: broker equity + Sharpe + trade book."""
import math
from datetime import datetime

import backtrader as bt
import numpy as np
import pandas as pd

from strategies.xaust_breakfree import XAUSTBreakFree
from utils.db.timescaledb_loader import load_timescaledb_1h


def sharpe_from_equity(equity: pd.Series, bars_per_year: float = 252 * 6.5) -> float:
    """Annualized Sharpe from hourly equity (RTH ~6.5 bars/day)."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * math.sqrt(bars_per_year))


def main():
    df = load_timescaledb_1h("GLD", "ALPACA", start_date="2023-10-01", end_date="2026-09-21")
    df = df[~df.index.duplicated(keep="first")]

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.set_checksubmit(False)
    data_1h = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_1h)
    cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)
    cerebro.addstrategy(XAUSTBreakFree, printlog=False)
    start = cerebro.broker.getvalue()
    st = cerebro.run()[0]
    end = cerebro.broker.getvalue()

    eq = pd.Series(
        [r["value"] for r in st._custom_portfolio_values],
        index=[r["date"] for r in st._custom_portfolio_values],
    )
    # Focus 2024-01-01 .. for Jesse target window
    eq_2024 = eq[eq.index >= datetime(2024, 1, 1)]
    stats = st.get_trade_statistics()
    trades = st._custom_trades
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    sides = {}
    for t in trades:
        sides[t.get("type", "?")] = sides.get(t.get("type", "?"), 0) + 1

    print("=== XAUSTBreakFree / GLD 1h proxy ===")
    print(f"bars={len(df)} {df.index.min()} -> {df.index.max()}")
    print(f"start={start:.2f} end={end:.2f} ret={(end/start-1)*100:.2f}%")
    print(f"MDD={stats['max_drawdown_pct']:.2f}%")
    print(f"Sharpe_full={sharpe_from_equity(eq):.3f}")
    print(f"Sharpe_2024plus={sharpe_from_equity(eq_2024):.3f}")
    print(f"trades={len(trades)} wins={len(wins)} losses={len(losses)} sides={sides}")
    if trades:
        pnls = [t["pnl"] for t in trades]
        print(f"sum_pnl={sum(pnls):.2f} mean={np.mean(pnls):.2f}")
        print("sample:")
        for t in trades[:3] + trades[-2:]:
            print(
                f"  {t['entry_date'].date()}->{t['exit_date'].date()} "
                f"px {t['entry_price']:.2f}->{t['exit_price']:.2f} "
                f"size={t['size']} pnl={t['pnl']:.2f}"
            )


if __name__ == "__main__":
    main()
