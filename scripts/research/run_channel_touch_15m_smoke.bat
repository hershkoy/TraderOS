@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting channel-touch 15m smoke at %DATE% %TIME% > logs\channel_touch_15m_smoke.log
venv\Scripts\python.exe scripts\research\backtest_channel_touch_trades.py --preset 15m --symbols AAPL,MSFT,AMZN,META,TSLA,AMD,GOOGL,NFLX,AVGO,JPM --workers 4 --load-workers 4 --start 2022-01-01 >> logs\channel_touch_15m_smoke.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\channel_touch_15m_smoke.log
echo Finished at %DATE% %TIME% >> logs\channel_touch_15m_smoke.log
