@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting channel-touch 15m n=300 at %DATE% %TIME% > logs\channel_touch_15m_n300.log
venv\Scripts\python.exe scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --workers 4 --load-workers 8 >> logs\channel_touch_15m_n300.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\channel_touch_15m_n300.log
echo Finished at %DATE% %TIME% >> logs\channel_touch_15m_n300.log
