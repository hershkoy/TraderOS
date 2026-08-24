@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting channel-touch edge-v2 long-history scan at %DATE% %TIME% > logs\channel_touch_edge_v2.log
venv\Scripts\python.exe scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --edge-v2 --workers 4 --load-workers 8 --start 2018-11-01 --end 2026-08-23 >> logs\channel_touch_edge_v2.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\channel_touch_edge_v2.log
echo Finished at %DATE% %TIME% >> logs\channel_touch_edge_v2.log
