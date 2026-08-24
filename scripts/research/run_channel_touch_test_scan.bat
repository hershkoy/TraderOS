@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting channel-touch test scan at %DATE% %TIME% > logs\channel_touch_test_scan.log
venv\Scripts\python.exe scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --workers 4 --load-workers 8 --start 2018-11-01 --end 2026-08-23 --max-entries-per-day 1 --friction-pct 0.25 >> logs\channel_touch_test_scan.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\channel_touch_test_scan.log
echo Finished at %DATE% %TIME% >> logs\channel_touch_test_scan.log
