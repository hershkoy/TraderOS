@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting channel-touch IB-fallback windowed rerun at %DATE% %TIME% > logs\channel_touch_ib_fallback_windowed.log
venv\Scripts\python.exe scripts\research\backtest_channel_touch_trades.py ^
  --all-symbols --workers 4 --load-workers 8 ^
  --squeeze-adaptive --atr-stop-mult 2.0 ^
  --require-in-channel --max-channel-span-days 365 ^
  --max-entries-per-day 1 --friction-pct 0.25 ^
  --start 2018-11-01 --end 2026-08-27 ^
  --fallback-provider IB --merge-mode prefix ^
  >> logs\channel_touch_ib_fallback_windowed.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\channel_touch_ib_fallback_windowed.log
echo Finished at %DATE% %TIME% >> logs\channel_touch_ib_fallback_windowed.log
