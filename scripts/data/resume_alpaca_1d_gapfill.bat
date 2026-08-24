@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
if not exist logs mkdir logs
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
echo Starting gap-fill resume at %DATE% %TIME% > logs\alpaca_1d_gapfill_resume.log
venv\Scripts\python.exe utils\data\update_universe_data.py --provider alpaca --timeframe 1d --since 2025-11-01 --universe-file reports\ascending_channels\alpaca_1d_need_update.txt --batch-size 25 --delay-tickers 0.2 --delay-batches 2 >> logs\alpaca_1d_gapfill_resume.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> logs\alpaca_1d_gapfill_resume.log
