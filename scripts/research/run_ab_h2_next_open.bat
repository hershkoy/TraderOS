@echo off
cd /d "%~dp0..\.."
set PYTHONPATH=.
set PYTHONUNBUFFERED=1
if not exist logs\research mkdir logs\research
set LOG=logs\research\ab_h2_2x2.log

echo ===== B12 close-fill ===== > %LOG%
venv\Scripts\python.exe scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --workers 4 --load-workers 8 >> %LOG% 2>&1
if errorlevel 1 exit /b 1

echo ===== B0 close-fill ===== >> %LOG%
venv\Scripts\python.exe scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --touch-error-pct 0 --workers 4 --load-workers 8 >> %LOG% 2>&1
if errorlevel 1 exit /b 1

echo ===== B12 next-open ===== >> %LOG%
venv\Scripts\python.exe scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --realistic-fill --realistic-fill-mode next-open --workers 4 --load-workers 8 >> %LOG% 2>&1
if errorlevel 1 exit /b 1

echo ===== B0 next-open ===== >> %LOG%
venv\Scripts\python.exe scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --realistic-fill --realistic-fill-mode next-open --touch-error-pct 0 --workers 4 --load-workers 8 >> %LOG% 2>&1
if errorlevel 1 exit /b 1

echo ===== DONE ===== >> %LOG%
exit /b 0
