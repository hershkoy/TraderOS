@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
call venv\Scripts\activate.bat
set PYTHONPATH=.
set CACHE=reports\ascending_channels\2026-10-09\last_15m_wheel_cache.jsonl
set OUT=reports\ascending_channels\2026-10-09

echo === mid full pool ===
python scripts\research\claimcheck_last_15m_wheel.py --n 1118 --min-buy-date 2022-01-01 --ensure-years 2022,2023 --min-adv 2000000 --source theta --strike-mode delta --mark-mode mid --cache %CACHE% --outdir %OUT%
if errorlevel 1 exit /b 1

echo === ask_bid full pool ===
python scripts\research\claimcheck_last_15m_wheel.py --n 1118 --min-buy-date 2022-01-01 --ensure-years 2022,2023 --min-adv 2000000 --source theta --strike-mode delta --mark-mode ask_bid --cache %CACHE% --outdir %OUT%
if errorlevel 1 exit /b 1

echo === DONE claimchecks ===
