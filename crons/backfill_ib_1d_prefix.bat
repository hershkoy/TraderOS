@echo off
REM IB daily prefix backfill back to 2006-01-01 for ALPACA names that start late.
REM Client 8827. Do not run during US RTH (live 15m is client 8826).
REM Stop file: logs\data\ib_1d_prefix.stop
REM Replaces the 5m universe fill until this prefix is done.
REM
REM   crons\backfill_ib_1d_prefix.bat
REM   crons\backfill_ib_1d_prefix.bat --limit 3 --dry-run
REM   crons\stop_ib_5m_backfill.bat

cd /d %~dp0..
if not exist logs\data mkdir logs\data

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

REM Leftover 09:15 ET stop file would exit immediately; watchdog/restart must clear it.
if exist logs\data\ib_1d_prefix.stop del /f /q logs\data\ib_1d_prefix.stop

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set datetime=%%I
set logfile=logs\data\ib_1d_prefix_backfill_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting IB 1d prefix backfill at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\data\backfill_ib_daily_prefix.py --sleep 1 --client-id 8827 %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
