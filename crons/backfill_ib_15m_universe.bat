@echo off
REM Incremental IB 15m universe backfill (gap from last stored bar to now).
REM IB 15m currently ends ~2025-12-02. Uses client id 8822 (8821 is single-symbol).
REM Resume file: logs\data\ib_15m_universe_resume.txt
REM Inventory only: add --inventory

cd /d %~dp0..
if not exist logs\data mkdir logs\data
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\data\ib_15m_universe_backfill_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting IB 15m universe backfill at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\data\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822 %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
