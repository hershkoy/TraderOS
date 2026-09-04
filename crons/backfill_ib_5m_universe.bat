@echo off
REM IB 5m universe backfill for the same names as IB 15m.
REM Resume is TimescaleDB MAX(ts) per symbol (Ctrl+C / stop file / 09:15 ET are safe).
REM Client 8823. Do not run during US RTH (live 15m is client 8826).
REM Stop file: logs\data\ib_5m_universe.stop
REM
REM   crons\backfill_ib_5m_universe.bat
REM   crons\backfill_ib_5m_universe.bat --limit 3 --dry-run
REM   crons\stop_ib_5m_backfill.bat

cd /d %~dp0..
if not exist logs\data mkdir logs\data
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1
set IB_PORT=4001

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\data\ib_5m_universe_backfill_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting IB 5m universe backfill at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\data\backfill_ib_5m_universe.py --sleep 1 --ib-client-id 8823 --ib-port 4001 %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
