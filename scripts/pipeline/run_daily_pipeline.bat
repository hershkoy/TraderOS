@echo off
REM Options Data Pipeline - Daily Tasks
REM This script should be scheduled to run daily after market close

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%venv\Scripts
set LOG_DIR=%PROJECT_DIR%logs

REM Set environment variables
set PYTHONPATH=%PROJECT_DIR%
set POLYGON_API_KEY=your_polygon_api_key_here

REM Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Get current date for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~2,2%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "MIN=%dt:~10,2%"
set "SEC=%dt:~12,2%"
set "datestamp=%YYYY%%MM%%DD%_%HH%%MIN%%SEC%"

echo [%datestamp%] Starting Options Data Pipeline >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log"

REM Step 1: Discover new contracts (run after market close ~4:00 PM ET)
echo [%datestamp%] Step 1: Discovering contracts... >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\activate.bat" && python "%PROJECT_DIR%scripts\polygon_discover_contracts.py" --underlying QQQ >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

REM Step 2: Ingest EOD quotes (run after market settle ~5:00 PM ET)
echo [%datestamp%] Step 2: Ingesting EOD quotes... >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\activate.bat" && python "%PROJECT_DIR%scripts\polygon_ingest_eod_quotes.py" --underlying QQQ --date %YYYY%-%MM%-%DD% >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

REM Step 3: Optional: Backfill missing Greeks
echo [%datestamp%] Step 3: Backfilling missing Greeks... >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\activate.bat" && python "%PROJECT_DIR%scripts\greeks_fill.py" --start-date %YYYY%-%MM%-%DD% --end-date %YYYY%-%MM%-%DD% --underlying QQQ >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

echo [%datestamp%] Options Data Pipeline completed >> "%LOG_DIR%\daily_pipeline_%YYYY%%MM%%DD%.log"
