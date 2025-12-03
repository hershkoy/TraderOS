@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

REM Create log file with timestamp (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\daily_scanner_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting Daily Scanner...
echo Logging to: %logfile%
echo Current directory: %CD%
echo.

python scripts\scanners\daily_scanner.py --output-report >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: Daily scanner failed with exit code %ERRORLEVEL%
) else (
    echo Daily scanner completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo.
pause
