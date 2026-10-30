@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

if not exist "logs\trading\spxw_iron_fly" mkdir "logs\trading\spxw_iron_fly"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set datetime=%%I
set logfile=logs\trading\spxw_iron_fly\combo_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting SPXW iron fly BAG combo...
echo Logging to: %logfile%

set PYTHONPATH=.
set live_arg=
if "%1"=="live-en" set live_arg=--live-en --place

REM Default: dry-run qualify. Pass live-en for live place (also needs --account).
REM Paper place: crons\run_spxw_iron_fly_combo.bat place
REM Park (no transmit): crons\run_spxw_iron_fly_combo.bat park
if "%1"=="place" (
  python scripts\trading\spxw_iron_fly_combo.py --place --wait-sec 120 >> "%logfile%" 2>&1
) else if "%1"=="park" (
  python scripts\trading\spxw_iron_fly_combo.py --place --no-transmit >> "%logfile%" 2>&1
) else if "%1"=="live-en" (
  echo LIVE place requires account id as 2nd arg.
  if "%2"=="" (
    echo ERROR: usage: run_spxw_iron_fly_combo.bat live-en Uxxxxxxx
    exit /b 2
  )
  python scripts\trading\spxw_iron_fly_combo.py --place --live-en --account %2 --wait-sec 120 >> "%logfile%" 2>&1
) else (
  python scripts\trading\spxw_iron_fly_combo.py >> "%logfile%" 2>&1
)

if errorlevel 1 (
  echo ERROR: combo failed with exit code %ERRORLEVEL%
) else (
  echo Combo completed. See %logfile%
)
echo.
pause
