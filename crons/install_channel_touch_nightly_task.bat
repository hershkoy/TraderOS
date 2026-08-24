@echo off
REM Register Windows Task Scheduler job for channel-touch nightly
REM Default: weekdays 23:00 local (after US cash close for Israel/UTC+2/+3)

set TASK_NAME=backTraderTest\ChannelTouchNightly
set BAT_PATH=%~dp0channel_touch_nightly.bat
set RUN_TIME=23:00

echo Creating scheduled task: %TASK_NAME%
echo Script: %BAT_PATH%
echo Schedule: Mon-Fri at %RUN_TIME% (local time)
echo.

schtasks /Create /F ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST %RUN_TIME% ^
  /RL LIMITED

if errorlevel 1 (
  echo FAILED to create task. Run this bat as your normal user (or Admin if needed).
  exit /b 1
)

echo.
echo Task created. Useful commands:
echo   schtasks /Query /TN "%TASK_NAME%" /V /FO LIST
echo   schtasks /Run /TN "%TASK_NAME%"
echo   schtasks /Delete /TN "%TASK_NAME%" /F
echo.
echo Fill TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env before first run.
echo Test ping: venv\Scripts\activate ^&^& set PYTHONPATH=. ^&^& python -m utils.notify.telegram_pinger --ping
exit /b 0
