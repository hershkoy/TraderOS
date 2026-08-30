@echo off
REM Register the one hidden every-minute Windows task that ticks crons\crontab.yaml
cd /d %~dp0..
if not exist logs\cron mkdir logs\cron
set PYTHONPATH=.
call venv\Scripts\activate.bat
python scripts\pipeline\cron_manager.py install-task
exit /b %ERRORLEVEL%
