@echo off
cd /d %~dp0..
set PYTHONPATH=.
call venv\Scripts\activate.bat
python scripts\pipeline\cron_manager.py uninstall-task
exit /b %ERRORLEVEL%
