@echo off
cd /d %~dp0..
set PYTHONPATH=.
call venv\Scripts\activate.bat
python scripts\pipeline\hot_price_service.py uninstall-task
exit /b %ERRORLEVEL%
