@echo off
REM Register the hidden logon Windows task that keeps hot_price_server.py running
cd /d %~dp0..
if not exist logs\hot_price mkdir logs\hot_price
set PYTHONPATH=.
call venv\Scripts\activate.bat
python scripts\pipeline\hot_price_service.py install-task
exit /b %ERRORLEVEL%
