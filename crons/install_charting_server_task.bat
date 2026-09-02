@echo off
REM Register the hidden logon Windows task that keeps charting_server.py running
cd /d %~dp0..
if not exist logs\charting_server mkdir logs\charting_server
set PYTHONPATH=.
call venv\Scripts\activate.bat
python scripts\pipeline\charting_server_service.py install-task
exit /b %ERRORLEVEL%
