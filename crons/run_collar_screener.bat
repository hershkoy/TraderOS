@echo off
REM Zero-Cost Collar Screener Launcher
REM This batch file runs the collar screener with proper environment setup

echo ========================================
echo Zero-Cost Collar Screener
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo And: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if TWS/IB Gateway is running
echo Checking IB connection...
python -c "import socket; s=socket.socket(); s.settimeout(2); result=s.connect_ex(('127.0.0.1', 7497)); s.close(); exit(0 if result==0 else 1)"
if %errorlevel% neq 0 (
    echo Warning: Could not connect to IB on port 7497
    echo Make sure TWS or IB Gateway is running
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo Running collar screener...
echo.

REM Run the enhanced screener
python utils\screening\zero_cost_collar.py

echo.
echo Screener completed. Check logs/collar_screener.log for details.
echo Results saved to reports/collar_opportunities.csv
echo.
pause
