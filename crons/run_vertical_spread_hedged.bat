@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

echo === Vertical Spread with Hedging (1x2 Ratio Spread) ===
echo.

REM Get symbol (required)
set /p "symbol=Symbol (required): "
if "%symbol%"=="" (
    echo ERROR: Symbol is required
    pause
    exit /b 1
)

REM Get DTE (default 2)
set "dte=2"
set /p "dte_input=DTE [2]: "
if not "%dte_input%"=="" set "dte=%dte_input%"

REM Get quantity (default 1)
set "quantity=1"
set /p "qty_input=Quantity [1]: "
if not "%qty_input%"=="" set "quantity=%qty_input%"

REM Get create_orders_en (default Y)
set "create_orders=--create-orders-en"
set /p "orders_input=Create orders? (Y/n) [Y]: "
if /i "%orders_input%"=="n" set "create_orders="

REM Create log file with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\vertical_spread_%symbol%_%datetime:~0,8%_%datetime:~8,6%.log

echo.
echo Running: %symbol% DTE=%dte% QTY=%quantity% CREATE_ORDERS=%create_orders%
echo Logging to: %logfile%
echo.

python scripts/options_strategy_trader.py --symbol %symbol% --strategy vertical_spread_with_hedging --dte %dte% --quantity %quantity% %create_orders%
if errorlevel 1 (
    echo ERROR: Vertical spread trader failed with exit code %ERRORLEVEL%
) else (
    echo Vertical spread trader completed successfully.
)

echo.
pause

