@echo off
REM Import tickers into universe

echo Importing tickers into TimescaleDB universe...
echo.

REM Run the import script
python scripts\data\import_tickers_to_universe.py --ticker-file "c:\Users\Hezi\Downloads\tickers.txt" --index-name "custom_universe" --stats

echo.
echo Press any key to exit...
pause
