@echo off
cd /d %~dp0..
.\venv\Scripts\activate
python scripts/spreads_trader.py --symbol QQQ --dte 7 --create-orders-en --quantity 2 --risk-profile balanced
python scripts/spreads_trader.py --symbol IWM --dte 7 --create-orders-en --quantity 2 --risk-profile balanced