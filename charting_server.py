"""
Charting Server for Backtrader Data
A Flask-based web server that provides charting capabilities for symbols in the data folder.
"""
from flask import Flask, render_template, request, jsonify, current_app
from flask_cors import CORS
from flask_sock import Sock
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.data.data_aggregator import DataAggregator
except ImportError:
    from utils.data_aggregator import DataAggregator
from indicators import SMA, EMA, WMA, RSI, MACD, Stochastic, Volume, OBV, VWAP, BollingerBands, ATR

app = Flask(__name__)
CORS(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 20}
sock = Sock(app)

# Available indicators
INDICATORS = {
    'SMA': {'name': 'Simple Moving Average', 'params': ['period'], 'defaults': {'period': 20}},
    'EMA': {'name': 'Exponential Moving Average', 'params': ['period'], 'defaults': {'period': 20}},
    'WMA': {'name': 'Weighted Moving Average', 'params': ['period'], 'defaults': {'period': 20}},
    'RSI': {'name': 'Relative Strength Index', 'params': ['period'], 'defaults': {'period': 14}},
    'MACD': {'name': 'MACD', 'params': ['fast', 'slow', 'signal'], 'defaults': {'fast': 12, 'slow': 26, 'signal': 9}},
    'Stochastic': {'name': 'Stochastic Oscillator', 'params': ['k_period', 'd_period'], 'defaults': {'k_period': 14, 'd_period': 3}},
    'Volume': {'name': 'Volume', 'params': [], 'defaults': {}},
    'OBV': {'name': 'On-Balance Volume', 'params': [], 'defaults': {}},
    'VWAP': {'name': 'Volume Weighted Average Price', 'params': [], 'defaults': {}},
    'BollingerBands': {'name': 'Bollinger Bands', 'params': ['period', 'std_dev'], 'defaults': {'period': 20, 'std_dev': 2}},
    'ATR': {'name': 'Average True Range', 'params': ['period'], 'defaults': {'period': 14}}
}

@app.route('/')
def index():
    """Main charting interface"""
    symbols = DataAggregator.get_available_symbols()
    return render_template('index.html', symbols=symbols, indicators=INDICATORS)

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols"""
    symbols = DataAggregator.get_available_symbols()
    return jsonify(symbols)

@app.route('/api/timeframes/<symbol>')
def get_timeframes(symbol):
    """Get available timeframes for a symbol"""
    timeframes = DataAggregator.get_available_timeframes(symbol)
    return jsonify(timeframes)

@app.route('/api/data')
def get_data():
    """Get chart data with indicators"""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')
    indicators = request.args.get('indicators', '[]')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        # Load data
        print(f"Loading data for {symbol} at {timeframe}")
        df = DataAggregator.get_data(symbol, timeframe)
        if df is None:
            print(f"ERROR: No data found for {symbol} at {timeframe}")
            return jsonify({'error': f'No data found for {symbol} at {timeframe}'}), 404
        
        # Data validation and debugging
        print(f"Data loaded successfully:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Index type: {type(df.index)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Check if DataFrame is empty
        if df.empty:
            print(f"ERROR: DataFrame is empty after loading data")
            return jsonify({'error': f'No data available for {symbol} at {timeframe}'}), 404
        
        print(f"  Sample data:")
        print(f"    First row: {df.iloc[0].to_dict()}")
        print(f"    Last row: {df.iloc[-1].to_dict()}")
        
        # Check for NaN values in the data
        nan_counts = df.isna().sum()
        if nan_counts.any():
            print(f"WARNING: Found NaN values in data:")
            print(f"  {nan_counts.to_dict()}")
        
        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_counts.any():
            print(f"WARNING: Found infinite values in data:")
            print(f"  {inf_counts.to_dict()}")
        
        # Clean the data by removing NaN and infinite values
        original_len = len(df)
        print(f"Original data length: {original_len}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        # Replace infinite values with NaN first
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        inf_replaced = original_len - len(df_clean.dropna())
        print(f"Replaced {inf_replaced} infinite values with NaN")
        
        # Now drop NaN values
        df_clean = df_clean.dropna()
        final_len = len(df_clean)
        print(f"After dropping NaN values: {final_len} rows")
        
        if final_len < original_len:
            print(f"WARNING: Removed {original_len - final_len} rows with NaN/infinite values")
            df = df_clean
        else:
            print(f"Data cleaning: no rows removed")
        
        # Check again if DataFrame is empty after cleaning
        if df.empty:
            print(f"ERROR: DataFrame is empty after cleaning")
            print(f"Original data sample:")
            print(f"  First few rows: {df.head() if not df.empty else 'Empty'}")
            print(f"  Data info: {df.info() if not df.empty else 'Empty'}")
            return jsonify({'error': f'No valid data available for {symbol} at {timeframe} after cleaning'}), 404
        
        # Prepare OHLCV data
        chart_data = {
            'datetime': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': df['open'].tolist() if 'open' in df.columns else [],
            'high': df['high'].tolist() if 'high' in df.columns else [],
            'low': df['low'].tolist() if 'low' in df.columns else [],
            'close': df['close'].tolist() if 'close' in df.columns else [],
            'volume': df['volume'].tolist() if 'volume' in df.columns else []
        }
        
        # Validate chart data
        print(f"Chart data prepared:")
        print(f"  Datetime points: {len(chart_data['datetime'])}")
        print(f"  Open points: {len(chart_data['open'])}")
        print(f"  Close points: {len(chart_data['close'])}")
        print(f"  Sample datetime: {chart_data['datetime'][:3]}")
        print(f"  Sample open: {chart_data['open'][:3]}")
        print(f"  Sample close: {chart_data['close'][:3]}")
        
        # Calculate indicators
        indicator_data = {}
        if indicators:
            try:
                indicator_list = json.loads(indicators)
                print(f"Processing {len(indicator_list)} indicators: {[ind['name'] for ind in indicator_list]}")
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid indicators JSON: {e}")
                indicator_list = []
            
            for indicator_config in indicator_list:
                indicator_name = indicator_config['name']
                params = indicator_config.get('params', {})
                
                print(f"Calculating {indicator_name} with params: {params}")
                
                if indicator_name not in INDICATORS:
                    print(f"WARNING: Unknown indicator '{indicator_name}', skipping")
                    continue
                
                try:
                    if indicator_name == 'SMA':
                        result = SMA(df['close'], params.get('period', 20))
                        print(f"  SMA calculated: {len(result)} values, NaN count: {result.isna().sum()}")
                        indicator_data[f'SMA_{params.get("period", 20)}'] = result.tolist()
                    
                    elif indicator_name == 'EMA':
                        result = EMA(df['close'], params.get('period', 20))
                        indicator_data[f'EMA_{params.get("period", 20)}'] = result.tolist()
                    
                    elif indicator_name == 'WMA':
                        result = WMA(df['close'], params.get('period', 20))
                        indicator_data[f'WMA_{params.get("period", 20)}'] = result.tolist()
                    
                    elif indicator_name == 'RSI':
                        result = RSI(df['close'], params.get('period', 14))
                        indicator_data[f'RSI_{params.get("period", 14)}'] = result.tolist()
                    
                    elif indicator_name == 'MACD':
                        result = MACD(df['close'], 
                                    params.get('fast', 12), 
                                    params.get('slow', 26), 
                                    params.get('signal', 9))
                        indicator_data['MACD_line'] = result['macd'].tolist()
                        indicator_data['MACD_signal'] = result['signal'].tolist()
                        indicator_data['MACD_histogram'] = result['histogram'].tolist()
                    
                    elif indicator_name == 'Stochastic':
                        result = Stochastic(df['high'], df['low'], df['close'],
                                          params.get('k_period', 14),
                                          params.get('d_period', 3))
                        indicator_data['Stoch_K'] = result['k'].tolist()
                        indicator_data['Stoch_D'] = result['d'].tolist()
                    
                    elif indicator_name == 'Volume':
                        result = Volume(df['volume'])
                        indicator_data['Volume'] = result.tolist()
                    
                    elif indicator_name == 'OBV':
                        result = OBV(df['close'], df['volume'])
                        indicator_data['OBV'] = result.tolist()
                    
                    elif indicator_name == 'VWAP':
                        result = VWAP(df['high'], df['low'], df['close'], df['volume'])
                        indicator_data['VWAP'] = result.tolist()
                    
                    elif indicator_name == 'BollingerBands':
                        result = BollingerBands(df['close'], 
                                              params.get('period', 20),
                                              params.get('std_dev', 2))
                        indicator_data['BB_upper'] = result['upper'].tolist()
                        indicator_data['BB_middle'] = result['middle'].tolist()
                        indicator_data['BB_lower'] = result['lower'].tolist()
                    
                    elif indicator_name == 'ATR':
                        result = ATR(df['high'], df['low'], df['close'],
                                   params.get('period', 14))
                        indicator_data['ATR'] = result.tolist()
                
                except Exception as e:
                    print(f"ERROR calculating {indicator_name}: {e}")
                    import traceback
                    print(f"  Traceback: {traceback.format_exc()}")
                    continue
        else:
            print("No indicators requested")
        
        print(f"Returning response with {len(chart_data['datetime'])} data points and {len(indicator_data)} indicators")
        
        return jsonify({
            'chart_data': chart_data,
            'indicators': indicator_data,
            'symbol': symbol,
            'timeframe': timeframe
        })
    
    except Exception as e:
        print(f"ERROR in get_data: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicators')
def get_indicators():
    """Get available indicators"""
    return jsonify(INDICATORS)


@app.route('/hot')
def hot_candidates_page():
    """15m + 1d channel-touch hot candidates dashboard."""
    return render_template('hot_candidates.html')


@app.route('/api/hot-candidates')
def api_hot_candidates():
    """Armed 15m and 1d setups from TimescaleDB with live Alpaca last (throttled)."""
    refresh = str(request.args.get('refresh', '1')).lower() not in ('0', 'false', 'no')
    try:
        from utils.scanning.channel_touch_hot_api import candidates_payload

        return jsonify(candidates_payload(refresh=refresh))
    except Exception as exc:
        return jsonify({'error': str(exc), 'rows': [], 'n_rows': 0, 'n_armed': 0, 'n_hot': 0}), 500


@app.route('/api/hot-candidates/settings', methods=['GET', 'POST'])
def api_hot_settings():
    """Persisted Telegram + filter settings (cron and UI share this row)."""
    try:
        from utils.scanning.channel_touch_hot_api import get_hot_hub, get_store

        store = get_store()
        if request.method == 'GET':
            return jsonify(store.load_settings())
        patch = request.get_json(silent=True) or {}
        saved = store.save_settings(patch)
        get_hot_hub().kick()
        return jsonify(saved)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@sock.route('/ws/hot-candidates')
def ws_hot_candidates(ws):
    """Push filtered hot-candidate snapshots; one Alpaca refresh loop for all tabs."""
    import time as _time

    from utils.scanning.channel_touch_hot_api import get_hot_hub

    hub = get_hot_hub()
    hub.register()
    last_seq = 0
    last_send = _time.monotonic()
    try:
        while True:
            last_seq, payload = hub.wait_next(last_seq, timeout=1.0)
            if payload is not None:
                ws.send(current_app.json.dumps(payload))
                last_send = _time.monotonic()
            elif (_time.monotonic() - last_send) >= 25.0:
                ws.send(current_app.json.dumps({'event': 'ping'}))
                last_send = _time.monotonic()
            if not getattr(ws, 'connected', True):
                break
    finally:
        hub.unregister()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Charting server for Backtrader data")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flask debug + reloader (ignored with --service)",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Background Windows service: no debug, log to logs/charting_server/",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    os.makedirs("templates", exist_ok=True)
    debug = False if args.service else True
    if args.debug:
        debug = not args.service
    if args.service:
        from utils.charting_server_service import attach_service_stdio, default_log_dir, write_pid

        log_dir = default_log_dir(ROOT)
        attach_service_stdio(log_dir)
        write_pid(log_dir)
    print("Starting Charting Server...")
    if not args.service:
        print("Available symbols:", DataAggregator.get_available_symbols())
    print("Server will be available at: http://localhost:%s" % args.port)
    app.run(
        debug=debug,
        host=args.host,
        port=args.port,
        use_reloader=debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
