"""
Charting Server for Backtrader Data
A Flask-based web server that provides charting capabilities for symbols in the data folder.
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
import os

from utils.data_aggregator import DataAggregator
from indicators import SMA, EMA, WMA, RSI, MACD, Stochastic, Volume, OBV, VWAP, BollingerBands, ATR

app = Flask(__name__)
CORS(app)

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
        df = DataAggregator.get_data(symbol, timeframe)
        if df is None:
            return jsonify({'error': f'No data found for {symbol} at {timeframe}'}), 404
        
        # Prepare OHLCV data
        chart_data = {
            'datetime': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': df['open'].tolist() if 'open' in df.columns else [],
            'high': df['high'].tolist() if 'high' in df.columns else [],
            'low': df['low'].tolist() if 'low' in df.columns else [],
            'close': df['close'].tolist() if 'close' in df.columns else [],
            'volume': df['volume'].tolist() if 'volume' in df.columns else []
        }
        
        # Calculate indicators
        indicator_data = {}
        if indicators:
            indicator_list = json.loads(indicators)
            
            for indicator_config in indicator_list:
                indicator_name = indicator_config['name']
                params = indicator_config.get('params', {})
                
                if indicator_name not in INDICATORS:
                    continue
                
                try:
                    if indicator_name == 'SMA':
                        result = SMA(df['close'], params.get('period', 20))
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
                    print(f"Error calculating {indicator_name}: {e}")
                    continue
        
        return jsonify({
            'chart_data': chart_data,
            'indicators': indicator_data,
            'symbol': symbol,
            'timeframe': timeframe
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicators')
def get_indicators():
    """Get available indicators"""
    return jsonify(INDICATORS)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Charting Server...")
    print("Available symbols:", DataAggregator.get_available_symbols())
    print("Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
