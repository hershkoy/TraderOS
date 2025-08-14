# 📈 Backtrader Charting Server

A modern web-based charting server for visualizing financial data from your Backtrader project. This server provides interactive charts with technical indicators and supports multiple timeframes.

## 🚀 Features

- **Interactive Charts**: Beautiful candlestick charts with Plotly.js
- **Technical Indicators**: 11+ common indicators including:
  - Moving Averages (SMA, EMA, WMA)
  - Momentum (RSI, MACD, Stochastic)
  - Volume (Volume, OBV, VWAP)
  - Trend (Bollinger Bands, ATR)
- **Timeframe Support**: Automatic data aggregation for different timeframes
- **Modern UI**: Responsive design with glassmorphism effects
- **Real-time Data**: Load data from your existing parquet files

## 📁 Project Structure

```
backTraderTest/
├── charting_server.py          # Main Flask server
├── test_charting_server.py     # Test script
├── indicators/                 # Technical indicators
│   ├── __init__.py
│   ├── moving_averages.py
│   ├── momentum.py
│   ├── volume.py
│   └── trend.py
├── utils/
│   └── data_aggregator.py      # Data loading and aggregation
├── templates/
│   └── index.html              # Web interface
└── data/                       # Your existing data folder
    └── ALPACA/
        ├── EXPE/
        └── NFLX/
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the Setup**:
   ```bash
   python test_charting_server.py
   ```

3. **Start the Server**:
   ```bash
   python charting_server.py
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:5000`

## 📊 Available Indicators

### Moving Averages
- **SMA (Simple Moving Average)**: Period parameter
- **EMA (Exponential Moving Average)**: Period parameter  
- **WMA (Weighted Moving Average)**: Period parameter

### Momentum
- **RSI (Relative Strength Index)**: Period parameter
- **MACD**: Fast, Slow, and Signal parameters
- **Stochastic Oscillator**: K and D period parameters

### Volume
- **Volume**: Simple volume display
- **OBV (On-Balance Volume)**: No parameters
- **VWAP (Volume Weighted Average Price)**: No parameters

### Trend
- **Bollinger Bands**: Period and Standard Deviation parameters
- **ATR (Average True Range)**: Period parameter

## 🔧 Usage

### 1. Select Symbol
Choose from available symbols in your `data/ALPACA/` folder.

### 2. Choose Timeframe
Select from available timeframes:
- 1m, 5m, 15m, 30m (minutes)
- 1h, 4h (hours)
- 1d (daily)
- 1w (weekly)
- 1M (monthly)

### 3. Add Indicators
Click on indicators in the sidebar to add them to your chart. Configure parameters as needed.

### 4. Load Chart
Click "Load Chart" to display the data with selected indicators.

## 📈 Data Aggregation

The server automatically handles data aggregation when switching timeframes:

- **Exact Match**: If data exists for the requested timeframe, it loads directly
- **Aggregation**: If data doesn't exist, it aggregates from smaller timeframes
- **OHLCV Rules**: 
  - Open: First value in period
  - High: Maximum value in period
  - Low: Minimum value in period
  - Close: Last value in period
  - Volume: Sum of volumes in period

## 🎨 Customization

### Adding New Indicators

1. Create a new function in the appropriate indicators file:
   ```python
   def MyIndicator(data: pd.Series, param1: int = 10) -> pd.Series:
       # Your calculation here
       return result
   ```

2. Add to `indicators/__init__.py`:
   ```python
   from .your_file import MyIndicator
   __all__ = [..., 'MyIndicator']
   ```

3. Add to the server's `INDICATORS` dictionary in `charting_server.py`

### Styling

The web interface uses modern CSS with:
- Glassmorphism effects
- Responsive design
- Smooth animations
- Color-coded indicators

## 🔍 API Endpoints

- `GET /` - Main charting interface
- `GET /api/symbols` - List available symbols
- `GET /api/timeframes/<symbol>` - Get timeframes for symbol
- `GET /api/data` - Get chart data with indicators
- `GET /api/indicators` - List available indicators

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_charting_server.py
```

This will test:
- Data loading from your parquet files
- Indicator calculations
- Timeframe aggregation

## 🐛 Troubleshooting

### Common Issues

1. **No symbols found**: Ensure your data is in `data/ALPACA/SYMBOL/TIMEFRAME/` structure
2. **Import errors**: Make sure all dependencies are installed
3. **Port already in use**: Change the port in `charting_server.py`

### Data Format

Your parquet files should contain:
- `open`, `high`, `low`, `close`, `volume` columns
- Datetime index or `datetime` column

## 📝 Example Data Structure

```
data/
└── ALPACA/
    ├── EXPE/
    │   └── 1h/
    │       └── expe_1h.parquet
    └── NFLX/
        └── 1h/
            └── nflx_1h.parquet
```

## 🤝 Contributing

Feel free to add new indicators or improve the interface! The modular design makes it easy to extend.

## 📄 License

This project is part of your Backtrader testing environment.
