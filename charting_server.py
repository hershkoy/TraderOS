"""
Charting Server for Backtrader Data
A Flask-based web server that provides charting capabilities for symbols in the data folder.
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import queue
import threading
import time
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
from indicators.atr_anchored_range import (
    atr_anchored_range,
    normalize_atr_tf,
    overlay_payload,
)

app = Flask(__name__)
CORS(app)
app.config["TEMPLATES_AUTO_RELOAD"] = True

_SYMBOLS_CACHE = {"ts": 0.0, "symbols": []}
_SYMBOLS_TTL_SEC = 600.0
_DB_LOCK = threading.Lock()

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
    'ATR': {'name': 'Average True Range', 'params': ['period'], 'defaults': {'period': 14}},
    'ATRAnchoredRange': {
        'name': 'ATR Anchored Range (session)',
        'params': ['mode', 'timeframe', 'period', 'show_gp'],
        'defaults': {'mode': 'Open', 'timeframe': '1D', 'period': 20, 'show_gp': False},
        'param_meta': {
            'mode': {
                'type': 'select',
                'options': ['Open', 'Prior Close'],
                'label': 'Mode',
                'title': 'Anchor to current session open or prior close.',
            },
            'timeframe': {
                'type': 'select',
                'options': ['1D', '1W', '1M'],
                'label': 'ATR Timeframe',
                'title': 'ATR timeframe. Common: 1D, 1W, 1M.',
            },
            'period': {
                'type': 'number',
                'label': 'ATR Period',
                'title': 'Bars in the ATR. 20 daily bars ~ 1 month.',
            },
            'show_gp': {
                'type': 'checkbox',
                'label': 'Golden pocket',
                'title': 'Fib 0.61-0.65 bands inside each ATR range.',
            },
        },
    },
}

@app.route('/')
def index():
    """Main charting interface. Symbol list loads async so the page is not blocked."""
    from utils.charting.ohlcv_window import MAX_PAD as CHART_MAX_PAD

    return render_template(
        'index.html',
        symbols=[],
        indicators=INDICATORS,
        max_pad=CHART_MAX_PAD,
    )

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols (optional ?q= prefix/substring filter)."""
    symbols = _chart_symbols()
    q = (request.args.get("q") or "").strip()
    if q:
        from utils.charting.channel_overlay import filter_symbols

        symbols = filter_symbols(q, symbols)
    return jsonify(symbols)


def _chart_symbols():
    """Fast symbol list: ticker_universe + memory cache (not DISTINCT on market_data)."""
    now = time.time()
    cached = _SYMBOLS_CACHE.get("symbols") or []
    if cached and (now - float(_SYMBOLS_CACHE.get("ts") or 0)) < _SYMBOLS_TTL_SEC:
        return list(cached)
    symbols = []
    try:
        from utils.db.timescaledb_client import get_timescaledb_client

        client = get_timescaledb_client()
        with _DB_LOCK:
            if client.ensure_connection():
                cursor = client.connection.cursor()
                try:
                    cursor.execute(
                        "SELECT DISTINCT symbol FROM ticker_universe "
                        "WHERE COALESCE(is_active, TRUE) = TRUE ORDER BY symbol"
                    )
                    symbols = [row[0] for row in cursor.fetchall() if row and row[0]]
                finally:
                    cursor.close()
    except Exception as exc:
        print("chart symbols from ticker_universe failed: %s" % exc)
        symbols = []
    if symbols:
        _SYMBOLS_CACHE["ts"] = now
        _SYMBOLS_CACHE["symbols"] = symbols
    return symbols


def _progress(cb, pct, msg):
    if cb is None:
        return
    try:
        cb(int(pct), str(msg))
    except Exception:
        pass


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _load_atr_htf_df(symbol, atr_tf, around_ts, period):
    """Extra HTF bars so session ATR has warmup beyond the visible window."""
    from utils.charting.ohlcv_window import clamp_pad, load_ohlcv_window

    pad = clamp_pad(max(int(period) + 80, 100), 100)
    around = ""
    if around_ts is not None:
        ts = pd.Timestamp(around_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        around = ts.strftime("%Y-%m-%d %H:%M:%S")
    with _DB_LOCK:
        win = load_ohlcv_window(
            symbol,
            atr_tf,
            around=around,
            before=pad,
            after=max(20, min(pad, 80)),
        )
    return win.get("df")


def _build_chart_payload(symbol, timeframe, around, before, after, indicators_raw, progress=None):
    """Return (payload_dict, http_status)."""
    from utils.charting.ohlcv_window import load_ohlcv_window

    _progress(progress, 6, "Connecting to TimescaleDB")
    with _DB_LOCK:
        win = load_ohlcv_window(
            symbol,
            timeframe,
            around=around,
            before=before,
            after=after,
            progress=progress,
        )
    df = win["df"]
    if df is None or df.empty:
        return (
            {"error": "No data found for %s at %s" % (symbol, timeframe)},
            404,
        )

    _progress(progress, 82, "Preparing %s bars" % len(df))
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df_clean.empty:
        return (
            {"error": "No valid data available for %s at %s after cleaning" % (symbol, timeframe)},
            404,
        )
    df = df_clean

    chart_data = {
        "datetime": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "open": df["open"].tolist() if "open" in df.columns else [],
        "high": df["high"].tolist() if "high" in df.columns else [],
        "low": df["low"].tolist() if "low" in df.columns else [],
        "close": df["close"].tolist() if "close" in df.columns else [],
        "volume": df["volume"].tolist() if "volume" in df.columns else [],
    }

    indicator_data = {}
    indicator_list = []
    if indicators_raw:
        try:
            indicator_list = json.loads(indicators_raw)
        except json.JSONDecodeError as e:
            print("ERROR: Invalid indicators JSON: %s" % e)
            indicator_list = []

    if indicator_list:
        _progress(progress, 90, "Calculating indicators")
        for indicator_config in indicator_list:
            indicator_name = indicator_config["name"]
            params = indicator_config.get("params", {})
            if indicator_name not in INDICATORS:
                continue
            try:
                if indicator_name == "SMA":
                    result = SMA(df["close"], params.get("period", 20))
                    indicator_data["SMA_%s" % params.get("period", 20)] = result.tolist()
                elif indicator_name == "EMA":
                    result = EMA(df["close"], params.get("period", 20))
                    indicator_data["EMA_%s" % params.get("period", 20)] = result.tolist()
                elif indicator_name == "WMA":
                    result = WMA(df["close"], params.get("period", 20))
                    indicator_data["WMA_%s" % params.get("period", 20)] = result.tolist()
                elif indicator_name == "RSI":
                    result = RSI(df["close"], params.get("period", 14))
                    indicator_data["RSI_%s" % params.get("period", 14)] = result.tolist()
                elif indicator_name == "MACD":
                    result = MACD(
                        df["close"],
                        params.get("fast", 12),
                        params.get("slow", 26),
                        params.get("signal", 9),
                    )
                    indicator_data["MACD_line"] = result["macd"].tolist()
                    indicator_data["MACD_signal"] = result["signal"].tolist()
                    indicator_data["MACD_histogram"] = result["histogram"].tolist()
                elif indicator_name == "Stochastic":
                    result = Stochastic(
                        df["high"],
                        df["low"],
                        df["close"],
                        params.get("k_period", 14),
                        params.get("d_period", 3),
                    )
                    indicator_data["Stoch_K"] = result["k"].tolist()
                    indicator_data["Stoch_D"] = result["d"].tolist()
                elif indicator_name == "Volume":
                    result = Volume(df["volume"])
                    indicator_data["Volume"] = result.tolist()
                elif indicator_name == "OBV":
                    result = OBV(df["close"], df["volume"])
                    indicator_data["OBV"] = result.tolist()
                elif indicator_name == "VWAP":
                    result = VWAP(df["high"], df["low"], df["close"], df["volume"])
                    indicator_data["VWAP"] = result.tolist()
                elif indicator_name == "BollingerBands":
                    result = BollingerBands(
                        df["close"],
                        params.get("period", 20),
                        params.get("std_dev", 2),
                    )
                    indicator_data["BB_upper"] = result["upper"].tolist()
                    indicator_data["BB_middle"] = result["middle"].tolist()
                    indicator_data["BB_lower"] = result["lower"].tolist()
                elif indicator_name == "ATR":
                    result = ATR(
                        df["high"],
                        df["low"],
                        df["close"],
                        params.get("period", 14),
                    )
                    indicator_data["ATR"] = result.tolist()
                elif indicator_name == "ATRAnchoredRange":
                    atr_tf = normalize_atr_tf(params.get("timeframe", "1D"))
                    period = int(params.get("period", 20) or 20)
                    atr_df = None
                    chart_tf = str(timeframe or "").strip()
                    if atr_tf != chart_tf:
                        try:
                            atr_df = _load_atr_htf_df(
                                symbol, atr_tf, df.index[-1], period
                            )
                        except Exception as exc:
                            print("ATR HTF load failed for %s %s: %s" % (symbol, atr_tf, exc))
                            atr_df = None
                    result = atr_anchored_range(
                        df,
                        atr_df,
                        mode=params.get("mode", "Open"),
                        atr_timeframe=atr_tf,
                        chart_timeframe=chart_tf,
                        period=period,
                    )
                    indicator_data["AAA"] = overlay_payload(
                        result, show_gp=_as_bool(params.get("show_gp"), False)
                    )
            except Exception as e:
                print("ERROR calculating %s: %s" % (indicator_name, e))
                continue

    _progress(progress, 100, "Ready (%s bars)" % len(chart_data["datetime"]))
    return (
        {
            "chart_data": chart_data,
            "indicators": indicator_data,
            "symbol": symbol,
            "timeframe": timeframe,
            "has_more_before": bool(win.get("has_more_before")),
            "has_more_after": bool(win.get("has_more_after")),
            "n_bars": len(chart_data["datetime"]),
        },
        200,
    )


@app.route("/api/timeframes/<symbol>")
def get_timeframes(symbol):
    """Get available timeframes for a symbol"""
    with _DB_LOCK:
        timeframes = DataAggregator.get_available_timeframes(symbol)
    return jsonify(timeframes)


@app.route("/api/data")
def get_data():
    """Get chart data with indicators (bar window, not full history)."""
    symbol = request.args.get("symbol")
    timeframe = request.args.get("timeframe", "1h")
    indicators = request.args.get("indicators", "[]")
    around = request.args.get("around", "") or ""
    stream = str(request.args.get("stream", "")).lower() in ("1", "true", "yes")
    from utils.charting.ohlcv_window import DEFAULT_AFTER, DEFAULT_BEFORE, clamp_pad

    before = clamp_pad(request.args.get("before", DEFAULT_BEFORE), DEFAULT_BEFORE)
    after = clamp_pad(request.args.get("after", DEFAULT_AFTER), DEFAULT_AFTER)

    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    print(
        "Loading window for %s %s around=%s before=%s after=%s stream=%s"
        % (symbol, timeframe, around or "(latest)", before, after, stream)
    )

    if not stream:
        try:
            payload, status = _build_chart_payload(
                symbol, timeframe, around, before, after, indicators
            )
            return jsonify(payload), status
        except Exception as e:
            print("ERROR in get_data: %s" % e)
            import traceback

            print("Traceback: %s" % traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    def generate():
        q = queue.Queue()

        def progress(pct, msg):
            q.put({"type": "progress", "pct": int(pct), "msg": str(msg)})

        def work():
            try:
                payload, status = _build_chart_payload(
                    symbol,
                    timeframe,
                    around,
                    before,
                    after,
                    indicators,
                    progress=progress,
                )
                if status != 200:
                    q.put({"type": "error", "error": payload.get("error") or "Chart load failed"})
                else:
                    q.put({"type": "result", "data": payload})
            except Exception as exc:
                q.put({"type": "error", "error": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, default=str) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/goto-spec")
def api_goto_spec():
    """Parse a pasted report/CTF clock into UTC-naive match keys."""
    q = request.args.get("q") or ""
    try:
        from utils.charting.channel_overlay import parse_goto_query

        return jsonify(parse_goto_query(q))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/channel-overlay", methods=["POST"])
def api_channel_overlay():
    """Normalize pasted /hot CTF Channel JSON for Plotly rails."""
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if text is None:
        text = body.get("json", body.get("channel", body.get("ch", "")))
    try:
        from utils.charting.channel_overlay import parse_channel_overlay

        return jsonify(parse_channel_overlay(text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/api/indicators')
def get_indicators():
    """Get available indicators"""
    return jsonify(INDICATORS)


def _kick_price_hub():
    """Wake the always-on price process; ignore if it is not running."""
    from utils.scanning.channel_touch_hot_api import kick_price_service

    kick_price_service()


@app.route('/hot')
def hot_candidates_page():
    """15m + 1d channel-touch hot candidates dashboard."""
    from utils.scanning.channel_touch_hot_api import price_ws_port

    return render_template('hot_candidates.html', price_ws_port=price_ws_port())


@app.route('/api/hot-candidates')
def api_hot_candidates():
    """Armed 15m and 1d setups from TimescaleDB (live Alpaca is the price hub)."""
    refresh = str(request.args.get('refresh', '0')).lower() in ('1', 'true', 'yes')
    try:
        from utils.scanning.channel_touch_hot_api import candidates_payload

        return jsonify(candidates_payload(refresh=refresh))
    except Exception as exc:
        return jsonify({'error': str(exc), 'rows': [], 'n_rows': 0, 'n_armed': 0, 'n_hot': 0}), 500


@app.route('/api/hot-candidates/settings', methods=['GET', 'POST'])
def api_hot_settings():
    """Persisted Telegram + filter settings (cron and UI share this row)."""
    try:
        from utils.scanning.channel_touch_hot_api import get_store

        store = get_store()
        if request.method == 'GET':
            return jsonify(store.load_settings())
        patch = request.get_json(silent=True) or {}
        saved = store.save_settings(patch)
        _kick_price_hub()
        return jsonify(saved)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/hot-candidates/bought', methods=['POST'])
def api_hot_bought_mark():
    """Mark a /hot row as bought so the ATR/trail stop can fire SELL NOW."""
    try:
        from utils.scanning.channel_touch_bought import mark_bought_from_candidate
        from utils.scanning.channel_touch_hot_api import get_store

        body = request.get_json(silent=True) or {}
        stock = str(body.get('stock') or '').strip()
        timeframe = str(body.get('timeframe') or '15m').strip() or '15m'
        entry_px = body.get('entry_px')
        trade = mark_bought_from_candidate(
            get_store(),
            stock=stock,
            timeframe=timeframe,
            entry_px=None if entry_px in (None, '') else float(entry_px),
        )
        _kick_price_hub()
        return jsonify(trade)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/hot-candidates/bought/close', methods=['POST'])
def api_hot_bought_close():
    """Drop a bought trade from the live stop watch (manual sold / unbuy)."""
    try:
        from utils.scanning.channel_touch_bought import close_bought_trade
        from utils.scanning.channel_touch_hot_api import get_store

        body = request.get_json(silent=True) or {}
        tid = body.get('id')
        if tid in (None, ''):
            return jsonify({'error': 'id is required'}), 400
        closed = close_bought_trade(get_store(), trade_id=int(tid))
        if closed is None:
            return jsonify({'error': 'trade not found'}), 404
        _kick_price_hub()
        return jsonify(closed)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


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
