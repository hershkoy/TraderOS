"""Always-on Alpaca last-price hub for the /hot dashboard.

Listens on its own port (default 5001). charting_server.py serves HTML/REST on
5000; the /hot page opens ws://<host>:5001/ws/hot-candidates for live quotes.
SELL NOW Telegram is flushed from this process even with no browser tab open.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flask import Flask, current_app, jsonify
from flask_cors import CORS
from flask_sock import Sock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hot_price_service import DEFAULT_HOST, DEFAULT_PORT
from utils.scanning.channel_touch_hot_api import (
    HotCandidatesHub,
    get_hot_hub,
    set_hot_hub,
)

app = Flask(__name__)
CORS(app)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 20}
sock = Sock(app)


@app.route("/health")
def health():
    hub = get_hot_hub()
    snap = hub.snapshot()
    snap["ok"] = True
    return jsonify(snap)


@app.route("/kick", methods=["POST", "GET"])
def kick():
    hub = get_hot_hub()
    hub.kick()
    return jsonify({"ok": True, "seq": hub.snapshot().get("seq")})


@sock.route("/ws/hot-candidates")
def ws_hot_candidates(ws):
    """Push filtered hot-candidate snapshots from the always-on Alpaca loop."""
    import time as _time

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
                ws.send(current_app.json.dumps({"event": "ping"}))
                last_send = _time.monotonic()
            if not getattr(ws, "connected", True):
                break
    finally:
        hub.unregister()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Always-on /hot Alpaca price hub")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flask debug + reloader (ignored with --service)",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Background Windows service: no debug, log to logs/hot_price/",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    debug = False if args.service else False
    if args.debug and not args.service:
        debug = True
    if args.service:
        from utils.hot_price_service import attach_service_stdio, default_log_dir, write_pid

        log_dir = default_log_dir(ROOT)
        attach_service_stdio(log_dir)
        write_pid(log_dir)
    hub = HotCandidatesHub(always_run=True)
    set_hot_hub(hub)
    hub.start()
    print("Starting hot price server...")
    print("WS: ws://localhost:%s/ws/hot-candidates" % args.port)
    app.run(
        debug=debug,
        host=args.host,
        port=args.port,
        use_reloader=debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
