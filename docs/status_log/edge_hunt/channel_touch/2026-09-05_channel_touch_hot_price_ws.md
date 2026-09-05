# /hot always-on price WebSocket (2026-09-05)

Moved the `/hot` Alpaca last-price loop and SELL NOW Telegram off `charting_server.py` into a logon Windows task, same pattern as ChartingServer (not a crontab.yaml job).

## What runs

| Process | Task | Port | Role |
|---------|------|------|------|
| `charting_server.py --service` | `backTraderTest\ChartingServer` | 5000 | `/hot` HTML + REST (DB snapshot, no Alpaca) |
| `hot_price_server.py --service` | `backTraderTest\HotPriceHub` | 5001 | Alpaca ~5s, WS `/ws/hot-candidates`, SELL NOW Telegram |

The hub loop runs with **zero** browser clients. The page opens `ws://<hostname>:5001/ws/hot-candidates`. Settings / Bought / Sold on Flask POST `http://127.0.0.1:5001/kick`.

H5 BUY NOW Telegram stays in `channel_touch_15m --mode run`. RTH-minute `channel_touch_15m_proximity` stays as a backup (`sell_notified_at` dedups).

## Install

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\hot_price_service.py install-task
```

Playbook: [15m live](../../../features/channel_touch_15m_live.md).
