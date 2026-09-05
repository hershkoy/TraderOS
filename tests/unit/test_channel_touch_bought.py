"""Bought-trade stop math and SELL NOW helpers (no live DB)."""
from __future__ import annotations

from datetime import datetime, timezone

from utils.scanning.channel_touch_bought import (
    apply_price_tick,
    atr_last,
    build_bought_trade,
    current_stop_price,
    dist_to_stop_pct,
    entry_px_from_candidate,
    flush_sell_notifications,
    format_sell_message,
    hard_stop_price,
    mark_bought_from_candidate,
    pending_sell_notifies,
    sell_alerts_from_trades,
    sell_keys_from_trades,
    sync_bought_prices,
    tag_candidates_bought,
)


class FakeBoughtStore:
    def __init__(self, rows=None, bought=None):
        self.rows = [dict(r) for r in (rows or [])]
        self.bought = [dict(t) for t in (bought or [])]
        self._next_id = 1 + max([int(t.get("id") or 0) for t in self.bought], default=0)
        self.notified = []

    def load_rows(self):
        return [dict(r) for r in self.rows]

    def load_bought(self, *, active_only=True):
        if not active_only:
            return [dict(t) for t in self.bought]
        return [dict(t) for t in self.bought if t.get("status") in ("open", "sell_now")]

    def find_active_bought(self, stock, timeframe="15m"):
        stock_u = str(stock).upper()
        tf = str(timeframe or "15m")
        for trade in self.bought:
            if (
                str(trade.get("stock") or "").upper() == stock_u
                and str(trade.get("timeframe") or "15m") == tf
                and trade.get("status") in ("open", "sell_now")
            ):
                return dict(trade)
        return None

    def upsert_bought(self, trade):
        existing = self.find_active_bought(trade.get("stock"), trade.get("timeframe") or "15m")
        if existing:
            return existing
        item = dict(trade)
        item["id"] = self._next_id
        self._next_id += 1
        self.bought.append(item)
        return dict(item)

    def update_bought_live(self, trade):
        tid = trade.get("id")
        for item in self.bought:
            if item.get("id") == tid or (
                tid is None
                and str(item.get("stock")) == str(trade.get("stock"))
                and str(item.get("timeframe") or "15m") == str(trade.get("timeframe") or "15m")
            ):
                item.update(trade)
                if tid is not None:
                    item["id"] = tid
                return dict(item)
        return None

    def close_bought(self, trade_id):
        for item in self.bought:
            if item.get("id") == int(trade_id):
                item["status"] = "closed"
                return dict(item)
        return None

    def mark_sell_notified(self, trade_ids, when=None):
        self.notified.extend(list(trade_ids))
        for item in self.bought:
            if item.get("id") in set(trade_ids):
                item["sell_notified_at"] = when or "marked"


def test_hard_stop_atr_clamp():
    assert round(hard_stop_price(100.0, atr_at_entry=10.0), 6) == 94.0
    assert round(hard_stop_price(100.0, atr_at_entry=0.5), 6) == 98.5
    assert round(hard_stop_price(100.0, atr_at_entry=None), 6) == 97.0


def test_trail_lifts_stop_after_peak():
    hard = hard_stop_price(100.0, atr_at_entry=2.0)
    assert round(hard, 6) == 96.0
    assert current_stop_price(hard, 100.0, 0.10) == hard
    lifted = current_stop_price(hard, 120.0, 0.10)
    assert round(lifted, 6) == 108.0


def test_apply_price_tick_sell_now_on_hard_stop():
    trade = build_bought_trade(stock="AAA", timeframe="1d", entry_px=100.0, atr_at_entry=2.0)
    assert trade["status"] == "open"
    hit = apply_price_tick(trade, 95.9)
    assert hit["status"] == "sell_now"
    assert hit["exit_reason"] == "hard_stop"
    assert hit["dist_to_stop_pct"] < 0


def test_apply_price_tick_trail_exit_and_no_downgrade():
    trade = build_bought_trade(stock="BBB", timeframe="15m", entry_px=100.0, atr_at_entry=2.0)
    up = apply_price_tick(trade, 120.0)
    assert up["status"] == "open"
    assert round(up["current_stop"], 6) == 108.0
    hit = apply_price_tick(up, 107.5)
    assert hit["status"] == "sell_now"
    assert hit["exit_reason"] == "trail_stop"
    bounce = apply_price_tick(hit, 110.0)
    assert bounce["status"] == "sell_now"


def test_dist_to_stop_pct_cushion():
    assert dist_to_stop_pct(100.0, 94.0) == 6.0
    assert dist_to_stop_pct(None, 94.0) is None


def test_atr_last_needs_enough_bars():
    high = [10 + i * 0.1 for i in range(20)]
    low = [9 + i * 0.1 for i in range(20)]
    close = [9.5 + i * 0.1 for i in range(20)]
    assert atr_last(high[:10], low[:10], close[:10]) is None
    got = atr_last(high, low, close)
    assert got is not None and got > 0


def test_entry_px_prefers_fill():
    assert entry_px_from_candidate({"fill_px": 11, "last_price": 12, "last_close": 10}) == 11.0
    assert entry_px_from_candidate({"last_close": 10}) == 10.0
    assert entry_px_from_candidate({}) is None


def test_mark_bought_and_sync_sell():
    store = FakeBoughtStore(
        rows=[{"stock": "AAA", "timeframe": "1d", "fill_px": 50.0, "last_price": 50.2, "h2_time": "t"}]
    )
    trade = mark_bought_from_candidate(
        store, stock="aaa", timeframe="1d", atr_loader=lambda s, t: 1.0
    )
    assert trade["stock"] == "AAA"
    assert trade["status"] == "open"
    assert store.find_active_bought("AAA", "1d")["id"] == trade["id"]
    again = mark_bought_from_candidate(
        store, stock="AAA", timeframe="1d", atr_loader=lambda s, t: 9.0
    )
    assert again["id"] == trade["id"]

    tagged = tag_candidates_bought(store.load_rows(), store.load_bought())
    assert tagged[0]["bought"] is True

    updated = sync_bought_prices(store, {"AAA": 40.0}, now=datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert updated[0]["status"] == "sell_now"
    assert sell_keys_from_trades(updated) == ["AAA|1d|%s" % trade["id"]]
    alerts = sell_alerts_from_trades(updated)
    assert alerts[0]["stock"] == "AAA"
    pending = pending_sell_notifies(updated)
    assert len(pending) == 1
    sent = []

    def fake_send(text, **kwargs):
        sent.append(text)

    msgs = flush_sell_notifications(
        store, {"telegram_on_sell": True, "desktop_notify": False}, send_fn=fake_send
    )
    assert len(msgs) == 1
    assert "SELL NOW" in msgs[0]
    assert store.notified == [trade["id"]]
    assert flush_sell_notifications(store, {"telegram_on_sell": True}, send_fn=fake_send) == []
    assert "SELL NOW" in format_sell_message(updated)
