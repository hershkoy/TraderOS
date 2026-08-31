"""Unit tests for Windows desktop toast helpers (no live toast)."""
from __future__ import annotations

from unittest import mock

from utils.notify.alerts import send_alert
from utils.notify.desktop import split_title_body, _xml_escape, send_desktop


def test_split_title_body_uses_first_line():
    title, body = split_title_body("Channel-touch 15m newly hot\nAAPL last=10")
    assert title == "Channel-touch 15m newly hot"
    assert "AAPL" in body


def test_xml_escape():
    assert _xml_escape('a <b> & "c"') == "a &lt;b&gt; &amp; &quot;c&quot;"


def test_send_desktop_plays_sound_and_toasts(monkeypatch):
    calls = {"sound": 0, "toast": 0}

    def fake_sound():
        calls["sound"] += 1

    def fake_toast(title, body):
        calls["toast"] += 1
        assert title == "Hello"
        assert "world" in body

    monkeypatch.setattr("utils.notify.desktop.play_alert_sound", fake_sound)
    monkeypatch.setattr("utils.notify.desktop.show_toast", fake_toast)
    send_desktop("Hello\nworld")
    assert calls == {"sound": 1, "toast": 1}


def test_send_alert_telegram_then_desktop(monkeypatch):
    order = []
    monkeypatch.setattr(
        "utils.notify.alerts.send_message", lambda text: order.append("tg:" + text)
    )
    monkeypatch.setattr(
        "utils.notify.alerts.send_desktop",
        lambda text, title=None: order.append("desk:" + text),
    )
    send_alert("fill AAPL")
    assert order == ["tg:fill AAPL", "desk:fill AAPL"]


def test_send_alert_can_skip_desktop(monkeypatch):
    order = []
    monkeypatch.setattr(
        "utils.notify.alerts.send_message", lambda text: order.append("tg")
    )
    monkeypatch.setattr(
        "utils.notify.alerts.send_desktop",
        lambda text, title=None: order.append("desk"),
    )
    send_alert("x", desktop=False)
    assert order == ["tg"]


def test_send_alert_dry_run_skips_both(monkeypatch):
    called = []
    monkeypatch.setattr("utils.notify.alerts.send_message", lambda text: called.append("tg"))
    monkeypatch.setattr(
        "utils.notify.alerts.send_desktop", lambda text, title=None: called.append("desk")
    )
    send_alert("x", dry_run=True)
    assert called == []
