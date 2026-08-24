"""Unit tests for utils.notify.telegram_pinger."""
from __future__ import annotations

from unittest import mock

import pytest

from utils.notify import telegram_pinger as tp


def test_send_message_requires_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with mock.patch.object(tp, "get_env_var", side_effect=lambda k, default=None: None):
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            tp.send_message("hello")


def test_send_message_posts_payload(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    fake = mock.Mock()
    fake.status_code = 200
    fake.json.return_value = {"ok": True, "result": {"message_id": 7}}

    with mock.patch.object(tp.requests, "post", return_value=fake) as post:
        out = tp.send_message("hello world")

    assert out["ok"] is True
    assert post.call_count == 1
    args, kwargs = post.call_args
    assert args[0] == "https://api.telegram.org/bottok123/sendMessage"
    assert kwargs["json"]["chat_id"] == "42"
    assert kwargs["json"]["text"] == "hello world"


def test_send_message_raises_on_api_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    fake = mock.Mock()
    fake.status_code = 400
    fake.json.return_value = {"ok": False, "description": "Bad Request"}

    with mock.patch.object(tp.requests, "post", return_value=fake):
        with pytest.raises(RuntimeError, match="Bad Request"):
            tp.send_message("x")


def test_ping_uses_send_message(monkeypatch):
    with mock.patch.object(tp, "send_message", return_value={"ok": True}) as sm:
        out = tp.ping()
    assert out["ok"] is True
    sm.assert_called_once()
    assert "OK" in sm.call_args.args[0]
