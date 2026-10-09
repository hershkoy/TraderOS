"""Unit tests for IB port detection defaults (no Gateway)."""
from __future__ import annotations

import inspect
from pathlib import Path

from utils.api.ib.ib_port_detector import (
    CONNECT_TIMEOUT,
    DEFAULT_PORTS,
    detect_ib_port,
    ordered_ib_ports,
    read_cached_ib_port,
    remember_ib_port,
)


def test_default_ports_cover_gateway_and_tws():
    assert DEFAULT_PORTS == (4001, 4002, 7496, 7497)
    assert DEFAULT_PORTS[0] == 4001


def test_connect_timeout_matches_ib_conn():
    assert CONNECT_TIMEOUT == 10.0
    assert inspect.signature(detect_ib_port).parameters["timeout"].default == CONNECT_TIMEOUT


def test_ordered_ib_ports_prefers_env_then_cache_then_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("IB_PORT", raising=False)
    cache = tmp_path / "ib_api_port.txt"
    cache.write_text("7496", encoding="utf-8")
    monkeypatch.setattr(
        "utils.api.ib.ib_port_detector.PORT_CACHE_PATH",
        cache,
    )
    assert ordered_ib_ports()[0] == 7496
    monkeypatch.setenv("IB_PORT", "7497")
    assert ordered_ib_ports()[:2] == [7497, 7496]
    assert ordered_ib_ports(4001) == [4001]


def test_remember_ib_port_writes_cache_and_env(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("IB_PORT", raising=False)
    cache = tmp_path / "ib_api_port.txt"
    remember_ib_port(7496, path=cache)
    assert Path(cache).read_text(encoding="utf-8").strip() == "7496"
    assert read_cached_ib_port(path=cache) == 7496
    assert os.environ["IB_PORT"] == "7496"
