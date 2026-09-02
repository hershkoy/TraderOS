"""Unit tests for IB port detection defaults (no Gateway)."""
from __future__ import annotations

import inspect

from utils.api.ib.ib_port_detector import CONNECT_TIMEOUT, DEFAULT_PORTS, detect_ib_port


def test_live_gateway_port_is_first():
    assert DEFAULT_PORTS[0] == 4001


def test_connect_timeout_matches_ib_conn():
    assert CONNECT_TIMEOUT == 10.0
    assert inspect.signature(detect_ib_port).parameters["timeout"].default == CONNECT_TIMEOUT
