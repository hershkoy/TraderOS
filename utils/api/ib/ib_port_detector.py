"""
Utilities for discovering a listening IB Gateway/TWS port by attempting real
IB connections (avoids raw socket probing that can interfere with TWS/Gateway).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from ib_insync import IB


LOGGER = logging.getLogger(__name__)
# Gateway live/paper, then TWS live/paper.
DEFAULT_PORTS = (4001, 4002, 7496, 7497)
PROBE_CLIENT_ID = 98
# Match tests/utils/ib_conn.py. A 2s handshake timeout false-fails when
# Gateway accepts TCP but is slow to send the API version.
CONNECT_TIMEOUT = 10.0

# Persist last successful API port across cron process restarts so a TWS
# session does not re-pay Gateway timeouts every 15m bar.
_REPO_ROOT = Path(__file__).resolve().parents[3]
PORT_CACHE_PATH = _REPO_ROOT / "logs" / "cron" / "ib_api_port.txt"


def read_cached_ib_port(path: Optional[Path] = None) -> Optional[int]:
    """Return the last successful IB API port, if cached."""
    cache_path = Path(path) if path is not None else PORT_CACHE_PATH
    try:
        raw = cache_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid cached IB port value: %s", raw)
        return None


def write_cached_ib_port(port: int, path: Optional[Path] = None) -> None:
    """Remember a working IB API port for the next process."""
    cache_path = Path(path) if path is not None else PORT_CACHE_PATH
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(str(int(port)), encoding="utf-8")
    except OSError as exc:
        LOGGER.debug("Unable to cache IB port %s: %s", port, exc)


def remember_ib_port(port: int, path: Optional[Path] = None) -> int:
    """Set IB_PORT env and persist the working port for later runs."""
    value = int(port)
    os.environ["IB_PORT"] = str(value)
    write_cached_ib_port(value, path=path)
    return value


def ordered_ib_ports(
    explicit_port: Optional[int] = None,
    *,
    env_port: Optional[str] = None,
    cached_port: Optional[int] = None,
    default_ports: Optional[Iterable[int]] = None,
) -> List[int]:
    """Ordered Gateway/TWS candidates without an extra handshake probe.

    Prefer explicit -> IB_PORT -> last-good cache -> DEFAULT_PORTS.
    Live connect with the real client id should try this list in order.
    """
    ports: List[int] = []

    def _add(value: Optional[int]) -> None:
        if value is None:
            return
        port = int(value)
        if port not in ports:
            ports.append(port)

    if explicit_port is not None:
        _add(explicit_port)
        return ports

    raw_env = env_port if env_port is not None else os.getenv("IB_PORT")
    if raw_env:
        try:
            _add(int(raw_env))
        except ValueError:
            LOGGER.warning("Invalid IB_PORT environment value: %s", raw_env)

    if cached_port is not None:
        _add(cached_port)
    else:
        _add(read_cached_ib_port())

    for default_port in list(default_ports) if default_ports is not None else list(DEFAULT_PORTS):
        _add(default_port)
    return ports


def _probe_port(host: str, port: int, timeout: float) -> bool:
    """Attempt to open and immediately close an IB connection on the port."""
    ib = IB()
    try:
        ib.connect(host, port, clientId=PROBE_CLIENT_ID, timeout=timeout)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Port probe failed for %s:%s -> %s", host, port, exc)
        return False
    finally:
        if ib.isConnected():
            try:
                ib.disconnect()
            except Exception:  # noqa: BLE001
                pass


def detect_ib_port(
    candidate_ports: Optional[Iterable[int]] = None,
    host: str = "127.0.0.1",
    timeout: float = CONNECT_TIMEOUT,
) -> Optional[int]:
    """
    Find the first IB port that is currently listening.

    Args:
        candidate_ports: Iterable of port numbers to try (defaults to
            4001, 4002, 7496, 7497).
        host: Hostname or IP address to probe.
        timeout: Timeout per connection attempt in seconds.

    Returns:
        The first listening port, or None if no ports responded.
    """
    ports = list(candidate_ports) if candidate_ports else ordered_ib_ports()

    for port in ports:
        if _probe_port(host, port, timeout):
            LOGGER.info("Detected listening IB port at %s:%s", host, port)
            remember_ib_port(port)
            return port

    LOGGER.warning(
        "Unable to detect an IB port on %s within %s seconds (ports tried: %s)",
        host,
        timeout,
        ", ".join(str(p) for p in ports),
    )
    return None


__all__ = [
    "detect_ib_port",
    "ordered_ib_ports",
    "read_cached_ib_port",
    "write_cached_ib_port",
    "remember_ib_port",
    "DEFAULT_PORTS",
    "CONNECT_TIMEOUT",
    "PROBE_CLIENT_ID",
    "PORT_CACHE_PATH",
]
