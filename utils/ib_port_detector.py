"""
Utilities for discovering a listening IB Gateway/TWS port by attempting real
IB connections (avoids raw socket probing that can interfere with TWS/Gateway).
"""

import logging
from typing import Iterable, Optional

from ib_insync import IB


LOGGER = logging.getLogger(__name__)
DEFAULT_PORTS = (4001, 4002, 7496, 4697)
PROBE_CLIENT_ID = 98


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
    timeout: float = 2.0,
) -> Optional[int]:
    """
    Find the first IB port that is currently listening.

    Args:
        candidate_ports: Iterable of port numbers to try (defaults to
            4001, 4002, 7496, 4697).
        host: Hostname or IP address to probe.
        timeout: Timeout per connection attempt in seconds.

    Returns:
        The first listening port, or None if no ports responded.
    """
    ports = list(candidate_ports) if candidate_ports else list(DEFAULT_PORTS)

    for port in ports:
        if _probe_port(host, port, timeout):
            LOGGER.info("Detected listening IB port at %s:%s", host, port)
            return port

    LOGGER.warning(
        "Unable to detect an IB port on %s within %s seconds (ports tried: %s)",
        host,
        timeout,
        ", ".join(str(p) for p in ports),
    )
    return None


__all__ = ["detect_ib_port", "DEFAULT_PORTS"]

