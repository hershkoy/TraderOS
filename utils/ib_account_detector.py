"""
Utilities for auto-detecting the default IB account from managed accounts.
"""

import logging
from typing import Optional

from utils.fetch_data import get_ib_connection

LOGGER = logging.getLogger(__name__)


def detect_ib_account(port: Optional[int] = None) -> Optional[str]:
    """
    Get the default IB account from managed accounts.
    
    Args:
        port: Optional IB API port number (default: from IB_PORT env var or auto-detect)
    
    Returns:
        The first managed account ID, or None if no accounts are available.
    """
    try:
        ib = get_ib_connection(port=port)
        
        if not ib.isConnected():
            LOGGER.warning("IB connection is not active, cannot detect account")
            return None
        
        accounts = ib.managedAccounts()
        
        if not accounts:
            LOGGER.warning("No managed accounts found")
            return None
        
        default_account = accounts[0]
        LOGGER.info("Detected default IB account: %s", default_account)
        
        if len(accounts) > 1:
            LOGGER.info("Available accounts: %s", ", ".join(accounts))
        
        return default_account
    
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Error detecting IB account: %s", exc)
        return None


__all__ = ["detect_ib_account"]


