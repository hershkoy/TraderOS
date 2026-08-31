"""Send scanner alerts to Telegram and the Windows desktop (sound + toast)."""
from __future__ import annotations

import logging
from typing import Optional

from utils.notify.desktop import send_desktop
from utils.notify.telegram_pinger import send_message

logger = logging.getLogger(__name__)


def send_alert(
    text: str,
    *,
    dry_run: bool = False,
    title: Optional[str] = None,
    desktop: bool = True,
) -> None:
    """Telegram first, then optional desktop toast + sound. Desktop failure does not raise."""
    if dry_run:
        logger.info("[dry-run] would notify:\n%s", text)
        return
    send_message(text)
    if desktop:
        send_desktop(text, title=title)
