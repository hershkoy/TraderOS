"""
Telegram pinger for cron / scanner alerts.

Reads credentials from the environment (see .env):
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Usage:
  from utils.notify.telegram_pinger import send_message, ping
  send_message("Channel-touch: TKO trigger")
  ping()  # connectivity check
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from utils.config.env_loader import get_env_var, load_env_file

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
DEFAULT_TIMEOUT_SEC = 30


def _credentials() -> tuple[str, str]:
    load_env_file()
    token = (get_env_var("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (get_env_var("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set (add it to .env)")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set (add it to .env)")
    return token, chat_id


def send_message(
    message: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    disable_notification: bool = False,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict:
    """
    Send a plain-text Telegram message via Bot API.

    Returns the parsed JSON response from Telegram.
    Raises ValueError on missing credentials, RuntimeError on API failure.
    """
    text = (message or "").strip()
    if not text:
        raise ValueError("message must be non-empty")

    token = (bot_token or "").strip()
    chat = (chat_id or "").strip()
    if not token or not chat:
        token, chat = _credentials()

    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
        "disable_notification": bool(disable_notification),
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)
    except requests.RequestException as exc:
        raise RuntimeError(f"Telegram request failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Telegram returned non-JSON (HTTP {resp.status_code})") from exc

    if resp.status_code != 200 or not data.get("ok"):
        desc = data.get("description") if isinstance(data, dict) else resp.text
        raise RuntimeError(f"Telegram sendMessage failed: {desc}")

    logger.info("Telegram message sent (message_id=%s)", data.get("result", {}).get("message_id"))
    return data


def ping(*, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> dict:
    """Send a short connectivity check message."""
    return send_message(
        "Telegram pinger OK",
        bot_token=bot_token,
        chat_id=chat_id,
    )


def main() -> int:
    """CLI: python -m utils.notify.telegram_pinger [--ping] [message...]"""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Send a Telegram message using TELEGRAM_* env vars")
    ap.add_argument("--ping", action="store_true", help="Send connectivity check only")
    ap.add_argument("message", nargs="*", help="Message text (joined with spaces)")
    args = ap.parse_args()

    try:
        if args.ping:
            ping()
        else:
            text = " ".join(args.message).strip()
            if not text:
                ap.error("Provide a message or use --ping")
            send_message(text)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
