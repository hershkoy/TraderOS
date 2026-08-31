"""Windows desktop toast + sound for scanner alerts (in addition to Telegram)."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_APP_ID = (
    "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"
)
_SOUND_ALIAS = "SystemNotification"


def _xml_escape(value: str) -> str:
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def split_title_body(message: str, *, title: Optional[str] = None) -> tuple:
    lines = [ln.strip() for ln in (message or "").splitlines() if ln.strip()]
    if title:
        head = str(title).strip()[:80]
        body = "\n".join(lines)[:240]
        return head, body
    if not lines:
        return "backTraderTest", ""
    head = lines[0][:80]
    body = "\n".join(lines[1:] if len(lines) > 1 else lines)[:240]
    return head, body


def play_alert_sound() -> None:
    """Play a Windows system sound (async). No-op on non-Windows."""
    if sys.platform != "win32":
        return
    try:
        import winsound

        winsound.PlaySound(_SOUND_ALIAS, winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception as exc:
            logger.warning("desktop sound failed: %s", exc)


def _toast_script(xml_path: Path) -> str:
    path = str(xml_path).replace("'", "''")
    app = _APP_ID.replace("'", "''")
    return (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] | Out-Null; "
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null; "
        "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
        "$xml.LoadXml((Get-Content -LiteralPath '%s' -Raw -Encoding UTF8)); "
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('%s').Show($toast)"
        % (path, app)
    )


def show_toast(title: str, body: str) -> None:
    """Show a Windows toast. Visual only; sound is play_alert_sound()."""
    if sys.platform != "win32":
        logger.info("desktop toast skipped (not Windows): %s", title)
        return
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<toast>\n"
        "  <visual>\n"
        '    <binding template="ToastGeneric">\n'
        "      <text>%s</text>\n"
        "      <text>%s</text>\n"
        "    </binding>\n"
        "  </visual>\n"
        '  <audio silent="true"/>\n'
        "</toast>\n"
        % (_xml_escape(title), _xml_escape(body))
    )
    tmp = tempfile.NamedTemporaryFile(
        prefix="bt_toast_", suffix=".xml", delete=False, mode="w", encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    try:
        tmp.write(xml)
        tmp.close()
        cmd = [
            "powershell",
            "-NoProfile",
            "-STA",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            _toast_script(tmp_path),
        ]
        kwargs = {
            "args": cmd,
            "capture_output": True,
            "text": True,
            "timeout": 15,
        }
        if os.name == "nt":
            kwargs["creationflags"] = CREATE_NO_WINDOW
        proc = subprocess.run(**kwargs)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError("toast powershell failed (%s): %s" % (proc.returncode, err[:300]))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def send_desktop(message: str, *, title: Optional[str] = None, sound: bool = True) -> None:
    """Toast + optional sound. Never raises to the caller (logs instead)."""
    head, body = split_title_body(message, title=title)
    try:
        if sound:
            play_alert_sound()
        show_toast(head, body)
    except Exception as exc:
        logger.warning("desktop notify failed: %s", exc)
