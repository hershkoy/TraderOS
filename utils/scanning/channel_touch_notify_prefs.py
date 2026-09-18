"""Per-detector Telegram / browser notification preferences for /hot.

Matrix: detector (15m | 1d | dp) x event (hot | armed | approaching | fill | sell)
x channel (telegram | browser).

Legacy flat flags (telegram_on_fill / telegram_on_hot / telegram_on_sell /
desktop_notify) stay in sync for older readers and are derived from this matrix.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

NOTIFY_DETECTORS = ("15m", "1d", "dp")
NOTIFY_EVENTS = ("hot", "armed", "approaching", "fill", "sell")
NOTIFY_CHANNELS = ("telegram", "browser")

# Events that are meaningful for each detector (UI greys out the rest).
NOTIFY_APPLICABLE: Dict[str, Dict[str, bool]] = {
    "15m": {
        "hot": True,
        "armed": True,
        "approaching": False,
        "fill": True,
        "sell": True,
    },
    "1d": {
        "hot": True,
        "armed": True,
        "approaching": False,
        "fill": True,
        "sell": True,
    },
    "dp": {
        "hot": False,
        "armed": True,
        "approaching": True,
        "fill": False,
        "sell": True,
    },
}

_CHANNEL_DEFAULTS = {"telegram": False, "browser": False}

DEFAULT_NOTIFY_PREFS: Dict[str, Dict[str, Dict[str, bool]]] = {
    "15m": {
        "hot": {"telegram": False, "browser": False},
        "armed": {"telegram": False, "browser": False},
        "approaching": {"telegram": False, "browser": False},
        "fill": {"telegram": True, "browser": True},
        "sell": {"telegram": True, "browser": True},
    },
    "1d": {
        "hot": {"telegram": False, "browser": False},
        "armed": {"telegram": False, "browser": False},
        "approaching": {"telegram": False, "browser": False},
        "fill": {"telegram": True, "browser": True},
        "sell": {"telegram": True, "browser": True},
    },
    "dp": {
        "hot": {"telegram": False, "browser": False},
        "armed": {"telegram": True, "browser": True},
        "approaching": {"telegram": False, "browser": False},
        "fill": {"telegram": False, "browser": False},
        "sell": {"telegram": True, "browser": True},
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return default


def _empty_prefs() -> Dict[str, Dict[str, Dict[str, bool]]]:
    out: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for det in NOTIFY_DETECTORS:
        out[det] = {}
        for ev in NOTIFY_EVENTS:
            out[det][ev] = dict(_CHANNEL_DEFAULTS)
    return out


def prefs_from_legacy(
    *,
    telegram_on_fill: bool = True,
    telegram_on_hot: bool = False,
    telegram_on_sell: bool = True,
    desktop_notify: bool = True,
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """Build prefs matching the old four global checkboxes."""
    prefs = copy.deepcopy(DEFAULT_NOTIFY_PREFS)
    prefs["15m"]["fill"]["telegram"] = bool(telegram_on_fill)
    prefs["1d"]["fill"]["telegram"] = bool(telegram_on_fill)
    prefs["dp"]["armed"]["telegram"] = bool(telegram_on_fill)
    prefs["dp"]["approaching"]["telegram"] = bool(telegram_on_hot)
    for det in NOTIFY_DETECTORS:
        prefs[det]["sell"]["telegram"] = bool(telegram_on_sell)
    # Browser historically gated fills/sells (and DP alerts when desktop was on).
    prefs["15m"]["fill"]["browser"] = bool(desktop_notify)
    prefs["1d"]["fill"]["browser"] = bool(desktop_notify)
    prefs["dp"]["armed"]["browser"] = bool(desktop_notify)
    prefs["dp"]["approaching"]["browser"] = bool(desktop_notify)
    for det in NOTIFY_DETECTORS:
        prefs[det]["sell"]["browser"] = bool(desktop_notify)
    return prefs


def legacy_from_prefs(prefs: Dict[str, Dict[str, Dict[str, bool]]]) -> Dict[str, bool]:
    """Derive flat flags so older scanners keep working."""
    p = normalize_notify_prefs(prefs)
    return {
        "telegram_on_fill": bool(
            p["15m"]["fill"]["telegram"]
            or p["1d"]["fill"]["telegram"]
            or p["dp"]["armed"]["telegram"]
        ),
        "telegram_on_hot": bool(p["dp"]["approaching"]["telegram"]),
        "telegram_on_sell": bool(
            p["15m"]["sell"]["telegram"]
            or p["1d"]["sell"]["telegram"]
            or p["dp"]["sell"]["telegram"]
        ),
        "desktop_notify": any(
            p[det][ev]["browser"]
            for det in NOTIFY_DETECTORS
            for ev in NOTIFY_EVENTS
            if NOTIFY_APPLICABLE.get(det, {}).get(ev)
        ),
    }


def _coerce_prefs_blob(raw: Any) -> Optional[dict]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(raw, dict):
        return None
    return raw


def normalize_notify_prefs(raw: Any = None) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """Fill missing detectors/events/channels; unknown keys ignored."""
    out = copy.deepcopy(DEFAULT_NOTIFY_PREFS)
    blob = _coerce_prefs_blob(raw)
    if not blob:
        return out
    for det in NOTIFY_DETECTORS:
        src_det = blob.get(det)
        if not isinstance(src_det, dict):
            continue
        for ev in NOTIFY_EVENTS:
            src_ev = src_det.get(ev)
            if not isinstance(src_ev, dict):
                continue
            for ch in NOTIFY_CHANNELS:
                if ch in src_ev:
                    out[det][ev][ch] = _as_bool(src_ev.get(ch), out[det][ev][ch])
    return out


def merge_notify_prefs(current: Any, patch: Any) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """Deep-merge a prefs patch onto current (missing keys keep current values)."""
    out = normalize_notify_prefs(current)
    blob = _coerce_prefs_blob(patch)
    if not blob:
        return out
    for det in NOTIFY_DETECTORS:
        src_det = blob.get(det)
        if not isinstance(src_det, dict):
            continue
        for ev in NOTIFY_EVENTS:
            src_ev = src_det.get(ev)
            if not isinstance(src_ev, dict):
                continue
            for ch in NOTIFY_CHANNELS:
                if ch in src_ev:
                    out[det][ev][ch] = _as_bool(src_ev.get(ch), out[det][ev][ch])
    return out


def apply_legacy_patch_to_prefs(
    prefs: Dict[str, Dict[str, Dict[str, bool]]],
    patch: dict,
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """When an old flat flag is POSTed without notify_prefs, mirror it into prefs."""
    out = normalize_notify_prefs(prefs)
    if "telegram_on_fill" in patch:
        v = _as_bool(patch.get("telegram_on_fill"), True)
        out["15m"]["fill"]["telegram"] = v
        out["1d"]["fill"]["telegram"] = v
        out["dp"]["armed"]["telegram"] = v
    if "telegram_on_hot" in patch:
        out["dp"]["approaching"]["telegram"] = _as_bool(patch.get("telegram_on_hot"), False)
    if "telegram_on_sell" in patch:
        v = _as_bool(patch.get("telegram_on_sell"), True)
        for det in NOTIFY_DETECTORS:
            out[det]["sell"]["telegram"] = v
    if "desktop_notify" in patch:
        v = _as_bool(patch.get("desktop_notify"), True)
        out["15m"]["fill"]["browser"] = v
        out["1d"]["fill"]["browser"] = v
        out["dp"]["armed"]["browser"] = v
        out["dp"]["approaching"]["browser"] = v
        for det in NOTIFY_DETECTORS:
            out[det]["sell"]["browser"] = v
    return out


def want_notify(
    settings: Optional[dict],
    detector: str,
    event: str,
    channel: str,
) -> bool:
    """True when settings allow ``channel`` alerts for ``detector``/``event``."""
    det = str(detector or "").lower()
    ev = str(event or "").lower()
    ch = str(channel or "").lower()
    if det not in NOTIFY_DETECTORS or ev not in NOTIFY_EVENTS or ch not in NOTIFY_CHANNELS:
        return False
    if not NOTIFY_APPLICABLE.get(det, {}).get(ev, False):
        return False
    settings = settings or {}
    prefs = settings.get("notify_prefs")
    if prefs is None:
        prefs = prefs_from_legacy(
            telegram_on_fill=_as_bool(settings.get("telegram_on_fill"), True),
            telegram_on_hot=_as_bool(settings.get("telegram_on_hot"), False),
            telegram_on_sell=_as_bool(settings.get("telegram_on_sell"), True),
            desktop_notify=_as_bool(settings.get("desktop_notify"), True),
        )
    else:
        prefs = normalize_notify_prefs(prefs)
    return bool(prefs.get(det, {}).get(ev, {}).get(ch, False))


def notify_prefs_for_api(prefs: Dict[str, Dict[str, Dict[str, bool]]]) -> dict:
    """Payload fragment including applicability metadata for the UI."""
    normalized = normalize_notify_prefs(prefs)
    return {
        "prefs": normalized,
        "applicable": copy.deepcopy(NOTIFY_APPLICABLE),
        "detectors": list(NOTIFY_DETECTORS),
        "events": list(NOTIFY_EVENTS),
        "channels": list(NOTIFY_CHANNELS),
    }
