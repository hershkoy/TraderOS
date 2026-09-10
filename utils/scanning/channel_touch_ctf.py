"""CTF (Channel Touch Fill Viewer) paste JSON for /hot candidates."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _as_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _as_int(val: Any) -> Optional[int]:
    x = _as_float(val)
    if x is None:
        return None
    return int(round(x))


def _iso_utc_ms(val: Any) -> Optional[Tuple[str, int]]:
    if val is None or val == "":
        return None
    text = str(val).strip()
    if text in ("", "None", "nan"):
        return None
    t = pd.to_datetime(text, utc=True, errors="coerce")
    if t is None or pd.isna(t):
        t = pd.to_datetime(text, errors="coerce")
        if t is None or pd.isna(t):
            return None
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
    else:
        t = pd.Timestamp(t)
    ms = int(round(t.timestamp() * 1000))
    return t.strftime("%Y-%m-%dT%H:%M:%SZ"), ms


def channel_rails_fingerprint(ch: Optional[dict]) -> str:
    """Stable id for copied-state; ignores fill/entry that move with as_of."""
    if not ch:
        return ""
    parts = (
        str(ch.get("sym") or ""),
        str(ch.get("src") or ""),
        str(ch.get("l1t") or ""),
        str(ch.get("l1p") or ""),
        str(ch.get("l2t") or ""),
        str(ch.get("l2p") or ""),
        str(ch.get("w") or ""),
        str(ch.get("h2t") or ""),
    )
    return "|".join(parts)


def channel_json_for_candidate(row: dict) -> Optional[Dict[str, Any]]:
    """Compact CTF paste payload from a /hot candidate row.

    Prefer explicit l1/l2 stamps when present. Otherwise rebuild L1 from
    support_y0/channel_start and a second support point at H2 (src=approx),
    matching the HTML report reconstruct path when L2 is missing.
    """
    if not row:
        return None
    sym = str(row.get("stock") or "").upper()
    width = _as_float(row.get("channel_width"))
    if not sym or width is None or width <= 0:
        return None

    l1 = _iso_utc_ms(row.get("l1_time")) or _iso_utc_ms(row.get("channel_start"))
    l2 = _iso_utc_ms(row.get("l2_time"))
    h2 = _iso_utc_ms(row.get("h2_time")) or _iso_utc_ms(row.get("channel_end"))
    l1p = _as_float(row.get("l1_price"))
    l2p = _as_float(row.get("l2_price"))
    src = "rails"

    sy0 = _as_float(row.get("support_y0"))
    sslope = _as_float(row.get("support_slope"))
    sx0 = _as_int(row.get("support_x0"))
    h2_idx = _as_int(row.get("h2_idx"))

    if l1p is None:
        l1p = sy0
    if l1 is None or l1p is None:
        return None

    if l2 is None or l2p is None:
        if h2 is None or sslope is None or sx0 is None or h2_idx is None or sy0 is None:
            return None
        l2 = h2
        l2p = float(sy0) + float(sslope) * float(h2_idx - sx0)
        src = "approx"

    if l2 is None or l2p is None or not (l2p == l2p) or l2[1] == l1[1]:
        return None

    fill = _iso_utc_ms(row.get("as_of"))
    if fill is None:
        fill = h2 if h2 is not None else l2

    out: Dict[str, Any] = {
        "v": 1,
        "sym": sym,
        "src": src,
        "fill": fill[0],
        "fill_ms": fill[1],
        "en": fill[0],
        "en_ms": fill[1],
        "l1t": l1[0],
        "l1_ms": l1[1],
        "l1p": round(float(l1p), 6),
        "l2t": l2[0],
        "l2_ms": l2[1],
        "l2p": round(float(l2p), 6),
        "w": round(float(width), 6),
    }
    entry_px = _as_float(row.get("fill_px"))
    if entry_px is not None:
        out["enp"] = round(float(entry_px), 6)
    if h2 is not None:
        out["h2t"] = h2[0]
        out["h2_ms"] = h2[1]
    out["fp"] = channel_rails_fingerprint(out)
    return out


def attach_channel_json(rows: list) -> list:
    """Add ``ch`` (CTF paste dict) on each candidate row in place."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        ch = channel_json_for_candidate(row)
        if ch is not None:
            row["ch"] = ch
        else:
            row.pop("ch", None)
    return rows
