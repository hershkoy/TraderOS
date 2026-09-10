"""Parse /hot CTF Channel JSON and report date-times for the Charts page."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from utils.scanning.channel_touch_ctf import _as_float, _iso_utc_ms

ET = ZoneInfo("America/New_York")
_TZ_SUFFIX = re.compile(r"\s+(ET|EDT|EST|NY|RTH|UTC|GMT)\s*$", re.I)
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HAS_TZ = re.compile(r"(Z|[+-]\d{2}:?\d{2})\s*$", re.I)


def filter_symbols(query: str, symbols: Sequence[str], limit: int = 40) -> List[str]:
    """Prefix matches first, then substring. Case-insensitive."""
    q = (query or "").strip().upper()
    uniq: List[str] = []
    seen = set()
    for s in symbols:
        name = str(s or "").strip()
        if not name:
            continue
        key = name.upper()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(name)
    if not q:
        return uniq[: max(0, int(limit))]
    starts = [s for s in uniq if s.upper().startswith(q)]
    contains = [s for s in uniq if q in s.upper() and not s.upper().startswith(q)]
    return (starts + contains)[: max(0, int(limit))]


def ms_to_utc_naive(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def parse_goto_query(text: str) -> Dict[str, Any]:
    """Turn a pasted report/CTF clock into match keys for chart datetime strings.

    Chart OHLCV is UTC naive ``YYYY-MM-DD HH:MM:SS`` (IB/TimescaleDB). HTML
    trade tables are America/New_York RTH. Naive pastes try both.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("date / date-time is required")
    raw = _TZ_SUFFIX.sub("", raw).strip()
    if _DATE_ONLY.fullmatch(raw):
        return {
            "query": text.strip(),
            "date": raw,
            "date_only": True,
            "candidates": [raw],
            "label": raw,
        }

    has_tz = bool(_HAS_TZ.search(raw.replace(" ", ""))) or raw.upper().endswith("Z")
    stamp = pd_to_datetime(raw)
    if stamp is None:
        raise ValueError("Could not parse date / date-time: %s" % raw)

    date = stamp.strftime("%Y-%m-%d")
    candidates: List[str] = []

    if stamp.tzinfo is not None or has_tz:
        utc = stamp.tz_convert("UTC") if stamp.tzinfo is not None else stamp.tz_localize("UTC")
        candidates.append(utc.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        naive = stamp.strftime("%Y-%m-%d %H:%M:%S")
        candidates.append(naive)
        et_utc = stamp.tz_localize(ET).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        if et_utc not in candidates:
            candidates.append(et_utc)

    return {
        "query": text.strip(),
        "date": date,
        "date_only": False,
        "candidates": candidates,
        "label": candidates[0],
    }


def pd_to_datetime(raw: str):
    import pandas as pd

    t = pd.to_datetime(raw, utc=False, errors="coerce")
    if t is None or pd.isna(t):
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    if t is None or pd.isna(t):
        return None
    return pd.Timestamp(t)


def parse_channel_overlay(text: Any) -> Dict[str, Any]:
    """Normalize pasted CTF /hot Channel JSON for Plotly rails."""
    ch = _load_channel_dict(text)
    l1p = _as_float(ch.get("l1p"))
    l2p = _as_float(ch.get("l2p"))
    width = _as_float(ch.get("w"))
    l1 = _stamp(ch, "l1_ms", "l1t")
    l2 = _stamp(ch, "l2_ms", "l2t")
    if l1p is None or l2p is None or width is None or l1 is None or l2 is None:
        raise ValueError("Need l1p, l2p, w, and L1/L2 times (l1_ms/l1t, l2_ms/l2t)")
    if width <= 0:
        raise ValueError("Channel width must be > 0")
    if l1[1] == l2[1]:
        raise ValueError("L1 and L2 times must differ")

    h2 = _stamp(ch, "h2_ms", "h2t")
    fill = _stamp(ch, "fill_ms", "fill")
    en = _stamp(ch, "en_ms", "en")
    ex = _stamp(ch, "ex_ms", "ex")
    if fill is None:
        fill = en if en is not None else (h2 if h2 is not None else l2)

    sym = str(ch.get("sym") or "").strip().upper()
    out: Dict[str, Any] = {
        "sym": sym,
        "src": str(ch.get("src") or ""),
        "w": float(width),
        "l1p": float(l1p),
        "l2p": float(l2p),
        "l1_ms": l1[1],
        "l2_ms": l2[1],
        "l1_utc": ms_to_utc_naive(l1[1]),
        "l2_utc": ms_to_utc_naive(l2[1]),
        "enp": _as_float(ch.get("enp")),
        "exp": _as_float(ch.get("exp")),
    }
    if h2 is not None:
        out["h2_ms"] = h2[1]
        out["h2_utc"] = ms_to_utc_naive(h2[1])
    if fill is not None:
        out["fill_ms"] = fill[1]
        out["fill_utc"] = ms_to_utc_naive(fill[1])
        out["goto"] = out["fill_utc"]
    if en is not None:
        out["en_ms"] = en[1]
        out["en_utc"] = ms_to_utc_naive(en[1])
        if "goto" not in out:
            out["goto"] = out["en_utc"]
    if ex is not None:
        out["ex_ms"] = ex[1]
        out["ex_utc"] = ms_to_utc_naive(ex[1])
    return out


def rail_y_at(overlay: Dict[str, Any], ts_ms: int) -> float:
    l1_ms = int(overlay["l1_ms"])
    l2_ms = int(overlay["l2_ms"])
    slope = (float(overlay["l2p"]) - float(overlay["l1p"])) / float(l2_ms - l1_ms)
    return float(overlay["l1p"]) + slope * float(int(ts_ms) - l1_ms)


def _load_channel_dict(text: Any) -> dict:
    if isinstance(text, dict):
        return dict(text)
    raw = (text or "").strip() if isinstance(text, str) else ""
    if not raw:
        raise ValueError("Channel JSON is empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid Channel JSON") from exc
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid Channel JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Channel JSON must be an object")
    return parsed


def _stamp(ch: dict, ms_key: str, iso_key: str) -> Optional[tuple]:
    ms_val = ch.get(ms_key)
    if ms_val not in (None, ""):
        try:
            ms = int(round(float(ms_val)))
        except (TypeError, ValueError):
            ms = None
        else:
            iso = str(ch.get(iso_key) or "") or ms_to_utc_naive(ms)
            iso_pair = _iso_utc_ms(iso)
            if iso_pair is not None:
                return iso_pair[0], ms
            return ms_to_utc_naive(ms), ms
    return _iso_utc_ms(ch.get(iso_key))
