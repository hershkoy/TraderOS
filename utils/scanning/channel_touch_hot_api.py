"""Dashboard helpers for 15m hot candidates (Flask + minute-tick share this)."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.scanning.channel_touch_15m import attach_last_prices, fetch_alpaca_last_prices
from utils.scanning.channel_touch_candidates_store import (
    ChannelTouchCandidatesStore,
    normalize_settings,
)

logger = logging.getLogger(__name__)

PRICE_STALE_SEC = 5.0
# UI "quotes stale" window: Alpaca snapshot of the full book can take tens of seconds.
PRICE_UI_STALE_SEC = 120.0

_store: Optional[ChannelTouchCandidatesStore] = None
_refresh_lock = threading.Lock()
_last_refresh_mono: Optional[float] = None


def get_store() -> ChannelTouchCandidatesStore:
    global _store
    if _store is None:
        _store = ChannelTouchCandidatesStore()
    return _store


def set_store(store: Optional[ChannelTouchCandidatesStore]) -> None:
    global _store
    _store = store


def reset_refresh_throttle() -> None:
    global _last_refresh_mono
    with _refresh_lock:
        _last_refresh_mono = None


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    text = str(value).replace("T", " ")
    try:
        ts = datetime.fromisoformat(text[:19])
    except ValueError:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def prices_are_stale(rows: Sequence[dict], *, now: Optional[datetime] = None, max_age_sec: float = PRICE_STALE_SEC) -> bool:
    if not rows:
        return False
    now_ts = now or datetime.now(timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    ages = []
    for row in rows:
        ts = _parse_ts(row.get("last_price_ts"))
        if ts is None:
            return True
        ages.append((now_ts - ts).total_seconds())
    return max(ages) > float(max_age_sec)


def sort_candidates(rows: Sequence[dict], *, sort_key: str = "abs_dist", sort_dir: str = "asc") -> List[dict]:
    desc = str(sort_dir).lower() == "desc"
    key = str(sort_key or "abs_dist")

    def _num(row: dict, field: str, missing: float) -> float:
        val = row.get(field)
        if val is None:
            return missing
        try:
            return float(val)
        except (TypeError, ValueError):
            return missing

    def key_fn(row: dict) -> tuple:
        if key == "abs_dist":
            dist = row.get("dist_live_pct")
            if dist is None:
                return (1, 1e9)
            try:
                return (0, abs(float(dist)))
            except (TypeError, ValueError):
                return (1, 1e9)
        if key == "dist_live_pct":
            return (_num(row, "dist_live_pct", -1e9 if desc else 1e9),)
        if key == "last_price":
            return (_num(row, "last_price", -1e9 if desc else 1e9),)
        if key == "wait_bars":
            return (_num(row, "wait_bars", -1e9 if desc else 1e9),)
        if key == "volume_rel_20":
            return (_num(row, "volume_rel_20", -1e9 if desc else 1e9),)
        if key == "resist":
            return (_num(row, "resist", -1e9 if desc else 1e9),)
        if key == "status":
            return (str(row.get("status") or ""),)
        if key == "stock":
            return (str(row.get("stock") or ""),)
        if key == "timeframe":
            return (str(row.get("timeframe") or "15m"),)
        if key == "hot":
            return (0 if row.get("hot") else 1,)
        return (str(row.get(key) or ""),)

    ordered = sorted(rows, key=key_fn, reverse=desc)
    return [dict(r) for r in ordered]


def hot_keys_from_rows(rows: Sequence[dict]) -> List[str]:
    """Unfiltered ``STOCK|timeframe`` keys for rows that are hot now."""
    keys: List[str] = []
    for row in rows:
        if not row.get("hot"):
            continue
        stock = str(row.get("stock") or "").upper()
        tf = str(row.get("timeframe") or "15m")
        if not stock:
            continue
        keys.append("%s|%s" % (stock, tf))
    return keys


def filter_candidates(
    rows: Sequence[dict],
    *,
    status_filter: str = "all",
    max_abs_dist_pct: Optional[float] = None,
    search: str = "",
    timeframe_filter: str = "all",
) -> List[dict]:
    status = str(status_filter or "all").lower()
    needle = str(search or "").strip().upper()
    cap = None
    if max_abs_dist_pct is not None:
        try:
            cap = abs(float(max_abs_dist_pct))
        except (TypeError, ValueError):
            cap = None
    out: List[dict] = []
    for row in rows:
        if status == "hot":
            if not row.get("hot"):
                continue
        elif status not in ("", "all"):
            if str(row.get("status") or "").lower() != status:
                continue
        tf_filter = str(timeframe_filter or "all").lower()
        if tf_filter in ("15m", "1d"):
            row_tf = str(row.get("timeframe") or "15m")
            if row_tf != tf_filter:
                continue
        if needle and needle not in str(row.get("stock") or "").upper():
            continue
        if cap is not None:
            dist = row.get("dist_live_pct")
            if dist is None:
                continue
            try:
                if abs(float(dist)) > cap:
                    continue
            except (TypeError, ValueError):
                continue
        out.append(dict(row))
    return out


def refresh_live_prices(
    store: ChannelTouchCandidatesStore,
    *,
    below_pct: float = 0.0,
    fetch_fn: Callable[..., Dict[str, float]] = fetch_alpaca_last_prices,
    now: Optional[datetime] = None,
    batch_size: int = 200,
) -> List[dict]:
    rows = store.load_rows()
    symbols = sorted({str(r.get("stock", "")).upper() for r in rows if r.get("stock")})
    if not symbols:
        return rows
    prices = fetch_fn(symbols, batch_size=int(batch_size))
    updated = attach_last_prices(rows, prices, below_pct=float(below_pct))
    ts = now or datetime.now(timezone.utc)
    store.update_live_prices(updated, price_ts=ts)
    return store.load_rows()


def maybe_refresh_live_prices(
    store: ChannelTouchCandidatesStore,
    settings: dict,
    *,
    refresh: bool = True,
    min_interval: float = PRICE_STALE_SEC,
    fetch_fn: Callable[..., Dict[str, float]] = fetch_alpaca_last_prices,
    now: Optional[datetime] = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> Tuple[List[dict], bool]:
    """Refresh Alpaca last if rows are stale. Process-level throttle for many tabs."""
    global _last_refresh_mono
    rows = store.load_rows()
    if not refresh or not rows:
        return rows, False
    below = float(settings.get("proximity_below_pct") or 0.0)
    with _refresh_lock:
        mono = float(monotonic_fn())
        if _last_refresh_mono is not None and (mono - _last_refresh_mono) < float(min_interval):
            return rows, False
        if not prices_are_stale(rows, now=now, max_age_sec=min_interval):
            return rows, False
        try:
            rows = refresh_live_prices(store, below_pct=below, fetch_fn=fetch_fn, now=now)
            _last_refresh_mono = mono
            return rows, True
        except Exception:
            logger.exception("Alpaca last-price refresh failed; serving stored prices")
            _last_refresh_mono = mono
            return store.load_rows(), False


def candidates_payload(
    store: Optional[ChannelTouchCandidatesStore] = None,
    *,
    refresh: bool = True,
    fetch_fn: Callable[..., Dict[str, float]] = fetch_alpaca_last_prices,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    st = store if store is not None else get_store()
    settings = normalize_settings(st.load_settings())
    rows, refreshed = maybe_refresh_live_prices(
        st, settings, refresh=refresh, fetch_fn=fetch_fn, now=now
    )
    filtered = filter_candidates(
        rows,
        status_filter=str(settings.get("status_filter") or "all"),
        max_abs_dist_pct=settings.get("max_abs_dist_pct"),
        search=str(settings.get("search") or ""),
        timeframe_filter=str(settings.get("timeframe_filter") or "all"),
    )
    ordered = sort_candidates(
        filtered,
        sort_key=str(settings.get("sort_key") or "abs_dist"),
        sort_dir=str(settings.get("sort_dir") or "asc"),
    )
    n_hot = sum(1 for r in rows if r.get("hot"))
    n_armed = sum(1 for r in rows if r.get("status") == "armed")
    n_armed_15m = sum(
        1 for r in rows if r.get("status") == "armed" and str(r.get("timeframe") or "15m") == "15m"
    )
    n_armed_1d = sum(
        1 for r in rows if r.get("status") == "armed" and str(r.get("timeframe") or "15m") == "1d"
    )
    price_ts = None
    for row in rows:
        ts = row.get("last_price_ts")
        if ts and (price_ts is None or str(ts) > str(price_ts)):
            price_ts = ts
    now_ts = now or datetime.now(timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    parsed_price_ts = _parse_ts(price_ts)
    price_age_sec = None
    if parsed_price_ts is not None:
        price_age_sec = (now_ts - parsed_price_ts).total_seconds()
    prices_stale = bool(rows) and (
        parsed_price_ts is None or float(price_age_sec or 0) > PRICE_UI_STALE_SEC
    )
    return {
        "rows": ordered,
        "all_rows": len(rows),
        "n_rows": len(ordered),
        "n_armed": n_armed,
        "n_armed_15m": n_armed_15m,
        "n_armed_1d": n_armed_1d,
        "n_hot": n_hot,
        "hot_keys": hot_keys_from_rows(rows),
        "as_of": settings.get("as_of"),
        "as_of_1d": settings.get("as_of_1d"),
        "n_universe": settings.get("n_universe") or 0,
        "n_universe_1d": settings.get("n_universe_1d") or 0,
        "stale_warning": settings.get("stale_warning"),
        "price_ts": price_ts,
        "price_age_sec": price_age_sec,
        "prices_stale": prices_stale,
        "refreshed": refreshed,
        "settings": settings,
    }
