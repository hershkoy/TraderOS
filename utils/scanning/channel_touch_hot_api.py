"""Dashboard helpers for 15m hot candidates (Flask + minute-tick share this)."""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.scanning.channel_touch_15m import (
    attach_last_prices,
    fetch_alpaca_last_prices,
    passes_h5_stack,
)
from utils.scanning.channel_touch_bought import (
    active_bought_symbols,
    flush_sell_notifications,
    load_bought_safe,
    sell_alerts_from_trades,
    sell_keys_from_trades,
    sync_bought_prices,
    tag_candidates_bought,
)
from utils.scanning.channel_touch_ctf import attach_channel_json
from utils.scanning.channel_touch_feed_status import build_feeds, feeds_fingerprint
from utils.scanning.channel_touch_candidates_store import (
    ChannelTouchCandidatesStore,
    normalize_settings,
)

logger = logging.getLogger(__name__)

PRICE_STALE_SEC = 5.0
# UI "quotes stale" window: Alpaca snapshot of the full book can take tens of seconds.
PRICE_UI_STALE_SEC = 120.0
# Match channel_touch_15m.py --stale-hours: H5 fills need completed IB 15m bars.
FILL_BARS_STALE_HOURS = 36.0

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


def _has_15m_watchlist(rows: Sequence[dict], n_universe: Any = 0) -> bool:
    try:
        if int(n_universe or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    for row in rows:
        if str(row.get("timeframe") or "15m") == "15m":
            return True
    return False


def resolve_15m_as_of(settings: dict, rows: Sequence[dict]) -> Optional[str]:
    """Scan as_of, else latest 15m row as_of (UTC naive/aware strings)."""
    as_of = settings.get("as_of")
    if as_of not in (None, ""):
        return str(as_of)
    times = [
        str(row.get("as_of"))
        for row in rows
        if str(row.get("timeframe") or "15m") == "15m" and row.get("as_of") not in (None, "")
    ]
    return max(times) if times else None


def fill_bars_warning(
    as_of: Optional[str],
    *,
    now: Optional[datetime] = None,
    stale_hours: float = FILL_BARS_STALE_HOURS,
    n_universe: int = 0,
    has_15m_rows: bool = False,
) -> Optional[str]:
    """Warn when IB 15m bars are too old (or missing) to evaluate H5 fills.

    Alpaca last can still mark names hot. Fills need a completed IB 15m bar.
    """
    if not has_15m_rows and int(n_universe or 0) <= 0:
        return None
    parsed = _parse_ts(as_of)
    if parsed is None:
        return (
            "15m fills cannot be evaluated: no IB 15m as_of. "
            "Hot names are Alpaca last-price proximity only, not fills. "
            "Run channel_touch_15m watchlist after backfill_ib_15m_universe.py"
        )
    now_ts = now or datetime.now(timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    age_h = (now_ts - parsed).total_seconds() / 3600.0
    if age_h <= float(stale_hours):
        return None
    return (
        "15m fills cannot be evaluated: IB 15m last bar is %s (%.1fh stale). "
        "Hot names are Alpaca last-price proximity only, not fills. "
        "If overnight backfill skipped via resume, run backfill_ib_15m_universe.py --reset-resume"
    ) % (parsed.strftime("%Y-%m-%d %H:%M:%S UTC"), age_h)


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


def is_strategy_fill(row: dict) -> bool:
    """True when a 15m row is a completed-bar H5 fill (buy-now)."""
    if str(row.get("timeframe") or "15m") != "15m":
        return False
    if str(row.get("status") or "").lower() != "filled":
        return False
    return bool(passes_h5_stack(row))


def fill_key(row: dict) -> str:
    stock = str(row.get("stock") or "").upper()
    as_of = str(row.get("as_of") or "")
    return "%s|15m|%s" % (stock, as_of)


def fill_keys_from_rows(rows: Sequence[dict]) -> List[str]:
    """Unfiltered keys for 15m H5 fills (status=filled + vol/overshoot/wait)."""
    keys: List[str] = []
    for row in rows:
        if not is_strategy_fill(row):
            continue
        stock = str(row.get("stock") or "").upper()
        if not stock:
            continue
        keys.append(fill_key(row))
    return keys


def fill_alerts_from_rows(rows: Sequence[dict]) -> List[dict]:
    """Unfiltered buy-now payloads for browser/Telegram-adjacent UI."""
    out: List[dict] = []
    for row in rows:
        if not is_strategy_fill(row):
            continue
        stock = str(row.get("stock") or "").upper()
        if not stock:
            continue
        out.append(
            {
                "stock": stock,
                "timeframe": "15m",
                "as_of": row.get("as_of"),
                "fill_px": row.get("fill_px") or row.get("buy_price"),
                "resist": row.get("resist"),
                "volume_rel_20": row.get("volume_rel_20"),
                "overshoot": row.get("overshoot"),
                "wait_bars": row.get("wait_bars"),
                "key": fill_key(row),
            }
        )
    return out


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
    symbols = sorted(
        {str(r.get("stock", "")).upper() for r in rows if r.get("stock")}
        | set(active_bought_symbols(store))
    )
    if not symbols:
        return rows
    prices = fetch_fn(symbols, batch_size=int(batch_size))
    updated = attach_last_prices(rows, prices, below_pct=float(below_pct))
    ts = now or datetime.now(timezone.utc)
    if updated:
        store.update_live_prices(updated, price_ts=ts)
    sync_bought_prices(store, prices, now=ts)
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
    bought = load_bought_safe(store, active_only=True)
    if not refresh or (not rows and not bought):
        return rows, False
    below = float(settings.get("proximity_below_pct") or 0.0)
    stale_src = rows if rows else [{"last_price_ts": t.get("last_price_ts")} for t in bought]
    with _refresh_lock:
        mono = float(monotonic_fn())
        if _last_refresh_mono is not None and (mono - _last_refresh_mono) < float(min_interval):
            return rows, False
        if not prices_are_stale(stale_src, now=now, max_age_sec=min_interval):
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
    jobs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    st = store if store is not None else get_store()
    settings = normalize_settings(st.load_settings())
    rows, refreshed = maybe_refresh_live_prices(
        st, settings, refresh=refresh, fetch_fn=fetch_fn, now=now
    )
    if not refreshed:
        prices = {}
        for row in rows:
            stock = str(row.get("stock") or "").upper()
            px = row.get("last_price")
            if stock and px is not None:
                try:
                    prices[stock] = float(px)
                except (TypeError, ValueError):
                    pass
        if prices:
            sync_bought_prices(st, prices, now=now)
    bought = load_bought_safe(st, active_only=True)
    tagged = tag_candidates_bought(rows, bought)
    filtered = filter_candidates(
        tagged,
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
    for item in ordered:
        item["h5_fill"] = is_strategy_fill(item)
    attach_channel_json(ordered)
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
    n_universe = settings.get("n_universe") or 0
    as_of_15m = resolve_15m_as_of(settings, rows)
    has_15m = _has_15m_watchlist(rows, n_universe)
    fill_warning = fill_bars_warning(
        as_of_15m,
        now=now_ts,
        n_universe=int(n_universe or 0),
        has_15m_rows=has_15m,
    )
    fill_data_ok = fill_warning is None
    parsed_as_of = _parse_ts(as_of_15m)
    fill_data_age_hours = None
    if parsed_as_of is not None:
        fill_data_age_hours = (now_ts - parsed_as_of).total_seconds() / 3600.0
    feeds = build_feeds(
        price_ts=price_ts,
        as_of_15m=as_of_15m,
        as_of_1d=settings.get("as_of_1d"),
        now=now_ts,
        jobs=jobs,
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
        "fill_keys": fill_keys_from_rows(rows),
        "fill_alerts": fill_alerts_from_rows(rows),
        "bought": bought,
        "n_bought": len(bought),
        "sell_keys": sell_keys_from_trades(bought),
        "sell_alerts": sell_alerts_from_trades(bought),
        "as_of": settings.get("as_of"),
        "as_of_1d": settings.get("as_of_1d"),
        "n_universe": n_universe,
        "n_universe_1d": settings.get("n_universe_1d") or 0,
        "stale_warning": fill_warning,
        "fill_data_ok": fill_data_ok,
        "fill_data_warning": fill_warning,
        "fill_data_age_hours": fill_data_age_hours,
        "price_ts": price_ts,
        "price_age_sec": price_age_sec,
        "prices_stale": prices_stale,
        "refreshed": refreshed,
        "feeds": feeds,
        "settings": settings,
    }


HUB_INTERVAL_SEC = 5.0
DEFAULT_PRICE_WS_PORT = 5001
PRICE_KICK_TIMEOUT_SEC = 1.0

_hub: Optional["HotCandidatesHub"] = None
_hub_lock = threading.Lock()


def price_ws_port() -> int:
    """Port the /hot page uses for live quotes (env HOT_PRICE_WS_PORT)."""
    raw = os.environ.get("HOT_PRICE_WS_PORT")
    if raw in (None, ""):
        return DEFAULT_PRICE_WS_PORT
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PRICE_WS_PORT


def kick_price_service(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    timeout: float = PRICE_KICK_TIMEOUT_SEC,
) -> bool:
    """Ask the always-on price process to push immediately. Failures are non-fatal."""
    dest = int(port if port is not None else price_ws_port())
    url = "http://%s:%s/kick" % (host, dest)
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("hot price service kick failed: %s", exc)
        return False


def payload_fingerprint(payload: Dict[str, Any]) -> str:
    """Stable id for push-skip; ignores clock fields that change every call."""
    rows = payload.get("rows") or []
    slim = []
    for row in rows:
        slim.append(
            (
                row.get("stock"),
                row.get("timeframe"),
                row.get("status"),
                bool(row.get("hot")),
                bool(row.get("h5_fill")),
                row.get("last_price"),
                row.get("dist_live_pct"),
                row.get("wait_bars"),
                row.get("volume_rel_20"),
                row.get("as_of"),
                row.get("resist"),
                bool(row.get("bought")),
            )
        )
    bought_slim = []
    for trade in payload.get("bought") or []:
        bought_slim.append(
            (
                trade.get("id"),
                trade.get("stock"),
                trade.get("timeframe"),
                trade.get("status"),
                trade.get("last_price"),
                trade.get("current_stop"),
                trade.get("dist_to_stop_pct"),
                trade.get("peak_px"),
            )
        )
    settings = payload.get("settings") or {}
    key = (
        payload.get("error"),
        payload.get("price_ts"),
        payload.get("as_of"),
        payload.get("as_of_1d"),
        payload.get("n_hot"),
        payload.get("n_armed"),
        payload.get("n_armed_15m"),
        payload.get("n_armed_1d"),
        payload.get("n_rows"),
        payload.get("n_universe"),
        payload.get("n_universe_1d"),
        payload.get("stale_warning"),
        payload.get("fill_data_ok"),
        payload.get("prices_stale"),
        feeds_fingerprint(payload.get("feeds") or []),
        tuple(payload.get("fill_keys") or []),
        tuple(payload.get("hot_keys") or []),
        tuple(payload.get("sell_keys") or []),
        tuple(bought_slim),
        settings.get("status_filter"),
        settings.get("timeframe_filter"),
        settings.get("search"),
        settings.get("max_abs_dist_pct"),
        settings.get("sort_key"),
        settings.get("sort_dir"),
        settings.get("proximity_below_pct"),
        tuple(slim),
    )
    return repr(key)


class HotCandidatesHub:
    """One Alpaca/DB refresh loop; WebSocket clients wait on updates."""

    def __init__(
        self,
        *,
        interval: float = HUB_INTERVAL_SEC,
        payload_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        always_run: bool = False,
    ) -> None:
        self.interval = float(interval)
        self.always_run = bool(always_run)
        self._payload_fn = payload_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._kick = threading.Event()
        self._stop = threading.Event()
        self._payload: Optional[Dict[str, Any]] = None
        self._seq = 0
        self._fp: Optional[str] = None
        self._clients = 0
        self._thread: Optional[threading.Thread] = None

    def _call_payload(self) -> Dict[str, Any]:
        fn = self._payload_fn
        if fn is None:
            return candidates_payload(refresh=True)
        return fn()

    def _client_count(self) -> int:
        with self._lock:
            return self._clients

    def _should_poll(self) -> bool:
        return bool(self.always_run) or self._client_count() > 0

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="hot-candidates-hub", daemon=True
        )
        self._thread.start()

    def start(self) -> None:
        """Run the refresh loop even before any WebSocket client registers."""
        self._ensure_thread()
        self.kick()

    def register(self) -> None:
        with self._lock:
            self._clients += 1
            self._ensure_thread()
        self.kick()

    def unregister(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def kick(self) -> None:
        self._kick.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            payload = self._payload or {}
            thread = self._thread
            return {
                "seq": self._seq,
                "clients": self._clients,
                "price_ts": payload.get("price_ts"),
                "running": bool(thread is not None and thread.is_alive()),
                "always_run": bool(self.always_run),
            }

    def wait_next(
        self, after_seq: int, timeout: float = 30.0
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        deadline = time.monotonic() + float(timeout)
        with self._cond:
            while self._seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return after_seq, None
                self._cond.wait(timeout=remaining)
            return self._seq, self._payload

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        self.kick()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._should_poll():
                self._kick.wait(timeout=1.0)
                self._kick.clear()
                continue
            try:
                payload = self._call_payload()
            except Exception as exc:
                logger.exception("hot-candidates hub payload failed")
                payload = {
                    "error": str(exc),
                    "rows": [],
                    "n_rows": 0,
                    "n_armed": 0,
                    "n_hot": 0,
                    "fill_keys": [],
                    "hot_keys": [],
                    "bought": [],
                    "sell_keys": [],
                }
            else:
                try:
                    settings = payload.get("settings") or {}
                    flush_sell_notifications(get_store(), settings)
                except Exception:
                    logger.exception("SELL NOW telegram flush failed")
            fp = payload_fingerprint(payload)
            if fp != self._fp:
                with self._cond:
                    self._payload = payload
                    self._seq += 1
                    self._fp = fp
                    self._cond.notify_all()
            self._kick.wait(timeout=self.interval)
            self._kick.clear()


def get_hot_hub() -> HotCandidatesHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = HotCandidatesHub()
        return _hub


def set_hot_hub(hub: Optional[HotCandidatesHub]) -> None:
    global _hub
    with _hub_lock:
        old = _hub
        _hub = hub
    if old is not None and old is not hub and hasattr(old, "stop"):
        old.stop()
