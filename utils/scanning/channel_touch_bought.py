"""Track /hot Bought positions and the live ATR/trail stop (SELL NOW).

Keeper exit (same as nightly occupancy): ATR hard-stop k=2.0 clamped
1.5%-6% (3% fallback if no ATR), plus a 10% trail from peak. Live peak
and stop use Alpaca last as a proxy for bar high/low so SELL NOW is
actionable while the bar is still forming. Nightly occupancy still sims
completed bars; this is the dashboard/proximity live book.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TRAIL_PCT = 0.10
ATR_LEN = 14
ATR_STOP_MULT = 2.0
STOP_PCT = 0.03
STOP_PCT_FLOOR = 0.015
STOP_PCT_CEIL = 0.06
ATR_LOOKBACK_BARS = 40
ACTIVE_STATUSES = ("open", "sell_now")


def hard_stop_price(
    entry_px: float,
    *,
    atr_at_entry: Optional[float] = None,
    atr_stop_mult: Optional[float] = ATR_STOP_MULT,
    stop_pct: float = STOP_PCT,
    stop_pct_floor: float = STOP_PCT_FLOOR,
    stop_pct_ceil: float = STOP_PCT_CEIL,
) -> float:
    """Fixed % stop, or ATR-scaled distance clamped to [floor, ceil] of price."""
    px = float(entry_px)
    if (
        atr_stop_mult is not None
        and atr_at_entry is not None
        and np.isfinite(float(atr_at_entry))
        and float(atr_at_entry) > 0
        and px > 0
    ):
        dist = float(atr_stop_mult) * float(atr_at_entry)
        dist = max(px * float(stop_pct_floor), min(px * float(stop_pct_ceil), dist))
        return px - dist
    return px * (1.0 - float(stop_pct))


def current_stop_price(
    hard_stop: float,
    peak_px: float,
    trail_pct: float = TRAIL_PCT,
) -> float:
    trail_stop = float(peak_px) * (1.0 - float(trail_pct))
    return max(float(hard_stop), trail_stop)


def dist_to_stop_pct(last_px: Optional[float], stop_px: Optional[float]) -> Optional[float]:
    """Remaining cushion as percent of last. Negative = last is through the stop."""
    if last_px is None or stop_px is None:
        return None
    try:
        last_f = float(last_px)
        stop_f = float(stop_px)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(last_f) or last_f <= 0 or not np.isfinite(stop_f):
        return None
    return round((last_f - stop_f) / last_f * 100.0, 4)


def atr_last(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    length: int = ATR_LEN,
) -> Optional[float]:
    """Wilder ATR of the last bar; None if too short or not finite."""
    n = min(len(high), len(low), len(close))
    if n < int(length) + 1:
        return None
    hi = np.asarray(high[:n], dtype=float)
    lo = np.asarray(low[:n], dtype=float)
    cl = np.asarray(close[:n], dtype=float)
    prev = cl[:-1]
    tr = np.empty(n, dtype=float)
    tr[0] = hi[0] - lo[0]
    tr[1:] = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - prev), np.abs(lo[1:] - prev)))
    atr = np.full(n, np.nan, dtype=float)
    atr[int(length) - 1] = float(np.nanmean(tr[: int(length)]))
    alpha = 1.0 / float(length)
    for i in range(int(length), n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    last = float(atr[-1])
    if not np.isfinite(last) or last <= 0:
        return None
    return last


def entry_px_from_candidate(row: dict) -> Optional[float]:
    for key in ("fill_px", "last_price", "last_close"):
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            px = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(px) and px > 0:
            return px
    return None


def apply_price_tick(trade: dict, last_px: Optional[float], *, last_ts: Any = None) -> dict:
    """Ratchet peak from last, recompute stop, flip open -> sell_now if last hits it."""
    out = dict(trade)
    if last_px is None:
        return out
    try:
        px = float(last_px)
    except (TypeError, ValueError):
        return out
    if not np.isfinite(px) or px <= 0:
        return out
    out["last_price"] = px
    if last_ts is not None:
        out["last_price_ts"] = last_ts
    entry = float(out.get("entry_px") or px)
    peak = max(float(out.get("peak_px") or 0.0), px, entry)
    out["peak_px"] = peak
    hard = float(out.get("hard_stop") or hard_stop_price(entry))
    trail = float(out.get("trail_pct") if out.get("trail_pct") is not None else TRAIL_PCT)
    stop = current_stop_price(hard, peak, trail)
    out["hard_stop"] = hard
    out["current_stop"] = stop
    out["dist_to_stop_pct"] = dist_to_stop_pct(px, stop)
    if str(out.get("status") or "open") == "open" and px <= stop + 1e-9:
        out["status"] = "sell_now"
        out["exit_reason"] = "trail_stop" if stop > hard + 1e-9 else "hard_stop"
    return out


def build_bought_trade(
    *,
    stock: str,
    timeframe: str,
    entry_px: float,
    h2_time: Optional[str] = None,
    atr_at_entry: Optional[float] = None,
    trail_pct: float = TRAIL_PCT,
    last_price: Optional[float] = None,
    last_price_ts: Any = None,
    entry_ts: Any = None,
) -> dict:
    px = float(entry_px)
    hard = hard_stop_price(px, atr_at_entry=atr_at_entry)
    peak = px
    last = float(last_price) if last_price is not None else px
    if np.isfinite(last) and last > peak:
        peak = last
    stop = current_stop_price(hard, peak, trail_pct)
    stop_pct_used = round((1.0 - hard / px) * 100.0, 3) if px > 0 else None
    ts = entry_ts or datetime.now(timezone.utc)
    trade = {
        "stock": str(stock).upper(),
        "timeframe": str(timeframe or "15m"),
        "h2_time": None if h2_time in (None, "") else str(h2_time),
        "entry_px": px,
        "entry_ts": ts,
        "hard_stop": hard,
        "atr_at_entry": None if atr_at_entry is None else float(atr_at_entry),
        "stop_pct_used": stop_pct_used,
        "trail_pct": float(trail_pct),
        "peak_px": peak,
        "current_stop": stop,
        "last_price": last,
        "last_price_ts": last_price_ts or ts,
        "dist_to_stop_pct": dist_to_stop_pct(last, stop),
        "status": "open",
        "exit_reason": None,
        "sell_notified_at": None,
        "closed_at": None,
    }
    return apply_price_tick(trade, last, last_ts=trade["last_price_ts"])


def sell_key(trade: dict) -> str:
    stock = str(trade.get("stock") or "").upper()
    tf = str(trade.get("timeframe") or "15m")
    tid = trade.get("id")
    if tid not in (None, ""):
        return "%s|%s|%s" % (stock, tf, tid)
    return "%s|%s" % (stock, tf)


def sell_keys_from_trades(trades: Sequence[dict]) -> List[str]:
    keys: List[str] = []
    for trade in trades:
        if str(trade.get("status") or "") != "sell_now":
            continue
        stock = str(trade.get("stock") or "").upper()
        if not stock:
            continue
        keys.append(sell_key(trade))
    return keys


def sell_alerts_from_trades(trades: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for trade in trades:
        if str(trade.get("status") or "") != "sell_now":
            continue
        stock = str(trade.get("stock") or "").upper()
        if not stock:
            continue
        out.append(
            {
                "stock": stock,
                "timeframe": str(trade.get("timeframe") or "15m"),
                "entry_px": trade.get("entry_px"),
                "last_price": trade.get("last_price"),
                "current_stop": trade.get("current_stop"),
                "dist_to_stop_pct": trade.get("dist_to_stop_pct"),
                "exit_reason": trade.get("exit_reason"),
                "id": trade.get("id"),
                "key": sell_key(trade),
            }
        )
    return out


def format_sell_message(trades: Sequence[dict]) -> str:
    rows = [t for t in trades if str(t.get("status") or "") == "sell_now"]
    n = len(rows)
    title = "SELL NOW — Channel-touch stop" if n == 1 else "SELL NOW — Channel-touch stops"
    lines = [
        title,
        "ATR k=2 hard-stop (1.5%-6%) + 10% trail from peak",
        "n=%d" % n,
        "",
    ]
    for trade in rows:
        last = trade.get("last_price")
        stop = trade.get("current_stop")
        entry = trade.get("entry_px")
        dist = trade.get("dist_to_stop_pct")
        last_s = "%.4f" % float(last) if last is not None else "n/a"
        stop_s = "%.4f" % float(stop) if stop is not None else "n/a"
        entry_s = "%.4f" % float(entry) if entry is not None else "n/a"
        dist_s = "%+.2f%%" % float(dist) if dist is not None else "n/a"
        lines.append(
            "{stock} {tf} last={last} stop={stop} entry={entry} dist={dist} reason={reason}".format(
                stock=trade.get("stock"),
                tf=trade.get("timeframe") or "15m",
                last=last_s,
                stop=stop_s,
                entry=entry_s,
                dist=dist_s,
                reason=trade.get("exit_reason") or "stop",
            )
        )
    return "\n".join(lines)


def pending_sell_notifies(trades: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for trade in trades:
        if str(trade.get("status") or "") != "sell_now":
            continue
        if trade.get("sell_notified_at") not in (None, "", "None"):
            continue
        out.append(dict(trade))
    return out


def load_recent_ohlc(
    symbol: str,
    timeframe: str,
    *,
    n: int = ATR_LOOKBACK_BARS,
    runner: Optional[Callable[..., Any]] = None,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """Latest N bars oldest-first. 15m prefers IB; 1d prefers ALPACA then IB."""
    stock = str(symbol or "").upper()
    tf = str(timeframe or "15m")
    if not stock:
        return None
    providers = ("IB",) if tf == "15m" else ("ALPACA", "IB")
    sql = (
        "SELECT high, low, close FROM ("
        " SELECT high, low, close, ts FROM market_data"
        " WHERE symbol = %s AND timeframe = %s AND provider = %s"
        " ORDER BY ts DESC LIMIT %s"
        ") x ORDER BY ts ASC"
    )
    sql_any = (
        "SELECT high, low, close FROM ("
        " SELECT high, low, close, ts FROM market_data"
        " WHERE symbol = %s AND timeframe = %s"
        " ORDER BY ts DESC LIMIT %s"
        ") x ORDER BY ts ASC"
    )

    def _rows(query: str, params: tuple) -> List[Any]:
        if runner is not None:
            got = runner(query, params, fetch=True)
            return list(got or [])
        from utils.db.timescaledb_client import get_timescaledb_client

        client = get_timescaledb_client()
        if not client.ensure_connection():
            return []
        from psycopg2.extras import RealDictCursor

        cur = client.connection.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute(query, params)
            return list(cur.fetchall() or [])
        finally:
            cur.close()

    rows: List[Any] = []
    for provider in providers:
        try:
            rows = _rows(sql, (stock, tf, provider, int(n)))
        except Exception:
            logger.debug("ATR OHLC load failed %s %s %s", stock, tf, provider, exc_info=True)
            rows = []
        if rows:
            break
    if not rows:
        try:
            rows = _rows(sql_any, (stock, tf, int(n)))
        except Exception:
            logger.debug("ATR OHLC load (any provider) failed %s %s", stock, tf, exc_info=True)
            rows = []
    if not rows:
        return None
    high, low, close = [], [], []
    for row in rows:
        if isinstance(row, dict):
            h, l, c = row.get("high"), row.get("low"), row.get("close")
        else:
            h, l, c = row[0], row[1], row[2]
        try:
            high.append(float(h))
            low.append(float(l))
            close.append(float(c))
        except (TypeError, ValueError):
            continue
    if len(close) < ATR_LEN + 1:
        return None
    return high, low, close


def atr_for_symbol(
    symbol: str,
    timeframe: str,
    *,
    runner: Optional[Callable[..., Any]] = None,
) -> Optional[float]:
    ohlc = load_recent_ohlc(symbol, timeframe, runner=runner)
    if ohlc is None:
        return None
    high, low, close = ohlc
    return atr_last(high, low, close)


def load_bought_safe(store: Any, *, active_only: bool = True) -> List[dict]:
    loader = getattr(store, "load_bought", None)
    if loader is None:
        return []
    try:
        return list(loader(active_only=active_only) or [])
    except TypeError:
        return list(loader() or [])
    except Exception:
        logger.exception("load_bought failed")
        return []


def active_bought_symbols(store: Any) -> List[str]:
    return sorted(
        {
            str(t.get("stock") or "").upper()
            for t in load_bought_safe(store, active_only=True)
            if t.get("stock")
        }
    )


def sync_bought_prices(
    store: Any,
    prices: Dict[str, float],
    *,
    now: Optional[datetime] = None,
) -> List[dict]:
    """Apply last prices to open/sell_now trades; persist; return updated rows."""
    trades = load_bought_safe(store, active_only=True)
    if not trades:
        return []
    ts = now or datetime.now(timezone.utc)
    saver = getattr(store, "update_bought_live", None)
    out: List[dict] = []
    for trade in trades:
        stock = str(trade.get("stock") or "").upper()
        px = prices.get(stock)
        updated = apply_price_tick(trade, px, last_ts=ts if px is not None else None)
        if saver is not None:
            try:
                saved = saver(updated)
                if saved:
                    updated = dict(saved)
            except Exception:
                logger.exception("update_bought_live failed for %s", stock)
        out.append(updated)
    return out


def mark_bought_from_candidate(
    store: Any,
    *,
    stock: str,
    timeframe: str = "15m",
    entry_px: Optional[float] = None,
    atr_loader: Optional[Callable[[str, str], Optional[float]]] = None,
) -> dict:
    stock_u = str(stock or "").upper()
    tf = str(timeframe or "15m")
    if not stock_u:
        raise ValueError("stock is required")
    existing = None
    finder = getattr(store, "find_active_bought", None)
    if finder is not None:
        existing = finder(stock_u, tf)
    else:
        for trade in load_bought_safe(store, active_only=True):
            if str(trade.get("stock") or "").upper() == stock_u and str(
                trade.get("timeframe") or "15m"
            ) == tf:
                existing = trade
                break
    if existing:
        return dict(existing)
    cand = None
    for row in store.load_rows() or []:
        if str(row.get("stock") or "").upper() == stock_u and str(
            row.get("timeframe") or "15m"
        ) == tf:
            cand = row
            break
    px = entry_px if entry_px is not None else (entry_px_from_candidate(cand) if cand else None)
    if px is None:
        raise ValueError("No entry price for %s %s (need fill, last, or close)" % (stock_u, tf))
    atr = None
    loader = atr_loader if atr_loader is not None else atr_for_symbol
    try:
        atr = loader(stock_u, tf)
    except Exception:
        logger.exception("ATR load failed for %s %s; using %% stop fallback", stock_u, tf)
    h2 = None if cand is None else cand.get("h2_time")
    last = None
    last_ts = None
    if cand is not None:
        last = cand.get("last_price")
        last_ts = cand.get("last_price_ts")
    trade = build_bought_trade(
        stock=stock_u,
        timeframe=tf,
        entry_px=float(px),
        h2_time=h2,
        atr_at_entry=atr,
        last_price=last,
        last_price_ts=last_ts,
    )
    upsert = getattr(store, "upsert_bought", None)
    if upsert is None:
        raise RuntimeError("store cannot persist bought trades")
    return dict(upsert(trade))


def close_bought_trade(store: Any, *, trade_id: int) -> Optional[dict]:
    closer = getattr(store, "close_bought", None)
    if closer is None:
        raise RuntimeError("store cannot close bought trades")
    return closer(int(trade_id))


def flush_sell_notifications(
    store: Any,
    settings: Optional[dict] = None,
    *,
    dry_run: bool = False,
    send_fn: Optional[Callable[..., None]] = None,
) -> List[str]:
    """Telegram SELL NOW once per trade. Browser alerts use sell_keys on the payload."""
    settings = settings or {}
    want = bool(settings.get("telegram_on_sell", True))
    pending = pending_sell_notifies(load_bought_safe(store, active_only=True))
    if not pending or not want:
        return []
    body = format_sell_message(pending)
    if dry_run:
        logger.info("[dry-run] would notify SELL NOW:\n%s", body)
        return [body]
    if send_fn is None:
        from utils.notify.alerts import send_alert

        send_fn = send_alert
    desktop = bool(settings.get("desktop_notify", True))
    send_fn(body, dry_run=False, title="SELL NOW", desktop=desktop)
    marker = getattr(store, "mark_sell_notified", None)
    if marker is not None:
        ids = [t.get("id") for t in pending if t.get("id") is not None]
        if ids:
            marker(ids)
    return [body]


def tag_candidates_bought(rows: Sequence[dict], trades: Sequence[dict]) -> List[dict]:
    owned = {
        (str(t.get("stock") or "").upper(), str(t.get("timeframe") or "15m"))
        for t in trades
        if str(t.get("status") or "") in ACTIVE_STATUSES
    }
    out: List[dict] = []
    for row in rows:
        item = dict(row)
        key = (str(item.get("stock") or "").upper(), str(item.get("timeframe") or "15m"))
        item["bought"] = key in owned
        out.append(item)
    return out
