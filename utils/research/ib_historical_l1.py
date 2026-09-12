"""Historical IB top-of-book (BID/ASK + trades). Not Level-2 / DOM.

IBKR does not store historical market depth. ``reqMktDepth`` is live only.
This module pages ``reqHistoricalTicks`` (Last / BidAsk, typically ~6 months)
and ``reqHistoricalData`` BID / ASK / BID_ASK / TRADES bars (years of 1m/5s).

Use America/New_York RTH clocks. IB ``endDateTime`` must carry an explicit
UTC suffix via ``_ib_end_datetime_utc``.
"""
from __future__ import annotations

import logging
import math
import time as time_mod
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
LAST_15M_START = time(15, 45)
OPEN_30_END = time(10, 0)

# IB duration caps (TWS API historical limitations).
BAR_DURATION_CAP = {
    "1 secs": timedelta(seconds=1800),
    "5 secs": timedelta(seconds=7200),
    "10 secs": timedelta(seconds=14400),
    "15 secs": timedelta(seconds=14400),
    "30 secs": timedelta(seconds=28800),
    "1 min": timedelta(days=1),
    "5 mins": timedelta(days=7),
}

TICK_PAGE = 1000
DEFAULT_SLEEP_S = 1.0
WIDE_SPREAD_BPS = 20.0
THIN_LAST_SIZE = 200.0

SYMBOL_ALIASES = {"APML": "AMPL", "TATS": "TARS"}

LOG = logging.getLogger("ib_historical_l1")


def canonical_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    return SYMBOL_ALIASES.get(raw, raw)


def parse_session_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(str(value)[:10]).date()


def et_dt(session: Any, hhmm: time) -> datetime:
    day = parse_session_date(session)
    return datetime(day.year, day.month, day.day, hhmm.hour, hhmm.minute, tzinfo=ET)


def rth_session_window(session: Any) -> Tuple[datetime, datetime]:
    """RTH [09:30, 16:00) America/New_York as timezone-aware datetimes."""
    return et_dt(session, RTH_OPEN), et_dt(session, RTH_CLOSE)


def last_15m_window(session: Any) -> Tuple[datetime, datetime]:
    """Last RTH 15m [15:45, 16:00) ET. Fill mid is known at 16:00."""
    return et_dt(session, LAST_15M_START), et_dt(session, RTH_CLOSE)


def open_30_window(session: Any) -> Tuple[datetime, datetime]:
    return et_dt(session, RTH_OPEN), et_dt(session, OPEN_30_END)


def to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ET).astimezone(UTC)
    return ts.astimezone(UTC)


def spread_bps(bid: float, ask: float) -> Optional[float]:
    if bid is None or ask is None:
        return None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bid_f) or not math.isfinite(ask_f) or bid_f <= 0 or ask_f <= 0:
        return None
    mid = (bid_f + ask_f) / 2.0
    if mid <= 0:
        return None
    return (ask_f - bid_f) / mid * 1.0e4


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def bars_to_frame(bars: Sequence[Any], *, what: str) -> pd.DataFrame:
    rows: List[dict] = []
    for bar in bars or []:
        raw_ts = getattr(bar, "date", None)
        if raw_ts is None and isinstance(bar, dict):
            raw_ts = bar.get("date") or bar.get("ts")
        ts = pd.Timestamp(raw_ts)
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        getter = bar.get if isinstance(bar, dict) else lambda k, d=None: getattr(bar, k, d)
        rows.append(
            {
                "ts": ts.to_pydatetime(),
                "open": _finite(getter("open")),
                "high": _finite(getter("high")),
                "low": _finite(getter("low")),
                "close": _finite(getter("close")),
                "volume": _finite(getter("volume")) or 0.0,
                "what": what,
            }
        )
    cols = ["ts", "open", "high", "low", "close", "volume", "what"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def ticks_to_frame(ticks: Sequence[Any], *, what: str) -> pd.DataFrame:
    rows: List[dict] = []
    for tick in ticks or []:
        raw_ts = getattr(tick, "time", None)
        if raw_ts is None and isinstance(tick, dict):
            raw_ts = tick.get("time") or tick.get("ts")
        ts = pd.Timestamp(raw_ts)
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        getter = tick.get if isinstance(tick, dict) else lambda k, d=None: getattr(tick, k, d)
        row = {
            "ts": ts.to_pydatetime(),
            "what": what,
            "price": _finite(getter("price")),
            "size": _finite(getter("size")),
            "bid": _finite(getter("priceBid")),
            "ask": _finite(getter("priceAsk")),
            "bid_size": _finite(getter("sizeBid")),
            "ask_size": _finite(getter("sizeAsk")),
            "exchange": getter("exchange") or "",
        }
        if str(what).upper() == "BID_ASK":
            row["spread_bps"] = spread_bps(row["bid"], row["ask"])
        rows.append(row)
    cols = [
        "ts",
        "what",
        "price",
        "size",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "exchange",
        "spread_bps",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    frame = pd.DataFrame(rows)
    for col in cols:
        if col not in frame.columns:
            frame[col] = None
    return frame.sort_values("ts").reset_index(drop=True)


def combine_bid_ask_bars(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
) -> pd.DataFrame:
    """Align BID and ASK 1m/5s bars on timestamp. Bid/ask use each bar's close."""
    empty = pd.DataFrame(
        columns=["ts", "bid", "ask", "mid", "spread_bps", "bid_high", "ask_low"]
    )
    if bid is None or bid.empty or ask is None or ask.empty:
        return empty
    left = bid[["ts", "close", "high", "low"]].rename(
        columns={"close": "bid", "high": "bid_high", "low": "bid_low"}
    )
    right = ask[["ts", "close", "high", "low"]].rename(
        columns={"close": "ask", "high": "ask_high", "low": "ask_low"}
    )
    merged = pd.merge(left, right, on="ts", how="inner")
    if merged.empty:
        return empty
    merged["mid"] = (merged["bid"] + merged["ask"]) / 2.0
    merged["spread_bps"] = [
        spread_bps(b, a) for b, a in zip(merged["bid"].tolist(), merged["ask"].tolist())
    ]
    return merged.sort_values("ts").reset_index(drop=True)


def bid_ask_from_combo_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """IB BID_ASK bars: open=TWAP bid, close=TWAP ask, high=max ask, low=min bid."""
    empty = pd.DataFrame(columns=["ts", "bid", "ask", "mid", "spread_bps"])
    if bars is None or bars.empty:
        return empty
    out = pd.DataFrame(
        {
            "ts": bars["ts"],
            "bid": bars["open"],
            "ask": bars["close"],
            "mid": (bars["open"] + bars["close"]) / 2.0,
        }
    )
    out["spread_bps"] = [
        spread_bps(b, a) for b, a in zip(out["bid"].tolist(), out["ask"].tolist())
    ]
    return out.sort_values("ts").reset_index(drop=True)


def clip_frame(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df is None or df.empty or "ts" not in df.columns:
        return df if df is not None else pd.DataFrame()
    start_utc = to_utc(start)
    end_utc = to_utc(end)
    ts = pd.to_datetime(df["ts"], utc=True)
    mask = (ts >= start_utc) & (ts < end_utc)
    return df.loc[mask].copy().reset_index(drop=True)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return None
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def window_metrics(
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    rail: Optional[float] = None,
    fill_px: Optional[float] = None,
    wide_bps: float = WIDE_SPREAD_BPS,
    thin_size: float = THIN_LAST_SIZE,
) -> Dict[str, Any]:
    """Summarize a quote/trade window (typically last 15m or the open)."""
    out: Dict[str, Any] = {
        "n_quote": 0 if quotes is None or quotes.empty else int(len(quotes)),
        "n_trade": 0 if trades is None or trades.empty else int(len(trades)),
        "median_spread_bps": None,
        "last_spread_bps": None,
        "last_bid": None,
        "last_ask": None,
        "last_mid": None,
        "last_trade": None,
        "last_trade_size": None,
        "fill_vs_mid_bps": None,
        "fill_vs_ask_bps": None,
        "last_vs_rail_bps": None,
        "wide": False,
        "thin": False,
        "last_through_ask": False,
        "paid_the_ask": False,
        "trade_volume": None,
    }
    if quotes is not None and not quotes.empty and "spread_bps" in quotes.columns:
        spreads = [_finite(v) for v in quotes["spread_bps"].tolist()]
        out["median_spread_bps"] = _median(spreads)
        last_q = _last_row(quotes)
        if last_q is not None:
            out["last_spread_bps"] = _finite(last_q.get("spread_bps"))
            out["last_bid"] = _finite(last_q.get("bid"))
            out["last_ask"] = _finite(last_q.get("ask"))
            out["last_mid"] = _finite(last_q.get("mid"))
            if out["last_mid"] is None and out["last_bid"] and out["last_ask"]:
                out["last_mid"] = (out["last_bid"] + out["last_ask"]) / 2.0
    last_t = _last_row(trades) if trades is not None and not trades.empty else None
    if last_t is not None:
        px = _finite(last_t.get("close"))
        if px is None:
            px = _finite(last_t.get("price"))
        out["last_trade"] = px
        size = _finite(last_t.get("volume"))
        if size is None:
            size = _finite(last_t.get("size"))
        out["last_trade_size"] = size
        if "volume" in trades.columns:
            out["trade_volume"] = float(
                pd.to_numeric(trades["volume"], errors="coerce").fillna(0).sum()
            )
        elif "size" in trades.columns:
            out["trade_volume"] = float(
                pd.to_numeric(trades["size"], errors="coerce").fillna(0).sum()
            )
    last_spread = out["last_spread_bps"]
    if last_spread is None:
        last_spread = out["median_spread_bps"]
    out["wide"] = bool(last_spread is not None and last_spread >= float(wide_bps))
    out["thin"] = bool(
        out["last_trade_size"] is not None and 0 < out["last_trade_size"] <= float(thin_size)
    )
    last_px = out["last_trade"]
    ask = out["last_ask"]
    mid = out["last_mid"]
    if last_px is not None and ask is not None and ask > 0 and last_px > ask * 1.0001:
        out["last_through_ask"] = True
    fill = _finite(fill_px)
    if fill is not None and mid is not None and mid > 0:
        out["fill_vs_mid_bps"] = (fill - mid) / mid * 1.0e4
    if fill is not None and ask is not None and ask > 0:
        out["fill_vs_ask_bps"] = (fill - ask) / ask * 1.0e4
        out["paid_the_ask"] = fill >= ask * 0.999
    rail_f = _finite(rail)
    ref = last_px if last_px is not None else fill
    if rail_f is not None and rail_f > 0 and ref is not None:
        out["last_vs_rail_bps"] = (ref - rail_f) / rail_f * 1.0e4
    return out


def flags_for_skip(metrics: Dict[str, Any]) -> List[str]:
    """Human tags: display-only ideas, not a promote rule."""
    tags: List[str] = []
    if metrics.get("wide"):
        tags.append("wide_spread")
    if metrics.get("thin"):
        tags.append("thin_print")
    if metrics.get("last_through_ask"):
        tags.append("last_through_ask")
    fill_ask = metrics.get("fill_vs_ask_bps")
    if fill_ask is not None and fill_ask > 5.0:
        tags.append("fill_above_ask")
    rail_bps = metrics.get("last_vs_rail_bps")
    if rail_bps is not None and rail_bps > 80.0:
        tags.append("extended_vs_rail")
    return tags


def chunk_windows(
    start: datetime,
    end: datetime,
    cap: timedelta,
) -> List[Tuple[datetime, datetime]]:
    """Split [start, end) into IB-legal duration chunks."""
    if end <= start:
        return []
    if cap is None or cap.total_seconds() <= 0:
        return [(start, end)]
    out: List[Tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + cap, end)
        out.append((cursor, nxt))
        cursor = nxt
    return out


def _duration_str(start: datetime, end: datetime, bar_size: str) -> str:
    sec = max(1, int((to_utc(end) - to_utc(start)).total_seconds()))
    if bar_size in ("1 secs", "5 secs", "10 secs", "15 secs", "30 secs"):
        return "%s S" % sec
    if sec <= 86400:
        return "1 D"
    days = int(math.ceil(sec / 86400.0))
    return "%s D" % days


def fetch_historical_bars(
    ib: Any,
    contract: Any,
    *,
    start: datetime,
    end: datetime,
    bar_size: str,
    what: str,
    use_rth: bool = True,
    sleep_s: float = DEFAULT_SLEEP_S,
    end_dt_fn: Optional[Callable[[datetime], str]] = None,
) -> pd.DataFrame:
    """Page ``reqHistoricalData`` using duration caps."""
    from utils.data.fetch_data import _ib_end_datetime_utc

    fmt = end_dt_fn or _ib_end_datetime_utc
    cap = BAR_DURATION_CAP.get(bar_size, timedelta(days=1))
    frames: List[pd.DataFrame] = []
    windows = chunk_windows(start, end, cap)
    for _win_start, win_end in windows:
        duration = _duration_str(_win_start, win_end, bar_size)
        end_str = fmt(to_utc(win_end))
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what,
                useRTH=bool(use_rth),
                formatDate=2,
                keepUpToDate=False,
            )
        except Exception:
            LOG.exception(
                "reqHistoricalData failed what=%s bar=%s end=%s duration=%s",
                what,
                bar_size,
                end_str,
                duration,
            )
            bars = []
        frame = bars_to_frame(bars, what=what)
        frame = clip_frame(frame, _win_start, win_end)
        if not frame.empty:
            frames.append(frame)
        if sleep_s and float(sleep_s) > 0:
            time_mod.sleep(float(sleep_s))
    if not frames:
        return bars_to_frame([], what=what)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(
        drop=True
    )


def fetch_historical_ticks(
    ib: Any,
    contract: Any,
    *,
    start: datetime,
    end: datetime,
    what: str,
    use_rth: bool = True,
    sleep_s: float = DEFAULT_SLEEP_S,
    ignore_size: bool = False,
    max_pages: int = 80,
) -> pd.DataFrame:
    """Forward-page ``reqHistoricalTicks`` (empty endDateTime, 1000 ticks/page).

    IB typically keeps tick-by-tick history for about six months. Older
    AMPL/VST/TARS fill days will often return empty; use BID/ASK bars instead.
    """
    pages: List[Any] = []
    cursor = to_utc(start)
    end_utc = to_utc(end)
    for _ in range(int(max_pages)):
        if cursor >= end_utc:
            break
        try:
            chunk = ib.reqHistoricalTicks(
                contract,
                cursor,
                "",
                TICK_PAGE,
                what,
                useRth=bool(use_rth),
                ignoreSize=bool(ignore_size),
            )
        except Exception:
            LOG.exception("reqHistoricalTicks failed what=%s start=%s", what, cursor)
            break
        if not chunk:
            break
        pages.extend(chunk)
        last_t = getattr(chunk[-1], "time", None)
        if last_t is None:
            break
        last_ts = pd.Timestamp(last_t)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        if last_ts.to_pydatetime() >= end_utc:
            break
        if len(chunk) < TICK_PAGE:
            break
        cursor = last_ts.to_pydatetime() + timedelta(milliseconds=1)
        if sleep_s and float(sleep_s) > 0:
            time_mod.sleep(float(sleep_s))
    frame = ticks_to_frame(pages, what=what)
    return clip_frame(frame, start, end)


PRIMARY_EXCHANGES = ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "ISLAND")


def qualify_stock(ib: Any, symbol: str) -> Any:
    from ib_insync import Stock

    from utils.data.fetch_data import create_ib_contract_with_primary_exchange

    sym = canonical_symbol(symbol)
    tried = []
    contract = create_ib_contract_with_primary_exchange(sym)
    tried.append(getattr(contract, "primaryExchange", None) or "SMART")
    qualified = ib.qualifyContracts(contract)
    if qualified:
        return qualified[0]
    for px in PRIMARY_EXCHANGES:
        if px in tried:
            continue
        fallback = Stock(sym, "SMART", "USD", primaryExchange=px)
        tried.append(px)
        qualified = ib.qualifyContracts(fallback)
        if qualified:
            LOG.info("qualified %s primaryExchange=%s", sym, px)
            return qualified[0]
    matcher = getattr(ib, "reqMatchingSymbols", None)
    if callable(matcher):
        try:
            hits = matcher(sym) or []
        except Exception:
            LOG.exception("reqMatchingSymbols failed for %s", sym)
            hits = []
        for hit in hits:
            cd = getattr(hit, "contract", None)
            if cd is None:
                continue
            if str(getattr(cd, "symbol", "")).upper() != sym:
                continue
            if str(getattr(cd, "secType", "STK")).upper() not in ("STK", ""):
                continue
            currency = str(getattr(cd, "currency", "") or "USD").upper()
            if currency and currency != "USD":
                continue
            qualified = ib.qualifyContracts(cd)
            if qualified:
                LOG.info("qualified %s via matchingSymbols %s", sym, cd)
                return qualified[0]
    raise RuntimeError("qualifyContracts empty for %s (tried %s)" % (sym, ",".join(str(t) for t in tried)))


def pull_session_top_of_book(
    ib: Any,
    symbol: str,
    session: Any,
    *,
    rail: Optional[float] = None,
    fill_px: Optional[float] = None,
    sleep_s: float = DEFAULT_SLEEP_S,
    want_ticks: bool = True,
    want_5s: bool = True,
) -> Dict[str, Any]:
    """Pull 1m BID/ASK/TRADES for RTH plus 5s (and ticks) on the last 15m."""
    sym = canonical_symbol(symbol)
    session_day = parse_session_date(session)
    rth_start, rth_end = rth_session_window(session_day)
    last_start, last_end = last_15m_window(session_day)
    contract = qualify_stock(ib, sym)
    payload: Dict[str, Any] = {
        "stock": sym,
        "session": session_day.isoformat(),
        "contract": str(contract),
        "rail": _finite(rail),
        "fill_px": _finite(fill_px),
        "bars_1m": {},
        "bars_5s": {},
        "ticks": {},
        "quotes_1m": pd.DataFrame(),
        "quotes_last15m": pd.DataFrame(),
        "metrics_session": {},
        "metrics_last15m": {},
        "tags_last15m": [],
        "errors": [],
    }

    def _bars(what: str, start: datetime, end: datetime, bar_size: str) -> pd.DataFrame:
        return fetch_historical_bars(
            ib,
            contract,
            start=start,
            end=end,
            bar_size=bar_size,
            what=what,
            sleep_s=sleep_s,
        )

    bid_1m = _bars("BID", rth_start, rth_end, "1 min")
    ask_1m = _bars("ASK", rth_start, rth_end, "1 min")
    trd_1m = _bars("TRADES", rth_start, rth_end, "1 min")
    quotes_1m = combine_bid_ask_bars(bid_1m, ask_1m)
    combo_1m = pd.DataFrame()
    if quotes_1m.empty:
        combo_1m = _bars("BID_ASK", rth_start, rth_end, "1 min")
        quotes_1m = bid_ask_from_combo_bars(combo_1m)
    payload["bars_1m"] = {"BID": bid_1m, "ASK": ask_1m, "TRADES": trd_1m, "BID_ASK": combo_1m}
    payload["quotes_1m"] = quotes_1m
    payload["metrics_session"] = window_metrics(
        quotes_1m, trd_1m, rail=rail, fill_px=fill_px
    )

    quotes_5s = pd.DataFrame()
    trd_5s = pd.DataFrame()
    if want_5s:
        bid_5s = _bars("BID", last_start, last_end, "5 secs")
        ask_5s = _bars("ASK", last_start, last_end, "5 secs")
        trd_5s = _bars("TRADES", last_start, last_end, "5 secs")
        quotes_5s = combine_bid_ask_bars(bid_5s, ask_5s)
        combo_5s = pd.DataFrame()
        if quotes_5s.empty:
            combo_5s = _bars("BID_ASK", last_start, last_end, "5 secs")
            quotes_5s = bid_ask_from_combo_bars(combo_5s)
        payload["bars_5s"] = {
            "BID": bid_5s,
            "ASK": ask_5s,
            "TRADES": trd_5s,
            "BID_ASK": combo_5s,
        }

    quotes_last = quotes_5s if quotes_5s is not None and not quotes_5s.empty else clip_frame(
        quotes_1m, last_start, last_end
    )
    trades_last = trd_5s if trd_5s is not None and not trd_5s.empty else clip_frame(
        trd_1m, last_start, last_end
    )
    if want_ticks:
        tick_ba = fetch_historical_ticks(
            ib, contract, start=last_start, end=last_end, what="BID_ASK", sleep_s=sleep_s
        )
        tick_tr = fetch_historical_ticks(
            ib, contract, start=last_start, end=last_end, what="TRADES", sleep_s=sleep_s
        )
        payload["ticks"] = {"BID_ASK": tick_ba, "TRADES": tick_tr}
        if tick_ba is not None and not tick_ba.empty:
            quotes_last = tick_ba.copy()
            if "mid" not in quotes_last.columns:
                quotes_last["mid"] = (quotes_last["bid"] + quotes_last["ask"]) / 2.0
        if tick_tr is not None and not tick_tr.empty:
            trades_last = tick_tr
    payload["quotes_last15m"] = quotes_last
    payload["metrics_last15m"] = window_metrics(
        quotes_last, trades_last, rail=rail, fill_px=fill_px
    )
    payload["tags_last15m"] = flags_for_skip(payload["metrics_last15m"])
    return payload


def metrics_row(payload: Dict[str, Any], *, note: str = "") -> dict:
    m = payload.get("metrics_last15m") or {}
    s = payload.get("metrics_session") or {}
    return {
        "stock": payload.get("stock"),
        "session": payload.get("session"),
        "note": note,
        "rail": payload.get("rail"),
        "fill_px": payload.get("fill_px"),
        "n_quote_last15m": m.get("n_quote"),
        "n_trade_last15m": m.get("n_trade"),
        "median_spread_bps": m.get("median_spread_bps"),
        "last_spread_bps": m.get("last_spread_bps"),
        "last_bid": m.get("last_bid"),
        "last_ask": m.get("last_ask"),
        "last_mid": m.get("last_mid"),
        "last_trade": m.get("last_trade"),
        "fill_vs_mid_bps": m.get("fill_vs_mid_bps"),
        "fill_vs_ask_bps": m.get("fill_vs_ask_bps"),
        "last_vs_rail_bps": m.get("last_vs_rail_bps"),
        "session_median_spread_bps": s.get("median_spread_bps"),
        "wide": m.get("wide"),
        "thin": m.get("thin"),
        "paid_the_ask": m.get("paid_the_ask"),
        "tags": "|".join(payload.get("tags_last15m") or []),
        "n_quote_session": s.get("n_quote"),
        "n_trade_session": s.get("n_trade"),
    }


def save_frames(payload: Dict[str, Any], outdir, *, stem: str) -> List[str]:
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _write(df: pd.DataFrame, name: str) -> None:
        if df is None or df.empty:
            return
        path = outdir / ("%s_%s.csv" % (stem, name))
        df.to_csv(path, index=False)
        written.append(str(path))

    for kind, frames in (("1m", payload.get("bars_1m") or {}), ("5s", payload.get("bars_5s") or {})):
        for what, df in frames.items():
            _write(df, "%s_%s" % (kind, str(what).lower()))
    for what, df in (payload.get("ticks") or {}).items():
        _write(df, "tick_%s" % str(what).lower())
    _write(payload.get("quotes_1m"), "quotes_1m")
    _write(payload.get("quotes_last15m"), "quotes_last15m")
    return written
