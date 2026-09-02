#!/usr/bin/env python3
"""
15m channel-touch live loop (research H5 stack; does not replace daily nightly).

  1) Load recent IB 15m from TimescaleDB (full watchlist on a new session)
  2) Build armed H2 resist-break watchlist (span<=10, wait-12)
  3) Alpaca last-price proximity (at-or-above resist; not a fill)
  4) At each RTH 15m close: IB hist on the armed/hot list only, drop the
     in-progress bar, upsert TimescaleDB, re-scan those names
  5) On that completed bar: unique-symbol/day + prior-bar vol>=2 + overshoot>=0.08
  6) Telegram + browser/desktop only on H5 fills (not on hot last-price)

Prerequisite: IB 15m must be current. If last bar is still 2025-12, run:
  python scripts\\data\\backfill_ib_15m_universe.py

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\channel_touch_15m.py --mode watchlist --dry-run --max-symbols 50
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\channel_touch_15m.py --mode run --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config.env_loader import load_env_file  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.notify.alerts import send_alert  # noqa: E402
from utils.scanning.channel_touch_15m import (  # noqa: E402
    LIVE_15M_DEFAULTS,
    armed_rows_for_symbol,
    attach_last_prices,
    drop_incomplete_15m_bars,
    fetch_alpaca_last_prices,
    filter_15m_rows,
    format_15m_message,
    is_hot_proximity,
    is_prior_et_session,
    live_refresh_symbols,
    lookback_start,
    merge_rescanned_15m_rows,
    notify_payloads,
    passes_h5_stack,
    rank_hot,
    unique_symbol_day_ok,
)
from utils.scanning.channel_touch_15m_refresh import (  # noqa: E402
    refresh_ib_15m_symbols,
)
from utils.scanning.channel_touch_candidates_store import (  # noqa: E402
    ChannelTouchCandidatesStore,
    DEFAULT_SETTINGS,
)

RESEARCH = ROOT / "scripts" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
from find_ascending_channels import list_symbols_fast  # noqa: E402

logger = logging.getLogger("channel_touch_15m")

DEFAULT_WATCHLIST = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_watchlist.json"
DEFAULT_FILLS_LOG = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_fills_log.csv"


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def build_arg_parser() -> argparse.ArgumentParser:
    d = LIVE_15M_DEFAULTS
    ap = argparse.ArgumentParser(description="15m channel-touch armed watchlist + H5 fill check")
    ap.add_argument(
        "--mode",
        choices=("watchlist", "proximity", "fills", "run"),
        default="run",
        help="watchlist=build armed JSON+DB; proximity=Alpaca last vs resist; fills=H5 on last bar; run=all",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not send Telegram")
    ap.add_argument("--skip-proximity", action="store_true", help="Skip Alpaca last-price poll")
    ap.add_argument("--provider", default=str(d["provider"]))
    ap.add_argument("--timeframe", default=str(d["timeframe"]))
    ap.add_argument("--lookback-sessions", type=int, default=int(d["lookback_sessions"]))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--min-l3-wait-bars", type=int, default=int(d["min_l3_wait_bars"]))
    ap.add_argument("--max-l3-wait-bars", type=int, default=int(d["max_l3_wait_bars"]))
    ap.add_argument("--max-channel-span-days", type=float, default=float(d["max_channel_span_days"]))
    ap.add_argument("--volume-rel-min", type=float, default=float(d["volume_rel_min"]))
    ap.add_argument("--overshoot-min", type=float, default=float(d["overshoot_min"]))
    ap.add_argument(
        "--proximity-below-pct",
        type=float,
        default=float(d["proximity_below_pct"]),
        help="Hot if last is within this percent BELOW resist (default 0 = at or above only)",
    )
    ap.add_argument("--entry-slip-pct", type=float, default=float(d["entry_slip_pct"]))
    ap.add_argument("--stale-hours", type=float, default=36.0)
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip parquet OHLCV cache (use after IB 15m backfill on the same calendar day)",
    )
    ap.add_argument("--alpaca-batch", type=int, default=200)
    ap.add_argument(
        "--skip-ib-refresh",
        action="store_true",
        help="Do not pull IB 15m on the armed/hot list (fills stay on stored bars)",
    )
    ap.add_argument(
        "--rebuild-watchlist",
        action="store_true",
        help="Force a full IB 15m universe scan instead of refreshing the stored watchlist",
    )
    ap.add_argument(
        "--ib-client-id",
        type=int,
        default=int(d["ib_client_id"]),
        help="Gateway client ID for live 15m refresh (default 8823; 8821/8822 are backfill)",
    )
    ap.add_argument("--ib-sleep", type=float, default=float(d["ib_sleep"]))
    ap.add_argument(
        "--ib-close-lag-sec",
        type=float,
        default=float(d["ib_close_lag_sec"]),
        help="Wait after bar-close cron tick so IB has finalized the last 15m bar",
    )
    ap.add_argument("--ib-overlap-bars", type=int, default=int(d["ib_overlap_bars"]))
    ap.add_argument("--ib-fetch-max-days", type=float, default=float(d["ib_fetch_max_days"]))
    ap.add_argument(
        "--refresh-below-pct",
        type=float,
        default=float(d["refresh_below_pct"]),
        help="Also IB-refresh armed names within this percent BELOW resist (default 0.5)",
    )
    ap.add_argument(
        "--watchlist",
        type=Path,
        default=DEFAULT_WATCHLIST,
    )
    ap.add_argument(
        "--fills-log",
        type=Path,
        default=DEFAULT_FILLS_LOG,
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "scanners",
    )
    return ap


def _notify(text: str, *, dry_run: bool, desktop: bool = True) -> None:
    send_alert(text, dry_run=dry_run, desktop=desktop)


def _open_store() -> Optional[ChannelTouchCandidatesStore]:
    try:
        store = ChannelTouchCandidatesStore()
        store.ensure_tables()
        return store
    except Exception as exc:
        logger.warning("TimescaleDB candidates store unavailable: %s", exc)
        return None


def _load_settings(store: Optional[ChannelTouchCandidatesStore]) -> dict:
    if store is None:
        return dict(DEFAULT_SETTINGS)
    try:
        return store.load_settings()
    except Exception as exc:
        logger.warning("Could not load 15m dashboard settings: %s", exc)
        return dict(DEFAULT_SETTINGS)


def _persist_candidates(
    store: Optional[ChannelTouchCandidatesStore],
    rows: List[dict],
    *,
    meta: dict,
    below_pct: float,
) -> List[dict]:
    if store is None:
        return rows
    try:
        return store.replace_candidates(
            rows, meta=meta, below_pct=float(below_pct), timeframe="15m"
        )
    except Exception as exc:
        logger.warning("Could not persist candidates: %s", exc)
        return rows


def _load_rows_from_store_or_json(
    args,
    store: Optional[ChannelTouchCandidatesStore],
) -> Tuple[dict, List[dict]]:
    if store is not None:
        try:
            rows = store.load_rows()
            settings = store.load_settings()
            if rows:
                payload = {
                    "as_of": settings.get("as_of") or "",
                    "n_universe": int(settings.get("n_universe") or 0),
                    "n_rows": len(rows),
                    "stale_warning": settings.get("stale_warning"),
                    "defaults": {
                        "proximity_below_pct": settings.get("proximity_below_pct"),
                    },
                    "rows": rows,
                }
                logger.info("Loaded %d candidates from TimescaleDB as_of=%s", len(rows), payload["as_of"])
                return payload, rows
        except Exception as exc:
            logger.warning("DB watchlist load failed: %s", exc)
    if args.watchlist.exists():
        payload = _read_watchlist(args.watchlist)
        rows = list(payload.get("rows") or [])
        logger.info("Loaded watchlist %s rows=%d as_of=%s", args.watchlist, len(rows), payload.get("as_of"))
        return payload, rows
    raise FileNotFoundError("No candidates in TimescaleDB and watchlist JSON missing: %s" % args.watchlist)


def _load_filled_today(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "stock" not in df.columns:
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    asof_col = "as_of" if "as_of" in df.columns else None
    for _, row in df.iterrows():
        stock = str(row["stock"]).upper()
        asof = str(row[asof_col]) if asof_col else ""
        if asof.startswith(today) or str(row.get("buy_date", "")).startswith(today):
            out.append("%s|%s" % (stock, asof[:10] or today))
    return out


def _append_fills(path: Path, fills: pd.DataFrame) -> None:
    if fills is None or fills.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    fills.to_csv(path, mode="a", header=header, index=False)


def _write_watchlist(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_watchlist(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_as_of(rows: List[dict]) -> str:
    times = [str(r.get("as_of") or "") for r in rows if r.get("as_of")]
    return max(times) if times else ""


def _should_rebuild_watchlist(
    args,
    rows: List[dict],
    as_of: str,
) -> bool:
    if args.mode == "watchlist" or bool(args.rebuild_watchlist):
        return True
    if not rows:
        return True
    if not filter_15m_rows(rows):
        return True
    return is_prior_et_session(as_of)


def _rescan_symbols(
    symbols: List[str],
    *,
    min_wait: int,
    max_wait: int,
    max_span: float,
    slip: float,
    lookback_sessions: int,
) -> List[dict]:
    if not symbols:
        return []
    start = lookback_start(sessions=int(lookback_sessions))
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    panels = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=start,
        end=end,
        use_cache=False,
        chunk_size=10,
        workers=min(8, max(1, len(symbols))),
    )
    now = datetime.now(timezone.utc)
    cleaned = {}
    for sym, df in panels.items():
        if df is None or df.empty:
            continue
        trimmed = drop_incomplete_15m_bars(df, now=now)
        if trimmed is None or trimmed.empty:
            continue
        cleaned[sym] = trimmed
    return build_watchlist_rows(
        cleaned,
        list(cleaned.keys()),
        min_wait=int(min_wait),
        max_wait=int(max_wait),
        max_span=float(max_span),
        slip=float(slip),
    )


def _refresh_and_rescan(
    rows: List[dict],
    symbols: List[str],
    *,
    args,
    below_pct: float,
    close_lag_sec: Optional[float] = None,
) -> Tuple[List[dict], List[str]]:
    """IB-refresh ``symbols``, rescan them, merge back into ``rows``."""
    if not symbols:
        return rows, []
    logger.info("IB 15m live refresh n=%d client=%s", len(symbols), args.ib_client_id)
    saved = refresh_ib_15m_symbols(
        symbols,
        client_id=int(args.ib_client_id),
        sleep_s=float(args.ib_sleep),
        overlap_bars=int(args.ib_overlap_bars),
        max_days=float(args.ib_fetch_max_days),
        close_lag_sec=float(close_lag_sec if close_lag_sec is not None else args.ib_close_lag_sec),
        dry_run=False,
    )
    # Always rescan the requested list: even 0 new bars may still have
    # completed the previous slot already in the DB.
    scan_syms = list(dict.fromkeys(list(saved) + list(symbols)))
    new_rows = _rescan_symbols(
        scan_syms,
        min_wait=int(args.min_l3_wait_bars),
        max_wait=int(args.max_l3_wait_bars),
        max_span=float(args.max_channel_span_days),
        slip=float(args.entry_slip_pct),
        lookback_sessions=int(args.lookback_sessions),
    )
    px_map: Dict[str, float] = {}
    for row in rows:
        stock = str(row.get("stock") or "").upper()
        px = row.get("last_price")
        if not stock or px is None:
            continue
        try:
            px_map[stock] = float(px)
        except (TypeError, ValueError):
            pass
    merged = merge_rescanned_15m_rows(rows, new_rows, scanned=scan_syms)
    merged = attach_last_prices(merged, px_map, below_pct=float(below_pct))
    return merged, saved


def _stale_warning(panels: Dict[str, pd.DataFrame], stale_hours: float) -> Optional[str]:
    lasts = []
    for df in panels.values():
        if df is None or df.empty:
            continue
        ts = pd.Timestamp(df.index.max())
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        else:
            ts = ts.tz_localize("UTC")
        lasts.append(ts)
    if not lasts:
        return "no IB 15m panels loaded"
    latest = max(lasts)
    age_h = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600.0
    if age_h > float(stale_hours):
        return "IB 15m last bar %s (%.1fh stale). Run backfill_ib_15m_universe.py" % (
            latest.isoformat(),
            age_h,
        )
    return None


def _rebuild_full_watchlist(args, *, below_pct: float, store) -> Tuple[dict, List[dict], str, int, Optional[str]]:
    t_sym = time.perf_counter()
    symbols = list_symbols_fast(args.provider, args.timeframe)
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: int(args.max_symbols)]
    symbols = [s.upper() for s in symbols]
    logger.info("Universe: %d IB 15m symbols (%.1fs)", len(symbols), time.perf_counter() - t_sym)

    start = lookback_start(sessions=int(args.lookback_sessions))
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    logger.info("Loading IB 15m %s -> %s ...", start.date(), end.date())
    t_load = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=start,
        end=end,
        use_cache=not bool(args.no_cache),
        chunk_size=int(args.chunk_size),
        workers=max(1, int(args.load_workers)),
    )
    now = datetime.now(timezone.utc)
    cleaned = {}
    for sym, df in panels.items():
        if df is None or df.empty:
            continue
        trimmed = drop_incomplete_15m_bars(df, now=now)
        if trimmed is None or trimmed.empty:
            continue
        cleaned[sym] = trimmed
    logger.info("Loaded %d/%d panels in %.1fs", len(cleaned), len(symbols), time.perf_counter() - t_load)
    stale = _stale_warning(cleaned, float(args.stale_hours))
    if stale:
        logger.warning("%s", stale)

    t_scan = time.perf_counter()
    rows = build_watchlist_rows(
        cleaned,
        list(cleaned.keys()),
        min_wait=int(args.min_l3_wait_bars),
        max_wait=int(args.max_l3_wait_bars),
        max_span=float(args.max_channel_span_days),
        slip=float(args.entry_slip_pct),
    )
    as_of = _max_as_of(rows)
    if not as_of and cleaned:
        lasts = [pd.Timestamp(df.index.max()) for df in cleaned.values()]
        if lasts:
            as_of = str(max(lasts))
    n_universe = len(cleaned)
    logger.info(
        "Watchlist built in %.1fs armed_or_waiting_or_fill=%d as_of=%s",
        time.perf_counter() - t_scan,
        len(rows),
        as_of,
    )
    payload = {
        "as_of": as_of,
        "n_universe": n_universe,
        "n_rows": len(rows),
        "stale_warning": stale,
        "defaults": {
            "min_wait": int(args.min_l3_wait_bars),
            "span_days": float(args.max_channel_span_days),
            "volume_rel_min": float(args.volume_rel_min),
            "overshoot_min": float(args.overshoot_min),
            "proximity_below_pct": below_pct,
        },
        "rows": rows,
    }
    _write_watchlist(args.watchlist, payload)
    logger.info("Wrote %s", args.watchlist)
    meta = {
        "as_of": as_of or None,
        "n_universe": n_universe,
        "stale_warning": stale,
    }
    rows = _persist_candidates(store, rows, meta=meta, below_pct=below_pct)
    payload["rows"] = rows
    return payload, rows, as_of, n_universe, stale


def build_watchlist_rows(
    panels: Dict[str, pd.DataFrame],
    symbols: List[str],
    *,
    min_wait: int,
    max_wait: int,
    max_span: float,
    slip: float,
) -> List[dict]:
    rows: List[dict] = []
    d = LIVE_15M_DEFAULTS
    for i, sym in enumerate(symbols, start=1):
        df = panels.get(sym)
        if df is None or df.empty:
            continue
        got = armed_rows_for_symbol(
            sym,
            df,
            min_wait=int(min_wait),
            max_wait=int(max_wait),
            max_span_days=float(max_span),
            error_pct=float(d["error_pct"]),
            slip=float(slip),
            window_bars=int(d["window_bars"]),
            window_step_bars=int(d["window_step_bars"]),
        )
        rows.extend(got)
        if i % 100 == 0:
            logger.info("Watchlist scanned %d/%d symbols, armed_rows=%d", i, len(symbols), len(rows))
    return rows


def fills_from_rows(
    rows: List[dict],
    *,
    volume_rel_min: float,
    overshoot_min: float,
    filled_today: List[str],
) -> pd.DataFrame:
    hits = []
    for row in rows:
        if str(row.get("timeframe") or "15m") != "15m":
            continue
        if row.get("status") != "filled":
            continue
        stock = str(row.get("stock", "")).upper()
        as_of = str(row.get("as_of", ""))
        if not unique_symbol_day_ok(stock, as_of, filled_today):
            continue
        gated = dict(row)
        if not passes_h5_stack(
            gated, volume_rel_min=float(volume_rel_min), overshoot_min=float(overshoot_min)
        ):
            continue
        hits.append(gated)
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame(hits)


def _split_refresh_batches(
    rows: Sequence[dict],
    *,
    refresh_below_pct: float,
) -> Tuple[List[str], List[str]]:
    ordered = live_refresh_symbols(rows, below_pct=float(refresh_below_pct))
    hot: set = set()
    for row in filter_15m_rows(rows):
        stock = str(row.get("stock") or "").upper()
        if not stock:
            continue
        if bool(row.get("hot")) or is_hot_proximity(
            row.get("last_price"), row.get("resist"), below_pct=float(refresh_below_pct)
        ):
            hot.add(stock)
    first = [s for s in ordered if s in hot]
    rest = [s for s in ordered if s not in hot]
    return first, rest


def _commit_live_state(
    args,
    store,
    payload: dict,
    rows: List[dict],
    *,
    as_of: str,
    n_universe: int,
    stale: Optional[str],
    below_pct: float,
) -> List[dict]:
    payload["rows"] = rows
    payload["as_of"] = as_of
    payload["n_rows"] = len(filter_15m_rows(rows))
    payload["stale_warning"] = stale
    _write_watchlist(args.watchlist, payload)
    meta = {
        "as_of": as_of or None,
        "n_universe": n_universe,
        "stale_warning": stale,
    }
    persisted = _persist_candidates(store, filter_15m_rows(rows), meta=meta, below_pct=below_pct)
    by_15m = {str(r.get("stock", "")).upper(): r for r in persisted}
    out = []
    for row in rows:
        if str(row.get("timeframe") or "15m") != "15m":
            out.append(row)
            continue
        stock = str(row.get("stock", "")).upper()
        out.append(by_15m.get(stock, row))
    payload["rows"] = out
    return out


def _notify_new_fills(
    *,
    args,
    store,
    settings,
    rows,
    fills,
    filled_today: List[str],
    as_of: str,
    n_universe: int,
    stale: Optional[str],
    dry_run: bool,
) -> List[str]:
    if fills is None or fills.empty:
        return filled_today
    _append_fills(args.fills_log, fills)
    logger.info("Fills this bar: %d (log %s)", len(fills), args.fills_log)
    n_hot = sum(1 for r in rows if r.get("hot") and str(r.get("timeframe") or "15m") == "15m")
    n_armed = sum(1 for r in rows if r.get("status") == "armed" and str(r.get("timeframe") or "15m") == "15m")
    _send_gated_telegram(
        store=store,
        settings=settings,
        rows=rows,
        fills=fills,
        as_of=as_of or "n/a",
        n_armed=n_armed,
        n_hot=n_hot,
        n_universe=n_universe,
        stale=stale,
        dry_run=dry_run,
    )
    extra = []
    for _, row in fills.iterrows():
        stock = str(row.get("stock", "")).upper()
        asof = str(row.get("as_of", ""))[:10]
        extra.append("%s|%s" % (stock, asof))
    return list(filled_today) + extra


def _send_gated_telegram(
    *,
    store: Optional[ChannelTouchCandidatesStore],
    settings: dict,
    rows: List[dict],
    fills: Optional[pd.DataFrame],
    as_of: str,
    n_armed: int,
    n_hot: int,
    n_universe: int,
    stale: Optional[str],
    dry_run: bool,
) -> None:
    msgs = notify_payloads(
        fills=fills,
        newly_hot=[],
        settings=settings,
        as_of=as_of or "n/a",
        n_armed=n_armed,
        n_hot=n_hot,
        n_universe=n_universe,
        stale_warning=stale,
    )
    for msg in msgs:
        logger.info("Notify payload:\n%s", msg)
        _notify(msg, dry_run=dry_run, desktop=bool(settings.get("desktop_notify", True)))


def main() -> int:
    load_env_file()
    args = build_arg_parser().parse_args()
    if args.mode == "proximity":
        log_name = "channel_touch_15m_proximity_%s.log" % datetime.now().strftime("%Y%m%d")
    else:
        log_name = "channel_touch_15m_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S")
    _configure_logging(args.log_dir / log_name)
    t0 = time.perf_counter()
    dry_run = bool(args.dry_run)
    store = _open_store()
    settings = _load_settings(store)
    below_pct = float(settings.get("proximity_below_pct") or args.proximity_below_pct)

    try:
        payload: dict = {}
        rows: List[dict] = []
        as_of = ""
        n_universe = 0
        stale: Optional[str] = None

        if args.mode in ("proximity", "fills"):
            payload, rows = _load_rows_from_store_or_json(args, store)
            as_of = str(payload.get("as_of") or "")
            n_universe = int(payload.get("n_universe") or 0)
            stale = payload.get("stale_warning")
        else:
            try:
                payload, rows = _load_rows_from_store_or_json(args, store)
                as_of = str(payload.get("as_of") or _max_as_of(rows))
                n_universe = int(payload.get("n_universe") or 0)
                stale = payload.get("stale_warning")
            except FileNotFoundError:
                payload, rows, as_of, n_universe, stale = {}, [], "", 0, None
            if _should_rebuild_watchlist(args, rows, as_of):
                logger.info("Rebuilding 15m watchlist from TimescaleDB as_of=%s", as_of or "empty")
                payload, rows, as_of, n_universe, stale = _rebuild_full_watchlist(
                    args, below_pct=below_pct, store=store
                )
            else:
                logger.info(
                    "Reusing stored 15m watchlist rows=%d as_of=%s (same ET session)",
                    len(filter_15m_rows(rows)),
                    as_of,
                )

        if args.mode == "watchlist":
            msg = format_15m_message(
                as_of=as_of or "n/a",
                n_armed=sum(1 for r in rows if r.get("status") == "armed"),
                n_hot=0,
                fills=pd.DataFrame(),
                n_universe=n_universe,
                stale_warning=stale,
            )
            msg = f"{msg}\nelapsed_sec={time.perf_counter() - t0:.1f}\nwatchlist={args.watchlist.name}"
            logger.info("Watchlist-only:\n%s", msg)
            return 0

        last_prices: Dict[str, float] = {}
        if not args.skip_proximity and args.mode in ("proximity", "run", "fills"):
            armed_syms = sorted(
                {
                    str(r["stock"]).upper()
                    for r in filter_15m_rows(rows)
                    if r.get("status") in ("armed", "waiting", "filled")
                }
            )
            if armed_syms:
                t_px = time.perf_counter()
                last_prices = fetch_alpaca_last_prices(armed_syms, batch_size=int(args.alpaca_batch))
                logger.info(
                    "Alpaca last prices %d/%d in %.2fs",
                    len(last_prices),
                    len(armed_syms),
                    time.perf_counter() - t_px,
                )
            rows = attach_last_prices(rows, last_prices, below_pct=below_pct)
            rows = rank_hot(rows, below_pct=below_pct)
            payload["rows"] = rows
            _write_watchlist(args.watchlist, payload)
            if store is not None:
                try:
                    store.update_live_prices(rows, price_ts=datetime.now(timezone.utc))
                except Exception as exc:
                    logger.warning("Could not update live prices: %s", exc)

        n_hot = sum(1 for r in filter_15m_rows(rows) if r.get("hot"))
        n_armed = sum(1 for r in filter_15m_rows(rows) if r.get("status") == "armed")
        if args.mode == "proximity":
            hot = [r for r in filter_15m_rows(rows) if r.get("hot")]
            logger.info("Hot names (at/above resist): %d", len(hot))
            for r in hot[:30]:
                logger.info(
                    "  %s last=%s resist=%s dist_live=%s wait=%s vol=%s",
                    r.get("stock"),
                    r.get("last_price"),
                    r.get("resist"),
                    r.get("dist_live_pct"),
                    r.get("wait_bars"),
                    r.get("volume_rel_20"),
                )
            _send_gated_telegram(
                store=store,
                settings=settings,
                rows=rows,
                fills=pd.DataFrame(),
                as_of=as_of or "n/a",
                n_armed=n_armed,
                n_hot=n_hot,
                n_universe=n_universe,
                stale=stale,
                dry_run=dry_run,
            )
            logger.info("proximity elapsed_sec=%.1f", time.perf_counter() - t0)
            return 0

        filled_today = _load_filled_today(args.fills_log)
        if not args.skip_ib_refresh:
            first, rest = _split_refresh_batches(
                rows, refresh_below_pct=float(args.refresh_below_pct)
            )
            logger.info("IB refresh batches hot/near=%d rest=%d", len(first), len(rest))
            if first:
                rows, _ = _refresh_and_rescan(
                    rows, first, args=args, below_pct=below_pct
                )
                as_of = _max_as_of(filter_15m_rows(rows)) or as_of
                stale = None
                rows = _commit_live_state(
                    args,
                    store,
                    payload,
                    rows,
                    as_of=as_of,
                    n_universe=n_universe,
                    stale=stale,
                    below_pct=below_pct,
                )
                fills = fills_from_rows(
                    rows,
                    volume_rel_min=float(args.volume_rel_min),
                    overshoot_min=float(args.overshoot_min),
                    filled_today=filled_today,
                )
                filled_today = _notify_new_fills(
                    args=args,
                    store=store,
                    settings=settings,
                    rows=rows,
                    fills=fills,
                    filled_today=filled_today,
                    as_of=as_of,
                    n_universe=n_universe,
                    stale=stale,
                    dry_run=dry_run,
                )
            if rest:
                rows, _ = _refresh_and_rescan(
                    rows,
                    rest,
                    args=args,
                    below_pct=below_pct,
                    close_lag_sec=0.0,
                )
                as_of = _max_as_of(filter_15m_rows(rows)) or as_of
                rows = _commit_live_state(
                    args,
                    store,
                    payload,
                    rows,
                    as_of=as_of,
                    n_universe=n_universe,
                    stale=stale,
                    below_pct=below_pct,
                )
                fills = fills_from_rows(
                    rows,
                    volume_rel_min=float(args.volume_rel_min),
                    overshoot_min=float(args.overshoot_min),
                    filled_today=filled_today,
                )
                filled_today = _notify_new_fills(
                    args=args,
                    store=store,
                    settings=settings,
                    rows=rows,
                    fills=fills,
                    filled_today=filled_today,
                    as_of=as_of,
                    n_universe=n_universe,
                    stale=stale,
                    dry_run=dry_run,
                )
        else:
            fills = fills_from_rows(
                rows,
                volume_rel_min=float(args.volume_rel_min),
                overshoot_min=float(args.overshoot_min),
                filled_today=filled_today,
            )
            filled_today = _notify_new_fills(
                args=args,
                store=store,
                settings=settings,
                rows=rows,
                fills=fills,
                filled_today=filled_today,
                as_of=as_of,
                n_universe=n_universe,
                stale=stale,
                dry_run=dry_run,
            )

        logger.info(
            "overshoot_min=%s (frozen live floor) elapsed_sec=%.1f watchlist=%s as_of=%s",
            args.overshoot_min,
            time.perf_counter() - t0,
            args.watchlist.name,
            as_of,
        )
        return 0
    except Exception as exc:
        logger.exception("15m channel-touch scan failed: %s", exc)
        try:
            _notify(f"Channel-touch 15m FAILED: {exc}", dry_run=dry_run)
        except Exception as notify_exc:
            logger.error("Also failed to notify failure: %s", notify_exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
