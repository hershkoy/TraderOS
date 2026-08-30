#!/usr/bin/env python3
"""
15m channel-touch live loop (research H5 stack; does not replace daily nightly).

  1) Load recent IB 15m from TimescaleDB
  2) Build armed H2 resist-break watchlist (span<=10, wait-12)
  3) Alpaca last-price proximity (at-or-above resist; not a fill)
  4) On completed bars: unique-symbol/day + prior-bar vol>=2 + overshoot>=0.08
  5) Optional Telegram

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
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config.env_loader import load_env_file  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.notify.telegram_pinger import send_message  # noqa: E402
from utils.scanning.channel_touch_15m import (  # noqa: E402
    LIVE_15M_DEFAULTS,
    armed_rows_for_symbol,
    attach_last_prices,
    fetch_alpaca_last_prices,
    format_15m_message,
    lookback_start,
    passes_h5_stack,
    rank_hot,
    unique_symbol_day_ok,
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
        help="watchlist=build armed JSON; proximity=Alpaca last vs resist; fills=H5 on last bar; run=all",
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
    ap.add_argument("--alpaca-batch", type=int, default=200)
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


def _notify(text: str, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would Telegram:\n%s", text)
        return
    send_message(text)


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


def main() -> int:
    load_env_file()
    args = build_arg_parser().parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _configure_logging(args.log_dir / f"channel_touch_15m_{stamp}.log")
    t0 = time.perf_counter()
    dry_run = bool(args.dry_run)

    try:
        if args.mode in ("proximity", "fills") and args.watchlist.exists():
            payload = _read_watchlist(args.watchlist)
            rows = list(payload.get("rows") or [])
            as_of = str(payload.get("as_of") or "")
            n_universe = int(payload.get("n_universe") or 0)
            stale = payload.get("stale_warning")
            logger.info("Loaded watchlist %s rows=%d as_of=%s", args.watchlist, len(rows), as_of)
        else:
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
                use_cache=True,
                chunk_size=int(args.chunk_size),
                workers=max(1, int(args.load_workers)),
            )
            logger.info("Loaded %d/%d panels in %.1fs", len(panels), len(symbols), time.perf_counter() - t_load)
            stale = _stale_warning(panels, float(args.stale_hours))
            if stale:
                logger.warning("%s", stale)

            t_scan = time.perf_counter()
            rows = build_watchlist_rows(
                panels,
                list(panels.keys()),
                min_wait=int(args.min_l3_wait_bars),
                max_wait=int(args.max_l3_wait_bars),
                max_span=float(args.max_channel_span_days),
                slip=float(args.entry_slip_pct),
            )
            as_of = ""
            if rows:
                as_of = str(rows[0].get("as_of") or "")
            elif panels:
                lasts = [pd.Timestamp(df.index.max()) for df in panels.values() if df is not None and not df.empty]
                if lasts:
                    as_of = str(max(lasts))
            n_universe = len(panels)
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
                    "proximity_below_pct": float(args.proximity_below_pct),
                },
                "rows": rows,
            }
            _write_watchlist(args.watchlist, payload)
            logger.info("Wrote %s", args.watchlist)

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
        if not args.skip_proximity and args.mode in ("proximity", "run"):
            armed_syms = sorted({str(r["stock"]).upper() for r in rows if r.get("status") in ("armed", "waiting", "filled")})
            if armed_syms:
                t_px = time.perf_counter()
                last_prices = fetch_alpaca_last_prices(armed_syms, batch_size=int(args.alpaca_batch))
                logger.info(
                    "Alpaca last prices %d/%d in %.2fs",
                    len(last_prices),
                    len(armed_syms),
                    time.perf_counter() - t_px,
                )
            rows = attach_last_prices(
                rows, last_prices, below_pct=float(args.proximity_below_pct)
            )
            rows = rank_hot(rows, below_pct=float(args.proximity_below_pct))
            payload["rows"] = rows
            _write_watchlist(args.watchlist, payload)

        n_hot = sum(1 for r in rows if r.get("hot"))
        n_armed = sum(1 for r in rows if r.get("status") == "armed")
        if args.mode == "proximity":
            hot = [r for r in rows if r.get("hot")]
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
            return 0

        filled_today = _load_filled_today(args.fills_log)
        fills = fills_from_rows(
            rows,
            volume_rel_min=float(args.volume_rel_min),
            overshoot_min=float(args.overshoot_min),
            filled_today=filled_today,
        )
        if not fills.empty:
            _append_fills(args.fills_log, fills)
            logger.info("Fills this bar: %d (log %s)", len(fills), args.fills_log)

        msg = format_15m_message(
            as_of=as_of or "n/a",
            n_armed=n_armed,
            n_hot=n_hot,
            fills=fills,
            n_universe=n_universe,
            stale_warning=stale,
        )
        msg = (
            f"{msg}\novershoot_min={args.overshoot_min} (frozen live floor, not expanding p80)\n"
            f"elapsed_sec={time.perf_counter() - t0:.1f}\n"
            f"watchlist={args.watchlist.name}"
        )
        logger.info("Notify payload:\n%s", msg)
        _notify(msg, dry_run=dry_run)
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
