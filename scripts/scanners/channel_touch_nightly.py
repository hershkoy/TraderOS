#!/usr/bin/env python3
"""
Nightly channel-touch job:
  1) Refresh ALPACA 1d bars for the stored daily universe
  2) Scan for H2 resist-break fills on the latest bar (min-wait 6,
     span365, unique-symbol/day, ATR k=2.0; skip in-channel / RSI / beyond-width)
  3) Telegram-notify triggers (and a no-signal heartbeat)

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\channel_touch_nightly.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\channel_touch_nightly.py --skip-update
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\channel_touch_nightly.py --dry-run --max-symbols 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config.env_loader import load_env_file
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.data.update_universe_data import UniverseDataUpdater
from utils.notify.telegram_pinger import send_message
from utils.scanning.channel_touch import (
    LIVE_DEFAULTS,
    format_triggers_message,
    resolve_as_of_from_panels,
    scan_live_triggers,
)

RESEARCH = ROOT / "scripts" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
from find_ascending_channels import list_symbols_fast  # noqa: E402
from backtest_channel_touch_trades import keep_one_per_symbol_day, select_same_day_rs  # noqa: E402

logger = logging.getLogger("channel_touch_nightly")


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


def _write_universe_file(symbols: List[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    return path


def _run_data_update(
    *,
    universe_file: Path,
    since: str,
    batch_size: int,
    delay_tickers: float,
    delay_batches: float,
    multi_symbol: bool,
) -> Dict:
    updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")
    stats = updater.update_universe_data(
        batch_size=batch_size,
        delay_between_batches=delay_batches,
        delay_between_tickers=delay_tickers,
        universe_file=str(universe_file),
        start_date=since,
        multi_symbol=multi_symbol,
    )
    return stats or {}


def _notify(text: str, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would Telegram:\n%s", text)
        return
    send_message(text)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Nightly channel-touch update + scan + Telegram")
    ap.add_argument("--skip-update", action="store_true", help="Skip ALPACA 1d refresh")
    ap.add_argument("--dry-run", action="store_true", help="Scan only; do not send Telegram")
    ap.add_argument("--no-notify", action="store_true", help="Alias of --dry-run for notify skip")
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default="2018-11-01", help="OHLCV load start for scan")
    ap.add_argument("--end", default="", help="OHLCV load end (default: today UTC date)")
    ap.add_argument("--update-lookback-days", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-symbols", type=int, default=0, help="Debug: limit universe size")
    ap.add_argument(
        "--max-entries-per-day",
        type=int,
        default=int(LIVE_DEFAULTS["max_entries_per_day"]),
        help="0=all unique symbols that day; N=RS vs SPY top-N (optional cap)",
    )
    ap.add_argument("--atr-stop-mult", type=float, default=float(LIVE_DEFAULTS["atr_stop_mult"]))
    ap.add_argument(
        "--entry-mode",
        default=str(LIVE_DEFAULTS["entry_mode"]),
        choices=("l3_touch", "pivot", "reclaim"),
        help="Live fill mode (default: l3_touch setups; trigger is H2 resist-break)",
    )
    ap.add_argument("--entry-touch", type=int, default=int(LIVE_DEFAULTS["entry_touch"]))
    ap.add_argument("--pivot-len", type=int, default=int(LIVE_DEFAULTS["pivot_len"]))
    ap.add_argument("--min-l3-wait-bars", type=int, default=int(LIVE_DEFAULTS["min_l3_wait_bars"]))
    ap.add_argument("--max-l3-wait-bars", type=int, default=int(LIVE_DEFAULTS["max_l3_wait_bars"]))
    ap.add_argument("--entry-slip-pct", type=float, default=float(LIVE_DEFAULTS["entry_slip_pct"]))
    ap.add_argument(
        "--h2-resist-break",
        action=argparse.BooleanOptionalAction,
        default=bool(LIVE_DEFAULTS["h2_resist_break"]),
        help="Fill close above resistance after H2 (default on)",
    )
    ap.add_argument(
        "--h2-resist-break-only",
        action=argparse.BooleanOptionalAction,
        default=bool(LIVE_DEFAULTS["h2_resist_break_only"]),
        help="Drop L3 support-tag fills; live trigger is resist-break only (default on)",
    )
    ap.add_argument(
        "--max-rsi",
        type=float,
        default=LIVE_DEFAULTS["max_rsi"],
        help="Reject rsi_14 above this (default: off for resist-break)",
    )
    ap.add_argument(
        "--require-in-channel",
        action=argparse.BooleanOptionalAction,
        default=bool(LIVE_DEFAULTS["require_in_channel"]),
    )
    ap.add_argument(
        "--max-channel-span-days",
        type=float,
        default=float(LIVE_DEFAULTS["max_channel_span_days"]),
    )
    ap.add_argument(
        "--max-beyond-width",
        type=float,
        default=LIVE_DEFAULTS["max_beyond_width"],
        help="Reject max (high-resist)/width above this (default: off for resist-break)",
    )
    ap.add_argument("--window-bars", type=int, default=int(LIVE_DEFAULTS["window_bars"]))
    ap.add_argument("--window-step-bars", type=int, default=int(LIVE_DEFAULTS["window_step_bars"]))
    ap.add_argument(
        "--no-window-scan",
        action="store_true",
        help="Disable sliding-window channel scan (legacy last-16-pivot pass)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Alpaca multi-symbol request size (also legacy per-ticker batch size)",
    )
    ap.add_argument("--delay-tickers", type=float, default=0.0)
    ap.add_argument("--delay-batches", type=float, default=0.25)
    ap.add_argument(
        "--multi-symbol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch many Alpaca symbols per HTTP request (default: on)",
    )
    ap.add_argument(
        "--universe-file",
        type=Path,
        default=ROOT / "reports" / "ascending_channels" / "alpaca_1d_symbols.txt",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "scanners",
    )
    return ap


def main() -> int:
    load_env_file()
    ap = build_arg_parser()
    args = ap.parse_args()
    dry_run = bool(args.dry_run or args.no_notify)
    window_bars = 0 if args.no_window_scan else int(args.window_bars)
    window_step = int(args.window_step_bars) if window_bars else 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _configure_logging(args.log_dir / f"channel_touch_nightly_{stamp}.log")
    t0 = time.perf_counter()

    try:
        t_sym = time.perf_counter()
        symbols = list_symbols_fast(args.provider, args.timeframe)
        if args.max_symbols and args.max_symbols > 0:
            symbols = symbols[: int(args.max_symbols)]
        if "SPY" not in {s.upper() for s in symbols}:
            symbols = list(symbols) + ["SPY"]
        symbols = [s.upper() for s in symbols]
        logger.info("Universe: %d symbols (%.1fs)", len(symbols), time.perf_counter() - t_sym)
        _write_universe_file(symbols, args.universe_file)

        update_stats: Optional[Dict] = None
        if not args.skip_update:
            since = (datetime.utcnow().date() - timedelta(days=int(args.update_lookback_days))).isoformat()
            logger.info("Updating ALPACA 1d since %s ...", since)
            t_upd = time.perf_counter()
            update_stats = _run_data_update(
                universe_file=args.universe_file,
                since=since,
                batch_size=int(args.batch_size),
                delay_tickers=float(args.delay_tickers),
                delay_batches=float(args.delay_batches),
                multi_symbol=bool(args.multi_symbol),
            )
            logger.info(
                "Update done in %.1fs | stats=%s",
                time.perf_counter() - t_upd,
                update_stats,
            )
        else:
            logger.info("Skipping data update (--skip-update)")

        end = args.end.strip() or datetime.utcnow().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        logger.info("Loading OHLCV %s -> %s ...", args.start, end)
        t_load = time.perf_counter()
        panels = load_ohlcv_many(
            symbols,
            timeframe=args.timeframe,
            provider=args.provider,
            start=start_dt,
            end=end_dt,
            use_cache=True,
            chunk_size=int(args.chunk_size),
            workers=max(1, int(args.load_workers)),
        )
        logger.info("Loaded %d/%d panels in %.1fs", len(panels), len(symbols), time.perf_counter() - t_load)

        spy_df = panels.get("SPY")
        if spy_df is None or spy_df.empty:
            msg = "Channel-touch nightly FAILED: SPY panel missing"
            logger.error(msg)
            _notify(msg, dry_run=dry_run)
            return 1

        as_of = resolve_as_of_from_panels(panels, spy_df)
        as_of_s = as_of.strftime("%Y-%m-%d")
        logger.info(
            "Scanning live triggers as_of=%s mode=%s h2_break=%s only=%s min_wait=%d "
            "max_rsi=%s in_channel=%s span<=%.0f beyond=%s window=%d/%d",
            as_of_s,
            args.entry_mode,
            bool(args.h2_resist_break),
            bool(args.h2_resist_break_only),
            int(args.min_l3_wait_bars),
            args.max_rsi,
            bool(args.require_in_channel),
            float(args.max_channel_span_days),
            args.max_beyond_width,
            window_bars,
            window_step,
        )
        t_scan = time.perf_counter()
        scan_stats: Dict = {}
        raw = scan_live_triggers(
            panels,
            symbols=symbols,
            spy_df=spy_df,
            as_of=as_of,
            workers=int(args.workers),
            max_entries_per_day=0,
            atr_stop_mult=float(args.atr_stop_mult),
            entry_mode=str(args.entry_mode),
            entry_touch=int(args.entry_touch),
            pivot_len=int(args.pivot_len),
            min_l3_wait_bars=int(args.min_l3_wait_bars),
            max_l3_wait_bars=int(args.max_l3_wait_bars),
            entry_slip_pct=float(args.entry_slip_pct),
            window_bars=window_bars if window_bars > 0 else None,
            window_step_bars=window_step if window_step > 0 else None,
            require_in_channel=bool(args.require_in_channel),
            max_channel_span_days=float(args.max_channel_span_days),
            max_beyond_width=args.max_beyond_width,
            max_rsi=args.max_rsi,
            h2_resist_break=bool(args.h2_resist_break),
            h2_resist_break_only=bool(args.h2_resist_break_only),
            stats=scan_stats,
        )
        n_raw = int(scan_stats.get("n_raw", 0 if raw.empty else len(raw)))
        n_cand = 0 if raw.empty else len(raw)
        triggers = raw
        if not raw.empty:
            triggers = keep_one_per_symbol_day(raw)
        if not triggers.empty and int(args.max_entries_per_day) > 0:
            triggers = select_same_day_rs(
                triggers, rs_col="rs_spy_126d", max_per_day=int(args.max_entries_per_day)
            )
        n_trig = 0 if triggers.empty else len(triggers)
        logger.info(
            "Scan done in %.1fs | raw=%d quality=%d triggers=%d",
            time.perf_counter() - t_scan,
            n_raw,
            n_cand,
            n_trig,
        )

        args.outdir.mkdir(parents=True, exist_ok=True)
        out_csv = args.outdir / f"channel_touch_nightly_{stamp}.csv"
        if triggers is not None and not triggers.empty:
            triggers.to_csv(out_csv, index=False)
        else:
            pd.DataFrame().to_csv(out_csv, index=False)
        logger.info("Wrote %s", out_csv)

        msg = format_triggers_message(
            triggers if triggers is not None else pd.DataFrame(),
            as_of=as_of_s,
            n_candidates=n_cand,
            update_stats=update_stats,
            n_raw=n_raw,
            entry_mode=str(args.entry_mode),
            min_l3_wait_bars=int(args.min_l3_wait_bars),
            max_rsi=args.max_rsi,
            max_beyond_width=args.max_beyond_width,
            max_channel_span_days=float(args.max_channel_span_days),
            h2_resist_break=bool(args.h2_resist_break),
        )
        msg = f"{msg}\nelapsed_sec={time.perf_counter() - t0:.1f}\ncsv={out_csv.name}"
        logger.info("Notify payload:\n%s", msg)
        _notify(msg, dry_run=dry_run)
        return 0
    except Exception as exc:
        logger.exception("Nightly channel-touch failed: %s", exc)
        try:
            _notify(f"Channel-touch nightly FAILED: {exc}", dry_run=dry_run)
        except Exception as notify_exc:
            logger.error("Also failed to notify failure: %s", notify_exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
