#!/usr/bin/env python3
"""
Nightly channel-touch job:
  1) Refresh ALPACA 1d bars for the stored daily universe
  2) Scan for pivot-confirmation entries on the latest bar (RS top1, ATR k=2.0)
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
    format_triggers_message,
    resolve_as_of_from_panels,
    scan_live_triggers,
)

RESEARCH = ROOT / "scripts" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
from find_ascending_channels import list_symbols_fast  # noqa: E402
from backtest_channel_touch_trades import select_same_day_rs  # noqa: E402

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


def main() -> int:
    load_env_file()
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
    ap.add_argument("--max-entries-per-day", type=int, default=1)
    ap.add_argument("--atr-stop-mult", type=float, default=2.0)
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
    args = ap.parse_args()
    dry_run = bool(args.dry_run or args.no_notify)

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
        logger.info("Scanning live triggers as_of=%s ...", as_of_s)
        t_scan = time.perf_counter()
        raw = scan_live_triggers(
            panels,
            symbols=symbols,
            spy_df=spy_df,
            as_of=as_of,
            workers=int(args.workers),
            max_entries_per_day=0,
            atr_stop_mult=float(args.atr_stop_mult),
        )
        n_cand = 0 if raw.empty else len(raw)
        triggers = raw
        if not raw.empty and int(args.max_entries_per_day) > 0:
            triggers = select_same_day_rs(
                raw, rs_col="rs_spy_126d", max_per_day=int(args.max_entries_per_day)
            )
        n_trig = 0 if triggers.empty else len(triggers)
        logger.info(
            "Scan done in %.1fs | candidates=%d triggers=%d",
            time.perf_counter() - t_scan,
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
