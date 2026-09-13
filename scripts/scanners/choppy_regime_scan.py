#!/usr/bin/env python3
"""Scan choppy-regime RSI-2 MOC and 3-day pullback limits.

RSI-2 MOC uses a developing daily close from IB 15m (last RTH bar, prefer
15:45 ET) overlaid on Alpaca daily history. That is the 15:45 scan path --
not the Alpaca 17:00 ET EOD snapshot. Live 15:45 should pull IB hist on the
liquid shortlist the same way channel_touch_15m uses client 8826 (do not
stream the full daily universe). This scanner reads TimescaleDB IB 15m.

3-day pullback limits stay an evening job on the last completed daily bar.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\scanners\\choppy_regime_scan.py
  python scripts\\scanners\\choppy_regime_scan.py --mode rsi2
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.choppy_regime import (  # noqa: E402
    DEFAULT_SMA_TREND_RSI,
    last_rth_15m_closes,
    overlay_developing_close,
    scan_pullback_limits,
    scan_rsi2_moc,
)
from utils.research.cta_trend import load_macro_closes  # noqa: E402
from utils.research.panel import list_daily_symbols, load_wide_panels  # noqa: E402
from utils.research.realistic_purchaser import as_et, is_last_rth_15m_bar  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("choppy_regime_scan")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FIELDS = ("open", "high", "low", "close", "volume")


def _latest_1545_session(frames: dict) -> Optional[pd.Timestamp]:
    latest = None
    for df in frames.values():
        if df is None or df.empty:
            continue
        work = df
        if "ts" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
            work = work.set_index("ts")
        for ts in reversed(work.index):
            try:
                if not is_last_rth_15m_bar(ts):
                    continue
                day = pd.Timestamp(as_et(ts).date())
            except (TypeError, ValueError):
                continue
            if latest is None or day > latest:
                latest = day
            break
    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description="Choppy-regime daily scan")
    ap.add_argument("--mode", choices=("both", "rsi2", "pullback"), default="both")
    ap.add_argument("--lookback-days", type=int, default=400)
    ap.add_argument("--n-symbols", type=int, default=0)
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--sma-trend", type=int, default=DEFAULT_SMA_TREND_RSI)
    ap.add_argument("--rsi-entry", type=float, default=5.0)
    ap.add_argument("--rsi-cross-back", type=float, default=None)
    ap.add_argument("--min-close-loc", type=float, default=0.0)
    ap.add_argument("--vix-max", type=float, default=None)
    ap.add_argument("--vix-skip", action="store_true")
    ap.add_argument("--no-cpi", action="store_true")
    ap.add_argument("--no-15m", action="store_true", help="Alpaca daily close only (not 15:45)")
    ap.add_argument("--allow-more-lows", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    symbols = list_daily_symbols("ALPACA", "1d")
    if args.n_symbols and args.n_symbols > 0:
        symbols = symbols[: int(args.n_symbols)]
    if len(symbols) < 20:
        LOG.error("Universe too small (%d)", len(symbols))
        return 1

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=int(args.lookback_days) + 30)
    panels = load_wide_panels(
        symbols,
        start=start,
        end=end,
        provider="ALPACA",
        timeframe="1d",
        fields=FIELDS,
        workers=args.workers,
    )
    close = panels["close"]
    volume = panels["volume"]
    asof = close.index.max()
    LOG.info("Alpaca 1d asof=%s names=%d", asof.date(), close.shape[1])

    vix = None
    vix_max = 30.0 if args.vix_skip else args.vix_max
    if vix_max is not None:
        macros = load_macro_closes(["VIX"], start, end)
        vix = macros.get("VIX")
        if vix is None:
            LOG.warning("VIX missing; skip filter off")
            vix_max = None
    else:
        macros = load_macro_closes(["VIX"], start, end)
        vix = macros.get("VIX")

    liq = dict(liquid_n=args.liquid_n, min_price=args.min_price, min_adv=args.min_adv)
    rsi_close = close
    rsi_vol = volume
    rsi_asof = asof

    if args.mode in ("both", "rsi2") and not args.no_15m:
        ib_syms = set(list_daily_symbols("IB", "15m"))
        want = [s for s in close.columns if s in ib_syms]
        LOG.info("IB 15m overlap for developing close: %d", len(want))
        if want:
            ib_start = end - timedelta(days=7)
            frames = load_ohlcv_many(
                want,
                timeframe="15m",
                provider="IB",
                start=ib_start,
                end=end,
                use_cache=True,
                workers=args.workers,
                chunk_size=50,
            )
            session = _latest_1545_session(frames)
            if session is None:
                LOG.warning("No IB 15m 15:45 ET bar in last 7d; RSI-2 uses Alpaca daily")
            else:
                session_px = last_rth_15m_closes(frames, session=session)
                LOG.info(
                    "Developing close session=%s names_with_15m=%d (IB 15m, not Alpaca EOD)",
                    session.date(),
                    int(session_px.notna().sum()) if not session_px.empty else 0,
                )
                rsi_close = overlay_developing_close(close, session_px, session)
                if session not in volume.index:
                    rsi_vol = volume.copy()
                    rsi_vol.loc[session] = volume.iloc[-1]
                    rsi_vol = rsi_vol.sort_index()
                rsi_asof = session

    if args.mode in ("both", "rsi2"):
        hits = scan_rsi2_moc(
            rsi_close,
            rsi_vol,
            vix=vix,
            asof=rsi_asof,
            vix_max=vix_max,
            skip_cpi=not args.no_cpi,
            sma_trend=args.sma_trend,
            rsi_entry=args.rsi_entry,
            rsi_cross_back=args.rsi_cross_back,
            **liq,
        )
        LOG.info(
            "RSI-2 MOC candidates: %d asof=%s sma_trend=%d cross_back=%s",
            len(hits),
            rsi_asof.date(),
            args.sma_trend,
            args.rsi_cross_back,
        )
        if not hits.empty:
            print(hits.to_string(index=False))
        else:
            print("RSI-2 MOC: none")

    if args.mode in ("both", "pullback"):
        hits = scan_pullback_limits(
            panels["high"],
            panels["low"],
            close,
            volume,
            vix=vix,
            asof=asof,
            vix_max=vix_max,
            skip_cpi=not args.no_cpi,
            exactly_three_lows=not args.allow_more_lows,
            min_close_loc=args.min_close_loc,
            **liq,
        )
        LOG.info("3-day pullback limits: %d asof=%s (completed daily)", len(hits), asof.date())
        if not hits.empty:
            print(hits.to_string(index=False))
        else:
            print("3-day pullback: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
