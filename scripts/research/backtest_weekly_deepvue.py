#!/usr/bin/env python3
"""Backtest the causal Deepvue-style weekend scan with realistic 15m fills.

Weekend signal (completed daily through week-end). Buy next session 09:30 ET
15m mid. Sell: 3% hard stop (15m N+1 mid) or two daily closes below 21 EMA
(next 15m mid). Occupancy max_open unique symbols.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\backtest_weekly_deepvue.py --n-symbols 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    apply_friction,
    summarize_by_year,
)
from find_ascending_channels import _pick_15m_by_daily_adv  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.last_15m_volume_geometry import book_stats, geom_mean_year_pf  # noqa: E402
from utils.research.panel import frames_to_wide, list_daily_symbols, to_naive_index  # noqa: E402
from utils.research.realistic_purchaser import (  # noqa: E402
    index_rth_15m_by_session,
    purchase_next_session_open_mid,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.weekly_deepvue import (  # noqa: E402
    DeepvueParams,
    build_signal_frame,
    daily_below_ema21,
    ema,
    load_symbol_sectors,
    occupancy_take,
    qqq_risk_on,
    simulate_deepvue_exit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("weekly_deepvue")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FRICTION_PCT = 0.25


def _load_index_close(symbol: str, start: datetime, end: datetime) -> pd.Series:
    for provider in ("IB", "ALPACA"):
        frames = load_ohlcv_many(
            [symbol],
            timeframe="1d",
            provider=provider,
            start=start,
            end=end,
            use_cache=True,
            workers=1,
        )
        df = frames.get(symbol)
        if df is None or df.empty or "close" not in df.columns:
            LOG.warning("%s close missing provider=%s", symbol, provider)
            continue
        s = to_naive_index(df["close"].astype(float))
        LOG.info(
            "Loaded %s %s n=%d %s -> %s",
            symbol,
            provider,
            len(s),
            s.index[0].date(),
            s.index[-1].date(),
        )
        return s
    raise RuntimeError("%s daily close not found" % symbol)


def _load_sectors() -> Dict[str, str]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        LOG.warning("No DB for ticker_universe sectors; theme filter skipped")
        return {}
    cur = client.connection.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT ON (upper(symbol))
                   upper(symbol) AS symbol,
                   COALESCE(sector, '')
            FROM ticker_universe
            WHERE is_active IS TRUE
            ORDER BY upper(symbol), last_updated DESC NULLS LAST
            """
        )
        rows = [{"symbol": r[0], "sector": r[1]} for r in cur.fetchall()]
    except Exception as exc:
        LOG.warning("ticker_universe sector query failed: %s", exc)
        return {}
    finally:
        cur.close()
    return load_symbol_sectors(rows)


def _wide(frames: Dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return frames_to_wide(frames, field)


def _stats_block(label: str, trades: pd.DataFrame, gain_col: str = "gain_pct") -> None:
    g = pd.to_numeric(trades[gain_col], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    s = book_stats(g)
    years = summarize_by_year(trades, gain_col=gain_col, buckets=YEAR_BUCKETS)
    geo = geom_mean_year_pf(years)
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s.get("win_rate_pct"),
    )
    if not years.empty:
        print(years.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Causal Deepvue weekend scan + realistic 15m fills")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--qqq-start", default="2017-01-03")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-open", type=int, default=4)
    ap.add_argument("--stop-pct", type=float, default=0.03)
    ap.add_argument("--adv-min", type=float, default=20_000_000.0)
    ap.add_argument("--adr-min", type=float, default=3.0)
    ap.add_argument("--rs-pct", type=float, default=85.0)
    ap.add_argument("--rmv-max", type=float, default=45.0)
    ap.add_argument("--friction-pct", type=float, default=FRICTION_PCT)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    params = DeepvueParams(
        stop_pct=float(args.stop_pct),
        adv_min=float(args.adv_min),
        adr_min=float(args.adr_min),
        rs_pct=float(args.rs_pct),
        rmv_max=float(args.rmv_max),
        max_open=int(args.max_open),
        daily_focus=int(args.max_open),
    )
    outdir = args.outdir or dated_outdir(ROOT / "reports" / "weekly_deepvue")
    outdir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    qqq_start = datetime.strptime(args.qqq_start, "%Y-%m-%d")

    t0 = time.perf_counter()
    if args.all_symbols:
        symbols = list_daily_symbols("IB", "15m")
    else:
        symbols = _pick_15m_by_daily_adv(int(args.n_symbols))
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("Universe %d symbols (%.1fs)", len(symbols), timings["list_symbols"])
    if len(symbols) < 20:
        LOG.error("Universe too small (%d)", len(symbols))
        return 1

    t0 = time.perf_counter()
    daily_frames = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
        fallback_provider="IB",
        merge_mode="prefix",
    )
    timings["load_daily"] = time.perf_counter() - t0
    LOG.info("Daily panels %d/%d (%.1fs)", len(daily_frames), len(symbols), timings["load_daily"])

    open_px = _wide(daily_frames, "open")
    high = _wide(daily_frames, "high")
    low = _wide(daily_frames, "low")
    close = _wide(daily_frames, "close")
    volume = _wide(daily_frames, "volume")
    if close.empty:
        LOG.error("Empty daily close panel")
        return 1

    t0 = time.perf_counter()
    qqq = _load_index_close("QQQ", qqq_start, end)
    qon = qqq_risk_on(qqq, span=21).reindex(close.index).ffill().fillna(False)
    sectors = _load_sectors()
    timings["qqq_sectors"] = time.perf_counter() - t0
    LOG.info("QQQ risk-on days %d / %d", int(qon.sum()), len(qon))

    t0 = time.perf_counter()
    signals = build_signal_frame(
        open_px=open_px,
        high=high,
        low=low,
        close=close,
        volume=volume,
        qqq_on=qon,
        sectors=sectors,
        params=params,
    )
    timings["signals"] = time.perf_counter() - t0
    LOG.info("Weekend focus rows %d (%.1fs)", len(signals), timings["signals"])
    if signals.empty:
        LOG.error("No weekend signals")
        return 1

    sig_path = outdir / "weekly_deepvue_signals.csv"
    signals.to_csv(sig_path, index=False)
    LOG.info("Wrote %s", sig_path)

    need = sorted({str(s).upper() for s in signals["stock"].unique()})
    t0 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        need,
        timeframe="15m",
        provider="IB",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
    )
    timings["load_15m"] = time.perf_counter() - t0
    LOG.info("15m panels %d/%d (%.1fs)", len(panels_15m), len(need), timings["load_15m"])

    indexed: Dict[str, dict] = {}
    for sym, df in panels_15m.items():
        if df is None or df.empty:
            continue
        indexed[str(sym).upper()] = index_rth_15m_by_session(df, naive_tz="UTC")

    ema21_by_sym: Dict[str, pd.Series] = {}
    close_by_sym: Dict[str, pd.Series] = {}
    for sym in close.columns:
        s = close[sym].dropna()
        close_by_sym[str(sym).upper()] = s
        ema21_by_sym[str(sym).upper()] = ema(s, 21)

    t0 = time.perf_counter()
    raw: List[dict] = []
    n_skip_15m = 0
    n_skip_fill = 0
    n_skip_exit = 0
    for rec in signals.to_dict("records"):
        sym = str(rec["stock"]).upper()
        by_day = indexed.get(sym)
        if not by_day:
            n_skip_15m += 1
            continue
        sig_day = pd.Timestamp(rec["signal_date"]).date()
        bought = purchase_next_session_open_mid(
            None,
            signal_session_date=sig_day,
            session_index=by_day,
        )
        if not bought.filled or bought.fill_px is None or bought.exec_bar_ts is None:
            n_skip_fill += 1
            continue
        below = daily_below_ema21(close_by_sym[sym], ema21_by_sym[sym])
        exited = simulate_deepvue_exit(
            by_day,
            entry_ts=bought.exec_bar_ts,
            entry_px=float(bought.fill_px),
            below_ema=below,
            stop_pct=params.stop_pct,
            ma_closes=params.ma_closes,
        )
        if not exited.filled or exited.sell_px is None or exited.exec_bar_ts is None:
            n_skip_exit += 1
            continue
        buy_px = float(bought.fill_px)
        sell_px = float(exited.sell_px)
        gain = (sell_px / buy_px - 1.0) * 100.0
        raw.append(
            {
                "stock": sym,
                "sector": rec.get("sector"),
                "signal_date": str(pd.Timestamp(rec["signal_date"]).date()),
                "buy_date": pd.Timestamp(bought.exec_bar_ts).tz_convert("UTC").strftime("%Y-%m-%d")
                if getattr(bought.exec_bar_ts, "tzinfo", None)
                else str(pd.Timestamp(bought.exec_bar_ts).date()),
                "buy_ts": bought.exec_bar_ts,
                "buy_time": _utc_clock(bought.exec_bar_ts),
                "buy_price": round(buy_px, 4),
                "sell_ts": exited.exec_bar_ts,
                "sell_time": _utc_clock(exited.exec_bar_ts),
                "sell_price": round(sell_px, 4),
                "gain_pct": round(gain, 4),
                "exit_reason": exited.exit_reason,
                "rmv": rec.get("rmv"),
                "rs_1m": rec.get("rs_1m"),
                "rs_3m": rec.get("rs_3m"),
                "focus_rank": rec.get("focus_rank"),
            }
        )
    timings["fills"] = time.perf_counter() - t0
    LOG.info(
        "Filled %d skip_15m=%d skip_buy=%d skip_exit=%d (%.1fs)",
        len(raw),
        n_skip_15m,
        n_skip_fill,
        n_skip_exit,
        timings["fills"],
    )

    occupied = occupancy_take(raw, max_open=params.max_open)
    trades = pd.DataFrame(occupied)
    if not trades.empty:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y-%m-%d")
    csv_path = outdir / "weekly_deepvue_trades.csv"
    trades.to_csv(csv_path, index=False)
    LOG.info("Occupancy kept %d / %d -> %s", len(trades), len(raw), csv_path)

    _stats_block("gross", trades)
    if not trades.empty:
        net = apply_friction(trades, float(args.friction_pct))
        _stats_block("fric%.2f" % float(args.friction_pct), net, gain_col="gain_pct_net")
        if "exit_reason" in trades.columns:
            print(trades["exit_reason"].value_counts().to_string())

    timings["wall"] = time.perf_counter() - t_all
    LOG.info("Wall %.1fs %s", timings["wall"], timings)
    (outdir / "weekly_deepvue_timings.txt").write_text(
        "\n".join("%s=%.3f" % (k, v) for k, v in timings.items()) + "\n",
        encoding="utf-8",
    )
    return 0


def _utc_clock(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    raise SystemExit(main())
