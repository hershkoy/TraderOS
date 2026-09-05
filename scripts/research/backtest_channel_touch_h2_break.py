#!/usr/bin/env python3
"""A/B: after H2, fill a close above resistance instead of cancelling.

MGNI 2020-10: H2 in Aug, first close>resist ~Oct 13 (currently aborts the L3 wait).
Nightly EOD detector now matches the daily H2 resist-break span365 sleeve.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_h2_break.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_h2_break.py --preset 15m --all-symbols
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _scan_trades,
    _summarize,
    apply_friction,
    enrich_rs,
    filter_trades,
    select_same_day_rs,
    keep_one_per_symbol_day,
    summarize_by_year,
)
from find_ascending_channels import list_symbols_fast  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_scale import (  # noqa: E402
    BARS_PER_RTH_SESSION,
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
    PRESET_15M,
)
from scripts.research.generate_channel_touch_tv_report import (  # noqa: E402
    summary_sidecar_path,
    write_summary_sidecar,
)
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_h2_break")

RAW_TRADES = resolve_artifact("channel_touch_trades_raw_20260828_194314.csv")
KEEPER_TRADES = resolve_artifact("channel_touch_trades_20260828_194314.csv")
RAW_TRADES_15M = resolve_artifact("channel_touch_15m_trades_raw_20260829_105511.csv")
KEEPER_TRADES_15M = resolve_artifact("channel_touch_15m_trades_20260829_105511.csv")
L3_FILTERS = dict(
    require_in_channel=True,
    max_channel_span_days=365.0,
    max_beyond_width=0.25,
    max_rsi=50.0,
)
L3_FILTERS_15M = dict(
    require_in_channel=True,
    max_channel_span_days=10.0,
)


def _fmt(s: dict) -> str:
    return "n=%s E=%s PF=%s WR=%s med=%s hard_stop=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct"),
        s.get("hard_stop_exits"),
    )


def _net(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    if df.empty:
        return df
    return apply_friction(df, friction) if friction else df


def _flat_scan_params(base: dict) -> dict:
    out = {k: v for k, v in base.items() if k != "channel_kwargs"}
    ck = base.get("channel_kwargs") or {}
    if isinstance(ck, dict):
        out.update(ck)
    return out


def _write_h2_summary(
    csv_path: Path,
    *,
    title: str,
    base: dict,
    extra: dict,
    results: dict | None = None,
    notes: list | None = None,
) -> Path:
    params = _flat_scan_params(base)
    params.update(extra)
    return write_summary_sidecar(
        summary_sidecar_path(csv_path),
        title=title,
        params=params,
        results=results,
        notes=notes,
    )


def _daily_base() -> dict:
    return {
        "entry_touch": 3,
        "stop_pct": 0.03,
        "trail_pct": 0.10,
        "trail_pct_wide": 0.18,
        "squeeze_adaptive": True,
        "squeeze_pctile": 75.0,
        "squeeze_lookback": 100,
        "pivot_len": 15,
        "entry_mode": "l3_touch",
        "atr_stop_mult": 2.0,
        "stop_pct_floor": 0.015,
        "stop_pct_ceil": 0.06,
        "window_bars": DAILY_WINDOW_BARS,
        "window_step_bars": DAILY_WINDOW_STEP_BARS,
        "entry_features": True,
        "entry_slip_pct": 0.001,
        "max_l3_wait_bars": 252,
        "min_l3_wait_bars": 6,
        "h2_resist_break": True,
        "h2_resist_break_only": True,
        "shakeout_breakout": False,
        "shakeout_breakout_min_inside": 1,
        "shakeout_breakout_hard_stop": False,
        "channel_kwargs": {
            "error_pct": 1.2,
            "flat_pct": 0.04,
            "min_bars_apart": 15,
            "min_intervening_rally_pct": 4.0,
            "min_intervening_pullback_pct": 3.0,
            "min_total_rise_pct": 3.0,
            "max_low_pivots": 16,
            "causal_h2": True,
        },
    }


def _preset_15m_base() -> dict:
    p = PRESET_15M
    return {
        "entry_touch": 3,
        "stop_pct": float(p["stop_pct"]),
        "trail_pct": float(p["trail_pct"]),
        "trail_pct_wide": float(p["trail_pct_wide"]),
        "squeeze_adaptive": True,
        "squeeze_pctile": 75.0,
        "squeeze_lookback": int(p["squeeze_lookback"]),
        "pivot_len": int(p["pivot_len"]),
        "entry_mode": "l3_touch",
        "atr_stop_mult": float(p["atr_stop_mult"]),
        "stop_pct_floor": float(p["stop_pct_floor"]),
        "stop_pct_ceil": float(p["stop_pct_ceil"]),
        "window_bars": int(p["window_bars"]),
        "window_step_bars": int(p["window_step_bars"]),
        "entry_features": True,
        "feature_asof_prior_bar": True,
        "include_time": True,
        "adv_lookback": int(p["adv_lookback"]),
        "entry_slip_pct": 0.001,
        "max_l3_wait_bars": 252 * BARS_PER_RTH_SESSION,
        "min_l3_wait_bars": 12,
        "h2_resist_break": True,
        "h2_resist_break_only": True,
        "channel_kwargs": {
            "error_pct": float(p["error_pct"]),
            "flat_pct": float(p["flat_pct"]),
            "min_bars_apart": int(p["min_bars_apart"]),
            "min_intervening_rally_pct": float(p["min_rally_pct"]),
            "min_intervening_pullback_pct": float(p["min_pullback_pct"]),
            "min_total_rise_pct": float(p["min_total_rise_pct"]),
            "max_low_pivots": int(p["max_low_pivots"]),
            "causal_h2": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="H2 resistance-break A/B")
    ap.add_argument("--preset", default="", choices=("", "15m"))
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--keeper", type=Path, default=None)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--friction-pct", type=float, default=None)
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument(
        "--all-symbols",
        action="store_true",
        help="Scan full IB 15m (or daily raw CSV) universe instead of n-symbols / keeper names",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument(
        "--realistic-fill",
        action="store_true",
        help="Use realistic purchase prices. Default 15m fill is signal-bar close; "
        "--realistic-fill-mode next-mid restores next-bar mid.",
    )
    ap.add_argument(
        "--realistic-fill-mode",
        choices=("signal-close", "next-mid"),
        default="signal-close",
        help="When --realistic-fill: signal-close (default) or next-mid (kept).",
    )
    ap.add_argument(
        "--max-low-to-mid-pct",
        type=float,
        default=0.005,
        help="Cancel when exec 15m (mid-low)/mid exceeds this (default 0.5%%)",
    )
    ap.add_argument(
        "--shakeout-breakout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After first H2 resist-break, re-arm for a second close above resist "
        "(any-closed, min inside closes). Off by default.",
    )
    ap.add_argument(
        "--shakeout-breakout-min-inside",
        type=int,
        default=1,
        help="Inside closes required before the second resist-break (default 1)",
    )
    ap.add_argument(
        "--shakeout-breakout-hard-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only keep extras whose parent exited on the ATR hard stop",
    )
    args = ap.parse_args()
    t0 = time.perf_counter()
    is_15m = (args.preset or "").strip() == "15m"
    raw_path = args.raw or (RAW_TRADES_15M if is_15m else RAW_TRADES)
    keeper_path = args.keeper or (KEEPER_TRADES_15M if is_15m else KEEPER_TRADES)
    friction = float(
        args.friction_pct if args.friction_pct is not None else (0.10 if is_15m else 0.25)
    )
    gain_col = "gain_pct_net" if friction else "gain_pct"
    start_s = args.start.strip() or "2018-11-01"
    end_s = args.end.strip() or ("2025-12-02" if is_15m else "2026-08-27")
    l3_filters = L3_FILTERS_15M if is_15m else L3_FILTERS
    span_cap = 10.0 if is_15m else 365.0
    span_label = "10" if is_15m else "365"
    rs_bars = BARS_PER_RTH_SESSION if is_15m else 1
    chunk_size = int(args.chunk_size) or (int(PRESET_15M["chunk_size"]) if is_15m else 50)

    keeper = pd.DataFrame()
    if keeper_path.exists() and not args.all_symbols:
        keeper = pd.read_csv(keeper_path)

    if args.all_symbols:
        provider = "IB" if is_15m else "ALPACA"
        tf = "15m" if is_15m else "1d"
        symbols = sorted({s.upper() for s in list_symbols_fast(provider, tf)} | {"SPY"})
        logger.info("Full universe %s %s: %d symbols", provider, tf, len(symbols) - 1)
    else:
        raw = pd.read_csv(raw_path)
        symbols = sorted(set(raw["stock"].astype(str).str.upper()) | {"SPY"})
        if is_15m and int(args.n_symbols) > 0:
            names = [s for s in symbols if s != "SPY"]
            if len(names) > int(args.n_symbols):
                names = names[: int(args.n_symbols)]
            symbols = sorted(set(names) | {"SPY"})
        logger.info(
            "preset=%s raw n=%d unique=%d keeper n=%d",
            args.preset or "1d",
            len(raw),
            len(symbols) - 1,
            0 if keeper.empty else len(keeper),
        )

    start = datetime.strptime(start_s, "%Y-%m-%d")
    end = datetime.strptime(end_s, "%Y-%m-%d")
    if is_15m:
        panels = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=start,
            end=end,
            workers=int(args.load_workers),
            chunk_size=chunk_size,
            use_cache=True,
        )
    else:
        panels = load_ohlcv_many(
            symbols,
            timeframe="1d",
            provider="ALPACA",
            start=start,
            end=end,
            fallback_provider="IB",
            merge_mode="prefix",
            workers=int(args.load_workers),
        )
    spy_df = panels.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("No SPY panel")
        return 1
    logger.info("OHLCV loaded in %.1fs", time.perf_counter() - t0)

    base = _preset_15m_base() if is_15m else _daily_base()
    base["realistic_fill"] = bool(args.realistic_fill)
    base["realistic_fill_mode"] = str(args.realistic_fill_mode)
    base["max_low_to_mid_pct"] = float(args.max_low_to_mid_pct)
    base["shakeout_breakout"] = bool(args.shakeout_breakout)
    base["shakeout_breakout_min_inside"] = int(args.shakeout_breakout_min_inside)
    base["shakeout_breakout_hard_stop"] = bool(args.shakeout_breakout_hard_stop)
    panels_15m = None
    if (not is_15m) and args.realistic_fill:
        t_15 = time.perf_counter()
        panels_15m = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=start,
            end=end,
            workers=int(args.load_workers),
            chunk_size=chunk_size,
            use_cache=True,
        )
        n_have = sum(1 for s in symbols if s != "SPY" and panels_15m.get(s) is not None and not panels_15m[s].empty)
        logger.info("IB 15m purchase panels %d/%d in %.1fs", n_have, len(symbols) - 1, time.perf_counter() - t_15)
    t_scan = time.perf_counter()
    scanned = _scan_trades(
        panels, symbols=symbols, workers=int(args.workers), base=base, panels_15m=panels_15m
    )
    logger.info("Scan n=%d in %.1fs", len(scanned), time.perf_counter() - t_scan)
    if scanned.empty:
        logger.error("No trades")
        return 1
    scanned = enrich_rs(
        scanned, panels, spy_df, lookbacks=(63, 126), bars_per_session=rs_bars
    )

    is_brk = scanned["resist_break"].fillna(False).astype(bool)
    l3 = scanned.loc[~is_brk].copy()
    brk = scanned.loc[is_brk].copy()
    l3_q = filter_trades(l3, **l3_filters)
    brk_span = filter_trades(brk, max_channel_span_days=span_cap) if not brk.empty else brk
    l3_q = _net(l3_q, friction)
    brk_n = _net(brk, friction)
    brk_span_n = _net(brk_span, friction)

    if not keeper.empty:
        keeper_n = _net(keeper, friction)
        print("=== keeper CSV (RS top1 L3, not same universe if --all-symbols) ===")
        print(_fmt(_summarize(keeper_n, gain_col=gain_col)))
    print("=== rescanned L3 quality (no RS) ===")
    print(_fmt(_summarize(l3_q, gain_col=gain_col)))
    print("=== resist-break only (no span cap) ===")
    print(_fmt(_summarize(brk_n, gain_col=gain_col)))
    if not brk_n.empty:
        print(summarize_by_year(brk_n, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== resist-break + span<=%s ===" % span_label)
    print(_fmt(_summarize(brk_span_n, gain_col=gain_col)))
    if not brk_span_n.empty:
        print(
            summarize_by_year(brk_span_n, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )
    brk_span_u = keep_one_per_symbol_day(brk_span_n)
    print("=== resist-break span%s unique-symbol/day ===" % span_label)
    print(_fmt(_summarize(brk_span_u, gain_col=gain_col)))
    if not brk_span_u.empty:
        print(
            summarize_by_year(brk_span_u, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )
    brk_span_rs = select_same_day_rs(brk_span_n, rs_col="rs_spy_126d", max_per_day=1)
    print("=== resist-break span%s + RS top1 (optional cap) ===" % span_label)
    print(_fmt(_summarize(brk_span_rs, gain_col=gain_col)))
    if not brk_span_rs.empty:
        print(
            summarize_by_year(brk_span_rs, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )

    l3_u = keep_one_per_symbol_day(l3_q)
    l3_rs = select_same_day_rs(l3_q, rs_col="rs_spy_126d", max_per_day=1)
    combo = pd.concat([l3_q, brk_span_n], ignore_index=True, sort=False)
    combo_u = keep_one_per_symbol_day(combo)
    combo_rs = select_same_day_rs(combo, rs_col="rs_spy_126d", max_per_day=1)
    print("=== L3 quality unique-symbol/day (rescanned) ===")
    print(_fmt(_summarize(l3_u, gain_col=gain_col)))
    if not l3_u.empty:
        print(summarize_by_year(l3_u, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== L3 quality + RS top1 (optional cap) ===")
    print(_fmt(_summarize(l3_rs, gain_col=gain_col)))
    print(summarize_by_year(l3_rs, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== L3 quality + resist-break span%s unique-symbol/day ===" % span_label)
    print(_fmt(_summarize(combo_u, gain_col=gain_col)))
    if not combo_u.empty:
        print(summarize_by_year(combo_u, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== L3 quality + resist-break span%s + RS top1 (optional cap) ===" % span_label)
    print(_fmt(_summarize(combo_rs, gain_col=gain_col)))
    print(summarize_by_year(combo_rs, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    n_brk_kept = (
        int(combo_rs["resist_break"].fillna(False).astype(bool).sum()) if not combo_rs.empty else 0
    )
    print(
        "resist-break kept after RS=%d / span%s=%d / raw_break=%d"
        % (n_brk_kept, span_label, len(brk_span_n), len(brk_n))
    )

    mgni = scanned[scanned["stock"] == "MGNI"].copy()
    if not mgni.empty:
        cols = [
            c
            for c in (
                "buy_date",
                "sell_date",
                "buy_price",
                "gain_pct",
                "exit_reason",
                "resist_break",
                "channel_start",
                "channel_end",
                "channel_span_days",
                "channel_pos",
                "rsi_14",
            )
            if c in mgni.columns
        ]
        print("=== MGNI fills (rescan) ===")
        print(mgni[cols].sort_values("buy_date").to_string(index=False))

    outdir = dated_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra_meta = {
        "provider": "IB" if is_15m else "ALPACA",
        "timeframe": "15m" if is_15m else "1d",
        "fallback_provider": "" if is_15m else "IB",
        "merge_mode": "" if is_15m else "prefix",
        "start": start_s,
        "end": end_s,
        "symbols": max(0, len(symbols) - 1),
        "friction_pct": friction,
        "max_channel_span_days": span_cap,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "all_symbols": bool(args.all_symbols),
        "causal_h2": True,
        "shakeout_breakout": bool(args.shakeout_breakout),
        "shakeout_breakout_min_inside": int(args.shakeout_breakout_min_inside),
        "shakeout_breakout_hard_stop": bool(args.shakeout_breakout_hard_stop),
        "realistic_fill": bool(args.realistic_fill),
        "realistic_fill_mode": str(args.realistic_fill_mode),
    }
    notes = [
        "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
        "H2 resist-break fills a close above resistance after H2 (not L3 support tag)",
    ]
    if args.shakeout_breakout:
        notes.append(
            "Shakeout-breakout: after first resist-break, N inside closes then next close above resist "
            "(any-closed unless --shakeout-breakout-hard-stop)"
        )
    if not brk.empty:
        tag = "15m_" if is_15m else ""
        uni = "full_" if args.all_symbols else ""
        path = outdir / ("channel_touch_%s%sh2_resist_break_%s.csv" % (tag, uni, stamp))
        brk.to_csv(path, index=False)
        logger.info("Wrote %s", path)
        _write_h2_summary(
            path,
            title="H2 resistance-break backtest (no span cap)",
            base=base,
            extra={**extra_meta, "max_channel_span_days": None},
            results=_summarize(brk_n, gain_col=gain_col) if not brk_n.empty else None,
            notes=notes,
        )
        if not brk_span.empty:
            span_path = outdir / (
                "channel_touch_%s%sh2_break_span%s_%s.csv" % (tag, uni, span_label, stamp)
            )
            brk_span.to_csv(span_path, index=False)
            logger.info("Wrote %s", span_path)
            _write_h2_summary(
                span_path,
                title="H2 resistance-break backtest (span<=%s)" % span_label,
                base=base,
                extra=extra_meta,
                results=_summarize(brk_span_n, gain_col=gain_col) if not brk_span_n.empty else None,
                notes=notes,
            )
            if not brk_span_u.empty:
                u_path = outdir / (
                    "channel_touch_%s%sh2_break_span%s_unique_%s.csv" % (tag, uni, span_label, stamp)
                )
                brk_span_u.to_csv(u_path, index=False)
                logger.info("Wrote %s", u_path)
                _write_h2_summary(
                    u_path,
                    title="H2 resistance-break unique-symbol/day (span<=%s)" % span_label,
                    base=base,
                    extra=extra_meta,
                    results=_summarize(brk_span_u, gain_col=gain_col),
                    notes=notes
                    + ["Occupancy: keep_one_per_symbol_day (earliest fill per name per calendar day)"],
                )
    print("elapsed_sec=%.1f" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
