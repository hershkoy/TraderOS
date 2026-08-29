#!/usr/bin/env python3
"""A/B: shakeout rebuy after L3 (close below support, reclaim within N bars).

Rebuy only (does not hold through the dip). Parent trades are the live 1d
keeper (quality + RS top1). Nightly stays off until this beats E and PF.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_shakeout.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _atr,
    _line_at,
    _shakeout_rebuy_fill,
    _simulate_trade,
    _summarize,
    apply_friction,
    enrich_rs,
    filter_trades,
    select_same_day_rs,
    summarize_by_year,
)
from indicators.ttm_squeeze import calculate_squeeze_momentum  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_entry_features import (  # noqa: E402
    max_beyond_width,
    snapshot_stock_features,
    stock_entry_feature_series,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_shakeout")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "channel_touch_trades_20260828_194314.csv"
)
KEEPER_FILTERS = dict(
    require_in_channel=True,
    max_channel_span_days=365.0,
    max_beyond_width=0.25,
    max_rsi=50.0,
)


def _idx_on(dates: pd.DatetimeIndex, ts: object) -> Optional[int]:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    t = pd.Timestamp(t.date())
    loc = dates.get_indexer([t], method="pad")
    if loc is None or len(loc) == 0 or int(loc[0]) < 0:
        return None
    return int(loc[0])


def _rail_from_row(df: pd.DataFrame, row: pd.Series) -> Optional[dict]:
    dates = df.index
    l1_i = _idx_on(dates, row["channel_start"])
    buy_i = _idx_on(dates, row["buy_date"])
    sell_i = _idx_on(dates, row["sell_date"])
    if l1_i is None or buy_i is None or sell_i is None or buy_i <= l1_i:
        return None
    y0 = float(df["low"].iloc[l1_i])
    if not np.isfinite(y0) or y0 <= 0:
        return None
    pos = float(row["channel_pos"]) if pd.notna(row.get("channel_pos")) else 0.0
    width_pct = float(row["channel_width_pct"]) if pd.notna(row.get("channel_width_pct")) else 0.0
    width = width_pct / 100.0 * y0
    entry = float(row["buy_price"])
    if np.isfinite(pos) and np.isfinite(width) and width > 0:
        support_buy = entry - pos * width
    else:
        support_buy = entry / 1.001
    slope = (support_buy - y0) / float(buy_i - l1_i)
    if slope <= 0:
        slope = float(row.get("slope_pct_per_bar") or 0.0) / 100.0 * y0
    if not np.isfinite(slope) or slope <= 0:
        return None
    return {
        "support_x0": l1_i,
        "support_y0": y0,
        "support_slope": slope,
        "width": width,
        "buy_i": buy_i,
        "sell_i": sell_i,
    }


def _simulate_rebuy(
    df: pd.DataFrame,
    *,
    fill_i: int,
    fill_px: float,
    rail: dict,
    squeeze_mom: np.ndarray,
    atr: np.ndarray,
) -> Optional[dict]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_i = float(atr[fill_i]) if fill_i < len(atr) else float("nan")
    sim = _simulate_trade(
        high,
        low,
        close,
        df.index,
        fill_i,
        stop_pct=0.03,
        trail_pct=0.10,
        trail_pct_wide=0.18,
        squeeze_mom=squeeze_mom,
        squeeze_pctile=75.0,
        squeeze_lookback=100,
        atr_at_entry=atr_i if np.isfinite(atr_i) else None,
        atr_stop_mult=2.0,
        stop_pct_floor=0.015,
        stop_pct_ceil=0.06,
        support_x0=rail["support_x0"],
        support_y0=rail["support_y0"],
        support_slope=rail["support_slope"],
        channel_width=rail["width"],
        entry_px=fill_px,
    )
    return sim


def _fmt_summary(s: dict) -> str:
    return (
        "n=%s E=%s PF=%s WR=%s med=%s hard_stop=%s"
        % (
            s.get("n_trades"),
            s.get("expectancy_pct"),
            s.get("profit_factor"),
            s.get("win_rate_pct"),
            s.get("median_gain_pct"),
            s.get("hard_stop_exits"),
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel-touch shakeout rebuy A/B")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--bars", type=int, default=10, help="Primary shakeout window (trading bars)")
    ap.add_argument("--sweep", default="5,10,15", help="Comma list of windows to A/B")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--load-workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.perf_counter()

    keeper = pd.read_csv(args.trades)
    if keeper.empty:
        logger.error("Empty trades CSV %s", args.trades)
        return 1
    keeper = keeper.copy()
    keeper["shakeout_rebuy"] = False
    symbols = sorted(set(keeper["stock"].astype(str).str.upper()) | {"SPY"})
    logger.info("Keeper n=%d symbols=%d (incl SPY)", len(keeper), len(symbols))

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
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

    sweep = [int(x.strip()) for x in str(args.sweep).split(",") if x.strip()]
    max_bars = max([int(args.bars)] + sweep)
    cache: Dict[str, dict] = {}

    def _pack(sym: str) -> Optional[dict]:
        if sym in cache:
            return cache[sym]
        df = panels.get(sym)
        if df is None or df.empty:
            cache[sym] = None  # type: ignore[assignment]
            return None
        feat = stock_entry_feature_series(df)
        mom = calculate_squeeze_momentum(df, lengthKC=20, use_logging=False)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        pack = {
            "df": df,
            "high": high,
            "low": low,
            "close": close,
            "atr": _atr(high, low, close, length=14),
            "squeeze": mom.to_numpy(dtype=float),
            "feat": feat,
        }
        cache[sym] = pack
        return pack

    rows_by_n: Dict[int, List[dict]] = {n: [] for n in sweep}
    n_skip = 0
    for _, row in keeper.iterrows():
        sym = str(row["stock"]).upper()
        pack = _pack(sym)
        if pack is None:
            n_skip += 1
            continue
        df = pack["df"]
        rail = _rail_from_row(df, row)
        if rail is None:
            n_skip += 1
            continue
        extra = _shakeout_rebuy_fill(
            pack["high"],
            pack["low"],
            pack["close"],
            support_x0=rail["support_x0"],
            support_y0=rail["support_y0"],
            support_slope=rail["support_slope"],
            width=rail["width"],
            l3_i=rail["buy_i"],
            n=len(df),
            error_pct=1.2,
            slip=0.001,
            shakeout_bars=max_bars,
        )
        if extra is None:
            continue
        fill_i, fill_px = extra
        if fill_i <= int(rail["sell_i"]):
            continue
        bars_after = int(fill_i - rail["buy_i"])
        sim = _simulate_rebuy(
            df,
            fill_i=fill_i,
            fill_px=fill_px,
            rail=rail,
            squeeze_mom=pack["squeeze"],
            atr=pack["atr"],
        )
        if sim is None:
            continue
        entry_px = float(sim["buy_price"])
        width = float(rail["width"])
        support_at = _line_at(
            rail["support_y0"], rail["support_x0"], rail["support_slope"], fill_i
        )
        channel_pos = (
            (entry_px - support_at) / width if width > 0 and np.isfinite(support_at) else float("nan")
        )
        beyond = max_beyond_width(
            pack["high"],
            rail["support_y0"],
            rail["support_x0"],
            rail["support_slope"],
            width,
            rail["support_x0"],
            fill_i,
        )
        feat_snap = snapshot_stock_features(pack["feat"], fill_i)
        rec = {
            "stock": sym,
            "channel_start": row["channel_start"],
            "channel_end": row["channel_end"],
            "touch_num": 3,
            "buy_date": sim["buy_date"],
            "sell_date": sim["sell_date"],
            "buy_price": sim["buy_price"],
            "sell_price": sim["sell_price"],
            "gain_pct": sim["gain_pct"],
            "exit_reason": sim["exit_reason"],
            "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
            "max_beyond_width": round(float(beyond), 4) if np.isfinite(beyond) else None,
            "channel_span_days": row.get("channel_span_days"),
            "shakeout_rebuy": True,
            "shakeout_bars_after_l3": bars_after,
            "parent_buy_date": row["buy_date"],
            **feat_snap,
        }
        for n in sweep:
            if bars_after <= n:
                rows_by_n[n].append(rec)

    logger.info("Skipped rows (no panel/rail)=%d", n_skip)
    friction = float(args.friction_pct)
    if "gain_pct_net" in keeper.columns:
        base = keeper
        base_col = "gain_pct_net"
    else:
        base = apply_friction(keeper, friction) if friction else keeper
        base_col = "gain_pct_net" if friction else "gain_pct"
    print("=== baseline keeper (RS top1, no shakeout) ===")
    print(_fmt_summary(_summarize(base, gain_col=base_col)))
    print(summarize_by_year(base, gain_col=base_col, buckets=YEAR_BUCKETS).to_string(index=False))

    outdir = ROOT / "reports" / "ascending_channels"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for n in sweep:
        sh = pd.DataFrame(rows_by_n[n])
        if sh.empty:
            print("=== shakeout bars=%d: no rebys ===" % n)
            continue
        if spy_df is not None:
            sh = enrich_rs(sh, panels, spy_df, lookbacks=(63, 126))
        sh_q = filter_trades(sh, **KEEPER_FILTERS)
        if friction and not sh_q.empty:
            sh_q = apply_friction(sh_q, friction)
        print("=== shakeout-only bars=%d (quality, before RS) n_raw=%d ===" % (n, len(sh)))
        print(_fmt_summary(_summarize(sh_q, gain_col=base_col)))
        if not sh_q.empty:
            print(
                summarize_by_year(sh_q, gain_col=base_col, buckets=YEAR_BUCKETS).to_string(
                    index=False
                )
            )

        combo = pd.concat([base, sh_q], ignore_index=True, sort=False)
        combo_rs = select_same_day_rs(combo, rs_col="rs_spy_126d", max_per_day=1)
        print("=== combined + re-RS top1 bars=%d ===" % n)
        print(_fmt_summary(_summarize(combo_rs, gain_col=base_col)))
        print(
            summarize_by_year(combo_rs, gain_col=base_col, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )
        n_sh_kept = int(combo_rs["shakeout_rebuy"].fillna(False).astype(bool).sum()) if not combo_rs.empty else 0
        print("shakeout fills kept after RS=%d / quality=%d" % (n_sh_kept, len(sh_q)))

        if n == int(args.bars) and not sh.empty:
            csv_path = outdir / ("channel_touch_shakeout_rebuy_%dbars_%s.csv" % (n, stamp))
            sh.to_csv(csv_path, index=False)
            logger.info("Wrote %s", csv_path)
            cstm = sh[sh["stock"] == "CSTM"]
            if len(cstm):
                print("CSTM shakeout fills:")
                print(
                    cstm[
                        [
                            c
                            for c in (
                                "buy_date",
                                "sell_date",
                                "buy_price",
                                "gain_pct",
                                "exit_reason",
                                "shakeout_bars_after_l3",
                                "parent_buy_date",
                                "rsi_14",
                            )
                            if c in cstm.columns
                        ]
                    ].to_string(index=False)
                )
            else:
                print("CSTM: no shakeout rebuy in this window")

    print("elapsed_sec=%.1f" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
