"""Last-15m earnings / biotech event overlays (same-list, no occupancy walk).

1. Flatten before known print inside 1 / 5 / 10 sessions
2. Entry skip: AMC today or BMO next session
3. Skip when overnight EM from name ATM IV > ATR k=2 stop (missing keeps)
4. Biotech / pharma sleeve off (SIC 2830-2836 or sector label)

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_earnings_event_filters.py
  python scripts\\research\\overlay_last_15m_earnings_event_filters.py --skip-15m
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.last_15m_earnings_event_filters import (  # noqa: E402
    FLATTEN_SESSIONS,
    EarningsEvent,
    apply_flatten_before_earnings,
    biotech_keep_mask,
    em_gt_stop_keep_mask,
    entry_skip_earnings_mask,
    events_from_yahoo_frame,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_early_exit,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_earn_evt")

DEFAULT_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
DEFAULT_OPTION_FEAT = (
    ROOT / "reports" / "ascending_channels" / "2026-10-18" / "last15m_option_features.csv"
)
DEFAULT_EARN_CACHE = (
    ROOT / "reports" / "ascending_channels" / "earnings_event_filters" / "earnings_cache.json"
)


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _year_slice(df: pd.DataFrame, gain: pd.Series, start: str, end: str) -> dict:
    m = _in_range(df, start, end)
    return book_stats(gain.loc[m])


def _score_book(
    label: str,
    trades: pd.DataFrame,
    gain: pd.Series,
    *,
    baseline_gain: Optional[pd.Series] = None,
    keep: Optional[pd.Series] = None,
    early_exit: bool = False,
) -> dict:
    years = summarize_by_year(
        trades.assign(gain_pct_ov=gain), gain_col="gain_pct_ov", buckets=YEAR_BUCKETS
    )
    s = book_stats(gain)
    geo = geom_mean_year_pf(years)
    s22 = _year_slice(trades, gain, "2022-01-01", "2022-12-31")
    s23 = _year_slice(trades, gain, "2023-01-01", "2023-12-31")
    s2223 = _year_slice(trades, gain, "2022-01-01", "2023-12-31")
    cut: dict = {}
    if baseline_gain is not None and keep is not None and not early_exit:
        cut = winner_cut_skip(baseline_gain, keep)
    elif baseline_gain is not None and early_exit:
        cut = winner_cut_early_exit(baseline_gain, gain)

    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s | 2022 n=%d E=%s PF=%s | 2023 n=%d E=%s PF=%s",
        label,
        s["n"],
        "n/a" if s["expectancy_pct"] is None else "%.3f" % s["expectancy_pct"],
        "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
        "n/a" if geo is None else "%.3f" % geo,
        s22["n"],
        "n/a" if s22["expectancy_pct"] is None else "%.3f" % s22["expectancy_pct"],
        "n/a" if s22["profit_factor"] is None else "%.3f" % s22["profit_factor"],
        s23["n"],
        "n/a" if s23["expectancy_pct"] is None else "%.3f" % s23["expectancy_pct"],
        "n/a" if s23["profit_factor"] is None else "%.3f" % s23["profit_factor"],
    )
    print("---- %s ----" % label)
    print(years.to_string(index=False))
    if cut:
        print(
            "cut n_drop=%s win$=%.1f lose$=%.1f blunt=%s dollar_ratio=%s"
            % (
                cut.get("n_dropped", cut.get("n_cut_earlier")),
                float(cut.get("winner_drop_gain_sum", cut.get("winner_gain_delta_sum") or 0.0)),
                float(cut.get("loser_drop_gain_sum", cut.get("loser_gain_delta_sum") or 0.0)),
                cut.get("blunt"),
                cut.get("dollar_ratio"),
            )
        )
    row = {
        "label": label,
        "n": s["n"],
        "expectancy_pct": s["expectancy_pct"],
        "profit_factor": s["profit_factor"],
        "geo_year_pf": geo,
        "n_2022": s22["n"],
        "e_2022": s22["expectancy_pct"],
        "pf_2022": s22["profit_factor"],
        "n_2023": s23["n"],
        "e_2023": s23["expectancy_pct"],
        "pf_2023": s23["profit_factor"],
        "n_2022_23": s2223["n"],
        "e_2022_23": s2223["expectancy_pct"],
        "pf_2022_23": s2223["profit_factor"],
    }
    for k, v in cut.items():
        row["cut_%s" % k] = v
    return row


def _load_earnings_cache(path: Path) -> Dict[str, List[dict]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("earnings cache read failed: %s", exc)
        return {}


def _save_earnings_cache(path: Path, cache: Mapping[str, List[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _fetch_one_symbol(sym: str) -> Tuple[str, List[EarningsEvent]]:
    import yfinance as yf

    try:
        frame = yf.Ticker(sym).get_earnings_dates(limit=80)
    except Exception as exc:
        LOG.warning("earnings fetch failed %s: %s", sym, exc)
        return sym, []
    return sym, events_from_yahoo_frame(frame)


def load_earnings_map(
    symbols: Sequence[str],
    cache_path: Path,
    *,
    workers: int = 8,
    refresh_empty: bool = True,
) -> Dict[str, List[EarningsEvent]]:
    raw = _load_earnings_cache(cache_path)
    out: Dict[str, List[EarningsEvent]] = {}
    missing: List[str] = []
    for sym in symbols:
        s = str(sym).upper()
        if s in raw and (raw[s] or not refresh_empty):
            out[s] = [
                EarningsEvent(
                    earn_date=date.fromisoformat(r["earn_date"]),
                    amc=bool(r["amc"]),
                )
                for r in (raw[s] or [])
            ]
        else:
            missing.append(s)
    if missing:
        LOG.info(
            "Fetching earnings for %d symbols (cache nonempty %d)",
            len(missing),
            len(out),
        )
        dirty = dict(raw)
        # Keep workers low: Yahoo rate-limits and caches empty lists.
        with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
            futs = {pool.submit(_fetch_one_symbol, s): s for s in missing}
            done = 0
            for fut in as_completed(futs):
                sym, events = fut.result()
                out[sym] = events
                dirty[sym] = [
                    {"earn_date": e.earn_date.isoformat(), "amc": bool(e.amc)} for e in events
                ]
                done += 1
                if done % 25 == 0 or done == len(missing):
                    LOG.info("earnings progress %d/%d nonempty_batch=%d", done, len(missing), len(events))
                    _save_earnings_cache(cache_path, dirty)
        _save_earnings_cache(cache_path, dirty)
    return out


def load_biotech_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB not connected (need sector/SIC for biotech skip)")
    cur = client.connection.cursor()
    sectors: Dict[str, str] = {}
    sics: Dict[str, str] = {}
    try:
        cur.execute(
            """
            SELECT DISTINCT ON (upper(symbol))
                   upper(symbol), COALESCE(sector, '')
            FROM ticker_universe
            WHERE sector IS NOT NULL AND btrim(sector) <> ''
            ORDER BY upper(symbol), last_updated DESC NULLS LAST
            """
        )
        for sym, sector in cur.fetchall():
            sectors[str(sym).upper()] = str(sector or "")
        cur.execute(
            """
            SELECT DISTINCT ON (upper(symbol))
                   upper(symbol), sic
            FROM fundamentals_quarterly
            WHERE sic IS NOT NULL AND btrim(sic) <> ''
            ORDER BY upper(symbol), period_end DESC NULLS LAST
            """
        )
        for sym, sic in cur.fetchall():
            sics[str(sym).upper()] = str(sic or "")
    finally:
        cur.close()
    return sectors, sics


def attach_name_iv(trades: pd.DataFrame, option_feat_path: Path) -> pd.DataFrame:
    out = trades.copy()
    if "name_atm_iv" in out.columns and out["name_atm_iv"].notna().any():
        return out
    if not option_feat_path.exists():
        LOG.warning("option features missing: %s", option_feat_path)
        out["name_atm_iv"] = float("nan")
        return out
    feat = pd.read_csv(option_feat_path)
    feat["stock"] = feat["stock"].astype(str).str.upper()
    feat["buy_date"] = pd.to_datetime(feat["buy_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["buy_date"] = pd.to_datetime(out["buy_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = ["stock", "buy_date", "name_atm_iv"]
    if "atr_1d_pct" in feat.columns and "atr_1d_pct" not in out.columns:
        cols.append("atr_1d_pct")
    merged = out.merge(feat[cols], on=["stock", "buy_date"], how="left", suffixes=("", "_feat"))
    if "name_atm_iv_feat" in merged.columns:
        merged["name_atm_iv"] = merged["name_atm_iv"].fillna(merged["name_atm_iv_feat"])
        merged = merged.drop(columns=["name_atm_iv_feat"])
    if "atr_1d_pct_feat" in merged.columns:
        if "atr_1d_pct" not in merged.columns:
            merged["atr_1d_pct"] = merged["atr_1d_pct_feat"]
        else:
            merged["atr_1d_pct"] = merged["atr_1d_pct"].fillna(merged["atr_1d_pct_feat"])
        merged = merged.drop(columns=["atr_1d_pct_feat"])
    LOG.info("name_atm_iv marked %d/%d", int(merged["name_atm_iv"].notna().sum()), len(merged))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--option-feat", type=Path, default=DEFAULT_OPTION_FEAT)
    ap.add_argument("--earn-cache", type=Path, default=DEFAULT_EARN_CACHE)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--earn-workers", type=int, default=8)
    ap.add_argument("--skip-15m", action="store_true", help="Skip flatten overlay (2/3/4 only)")
    ap.add_argument(
        "--no-refresh-empty",
        action="store_true",
        help="Keep empty Yahoo earnings cache rows (default refetches empties)",
    )
    ap.add_argument("--symbols", type=str, default="", help="Optional comma list (smoke)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_wall0 = time.perf_counter()

    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    want = str(args.symbols or "").strip().upper()
    if want:
        smoke = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(smoke)].copy()
        LOG.info("Restricted to %s n=%d", sorted(smoke), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    trades = attach_name_iv(trades, Path(args.option_feat))
    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    rows: List[dict] = []
    rows.append(_score_book("baseline", trades, baseline_gain))

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    earn_map = load_earnings_map(
        symbols,
        Path(args.earn_cache),
        workers=int(args.earn_workers),
        refresh_empty=not bool(args.no_refresh_empty),
    )
    n_with = sum(1 for s in symbols if earn_map.get(s))
    LOG.info("Earnings coverage %d/%d symbols", n_with, len(symbols))

    # ---- 2) tight entry skip ----
    keep2 = entry_skip_earnings_mask(trades, earn_map)
    part2 = trades.loc[keep2].copy()
    g2 = pd.to_numeric(part2["gain_pct"], errors="coerce")
    rows.append(
        _score_book(
            "2_entry_skip_amc_bmo",
            part2,
            g2,
            baseline_gain=baseline_gain,
            keep=keep2,
        )
    )

    # ---- 3) EM > stop ----
    keep3 = em_gt_stop_keep_mask(trades)
    part3 = trades.loc[keep3].copy()
    g3 = pd.to_numeric(part3["gain_pct"], errors="coerce")
    rows.append(
        _score_book(
            "3_em_gt_stop_skip",
            part3,
            g3,
            baseline_gain=baseline_gain,
            keep=keep3,
        )
    )

    # ---- 4) biotech sleeve off ----
    sectors, sics = load_biotech_maps()
    keep4 = biotech_keep_mask(trades, sector_by_symbol=sectors, sic_by_symbol=sics)
    part4 = trades.loc[keep4].copy()
    g4 = pd.to_numeric(part4["gain_pct"], errors="coerce")
    rows.append(
        _score_book(
            "4_biotech_sleeve_off",
            part4,
            g4,
            baseline_gain=baseline_gain,
            keep=keep4,
        )
    )
    n_bio = int((~keep4).sum())
    LOG.info("Biotech drops %d / %d", n_bio, len(trades))

    # ---- 1) flatten before print (needs 15m) ----
    if not args.skip_15m:
        buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(days=5)
        sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(days=21)
        LOG.info(
            "Loading IB 15m symbols=%d window %s -> %s",
            len(symbols),
            buy_min.strftime("%Y-%m-%d"),
            sell_max.strftime("%Y-%m-%d"),
        )
        t0 = time.perf_counter()
        panels = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=buy_min,
            end=sell_max,
            workers=int(args.workers),
            use_cache=True,
        )
        n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
        LOG.info("Loaded IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)
        indexed = build_hold_session_index(panels, trades, pad_before=5, pad_after=21)
        LOG.info("Indexed 15m sessions for %d symbols", len(indexed))

        for max_sess in FLATTEN_SESSIONS:
            gains = []
            keep_mask = []
            detail_rows = []
            for idx, row in trades.iterrows():
                sym = str(row["stock"]).upper()
                by_day = indexed.get(sym) or {}
                got = apply_flatten_before_earnings(
                    row,
                    earn_map.get(sym) or [],
                    by_day,
                    max_sessions=int(max_sess),
                )
                if got.skipped:
                    keep_mask.append(False)
                    gains.append(float("nan"))
                else:
                    keep_mask.append(True)
                    gains.append(got.gain_pct)
                detail_rows.append(
                    {
                        "stock": sym,
                        "buy_date": str(row.get("buy_date") or ""),
                        "max_sessions": int(max_sess),
                        "used": bool(got.used),
                        "skipped": bool(got.skipped),
                        "reason": got.reason,
                        "sessions_ahead": got.sessions_ahead,
                        "flatten_session": (
                            got.flatten_session.isoformat() if got.flatten_session else ""
                        ),
                        "gain_before": float(row["gain_pct"]),
                        "gain_after": None if got.skipped else got.gain_pct,
                        "sell_px_after": got.sell_px,
                    }
                )
            keep_s = pd.Series(keep_mask, index=trades.index, dtype=bool)
            gain_s = pd.Series(gains, index=trades.index, dtype=float)
            # Score kept rows only (skips drop); early-exit rows keep modified gain.
            part = trades.loc[keep_s].copy()
            g = gain_s.loc[keep_s]
            # For winner-cut on early exits, compare full-index overlay gain
            # (skipped = dropped from book via keep mask).
            overlay_for_cut = gain_s.where(keep_s, other=float("nan"))
            # Rebuild a same-length series for early-exit cut: use overlay gain where
            # kept, and drop skipped from baseline via keep mask.
            label = "1_flatten_sess%d" % int(max_sess)
            row_score = _score_book(
                label,
                part,
                g,
                baseline_gain=baseline_gain,
                keep=keep_s,
            )
            # Also report early-exit dollar move on the non-skipped flattened subset.
            flattened = [d for d in detail_rows if d["used"] and not d["skipped"]]
            row_score["n_flattened"] = len(flattened)
            row_score["n_skipped_entry"] = int(sum(1 for d in detail_rows if d["skipped"]))
            rows.append(row_score)

            outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
            outdir.mkdir(parents=True, exist_ok=True)
            det_path = outdir / ("last_15m_flatten_sess%d_detail.csv" % int(max_sess))
            pd.DataFrame(detail_rows).to_csv(det_path, index=False)
            LOG.info("Wrote %s", det_path)

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    sum_path = outdir / "last_15m_earnings_event_filters_summary.csv"
    summary.to_csv(sum_path, index=False)
    wall = time.perf_counter() - t_wall0
    meta = {
        "trades": str(args.trades),
        "n_trades": int(len(trades)),
        "earn_symbols": int(n_with),
        "wall_s": round(wall, 1),
        "skip_15m": bool(args.skip_15m),
    }
    (outdir / "last_15m_earnings_event_filters_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    LOG.info("Wrote %s wall=%.1fs", sum_path, wall)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
