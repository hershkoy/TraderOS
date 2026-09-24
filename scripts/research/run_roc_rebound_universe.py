#!/usr/bin/env python3
"""ROC rebound multi-symbol sample: ETFs / large caps / smaller caps.

Draws a seeded random sample of 50 names (default 12 ETF + 19 large + 19 small),
applies round-turn slippage (friction), and reports pooled books by bucket.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_roc_rebound_universe.py
  python scripts\\research\\run_roc_rebound_universe.py --n-total 50 --friction 0.25 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.panel import list_daily_symbols  # noqa: E402
from utils.research.roc_rebound import (  # noqa: E402
    ERA_BUCKETS,
    backtest_roc_rebound,
    book_from_trades,
    geom_mean_year_pf,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("roc_rebound_universe")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "roc_rebound"
YAHOO_CACHE_DIR = DEFAULT_OUT / "yahoo_cache"

# Liquid ETFs to sample from (many are missing from stock ticker_universe).
ETF_CANDIDATES: Sequence[str] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "MDY",
    "GLD",
    "TLT",
    "IEF",
    "UUP",
    "USO",
    "DBC",
    "EFA",
    "EEM",
    "HYG",
    "LQD",
    "VNQ",
    "SMH",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
    "XLC",
    "XLRE",
    "KRE",
    "ARKK",
)

# Round-trip friction %% by bucket (slippage + commission proxy).
DEFAULT_FRICTION_BY_BUCKET: Dict[str, float] = {
    "etf": 0.10,
    "large": 0.25,
    "small": 0.40,
}


def _frame_to_bt(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "datetime" in out.columns and "ts" not in out.columns:
                out = out.rename(columns={"datetime": "ts"})
            elif "index" in out.columns and "ts" not in out.columns:
                out = out.rename(columns={"index": "ts"})
            elif out.columns[0] != "ts":
                out = out.rename(columns={out.columns[0]: "ts"})
    return out


def _rank_symbols_by_adv(
    *,
    lookback_days: int = 45,
    provider: str = "ALPACA",
    timeframe: str = "1d",
    min_bars: int = 10,
) -> List[Tuple[str, float]]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")
    q = """
    SELECT symbol, AVG(close::float * volume::float) AS adv, COUNT(*) AS n
    FROM market_data
    WHERE timeframe = %s
      AND provider = %s
      AND ts >= NOW() - (%s || ' days')::interval
    GROUP BY symbol
    HAVING COUNT(*) >= %s
    ORDER BY adv DESC NULLS LAST
    """
    rows = client.execute_query(
        q, (timeframe, provider, str(int(lookback_days)), int(min_bars))
    )
    out: List[Tuple[str, float]] = []
    for r in rows or []:
        sym = str(r[0]).upper()
        if r[1] is None:
            continue
        out.append((sym, float(r[1])))
    return out


def _index_members(index_names: Sequence[str]) -> Set[str]:
    client = get_timescaledb_client()
    q = """
    SELECT DISTINCT UPPER(symbol)
    FROM ticker_universe
    WHERE is_active = TRUE AND index_name = ANY(%s)
    """
    rows = client.execute_query(q, (list(index_names),))
    return {str(r[0]).upper() for r in (rows or [])}


def _etfs_present_in_md(candidates: Sequence[str]) -> Set[str]:
    client = get_timescaledb_client()
    q = """
    SELECT DISTINCT UPPER(symbol)
    FROM market_data
    WHERE timeframe = '1d' AND symbol = ANY(%s)
    """
    rows = client.execute_query(q, ([s.upper() for s in candidates],))
    return {str(r[0]).upper() for r in (rows or [])}


def build_sample_cohorts(
    *,
    n_total: int = 50,
    n_etf: int = 12,
    n_large: int = 19,
    n_small: int = 19,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Seeded sample: ETFs + large (SPX/NDX) + smaller (rest of daily univ)."""
    if n_etf + n_large + n_small != int(n_total):
        # Rebalance to n_total if caller mismatch.
        n_etf = min(n_etf, n_total)
        rem = n_total - n_etf
        n_large = rem // 2
        n_small = rem - n_large

    rng = random.Random(int(seed))
    daily = {s.upper() for s in list_daily_symbols(provider="ALPACA", timeframe="1d")}
    large_pool = _index_members(("sp500", "nasdaq100")) & daily
    etf_md = _etfs_present_in_md(ETF_CANDIDATES)
    # Prefer ETFs already in MD; allow Yahoo for the rest of candidates.
    etf_pool = sorted(set(ETF_CANDIDATES) | etf_md)
    etf_pool = [s for s in etf_pool if s not in large_pool or s in etf_md or s in ETF_CANDIDATES]
    # Keep ETFs out of stock pools.
    etf_set = set(etf_pool)
    large_pool = sorted(s for s in large_pool if s not in etf_set)
    small_pool = sorted(s for s in daily if s not in large_pool and s not in etf_set)

    def _take(pool: List[str], n: int) -> List[str]:
        if n <= 0 or not pool:
            return []
        if n >= len(pool):
            return sorted(pool)
        return sorted(rng.sample(pool, n))

    etfs = _take(etf_pool, n_etf)
    # Prefer higher ADV within large/small when sampling (rank then sample from top half + full).
    adv = dict(_rank_symbols_by_adv())
    large_ranked = sorted(large_pool, key=lambda s: adv.get(s, 0.0), reverse=True)
    small_ranked = sorted(small_pool, key=lambda s: adv.get(s, 0.0), reverse=True)
    # Large: sample from top 200 ADV among SPX/NDX members (true large/liquid).
    large_src = large_ranked[: max(200, n_large * 4)] or large_ranked
    # Small: sample from bottom 60% ADV of non-index names (smaller / less liquid).
    if len(small_ranked) > 20:
        cut = max(n_small * 2, int(len(small_ranked) * 0.4))
        small_src = small_ranked[cut:] or small_ranked
    else:
        small_src = small_ranked

    large = _take(large_src, n_large)
    # Avoid collisions if RNG overlap somehow.
    used = set(etfs) | set(large)
    small_src2 = [s for s in small_src if s not in used]
    small = _take(small_src2, n_small)

    LOG.info(
        "Sample cohorts seed=%s: etf=%d/%d large=%d/%d small=%d/%d (daily=%d)",
        seed,
        len(etfs),
        len(etf_pool),
        len(large),
        len(large_src),
        len(small),
        len(small_src2),
        len(daily),
    )
    return {"etf": etfs, "large": large, "small": small}


def load_yahoo_daily(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    YAHOO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = YAHOO_CACHE_DIR / ("%s.json" % symbol.upper())
    if refresh or not path.exists() or path.stat().st_size < 5_000:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/%s"
            "?period1=727876800&period2=9999999999&interval=1d&events=history"
            % symbol.upper()
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
        LOG.info("Downloaded Yahoo %s", symbol)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(result["timestamp"], unit="s", utc=True),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    )
    return (
        df.dropna(subset=["close", "open"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )


def load_frames(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: Optional[datetime],
    workers: int,
    chunk_size: int,
    yahoo_fallback: bool,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    syms = [s.upper() for s in symbols]
    if not syms:
        return out

    frames = load_ohlcv_many(
        syms,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=workers,
        chunk_size=chunk_size,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    for s, df in frames.items():
        if df is not None and not df.empty:
            out[s.upper()] = _frame_to_bt(df)

    if yahoo_fallback:
        for s in syms:
            df = out.get(s)
            if df is not None and not df.empty:
                continue
            try:
                ydf = load_yahoo_daily(s)
                if start is not None:
                    ydf = ydf[ydf["ts"] >= pd.Timestamp(start, tz="UTC")]
                if end is not None:
                    ydf = ydf[ydf["ts"] <= pd.Timestamp(end, tz="UTC")]
                out[s] = ydf.reset_index(drop=True)
                LOG.info(
                    "Yahoo fallback %s: %d bars",
                    s,
                    len(out[s]),
                )
            except Exception as exc:
                LOG.warning("Yahoo failed for %s: %s", s, exc)
                out[s] = pd.DataFrame()
    return out


def _parse_friction_map(raw: Optional[str], default_friction: float) -> Dict[str, float]:
    if not raw:
        if default_friction is not None and default_friction > 0:
            return {k: float(default_friction) for k in ("etf", "large", "small")}
        return dict(DEFAULT_FRICTION_BY_BUCKET)
    out = dict(DEFAULT_FRICTION_BY_BUCKET)
    for part in str(raw).split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip().lower()] = float(v)
    return out


def run(
    cohorts: Dict[str, List[str]],
    *,
    start: datetime,
    end: Optional[datetime],
    friction_by_bucket: Dict[str, float],
    workers: int,
    chunk_size: int,
    yahoo_fallback: bool,
) -> Dict[str, Any]:
    cohort_of: Dict[str, str] = {}
    for name, syms in cohorts.items():
        for s in syms:
            cohort_of.setdefault(s.upper(), name)

    symbols = sorted(cohort_of.keys())
    t0 = time.perf_counter()
    frames = load_frames(
        symbols,
        start=start,
        end=end,
        workers=workers,
        chunk_size=chunk_size,
        yahoo_fallback=yahoo_fallback,
    )
    load_s = time.perf_counter() - t0
    LOG.info("Load done in %.1fs (%d/%d frames)", load_s, len(frames), len(symbols))

    per_symbol: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    trades_by: Dict[str, List[Dict[str, Any]]] = {
        "etf": [],
        "large": [],
        "small": [],
        "all": [],
    }

    t1 = time.perf_counter()
    for i, sym in enumerate(symbols, 1):
        cohort = cohort_of[sym]
        fric = float(friction_by_bucket.get(cohort, 0.25))
        df = _frame_to_bt(frames.get(sym))
        res = backtest_roc_rebound(df, symbol=sym, friction_pct=fric)
        row = {
            "symbol": sym,
            "cohort": cohort,
            "friction_pct": fric,
            "n": int(res.get("n") or 0),
            "E": res.get("expectancy_pct"),
            "PF": res.get("profit_factor"),
            "WR": res.get("win_rate_pct"),
            "median": res.get("median_pct"),
            "mdd": res.get("mdd_trade_eq_pct"),
            "eq": res.get("final_equity"),
            "avg_hold": res.get("avg_hold_days"),
            "bars": res.get("bars"),
            "first_bar": res.get("first_bar"),
            "last_bar": res.get("last_bar"),
            "reason": res.get("reason"),
        }
        per_symbol.append(row)
        for tr in res.get("trades") or []:
            tr = dict(tr)
            tr["cohort"] = cohort
            tr["friction_pct"] = fric
            all_trades.append(tr)
            trades_by[cohort].append(tr)
            trades_by["all"].append(tr)
        if i % 10 == 0 or i == len(symbols):
            LOG.info("Backtested %d/%d", i, len(symbols))

    bt_s = time.perf_counter() - t1
    books: Dict[str, Any] = {}
    for name, trs in trades_by.items():
        book = book_from_trades(trs)
        eras = year_bucket_stats(trs, ERA_BUCKETS)
        years = year_bucket_stats(trs)
        books[name] = {
            **book,
            "by_era": eras,
            "by_year_bucket": years,
            "geo_pf_year_min30": geom_mean_year_pf(years, min_n=30),
            "geo_pf_year_min10": geom_mean_year_pf(years, min_n=10),
            "n_symbols": len({t["symbol"] for t in trs}) if trs else 0,
        }

    return {
        "load_sec": round(load_s, 2),
        "bt_sec": round(bt_s, 2),
        "friction_by_bucket": friction_by_bucket,
        "cohorts": {k: list(v) for k, v in cohorts.items()},
        "per_symbol": per_symbol,
        "books": books,
        "n_trades": len(all_trades),
        "trades": all_trades,
    }


def _print_books(payload: Dict[str, Any]) -> None:
    LOG.info("--- Pooled books (with slippage) ---")
    for name in ("etf", "large", "small", "all"):
        b = payload["books"].get(name) or {}
        LOG.info(
            "  %-6s  syms=%s  n=%s  WR=%s%%  PF=%s  E=%s%%  med=%s  eq=%s  MDD=%s%%  geoPF10=%s",
            name,
            b.get("n_symbols"),
            b.get("n"),
            b.get("win_rate_pct"),
            b.get("profit_factor"),
            b.get("expectancy_pct"),
            b.get("median_pct"),
            b.get("final_equity"),
            b.get("mdd_trade_eq_pct"),
            None
            if b.get("geo_pf_year_min10") is None
            else round(float(b["geo_pf_year_min10"]), 3),
        )
        for era in b.get("by_era") or []:
            if int(era.get("n") or 0) <= 0:
                continue
            LOG.info(
                "           era %-10s n=%s WR=%s PF=%s E=%s",
                era.get("bucket"),
                era.get("n"),
                era.get("win_rate_pct"),
                era.get("profit_factor"),
                era.get("expectancy_pct"),
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="ROC rebound 50-name bucket sample")
    ap.add_argument("--n-total", type=int, default=50)
    ap.add_argument("--n-etf", type=int, default=12)
    ap.add_argument("--n-large", type=int, default=19)
    ap.add_argument("--n-small", type=int, default=19)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--friction",
        type=float,
        default=0.0,
        help="Uniform round-trip friction %% (0 = use per-bucket defaults)",
    )
    ap.add_argument(
        "--friction-by-bucket",
        type=str,
        default=None,
        help="Override map e.g. etf:0.10,large:0.25,small:0.40",
    )
    ap.add_argument("--start", type=str, default="2006-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--no-yahoo", action="store_true")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    friction_map = _parse_friction_map(args.friction_by_bucket, args.friction)
    LOG.info("Friction by bucket: %s", friction_map)

    cohorts = build_sample_cohorts(
        n_total=args.n_total,
        n_etf=args.n_etf,
        n_large=args.n_large,
        n_small=args.n_small,
        seed=args.seed,
    )
    for k, v in cohorts.items():
        LOG.info("  %s (%d): %s", k, len(v), ",".join(v))

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None
    t0 = time.perf_counter()
    payload = run(
        cohorts,
        start=start,
        end=end,
        friction_by_bucket=friction_map,
        workers=args.workers,
        chunk_size=args.chunk_size,
        yahoo_fallback=not args.no_yahoo,
    )
    elapsed = time.perf_counter() - t0
    payload["elapsed_sec"] = round(elapsed, 2)
    payload["seed"] = args.seed
    payload["start"] = args.start
    payload["end"] = args.end

    _print_books(payload)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in payload.items() if k != "trades"}
    summary_path = outdir / "roc_rebound_universe_summary.json"
    trades_path = outdir / "roc_rebound_universe_trades.csv"
    per_path = outdir / "roc_rebound_universe_per_symbol.csv"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(payload.get("trades") or []).to_csv(trades_path, index=False)
    pd.DataFrame(payload.get("per_symbol") or []).to_csv(per_path, index=False)
    LOG.info("Wrote %s", summary_path)
    LOG.info("Wrote %s (%d trades)", trades_path, payload.get("n_trades", 0))
    LOG.info("Wall-clock %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
