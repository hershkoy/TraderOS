#!/usr/bin/env python3
"""Deep Pullback multi-symbol TimescaleDB research.

Cohorts:
  1) big equity indexes (SPY/QQQ/IWM/DIA/VTI/MDY)
  2) top ADV liquid names (~150)
  3) random sample from the remaining daily universe (~150)

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_deep_pullback_universe.py
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
from utils.research.deep_pullback import (  # noqa: E402
    INDEX_SYMBOLS,
    backtest_deep_pullback,
    book_from_trades,
    geom_mean_year_pf,
    year_bucket_stats,
)
from utils.research.panel import list_daily_symbols  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402

YAHOO_CACHE_DIR = ROOT / "reports" / "research" / "deep_pullback" / "yahoo_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("deep_pullback_universe")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "deep_pullback"
STATUS_DIR = ROOT / "docs" / "status_log" / "edge_hunt"


def _rank_symbols_by_adv(
    *,
    lookback_days: int = 45,
    provider: str = "ALPACA",
    timeframe: str = "1d",
    min_bars: int = 10,
) -> List[Tuple[str, float]]:
    """Recent ADV ranking via a bounded window (avoids full-table GROUP BY)."""
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
    rows = client.execute_query(q, (timeframe, provider, str(int(lookback_days)), int(min_bars)))
    out: List[Tuple[str, float]] = []
    for r in rows or []:
        sym = str(r[0]).upper()
        adv = r[1]
        if adv is None:
            continue
        out.append((sym, float(adv)))
    LOG.info("ADV rank: %d symbols (lookback=%dd)", len(out), lookback_days)
    return out


def build_symbol_cohorts(
    *,
    n_extra: int = 300,
    big_frac: float = 0.5,
    seed: int = 42,
    indexes: Sequence[str] = INDEX_SYMBOLS,
) -> Dict[str, List[str]]:
    """indexes + top-ADV big names + random remainder."""
    idx = [s.upper() for s in indexes]
    universe = [s.upper() for s in list_daily_symbols(provider="ALPACA", timeframe="1d")]
    if not universe:
        raise RuntimeError("Empty ALPACA 1d universe")

    ranked = _rank_symbols_by_adv()
    ranked_syms = [s for s, _ in ranked]
    # Prefer ADV order; fall back to universe order for unranked names.
    seen: Set[str] = set()
    ordered: List[str] = []
    for s in ranked_syms + universe:
        if s in seen:
            continue
        seen.add(s)
        ordered.append(s)

    idx_set = set(idx)
    pool = [s for s in ordered if s not in idx_set]
    n_big = int(round(n_extra * float(big_frac)))
    n_rand = int(n_extra) - n_big
    big = pool[:n_big]
    remainder = pool[n_big:]
    rng = random.Random(int(seed))
    if n_rand > len(remainder):
        n_rand = len(remainder)
    rand = sorted(rng.sample(remainder, n_rand)) if n_rand else []

    # Keep indexes that actually exist in DB (or still try load).
    indexes_present = [s for s in idx if s in seen or s in idx_set]
    LOG.info(
        "Cohorts: indexes=%d big=%d random=%d (universe=%d)",
        len(indexes_present),
        len(big),
        len(rand),
        len(universe),
    )
    return {
        "index": indexes_present,
        "big": big,
        "random": rand,
    }


def _frame_to_bt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            # common loader names
            if "datetime" in out.columns and "ts" not in out.columns:
                out = out.rename(columns={"datetime": "ts"})
            elif "index" in out.columns and "ts" not in out.columns:
                out = out.rename(columns={"index": "ts"})
            elif out.columns[0] != "ts":
                out = out.rename(columns={out.columns[0]: "ts"})
    return out


def load_yahoo_daily(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    """Yahoo daily OHLCV for index ETFs missing from TimescaleDB."""
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
        LOG.info("Downloaded Yahoo %s -> %s", symbol, path.name)
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
        df.dropna(subset=["close", "high"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )


def load_index_frames(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: Optional[datetime],
) -> Dict[str, pd.DataFrame]:
    """TimescaleDB first (ALPACA then IB), Yahoo fallback for missing index ETFs."""
    out: Dict[str, pd.DataFrame] = {}
    need_yahoo: List[str] = []
    syms = [s.upper() for s in symbols]
    if not syms:
        return out

    for provider, fallback in (("ALPACA", "IB"), ("IB", None)):
        missing = [s for s in syms if s not in out or out[s].empty]
        if not missing:
            break
        frames = load_ohlcv_many(
            missing,
            timeframe="1d",
            provider=provider,
            start=start,
            end=end,
            use_cache=True,
            workers=2,
            chunk_size=20,
            fallback_provider=fallback,
            merge_mode="prefix" if fallback else None,
        )
        for s, df in frames.items():
            if df is not None and not df.empty:
                out[s.upper()] = _frame_to_bt(df)

    for s in syms:
        df = out.get(s)
        if df is None or df.empty:
            need_yahoo.append(s)
    for s in need_yahoo:
        try:
            ydf = load_yahoo_daily(s)
            if start is not None:
                ydf = ydf[ydf["ts"] >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                ydf = ydf[ydf["ts"] <= pd.Timestamp(end, tz="UTC")]
            out[s] = ydf.reset_index(drop=True)
            LOG.info(
                "Index %s via Yahoo: %d bars %s -> %s",
                s,
                len(ydf),
                ydf["ts"].iloc[0].date() if len(ydf) else "?",
                ydf["ts"].iloc[-1].date() if len(ydf) else "?",
            )
        except Exception as exc:
            LOG.warning("Yahoo load failed for %s: %s", s, exc)
            out[s] = pd.DataFrame()
    return out


def run_universe(
    cohorts: Dict[str, List[str]],
    *,
    start: datetime,
    end: Optional[datetime],
    friction_pct: float,
    workers: int,
    chunk_size: int,
) -> Dict[str, Any]:
    cohort_of: Dict[str, str] = {}
    for name, syms in cohorts.items():
        for s in syms:
            # Prefer index label if a name appears in multiple (shouldn't).
            cohort_of.setdefault(s, name)

    index_syms = [s for s, c in cohort_of.items() if c == "index"]
    stock_syms = sorted(s for s, c in cohort_of.items() if c != "index")
    symbols = sorted(cohort_of.keys())

    t0 = time.perf_counter()
    LOG.info(
        "Loading indexes=%d stocks=%d  start=%s end=%s",
        len(index_syms),
        len(stock_syms),
        start.date(),
        end.date() if end else "now",
    )
    frames: Dict[str, pd.DataFrame] = {}
    frames.update(load_index_frames(index_syms, start=start, end=end))
    if stock_syms:
        stock_frames = load_ohlcv_many(
            stock_syms,
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
        for s, df in stock_frames.items():
            frames[s.upper()] = df
    load_s = time.perf_counter() - t0
    LOG.info("Load done in %.1fs (%d frames)", load_s, len(frames))

    per_symbol: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    trades_by_cohort: Dict[str, List[Dict[str, Any]]] = {
        "index": [],
        "big": [],
        "random": [],
        "all": [],
    }

    t1 = time.perf_counter()
    for i, sym in enumerate(symbols, 1):
        raw = frames.get(sym)
        df = _frame_to_bt(raw if raw is not None else pd.DataFrame())
        res = backtest_deep_pullback(df, symbol=sym, friction_pct=friction_pct)
        cohort = cohort_of.get(sym, "other")
        row = {
            "symbol": sym,
            "cohort": cohort,
            "n": int(res.get("n") or 0),
            "E": res.get("avg_gain_pct"),
            "PF": res.get("profit_factor"),
            "WR": res.get("win_rate_pct"),
            "median": res.get("median_gain_pct"),
            "mdd": res.get("mdd_trade_eq_pct"),
            "exposure": res.get("exposure_pct"),
            "avg_hold": res.get("avg_hold_days"),
            "bars": res.get("bars"),
            "first_bar": res.get("first_bar"),
            "last_bar": res.get("last_bar"),
            "first_trade": res.get("first_trade"),
            "last_trade": res.get("last_trade"),
            "reason": res.get("reason"),
        }
        per_symbol.append(row)
        trades = res.get("trades") or []
        for tr in trades:
            tr = dict(tr)
            tr["cohort"] = cohort
            all_trades.append(tr)
            trades_by_cohort[cohort].append(tr)
            trades_by_cohort["all"].append(tr)
        if i % 50 == 0 or i == len(symbols):
            LOG.info("Backtested %d/%d", i, len(symbols))

    bt_s = time.perf_counter() - t1
    cohort_books: Dict[str, Any] = {}
    for name, trs in trades_by_cohort.items():
        book = book_from_trades(trs)
        buckets = year_bucket_stats(trs)
        geo = geom_mean_year_pf(buckets, min_n=30)
        # skip n<30 buckets already; also report with min_n=10 for rare index sleeve
        geo10 = geom_mean_year_pf(buckets, min_n=10)
        n_syms = len({t["symbol"] for t in trs}) if trs else 0
        n_syms_tested = sum(1 for r in per_symbol if r["cohort"] == name) if name != "all" else len(per_symbol)
        n_syms_with_trades = sum(
            1 for r in per_symbol if (name == "all" or r["cohort"] == name) and int(r["n"] or 0) > 0
        )
        cohort_books[name] = {
            **book,
            "geo_pf_n30": None if geo is None else round(geo, 3),
            "geo_pf_n10": None if geo10 is None else round(geo10, 3),
            "year_buckets": buckets,
            "n_symbols_tested": n_syms_tested,
            "n_symbols_with_trades": n_syms_with_trades,
            "n_unique_trade_symbols": n_syms,
        }

    # Per-index detail (always useful)
    index_detail = [r for r in per_symbol if r["cohort"] == "index"]
    index_detail.sort(key=lambda r: (-(r["n"] or 0), r["symbol"]))

    return {
        "friction_pct": friction_pct,
        "start": start.isoformat(),
        "end": None if end is None else end.isoformat(),
        "n_symbols": len(symbols),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "wall_clock_s": {
            "load": round(load_s, 1),
            "backtest": round(bt_s, 1),
            "total": round(load_s + bt_s, 1),
        },
        "cohort_books": cohort_books,
        "index_detail": index_detail,
        "per_symbol": per_symbol,
        "trades": all_trades,
    }


def _print_summary(payload: Dict[str, Any]) -> None:
    LOG.info("=== Deep Pullback universe (fric=%.2f%%) ===", float(payload["friction_pct"]))
    LOG.info(
        "Wall clock: load=%.1fs bt=%.1fs total=%.1fs  symbols=%d",
        payload["wall_clock_s"]["load"],
        payload["wall_clock_s"]["backtest"],
        payload["wall_clock_s"]["total"],
        payload["n_symbols"],
    )
    for name in ("index", "big", "random", "all"):
        b = payload["cohort_books"][name]
        LOG.info(
            "%-7s n=%s E=%s PF=%s geo30=%s geo10=%s WR=%s med=%s syms_w_trades=%s/%s",
            name,
            b.get("n"),
            b.get("expectancy_pct"),
            b.get("profit_factor"),
            b.get("geo_pf_n30"),
            b.get("geo_pf_n10"),
            b.get("win_rate_pct"),
            b.get("median_pct"),
            b.get("n_symbols_with_trades"),
            b.get("n_symbols_tested"),
        )
        for yb in b.get("year_buckets") or []:
            LOG.info(
                "  %-9s n=%s E=%s PF=%s WR=%s",
                yb["bucket"],
                yb["n_trades"],
                yb["expectancy_pct"],
                yb["profit_factor"],
                yb["win_rate_pct"],
            )
    LOG.info("--- Index detail ---")
    for r in payload["index_detail"]:
        LOG.info(
            "  %s n=%s E=%s PF=%s WR=%s med=%s bars=%s %s->%s",
            r["symbol"],
            r["n"],
            r["E"],
            r["PF"],
            r["WR"],
            r["median"],
            r["bars"],
            r.get("first_bar"),
            r.get("last_bar"),
        )


def write_outputs(payload: Dict[str, Any], outdir: Path, *, seed: int) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    stem = "deep_pullback_universe_%s" % day

    per = pd.DataFrame(payload["per_symbol"])
    per_path = outdir / ("%s_per_symbol.csv" % stem)
    per.to_csv(per_path, index=False)

    trades = pd.DataFrame(payload["trades"])
    trades_path = outdir / ("%s_trades.csv" % stem)
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        trades_path.write_text("symbol,entry_date,exit_date,entry_px,exit_px,ret_pct,hold_days,cohort\n")

    slim = {
        "friction_pct": payload["friction_pct"],
        "start": payload["start"],
        "end": payload["end"],
        "n_symbols": payload["n_symbols"],
        "cohort_sizes": payload["cohort_sizes"],
        "wall_clock_s": payload["wall_clock_s"],
        "seed": seed,
        "cohort_books": payload["cohort_books"],
        "index_detail": payload["index_detail"],
    }
    json_path = outdir / ("%s_summary.json" % stem)
    json_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    # Status log rollup
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = STATUS_DIR / ("%s_deep_pullback_universe.md" % day)
    lines = [
        "# Deep Pullback universe scan (%s)" % day,
        "",
        "QS rules on TimescaleDB daily (ALPACA + IB prefix). Equal-weight trade pool (not portfolio occupancy).",
        "",
        "- Friction: **%.2f%%** round-turn" % float(payload["friction_pct"]),
        "- Symbols: **%d** (indexes=%s big=%s random=%s)"
        % (
            payload["n_symbols"],
            payload["cohort_sizes"].get("index"),
            payload["cohort_sizes"].get("big"),
            payload["cohort_sizes"].get("random"),
        ),
        "- Start: `%s`  seed=%s" % (payload["start"], seed),
        "- Wall clock: load **%.1fs** / backtest **%.1fs** / total **%.1fs**"
        % (
            payload["wall_clock_s"]["load"],
            payload["wall_clock_s"]["backtest"],
            payload["wall_clock_s"]["total"],
        ),
        "",
        "## Cohort books",
        "",
        "| Cohort | n | E%% | PF | geoPF(n>=30) | geoPF(n>=10) | WR%% | median%% | syms w/ trades |",
        "|--------|--:|---:|---:|-------------:|-------------:|----:|---------:|---------------:|",
    ]
    for name in ("index", "big", "random", "all"):
        b = payload["cohort_books"][name]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |"
            % (
                name,
                b.get("n"),
                b.get("expectancy_pct"),
                b.get("profit_factor"),
                b.get("geo_pf_n30"),
                b.get("geo_pf_n10"),
                b.get("win_rate_pct"),
                b.get("median_pct"),
                b.get("n_symbols_with_trades"),
                b.get("n_symbols_tested"),
            )
        )
    lines.extend(["", "## Indexes", ""])
    lines.append("| Sym | n | E%% | PF | WR%% | median | bars | range |")
    lines.append("|-----|--:|---:|---:|----:|-------:|-----:|-------|")
    for r in payload["index_detail"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s -> %s |"
            % (
                r["symbol"],
                r["n"],
                r["E"],
                r["PF"],
                r["WR"],
                r["median"],
                r["bars"],
                r.get("first_bar"),
                r.get("last_bar"),
            )
        )
    lines.extend(["", "## Year buckets (all)", ""])
    for yb in payload["cohort_books"]["all"].get("year_buckets") or []:
        lines.append(
            "- **%s**: n=%s E=%s PF=%s WR=%s"
            % (
                yb["bucket"],
                yb["n_trades"],
                yb["expectancy_pct"],
                yb["profit_factor"],
                yb["win_rate_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `%s`" % per_path.as_posix(),
            "- `%s`" % trades_path.as_posix(),
            "- `%s`" % json_path.as_posix(),
            "",
            "Treated as a selective sleeve check, not Sharpe>1 edge-hunt promote.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", md_path)
    LOG.info("Wrote %s", per_path)
    LOG.info("Wrote %s", json_path)
    return md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deep Pullback multi-symbol universe scan")
    ap.add_argument("--n-extra", type=int, default=300, help="Extra symbols beyond indexes")
    ap.add_argument("--big-frac", type=float, default=0.5, help="Fraction of extra that are top-ADV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--friction", type=float, default=0.0, help="Round-turn friction %%")
    ap.add_argument("--start", type=str, default="2006-01-01")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output dir (default reports/research/deep_pullback/YYYY-MM-DD)",
    )
    ap.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional JSON path to reuse cohorts {index,big,random}",
    )
    args = ap.parse_args(argv)

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None
    outdir = dated_outdir(args.outdir or DEFAULT_OUT)

    if args.symbols_file and args.symbols_file.exists():
        cohorts = json.loads(args.symbols_file.read_text(encoding="utf-8"))
        LOG.info("Loaded cohorts from %s", args.symbols_file)
    else:
        cohorts = build_symbol_cohorts(
            n_extra=args.n_extra,
            big_frac=args.big_frac,
            seed=args.seed,
        )
        sym_path = outdir / "deep_pullback_universe_symbols.json"
        sym_path.write_text(json.dumps(cohorts, indent=2), encoding="utf-8")
        LOG.info("Wrote symbol cohorts -> %s", sym_path)

    payload = run_universe(
        cohorts,
        start=start,
        end=end,
        friction_pct=float(args.friction),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
    )
    _print_summary(payload)
    write_outputs(payload, outdir, seed=int(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
