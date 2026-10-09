"""One-cycle wheel on current_best last-15m fills (Theta + Polygon marks).

Replaces buying stock with CSP -> assign -> CC. Prefer Theta greeks/eod;
fall back to Polygon day-bar mid (+ half-spread for ask/bid marks).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=5
  python scripts\\research\\claimcheck_last_15m_wheel.py --smoke
  python scripts\\research\\claimcheck_last_15m_wheel.py --n 50 --min-buy-date 2024-01-02
  python scripts\\research\\claimcheck_last_15m_wheel.py --n 50 --mark-mode ask_bid
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

load_dotenv(ROOT / ".env")
# Client default is 50; only raise if unset (do not force free-tier 5).
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "30")

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
)
from utils.research.last_15m_wheel import (  # noqa: E402
    DEFAULT_DTE,
    HybridWheelSource,
    PolygonDaySource,
    ThetaEquitySource,
    result_to_record,
    simulate_one_cycle_wheel,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("claimcheck_last_15m_wheel")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
)
SMOKE_SYMBOLS = ("AMD", "NVDA", "META", "TSLA", "AMZN", "AAPL", "MSFT", "AVGO")


def _row_key(stock: str, buy_date: Any, mark_mode: str) -> str:
    d = pd.Timestamp(buy_date).strftime("%Y-%m-%d")
    return "%s|%s|%s" % (str(stock).upper(), d, str(mark_mode).lower())


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get("key")
            if k:
                out[k] = obj
    return out


def _append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _print_block(label: str, df: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    g = (
        pd.to_numeric(df[gain_col], errors="coerce")
        if (not df.empty and gain_col in df.columns)
        else pd.Series(dtype=float)
    )
    years = (
        summarize_by_year(
            df.dropna(subset=[gain_col]), gain_col=gain_col, buckets=YEAR_BUCKETS
        )
        if not df.empty and gain_col in df.columns
        else pd.DataFrame()
    )
    s = book_stats(g)
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
    print("---- %s ----" % label)
    if not years.empty:
        print(years.to_string(index=False))
    return {
        "label": label,
        "n": s.get("n"),
        "expectancy_pct": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_year_pf": geo,
        "win_rate_pct": s.get("win_rate_pct"),
        "median_pct": s.get("median_pct"),
    }


def _select_trades(
    df: pd.DataFrame,
    *,
    n: Optional[int],
    min_adv: float,
    min_buy_date: Optional[str],
    smoke: bool,
    symbols: Optional[List[str]],
    seed: int,
    ensure_years: Optional[List[int]] = None,
) -> pd.DataFrame:
    out = df.copy()
    out["stock"] = out["stock"].astype(str).str.upper()
    out["buy_date"] = pd.to_datetime(out["buy_date"], errors="coerce")
    out["adv_20"] = pd.to_numeric(out.get("adv_20"), errors="coerce")
    if min_adv > 0:
        out = out[out["adv_20"].fillna(0) >= float(min_adv)]
    if min_buy_date:
        out = out[out["buy_date"] >= pd.Timestamp(min_buy_date)]
    if symbols:
        want = {s.upper() for s in symbols}
        out = out[out["stock"].isin(want)]
    elif smoke:
        hit = out[out["stock"].isin(SMOKE_SYMBOLS)]
        # Mega-caps are rare on last-15m; fall back to top-ADV names.
        out = hit if len(hit) >= 3 else out.nlargest(min(40, len(out)), "adv_20")

    ensure = [int(y) for y in (ensure_years or [])]
    if ensure and n is not None and int(n) > 0:
        years = out["buy_date"].dt.year
        forced = out[years.isin(ensure)]
        rest = out[~years.isin(ensure)]
        # Take all ensure-year rows first (cap at n), then fill from the rest.
        if len(forced) >= int(n):
            out = forced.sample(n=int(n), random_state=int(seed))
        else:
            need = int(n) - len(forced)
            if len(rest) > need:
                rest = rest.sample(n=need, random_state=int(seed))
            out = pd.concat([forced, rest], ignore_index=True)
    elif n is not None and len(out) > int(n):
        out = out.sample(n=int(n), random_state=int(seed))

    return out.sort_values(["buy_date", "stock"]).reset_index(drop=True)


def _year_slices(
    df: pd.DataFrame,
    gain_col: str = "gain_pct",
    *,
    prefix: str = "slice",
) -> Dict[str, Any]:
    """Compact year / regime slices for the expanded pilot."""
    if df is None or df.empty or gain_col not in df.columns:
        return {}
    d = df.copy()
    d["_bd"] = pd.to_datetime(d.get("buy_date"), errors="coerce")
    out: Dict[str, Any] = {}
    slices = [
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2022-23", "2022-01-01", "2023-12-31"),
        ("2024-26", "2024-01-01", "2026-12-31"),
    ]
    for label, start, end in slices:
        sub = d[(d["_bd"] >= start) & (d["_bd"] <= end)]
        out[label] = _print_block("%s_%s" % (prefix, label), sub, gain_col=gain_col)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=None, help="Sample size (after filters)")
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument(
        "--min-buy-date",
        type=str,
        default="2022-01-01",
        help="Inclusive start (default 2022-01-01 to include bad years)",
    )
    ap.add_argument(
        "--ensure-years",
        type=str,
        default="2022,2023",
        help="Comma years to keep in full before sampling the rest (empty to disable)",
    )
    ap.add_argument("--mark-mode", choices=("mid", "ask_bid"), default="mid")
    ap.add_argument("--dte", type=int, default=DEFAULT_DTE)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument(
        "--strike-mode",
        choices=("delta", "atm"),
        default="delta",
        help="delta needs Theta greeks; atm uses nearest strike (Polygon-friendly)",
    )
    ap.add_argument(
        "--source",
        choices=("hybrid", "theta", "polygon"),
        default="theta",
        help="Default theta (fast when Terminal is up). hybrid adds Polygon fallback (slow on free RPM).",
    )
    ap.add_argument("--entry-slack-days", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--symbols", type=str, default="", help="Comma tickers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.trades.exists():
        LOG.error("trades CSV missing: %s", args.trades)
        return 2

    trades_all = pd.read_csv(args.trades)
    symbols_arg = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    min_buy = (args.min_buy_date or "").strip() or None
    ensure_years = [
        int(x.strip())
        for x in str(args.ensure_years or "").split(",")
        if x.strip().isdigit()
    ]
    trades = _select_trades(
        trades_all,
        n=5 if args.smoke and args.n is None else args.n,
        min_adv=args.min_adv,
        min_buy_date=min_buy,
        smoke=args.smoke,
        symbols=symbols_arg,
        seed=args.seed,
        ensure_years=None if args.smoke else ensure_years,
    )
    if trades.empty:
        LOG.error("no trades after filters")
        return 2
    year_counts = trades["buy_date"].dt.year.value_counts().sort_index().to_dict()
    LOG.info(
        "selected n=%s symbols=%s buy %s..%s years=%s ensure=%s",
        len(trades),
        trades["stock"].nunique(),
        trades["buy_date"].min().date(),
        trades["buy_date"].max().date(),
        year_counts,
        ensure_years,
    )

    outdir = args.outdir or dated_outdir(ROOT / "reports" / "ascending_channels")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (outdir / "last_15m_wheel_cache.jsonl")
    cache = _load_cache(cache_path)
    LOG.info("cache %s hits=%s", cache_path, len(cache))

    theta_cache = outdir / "theta_wheel_cache"
    theta_cache.mkdir(parents=True, exist_ok=True)

    theta_ok = False
    try:
        theta_ok = bool(get_thetadata_client().is_ready())
    except Exception as exc:  # noqa: BLE001
        LOG.warning("theta not ready: %s", exc)
    if args.source in ("hybrid", "theta") and not theta_ok:
        LOG.warning(
            "Theta Terminal not ready -- hybrid will lean on Polygon; "
            "start scripts/pipeline/thetadata_terminal.py for NBBO/greeks"
        )

    if args.source == "theta":
        source = ThetaEquitySource(cache_dir=theta_cache)
    elif args.source == "polygon":
        source = PolygonDaySource()
    else:
        source = HybridWheelSource(
            theta=ThetaEquitySource(cache_dir=theta_cache),
            polygon=PolygonDaySource(),
        )

    # Daily panels for spot at expiries
    syms = sorted(trades["stock"].astype(str).str.upper().unique())
    start = (trades["buy_date"].min() - pd.Timedelta(days=5)).date()
    end = (trades["buy_date"].max() + pd.Timedelta(days=args.dte * 2 + 40)).date()
    LOG.info("loading 1d panels n=%s %s..%s", len(syms), start, end)
    t0 = time.time()
    panels = load_ohlcv_many(
        syms,
        timeframe="1d",
        start=start,
        end=end,
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    LOG.info("panels loaded in %.1fs (%s ok)", time.time() - t0, len(panels))

    rows: List[Dict[str, Any]] = []
    t1 = time.time()
    for i, (_, row) in enumerate(trades.iterrows(), start=1):
        key = _row_key(row["stock"], row["buy_date"], args.mark_mode)
        if key in cache and cache[key].get("wheel_reason") is not None:
            rec = dict(cache[key])
            # Re-attach stock gain from live row
            rec["gain_pct_stock"] = float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else None
            rows.append(rec)
            continue
        sym = str(row["stock"]).upper()
        got = simulate_one_cycle_wheel(
            row,
            panels.get(sym),
            source,
            mark_mode=args.mark_mode,
            target_dte=int(args.dte),
            target_delta=float(args.delta),
            strike_mode=str(args.strike_mode),
            entry_slack_days=int(args.entry_slack_days),
        )
        rec = result_to_record(row, got)
        rec["key"] = key
        _append_cache(cache_path, rec)
        cache[key] = rec
        rows.append(rec)
        if i % 5 == 0 or i == len(trades):
            LOG.info(
                "progress %s/%s ok=%s last=%s %s",
                i,
                len(trades),
                sum(1 for r in rows if r.get("wheel_ok")),
                sym,
                got.reason if not got.filled else got.exit_reason,
            )

    wall = time.time() - t1
    out = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / ("last_15m_wheel_%s_%s.csv" % (args.mark_mode, stamp))
    out.to_csv(csv_path, index=False)

    filled = out[out["wheel_ok"] == True]  # noqa: E712
    missed = out[out["wheel_ok"] != True]  # noqa: E712
    summary: Dict[str, Any] = {
        "wall_s": round(wall, 1),
        "attempted": int(len(out)),
        "filled": int(len(filled)),
        "missed": int(len(missed)),
        "mark_mode": args.mark_mode,
        "dte": args.dte,
        "delta": args.delta,
        "strike_mode": args.strike_mode,
        "source": args.source,
        "theta_ready": theta_ok,
        "min_buy_date": min_buy,
        "min_adv": args.min_adv,
        "csv": str(csv_path),
    }
    if not missed.empty:
        summary["miss_reasons"] = (
            missed["wheel_reason"].value_counts().to_dict()
        )
    if not filled.empty:
        summary["exit_reasons"] = (
            filled["wheel_exit_reason"].value_counts().to_dict()
        )
        summary["wheel"] = _print_block("wheel_%s" % args.mark_mode, filled)
        summary["wheel_year_slices"] = _year_slices(
            filled, gain_col="gain_pct", prefix="wheel"
        )
        stock_cmp = filled.copy()
        stock_cmp["gain_pct"] = pd.to_numeric(
            stock_cmp.get("gain_pct_stock"), errors="coerce"
        )
        summary["stock_same_rows"] = _print_block(
            "stock_same_rows", stock_cmp
        )
        summary["stock_year_slices"] = _year_slices(
            stock_cmp, gain_col="gain_pct", prefix="stock"
        )
        # Sources used
        if "put_source" in filled.columns:
            summary["put_sources"] = filled["put_source"].value_counts().to_dict()
        if "buy_date" in filled.columns:
            summary["filled_year_counts"] = (
                pd.to_datetime(filled["buy_date"], errors="coerce")
                .dt.year.value_counts()
                .sort_index()
                .to_dict()
            )

    sum_path = outdir / ("last_15m_wheel_%s_%s_summary.json" % (args.mark_mode, stamp))
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("wrote %s", csv_path)
    LOG.info("wrote %s", sum_path)
    LOG.info("summary: %s", json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
