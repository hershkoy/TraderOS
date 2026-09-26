#!/usr/bin/env python3
"""Cup & Handle swing claim-check (Cashflow Academy video mExyCQrddTU).

Causal daily detector + buy-stop fill at rim. Claim exit: 1.5x ATR trail with
50/50 scale-out at 1/2 and 2/3 cup depth. Keeper exit: ATR k=2 clamp + 10pct trail.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\backtest_cup_handle.py --jpm-sanity
  python scripts\\research\\backtest_cup_handle.py --n-symbols 300
  python scripts\\research\\backtest_cup_handle.py --n-symbols 100 --trifecta
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import _pick_15m_by_daily_adv  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.cup_handle import (  # noqa: E402
    EXIT_CLAIM,
    EXIT_KEEPER,
    CupParams,
    SECTOR_TO_ETF,
    arm_sessions,
    max_concurrent,
    occupancy_fcfs,
    params_dict,
    sector_etf_for,
    simulate_cup_handle_trade,
    summarize_book,
    trades_for_symbol,
    trifecta_ok,
)
from utils.research.metrics import (  # noqa: E402
    format_elapsed,
    passes_phase5_gates,
    perf_stats,
)
from utils.research.panel import to_naive_index  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.roc_rebound import (  # noqa: E402
    YEAR_BUCKETS,
    closes_panel_from_frames,
    drop_top_n_book,
    geom_mean_year_pf,
    single_slot_equity,
    spy_buy_hold_equity,
    year_bucket_stats,
)
from utils.research.roc_rebound_capital import n_slot_equity  # noqa: E402
from utils.research.weekly_deepvue import load_symbol_sectors  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("cup_handle")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FRICTION_PCT = 0.25
DEFAULT_MIN_PRICE = 20.0
DEFAULT_MIN_ADV_SHARES = 2_000_000.0


def _naive_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if out.columns[0] != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
    out = to_naive_index(out.set_index("ts") if "ts" in out.columns else out)
    out = out.reset_index().rename(columns={"index": "ts"})
    if out.columns[0] != "ts":
        out = out.rename(columns={out.columns[0]: "ts"})
    return out


def _load_sectors() -> Dict[str, str]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        LOG.warning("No DB for ticker_universe sectors")
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


def _passes_liquid_screen(df: pd.DataFrame, *, min_price: float, min_adv_shares: float) -> bool:
    if df is None or df.empty or len(df) < 30:
        return False
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df.get("volume"), errors="coerce")
    last = float(close.iloc[-1]) if len(close) else 0.0
    if last < float(min_price):
        return False
    adv = float((vol.tail(30)).mean()) if vol is not None and len(vol) else 0.0
    if adv < float(min_adv_shares):
        return False
    return True


def _log_book(label: str, trades: List[dict]) -> Dict[str, Any]:
    book = summarize_book(trades)
    years = year_bucket_stats(trades, YEAR_BUCKETS)
    geo = geom_mean_year_pf(years)
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s max_concurrent=%s",
        label,
        book.get("n"),
        book.get("expectancy_pct"),
        book.get("profit_factor"),
        None if geo is None else round(geo, 3),
        book.get("win_rate_pct"),
        book.get("median_pct"),
        book.get("max_concurrent"),
    )
    for row in years:
        LOG.info(
            "  %s n=%s E=%s PF=%s WR=%s",
            row.get("bucket"),
            row.get("n"),
            row.get("expectancy_pct"),
            row.get("profit_factor"),
            row.get("win_rate_pct"),
        )
    drop3 = drop_top_n_book(trades, 3)
    LOG.info(
        "  drop-top-3 n=%s E=%s PF=%s",
        drop3.get("n"),
        drop3.get("expectancy_pct"),
        drop3.get("profit_factor"),
    )
    return {
        "book": book,
        "years": years,
        "geo_pf": geo,
        "drop_top_3": drop3,
    }


def jpm_sanity(df: pd.DataFrame, params: CupParams) -> List[dict]:
    """Print recent JPM arms / fills near the video rim (~340)."""
    d = _naive_daily(df)
    if d.empty:
        LOG.warning("JPM empty")
        return []
    LOG.info(
        "JPM bars %d  %s -> %s  last_close=%.2f",
        len(d),
        pd.Timestamp(d["ts"].iloc[0]).date(),
        pd.Timestamp(d["ts"].iloc[-1]).date(),
        float(d["close"].iloc[-1]),
    )
    arms = arm_sessions(d, params)
    LOG.info("JPM armed setups: %d", len(arms))
    near = [a for a in arms if 300.0 <= float(a.get("rim", 0)) <= 380.0]
    show = near if near else arms[-5:]
    for a in show[-10:]:
        LOG.info(
            "  arm=%s rim=%.2f depth=%.1f%% cup=%s handle=%s trough=%.2f",
            a.get("arm_date"),
            float(a.get("rim", 0)),
            float(a.get("cup_depth_pct", 0)),
            a.get("cup_bars"),
            a.get("handle_bars"),
            float(a.get("trough", 0)),
        )
        tr = simulate_cup_handle_trade(d, a, params, symbol="JPM", exit_mode=EXIT_CLAIM)
        if tr:
            LOG.info(
                "    FILL %s @ %.2f -> %s @ %.2f ret=%.2f%% (%s)",
                tr["entry_date"],
                tr["entry_px"],
                tr["exit_date"],
                tr["exit_px"],
                tr["ret_pct"],
                tr["exit_reason"],
            )
        else:
            LOG.info("    no fill/exit")
    trades = trades_for_symbol(d, params, symbol="JPM", exit_mode=EXIT_CLAIM)
    LOG.info("JPM completed claim trades: %d", len(trades))
    return trades


def _collect_trades(
    frames: Dict[str, pd.DataFrame],
    params: CupParams,
    *,
    exit_mode: str,
    friction_pct: float,
    min_price: float,
    min_adv_shares: float,
) -> List[dict]:
    trades: List[dict] = []
    n_sym = 0
    for i, (sym, raw) in enumerate(frames.items(), 1):
        d = _naive_daily(raw)
        if not _passes_liquid_screen(d, min_price=min_price, min_adv_shares=min_adv_shares):
            continue
        n_sym += 1
        got = trades_for_symbol(
            d,
            params,
            symbol=str(sym).upper(),
            friction_pct=friction_pct,
            exit_mode=exit_mode,
        )
        trades.extend(got)
        if i % 50 == 0:
            LOG.info(
                "  scanned %d/%d symbols (%d liquid) trades=%d",
                i,
                len(frames),
                n_sym,
                len(trades),
            )
    LOG.info("Liquid symbols used: %d  raw trades: %d (%s)", n_sym, len(trades), exit_mode)
    return trades


def _apply_trifecta(
    trades: List[dict],
    *,
    sectors: Dict[str, str],
    spy_close: pd.Series,
    etf_closes: Dict[str, pd.Series],
) -> List[dict]:
    kept: List[dict] = []
    for tr in trades:
        sym = str(tr.get("symbol") or "").upper()
        sec = sectors.get(sym, "")
        etf = sector_etf_for(sec)
        etf_s = etf_closes.get(etf) if etf else None
        arm = str(tr.get("arm_date") or tr.get("entry_date"))
        if trifecta_ok(arm_date=arm, spy_close=spy_close, sector_etf_close=etf_s):
            out = dict(tr)
            out["sector"] = sec
            out["sector_etf"] = etf
            kept.append(out)
    return kept


def _occupancy_gate(
    trades: List[dict],
    close_panel: pd.DataFrame,
    spy_close: pd.Series,
    *,
    max_open: int,
    label: str,
) -> Dict[str, Any]:
    kept, meta = occupancy_fcfs(trades, max_open=max_open, unique_symbol=True)
    if int(max_open) <= 1:
        eq, inv = single_slot_equity(kept, close_panel)
    else:
        eq, inv = n_slot_equity(kept, close_panel, max_open=int(max_open))
    spy_eq, spy_inv = spy_buy_hold_equity(
        spy_close,
        eval_start=str(eq.index[0].date()) if len(eq) else None,
        eval_end=str(eq.index[-1].date()) if len(eq) else None,
    )
    spy_st = perf_stats("SPY", spy_eq, invested=spy_inv)
    st = perf_stats(
        label,
        eq,
        invested=inv,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        trade_rets=[float(t["ret_pct"]) for t in kept],
    )
    ok, reason = passes_phase5_gates(st, spy_st)
    LOG.info(
        "%s occ max_open=%s kept=%s skip=%s Sharpe=%.3f MDD=%.2f%% vs SPY Sharpe=%.3f MDD=%.2f%% -> %s",
        label,
        max_open,
        meta.get("n_kept"),
        meta.get("n_skipped"),
        st.sharpe,
        st.max_drawdown * 100.0,
        spy_st.sharpe,
        spy_st.max_drawdown * 100.0,
        reason,
    )
    return {
        "meta": meta,
        "book": summarize_book(kept),
        "sharpe": st.sharpe,
        "mdd": st.max_drawdown,
        "cagr": st.cagr,
        "spy_sharpe": spy_st.sharpe,
        "spy_mdd": spy_st.max_drawdown,
        "gate": reason,
        "pass": ok,
        "trades": kept,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cup & Handle claim-check")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--symbols", default="", help="Comma list (overrides ADV pick)")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--friction-pct", type=float, default=0.0)
    ap.add_argument("--friction-stress", type=float, default=FRICTION_PCT)
    ap.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    ap.add_argument("--min-adv-shares", type=float, default=DEFAULT_MIN_ADV_SHARES)
    ap.add_argument("--jpm-sanity", action="store_true")
    ap.add_argument("--trifecta", action="store_true", help="Also report trifecta-filtered book")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--skip-keeper", action="store_true")
    args = ap.parse_args()

    params = CupParams()
    outdir = args.outdir or dated_outdir(ROOT / "reports" / "research" / "cup_handle")
    outdir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # --- JPM sanity (always load JPM; exit early if only --jpm-sanity) ---
    t0 = time.perf_counter()
    jpm_frames = load_ohlcv_many(
        ["JPM"],
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=1,
        chunk_size=1,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    timings["jpm_load"] = time.perf_counter() - t0
    jpm_df = jpm_frames.get("JPM")
    jpm_trades = jpm_sanity(_naive_daily(jpm_df) if jpm_df is not None else pd.DataFrame(), params)
    if args.jpm_sanity and (
        (args.symbols.strip().upper() in ("", "JPM"))
        and int(args.n_symbols) <= 1
    ):
        summary = {"jpm_trades": jpm_trades, "timings": timings}
        (outdir / "jpm_sanity.json").write_text(
            json.dumps(summary, default=str, indent=2), encoding="utf-8"
        )
        LOG.info(
            "JPM-only done in %s -> %s",
            format_elapsed(time.perf_counter() - t_all),
            outdir,
        )
        return 0

    # --- Universe ---
    t0 = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _pick_15m_by_daily_adv(int(args.n_symbols))
    for extra in ("JPM", "SPY"):
        if extra not in symbols:
            symbols.append(extra)
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("Universe %d symbols (%.1fs)", len(symbols), timings["list_symbols"])

    t0 = time.perf_counter()
    frames = load_ohlcv_many(
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
    LOG.info("Daily panels %d/%d (%.1fs)", len(frames), len(symbols), timings["load_daily"])

    spy_df = _naive_daily(frames.get("SPY"))
    if spy_df.empty:
        LOG.error("SPY daily missing")
        return 1
    spy_close = spy_df.set_index("ts")["close"].astype(float)
    spy_close.index = pd.DatetimeIndex(spy_close.index).tz_localize(None).normalize()

    # Claim book
    t0 = time.perf_counter()
    claim_trades = _collect_trades(
        frames,
        params,
        exit_mode=EXIT_CLAIM,
        friction_pct=float(args.friction_pct),
        min_price=float(args.min_price),
        min_adv_shares=float(args.min_adv_shares),
    )
    timings["claim_scan"] = time.perf_counter() - t0

    # Unique-symbol uncapped (skip while open already in trades_for_symbol;
    # cross-symbol occupancy applied below).
    claim_unique, u_meta = occupancy_fcfs(claim_trades, max_open=0, unique_symbol=True)
    LOG.info("Claim unique-symbol uncapped kept=%s skip=%s", u_meta["n_kept"], u_meta["n_skipped"])

    results: Dict[str, Any] = {
        "params": params_dict(params),
        "start": args.start,
        "end": args.end,
        "n_symbols_requested": args.n_symbols,
        "n_frames": len(frames),
        "jpm_n": len(jpm_trades),
    }
    results["claim"] = _log_book("CLAIM fric0", claim_unique)

    # Friction stress (post-hoc on completed fills; avoids a full rescan)
    if float(args.friction_stress) > 0 and claim_unique:
        claim_fric = []
        fr = float(args.friction_stress)
        for tr in claim_unique:
            out = dict(tr)
            out["ret_pct"] = round(float(tr["ret_pct"]) - fr, 4)
            claim_fric.append(out)
        results["claim_fric025"] = _log_book(
            "CLAIM fric%.2f" % args.friction_stress, claim_fric
        )

    # Keeper robustness
    if not args.skip_keeper:
        t0 = time.perf_counter()
        keeper_trades = _collect_trades(
            frames,
            params,
            exit_mode=EXIT_KEEPER,
            friction_pct=float(args.friction_pct),
            min_price=float(args.min_price),
            min_adv_shares=float(args.min_adv_shares),
        )
        keeper_unique, _ = occupancy_fcfs(keeper_trades, max_open=0, unique_symbol=True)
        timings["keeper_scan"] = time.perf_counter() - t0
        results["keeper"] = _log_book("KEEPER fric0", keeper_unique)

    # Occupancy vs SPY (before optional trifecta so a sector failure still gates)
    close_panel = closes_panel_from_frames(
        {k: _naive_daily(v) for k, v in frames.items() if v is not None}
    )
    results["occ_1"] = _occupancy_gate(
        claim_unique, close_panel, spy_close, max_open=1, label="CLAIM_occ1"
    )
    results["occ_2"] = _occupancy_gate(
        claim_unique, close_panel, spy_close, max_open=2, label="CLAIM_occ2"
    )

    # Trifecta overlay on claim
    if args.trifecta:
        try:
            sectors = _load_sectors()
            etf_syms = sorted({e for e in SECTOR_TO_ETF.values()})
            t0 = time.perf_counter()
            etf_frames = load_ohlcv_many(
                etf_syms,
                timeframe="1d",
                provider="ALPACA",
                start=start,
                end=end,
                use_cache=True,
                workers=4,
                chunk_size=20,
                fallback_provider="IB",
                merge_mode="prefix",
            )
            timings["etf_load"] = time.perf_counter() - t0
            etf_closes: Dict[str, pd.Series] = {}
            for es, edf in etf_frames.items():
                nd = _naive_daily(edf)
                if nd.empty:
                    continue
                s = nd.set_index("ts")["close"].astype(float)
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                etf_closes[str(es).upper()] = s
            tri = _apply_trifecta(
                claim_unique, sectors=sectors, spy_close=spy_close, etf_closes=etf_closes
            )
            results["trifecta"] = _log_book("TRIFECTA claim", tri)
        except Exception as exc:
            LOG.warning("Trifecta overlay failed: %s", exc)
            results["trifecta_error"] = str(exc)

    timings["total"] = time.perf_counter() - t_all
    results["timings"] = {k: round(v, 2) for k, v in timings.items()}
    results["max_concurrent_claim"] = max_concurrent(claim_unique)

    trades_path = outdir / "claim_trades.csv"
    if claim_unique:
        pd.DataFrame(claim_unique).to_csv(trades_path, index=False)
    summary_path = outdir / "summary.json"
    # Strip heavy trade lists from JSON
    slim = {k: v for k, v in results.items() if k not in ("occ_1", "occ_2")}
    for key in ("occ_1", "occ_2"):
        if key in results:
            slim[key] = {kk: vv for kk, vv in results[key].items() if kk != "trades"}
    summary_path.write_text(json.dumps(slim, default=str, indent=2), encoding="utf-8")
    LOG.info(
        "Done in %s  max_concurrent=%s  -> %s",
        format_elapsed(timings["total"]),
        results["max_concurrent_claim"],
        outdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
