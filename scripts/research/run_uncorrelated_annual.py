#!/usr/bin/env python3
"""
Yearly uncorrelated sleeve vs SPY.

Each year-end: among PIT top liquid names, greedily pick X lowest-avg-corr
symbols from the prior calendar year's daily returns; equal-weight hold the
next calendar year.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_uncorrelated_annual.py
  python scripts\\research\\run_uncorrelated_annual.py --x 10 --liquid-n 300 --start 2019-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.metrics import (  # noqa: E402
    annual_returns,
    format_annual_table,
    format_elapsed,
    format_table,
    passes_phase5_gates,
    perf_stats,
    rebase_equity,
)
from utils.research.panel import (  # noqa: E402
    list_daily_symbols,
    load_spy_close,
    load_wide_panels,
)
from utils.research.uncorrelated_annual import (  # noqa: E402
    picks_to_frame,
    simulate_uncorrelated_annual,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("uncorrelated_annual")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _spy_buy_hold(spy: pd.Series, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    px = spy.astype(float).sort_index()
    px = px[(px.index >= s) & (px.index <= e)].dropna()
    rets = px.pct_change().fillna(0.0)
    eq = rebase_equity((1.0 + rets).cumprod())
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv


def main() -> int:
    ap = argparse.ArgumentParser(description="Yearly uncorrelated sleeve vs SPY")
    ap.add_argument("--start", default="2023-01-01", help="Hold eval start (needs prior year)")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--x", type=int, default=10, help="Number of uncorrelated names to hold")
    ap.add_argument("--x-sweep", default="", help="Comma list of x values (overrides --x)")
    ap.add_argument("--liquid-n", type=int, default=300, help="PIT ADV universe size")
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-obs", type=int, default=180)
    ap.add_argument("--cost-bps-rt", type=float, default=10.0)
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument(
        "--ib-prefix",
        action="store_true",
        help="Stitch IB daily bars before first Alpaca bar (slower; deeper history)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-symbols", type=int, default=0, help="Cap loaded symbols (0=all daily)")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "research" / "uncorrelated_annual",
    )
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    x_list = [args.x]
    if args.x_sweep.strip():
        x_list = [int(x) for x in args.x_sweep.split(",") if x.strip()]

    t_all = time.perf_counter()
    timings: dict = {}

    # Load panel with one year warmup before eval start
    load_start = (pd.Timestamp(args.start) - pd.DateOffset(years=1)).to_pydatetime()
    if args.ib_prefix:
        # Unclipped early start so IB prefix actually stitches (AGENTS.md)
        load_start = datetime(2006, 1, 1)
    load_end = pd.Timestamp(args.end).to_pydatetime()

    t0 = time.perf_counter()
    symbols = list_daily_symbols(provider=args.provider, timeframe="1d")
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: int(args.max_symbols)]
    LOG.info("Universe %d symbols (%s 1d)%s", len(symbols), args.provider, " +IB prefix" if args.ib_prefix else "")
    panels = load_wide_panels(
        symbols,
        start=load_start,
        end=load_end,
        provider=args.provider,
        timeframe="1d",
        fields=("close", "volume"),
        workers=args.workers,
        chunk_size=args.chunk_size,
        fallback_provider="IB" if args.ib_prefix else None,
        merge_mode="prefix" if args.ib_prefix else None,
    )
    close = panels["close"]
    volume = panels["volume"]
    spy = load_spy_close(load_start, load_end)
    timings["load"] = time.perf_counter() - t0
    LOG.info(
        "Panel close shape=%s %s -> %s (load %s)",
        close.shape,
        close.index[0].date() if len(close) else "?",
        close.index[-1].date() if len(close) else "?",
        format_elapsed(timings["load"]),
    )

    eq_spy, inv_spy = _spy_buy_hold(spy, args.start, args.end)
    spy_stats = perf_stats("SPY_buy_hold", eq_spy, inv_spy, notes="100% SPY")

    def _df_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_no picks_"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.4f}" if pd.notna(v) else "")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_stats = [spy_stats]
    annual = {"SPY_buy_hold": annual_returns(eq_spy)}
    sections: list[str] = []
    last_gate = ""
    last_status = args.outdir / f"2026-09-24_uncorrelated_annual.md"

    for x in x_list:
        t0 = time.perf_counter()
        result = simulate_uncorrelated_annual(
            close,
            volume,
            eval_start=args.start,
            eval_end=args.end,
            x=x,
            liquid_n=args.liquid_n,
            min_adv=args.min_adv,
            min_price=args.min_price,
            min_obs=args.min_obs,
            cost_bps_rt=args.cost_bps_rt,
        )
        sim_s = time.perf_counter() - t0
        LOG.info("Sim x=%d in %s | %s", x, format_elapsed(sim_s), result.notes)

        eq_g = rebase_equity(result.equity_gross.reindex(eq_spy.index).ffill().bfill())
        eq_n = rebase_equity(result.equity_net.reindex(eq_spy.index).ffill().bfill())
        inv = result.invested.reindex(eq_spy.index).fillna(0.0)

        trade_rets = []
        for p in result.picks:
            if not p.symbols:
                continue
            days = eq_n.index[eq_n.index.year == p.hold_year]
            if len(days) < 2:
                continue
            trade_rets.append(float(eq_n.loc[days[-1]] / eq_n.loc[days[0]] - 1.0))

        name = f"UncorrAnnual_x{x}_L{args.liquid_n}"
        stats_g = perf_stats(
            name + "_gross",
            eq_g,
            inv,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=result.notes + "; gross",
            trade_rets=trade_rets,
        )
        stats_n = perf_stats(
            name + "_net",
            eq_n,
            inv,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=result.notes + "; net",
            trade_rets=trade_rets,
        )
        ok, gate = passes_phase5_gates(stats_n, spy_stats)
        LOG.info("Gate %s: %s", name, gate)
        last_gate = gate
        all_stats.extend([stats_g, stats_n])
        annual[f"{name}_net"] = annual_returns(eq_n)

        picks_df = picks_to_frame(result.picks)
        picks_path = args.reports_dir / f"picks_x{x}_L{args.liquid_n}_{stamp}.csv"
        picks_df.to_csv(picks_path, index=False)
        eq_path = args.reports_dir / f"equity_x{x}_L{args.liquid_n}_{stamp}.csv"
        pd.DataFrame(
            {
                "SPY_buy_hold": eq_spy,
                f"{name}_gross": eq_g,
                f"{name}_net": eq_n,
                "invested": inv,
            }
        ).to_csv(eq_path)

        sections.append(
            "\n".join(
                [
                    f"## x={x}",
                    "",
                    f"- Gate (net): **{gate}**",
                    f"- Notes: {result.notes}",
                    f"- Picks CSV: `{picks_path.as_posix()}`",
                    f"- Equity CSV: `{eq_path.as_posix()}`",
                    "",
                    _df_md(picks_df),
                    "",
                ]
            )
        )

        summary = {
            "gate": gate,
            "pass": ok,
            "spy": asdict(spy_stats),
            "net": asdict(stats_n),
            "gross": asdict(stats_g),
            "timings": timings,
            "picks_csv": str(picks_path),
            "equity_csv": str(eq_path),
        }
        summary_path = args.reports_dir / f"summary_x{x}_L{args.liquid_n}_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    timings["total"] = time.perf_counter() - t_all
    table = format_table(all_stats, spy_stats)
    annual_tbl = format_annual_table(annual)
    xs = ",".join(str(x) for x in x_list)
    last_status = args.outdir / f"2026-09-24_uncorrelated_annual_x{xs.replace(',', '_')}.md"
    md = "\n".join(
        [
            f"# Uncorrelated annual sleeve (x={xs})",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Eval: {args.start} -> {args.end}",
            f"- Provider: {args.provider}"
            + (" +IB prefix" if args.ib_prefix else "")
            + f"; liquid_n={args.liquid_n}; cost_bps_rt={args.cost_bps_rt}",
            f"- Panel symbols loaded: {len(symbols)}; close shape={tuple(close.shape)}",
            f"- Wall clock: load={format_elapsed(timings['load'])} "
            f"total={format_elapsed(timings['total'])}",
            "",
            "## Performance",
            "",
            table,
            "",
            "## Annual returns",
            "",
            annual_tbl,
            "",
            "## Year picks by x",
            "",
            *sections,
            "## Design",
            "",
            "- Lookback = prior calendar year daily returns among history-ok then top-ADV names.",
            "- Selection = greedy minimize mean pairwise Pearson corr.",
            "- Hold = equal-weight next calendar year; lag-1 implement + RT costs.",
            "- ALPACA daily is thin before ~2022; use `--ib-prefix` + early `--start` for longer history.",
            "",
        ]
    )
    last_status.write_text(md, encoding="utf-8")
    LOG.info("Wrote status %s", last_status)
    print(table)
    print(f"\nLast gate: {last_gate}")
    print(f"Status: {last_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
