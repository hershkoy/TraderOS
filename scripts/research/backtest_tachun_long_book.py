"""Run the Tachun long-only screen + 20y DCF equal-weight book.

Survivor-biased and estimate-free (no forward EPS, no delisted tape).

Example::

    venv\\Scripts\\activate
    set PYTHONPATH=.
    python scripts\\research\\backtest_tachun_long_book.py
    python scripts\\research\\backtest_tachun_long_book.py --fp-mode Q --max-symbols 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.tachun_long_book import run_tachun_backtest  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_tachun_long_book")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--fp-mode", choices=["FY", "Q", "BOTH"], default="FY")
    ap.add_argument("--max-names", type=int, default=15)
    ap.add_argument("--trim-mult", type=float, default=1.5)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--outdir",
        default="reports/research/tachun_long_book",
    )
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = run_tachun_backtest(
        start=start,
        end=end,
        max_names=int(args.max_names),
        trim_mult=float(args.trim_mult),
        fp_mode=args.fp_mode,
        max_symbols=int(args.max_symbols),
        workers=int(args.workers),
    )

    summary_path = outdir / f"tachun_summary_{stamp}.json"
    equity_path = outdir / f"tachun_equity_{stamp}.csv"
    holdings_path = outdir / f"tachun_holdings_{stamp}.csv"
    events_path = outdir / f"tachun_screen_events_{stamp}.csv"
    vals_path = outdir / f"tachun_valuations_{stamp}.csv"

    summary_path.write_text(json.dumps(result.stats, indent=2, default=str), encoding="utf-8")
    eq = result.equity.rename("equity").to_frame()
    eq["invested"] = result.invested.reindex(eq.index)
    eq["spy"] = result.spy_equity.reindex(eq.index)
    eq.to_csv(equity_path, index_label="date")
    if not result.holdings.empty:
        result.holdings.to_csv(holdings_path, index=False)
    if not result.screen_events.empty:
        result.screen_events.to_csv(events_path, index=False)
    if not result.valuations.empty:
        result.valuations.to_csv(vals_path, index=False)

    sleeve = result.stats["sleeve"]
    spy = result.stats["spy"]
    print(result.notes)
    print(
        "sleeve CAGR={:.2%} Sharpe={:.2f} MDD={:.2%} invested={:.1%} | "
        "SPY CAGR={:.2%} Sharpe={:.2f} MDD={:.2%} | {} | pass_syms={} vals={}".format(
            sleeve["cagr"],
            sleeve["sharpe"],
            sleeve["max_drawdown"],
            sleeve["pct_time_invested"],
            spy["cagr"],
            spy["sharpe"],
            spy["max_drawdown"],
            result.stats["phase5_reason"],
            result.stats["n_pass_symbols"],
            result.stats["n_valuations"],
        )
    )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
