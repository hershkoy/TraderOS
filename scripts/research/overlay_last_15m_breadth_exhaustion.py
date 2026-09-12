"""Internal SMA20 breadth + fill-session RTH exhaustion skips on last-15m.

Pre-registered (same-list, occupancy not re-walked). Last-15m decides at
15:45 ET: daily breadth uses prior-session close; 15:45 open is known;
15:45 close is not.

  B50     skip if universe pct-above-SMA20 < 0.50
  B_med   skip if breadth < expanding median through t-1
  X_m2    skip if (15:45 open - 09:30 open) / 15m ATR14 >= 2
  X_d1    skip if same run / prior-session daily ATR >= 1

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python -u scripts\\research\\overlay_last_15m_breadth_exhaustion.py --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from overlay_last_15m_hmm_vix_skip import (  # noqa: E402
    DEFAULT_TRADES,
    _score_keep,
    _print_block,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_exhaustion import (  # noqa: E402
    attach_exhaustion,
    keep_unless_run_atr,
)
from utils.research.panel import list_daily_symbols, load_wide_panels  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.universe_breadth import (  # noqa: E402
    attach_breadth,
    breadth_above_sma,
    breadth_expanding_median,
    keep_if_breadth_at_least,
    keep_if_breadth_at_least_median,
)

LOG = logging.getLogger("overlay_last_15m_breadth")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=3)
    ap.add_argument("--pad-after", type=int, default=2)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    rows = []
    rows.append(_print_block("baseline_last15m", trades))

    start = datetime(2017, 1, 1)
    end = datetime(2026, 9, 12)
    LOG.info("Loading ALPACA daily close panel for breadth...")
    t0 = time.perf_counter()
    symbols = list_daily_symbols(provider="ALPACA", timeframe="1d")
    if not symbols:
        raise RuntimeError("no ALPACA 1d symbols")
    wide = load_wide_panels(
        symbols,
        start=start,
        end=end,
        provider="ALPACA",
        timeframe="1d",
        fields=("close",),
        workers=int(args.workers),
    )
    close = wide.get("close")
    if close is None or close.empty:
        raise RuntimeError("empty daily close panel")
    LOG.info(
        "Daily close panel %s -> %s names=%d in %.1fs",
        close.index.min().date(),
        close.index.max().date(),
        close.shape[1],
        time.perf_counter() - t0,
    )
    breadth = breadth_above_sma(close, sma_len=20)
    med = breadth_expanding_median(breadth, min_periods=60)
    tagged_b = attach_breadth(trades, breadth, median=med)
    n_b = pd.to_numeric(tagged_b["breadth_sma20"], errors="coerce").notna().sum()
    LOG.info("breadth joined %d/%d  mean=%.3f", n_b, len(tagged_b), float(pd.to_numeric(tagged_b["breadth_sma20"], errors="coerce").mean()))
    _score_keep("B50", tagged_b, keep_if_breadth_at_least(tagged_b, min_breadth=0.50), baseline_gain, rows)
    _score_keep("B_med", tagged_b, keep_if_breadth_at_least_median(tagged_b), baseline_gain, rows)

    symbols_tr = sorted(set(trades["stock"].astype(str).str.upper()))
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(days=10)
    sell_max = pd.to_datetime(trades.get("sell_date", trades["buy_date"]), errors="coerce").max() + timedelta(days=5)
    LOG.info("Loading IB 15m for %d names...", len(symbols_tr))
    t1 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        symbols_tr,
        timeframe="15m",
        provider="IB",
        start=buy_min.to_pydatetime() if hasattr(buy_min, "to_pydatetime") else buy_min,
        end=sell_max.to_pydatetime() if hasattr(sell_max, "to_pydatetime") else sell_max,
        workers=int(args.workers),
        use_cache=True,
    )
    n_have = sum(1 for s in symbols_tr if panels_15m.get(s) is not None and not panels_15m[s].empty)
    LOG.info("IB 15m %d/%d in %.1fs", n_have, len(symbols_tr), time.perf_counter() - t1)
    indexed = build_hold_session_index(
        panels_15m,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )
    LOG.info("Loading IB/ALPACA daily OHLC for prior ATR...")
    daily = load_ohlcv_many(
        symbols_tr,
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2017, 1, 1),
        end=end,
        fallback_provider="IB",
        merge_mode="prefix",
        workers=int(args.workers),
        use_cache=True,
    )
    tagged_x = attach_exhaustion(trades, indexed, daily_panels=daily)
    n_run = pd.to_numeric(tagged_x["run_atr_15m"], errors="coerce").notna().sum()
    n_d = pd.to_numeric(tagged_x["run_atr_1d"], errors="coerce").notna().sum()
    LOG.info("exhaustion 15m ATR %d/%d  daily ATR %d/%d", n_run, len(tagged_x), n_d, len(tagged_x))
    _score_keep(
        "X_m2",
        tagged_x,
        keep_unless_run_atr(tagged_x, col="run_atr_15m", max_run_atr=2.0),
        baseline_gain,
        rows,
    )
    _score_keep(
        "X_d1",
        tagged_x,
        keep_unless_run_atr(tagged_x, col="run_atr_1d", max_run_atr=1.0),
        baseline_gain,
        rows,
    )

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "last_15m_breadth_exhaustion_summary.csv", index=False)
    tagged_b.to_csv(outdir / "last_15m_breadth_trades.csv", index=False)
    tagged_x.to_csv(outdir / "last_15m_exhaustion_trades.csv", index=False)
    breadth.to_csv(outdir / "universe_breadth_sma20_daily.csv", header=True)
    lines = ["last-15m breadth + RTH exhaustion skip (same-list, occupancy not re-walked)"]
    for rec in rows:
        lines.append(
            "%s n=%s E=%s PF=%s geo=%s skipped=%s 2022 n=%s E=%s PF=%s 2023 n=%s E=%s PF=%s 2022-23 E=%s PF=%s blunt=%s"
            % (
                rec.get("label"),
                rec.get("n"),
                rec.get("expectancy_pct"),
                rec.get("profit_factor"),
                rec.get("geo_year_pf"),
                rec.get("n_skipped"),
                rec.get("n_2022"),
                rec.get("e_2022"),
                rec.get("pf_2022"),
                rec.get("n_2023"),
                rec.get("e_2023"),
                rec.get("pf_2023"),
                rec.get("e_2022_23"),
                rec.get("pf_2022_23"),
                rec.get("cut_blunt"),
            )
        )
    txt = outdir / "last_15m_breadth_exhaustion_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", outdir, flush=True)
    LOG.info("Wrote %s", txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
