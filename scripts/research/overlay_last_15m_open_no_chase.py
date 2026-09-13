"""Last-15m next-session MOO without chasing an overnight gap.

Signal stays last-RTH 15m open-above-rail (current_best 1d). Fill is next
09:30 ET open when that open does not gap up vs the signal session close and
still holds the rail. Gap-up (or open already below the rail) waits the same
session for a pullback plus a positive 15m candle holding above the rail.

Pre-registered policies (occupancy not re-walked; 15m N+1 mid sells):

  chase_moo   buy next 09:30 open even if it gapped (control)
  skip_gap    MOO only when there is no gap-up and the open holds the rail
  pullback    same MOO, else wait same-session pullback + green/hammer/engulf

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_open_no_chase.py --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from overlay_last_15m_hmm_vix_skip import _print_block  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_open_no_chase import (  # noqa: E402
    POLICIES,
    POLICY_CHASE_MOO,
    POLICY_PULLBACK,
    POLICY_SKIP_GAP,
    apply_open_no_chase,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    FRICTION_PCT,
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_open_no_chase")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
SMOKE_DEFAULT = ("AMPL", "VST", "TARS")


def _apply_policy(
    trades: pd.DataFrame,
    indexed: Dict[str, dict],
    *,
    policy: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, row in trades.iterrows():
        sym = str(row["stock"]).upper()
        got = apply_open_no_chase(row, indexed.get(sym) or {}, policy=policy)
        rec = row.to_dict()
        rec["src_idx"] = idx
        rec["policy"] = policy
        rec["fill_ok"] = bool(got.filled)
        rec["fill_reason"] = got.reason
        rec["gap_up"] = bool(got.gap_up)
        rec["open_holds"] = bool(got.open_holds)
        rec["fill_pattern"] = got.pattern or ""
        rec["gap_pct"] = got.gap_pct
        rec["next_open"] = got.open_px
        rec["prior_close"] = got.prior_close
        rec["next_resist"] = got.resist
        rec["orig_gain_pct"] = row.get("gain_pct")
        rec["orig_buy_price"] = row.get("buy_price")
        rec["orig_buy_time"] = row.get("buy_time")
        if got.filled and got.gain_pct is not None and got.entry_px is not None:
            rec["buy_date"] = (
                got.next_day.isoformat() if got.next_day is not None else rec.get("buy_date")
            )
            rec["buy_time"] = str(got.entry_ts) if got.entry_ts is not None else ""
            rec["buy_price"] = round(float(got.entry_px), 4)
            rec["sell_price"] = (
                round(float(got.sell_px), 4) if got.sell_px is not None else rec.get("sell_price")
            )
            rec["sell_time"] = str(got.sell_ts) if got.sell_ts is not None else rec.get("sell_time")
            rec["gain_pct"] = round(float(got.gain_pct), 4)
            rec["exit_reason"] = got.exit_reason or rec.get("exit_reason")
        else:
            rec["gain_pct"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _score_policy(
    label: str,
    tagged: pd.DataFrame,
    baseline: pd.DataFrame,
    rows: List[dict],
) -> None:
    filled = tagged.loc[tagged["fill_ok"]].copy()
    rec = _print_block(label, filled)
    keep = tagged["fill_ok"].reindex(tagged.index).fillna(False).astype(bool)
    # winner-cut vs original last-15m gains (skip = no overlay fill)
    orig = pd.to_numeric(tagged["orig_gain_pct"], errors="coerce")
    cut = winner_cut_skip(orig, keep)
    rec.update({("cut_" + k): v for k, v in cut.items()})
    print(
        "  fills moo=%s pullback=%s skip=%s gap_up=%s | winner-cut dropped=%s win$=%s lose$=%s blunt=%s"
        % (
            int((tagged["fill_reason"] == "moo").sum()) if "fill_reason" in tagged.columns else 0,
            int((tagged["fill_reason"] == "pullback_confirm").sum())
            if "fill_reason" in tagged.columns
            else 0,
            int((~keep).sum()),
            int(tagged["gap_up"].fillna(False).astype(bool).sum()),
            cut["n_dropped"],
            cut["winner_drop_gain_sum"],
            cut["loser_drop_gain_sum"],
            cut["blunt"],
        )
    )
    if not filled.empty:
        net = pd.to_numeric(filled["gain_pct"], errors="coerce") - FRICTION_PCT
        s_net = book_stats(net)
        years_n = summarize_by_year(
            filled.assign(gain_pct_net=net),
            gain_col="gain_pct_net",
            buckets=YEAR_BUCKETS,
        )
        rec["fric025_expectancy_pct"] = s_net["expectancy_pct"]
        rec["fric025_profit_factor"] = s_net["profit_factor"]
        rec["fric025_geo_year_pf"] = geom_mean_year_pf(years_n)
        LOG.info(
            "%s fric0.25 E=%s PF=%s geo=%s",
            label,
            s_net["expectancy_pct"],
            s_net["profit_factor"],
            rec["fric025_geo_year_pf"],
        )
        reasons = tagged["fill_reason"].value_counts(dropna=False)
        print("  reasons:")
        print(reasons.to_string())
        if "fill_pattern" in filled.columns:
            pats = filled["fill_pattern"].value_counts(dropna=False)
            print("  patterns:")
            print(pats.to_string())
    rec["n_baseline"] = int(len(baseline))
    rec["n_skipped"] = int((~keep).sum())
    rows.append(rec)


def _smoke(tagged: pd.DataFrame, symbols: tuple) -> None:
    for sym in symbols:
        part = tagged.loc[tagged["stock"].astype(str).str.upper() == sym]
        if part.empty:
            print("  SMOKE %s: no last-15m row" % sym)
            continue
        for _, row in part.iterrows():
            print(
                "  SMOKE %s orig=%s px=%s gain=%s -> ok=%s reason=%s gap_up=%s "
                "open=%s prior_c=%s gap_pct=%s pattern=%s new_px=%s new_gain=%s"
                % (
                    sym,
                    row.get("orig_buy_time") or row.get("buy_date"),
                    row.get("orig_buy_price"),
                    row.get("orig_gain_pct"),
                    row.get("fill_ok"),
                    row.get("fill_reason"),
                    row.get("gap_up"),
                    row.get("next_open"),
                    row.get("prior_close"),
                    row.get("gap_pct"),
                    row.get("fill_pattern"),
                    row.get("buy_price") if row.get("fill_ok") else "",
                    row.get("gain_pct") if row.get("fill_ok") else "",
                )
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=3)
    ap.add_argument("--pad-after", type=int, default=21)
    ap.add_argument("--symbols", default="", help="Optional CSV of symbols (smoke)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    want = str(args.symbols or "").strip().upper()
    if want:
        smoke_syms = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(smoke_syms)].copy()
        LOG.info("Restricted to %s n=%d", sorted(smoke_syms), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    rows: List[dict] = []
    rows.append(_print_block("baseline_last15m", trades))

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
        days=int(args.pad_before)
    )
    sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
        days=int(args.pad_after)
    )
    LOG.info(
        "Loading IB 15m symbols=%d window %s -> %s",
        len(symbols),
        buy_min.date() if hasattr(buy_min, "date") else buy_min,
        sell_max.date() if hasattr(sell_max, "date") else sell_max,
    )
    t0 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min.to_pydatetime() if hasattr(buy_min, "to_pydatetime") else buy_min,
        end=sell_max.to_pydatetime() if hasattr(sell_max, "to_pydatetime") else sell_max,
        workers=int(args.workers),
        use_cache=True,
    )
    n_have = sum(1 for s in symbols if panels_15m.get(s) is not None and not panels_15m[s].empty)
    LOG.info("IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)
    indexed = build_hold_session_index(
        panels_15m,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )
    LOG.info("session index symbols=%d", len(indexed))

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    tagged_by_policy: Dict[str, pd.DataFrame] = {}
    t1 = time.perf_counter()
    for policy in (POLICY_CHASE_MOO, POLICY_SKIP_GAP, POLICY_PULLBACK):
        tagged = _apply_policy(trades, indexed, policy=policy)
        tagged_by_policy[policy] = tagged
        _score_policy(policy, tagged, trades, rows)
        tagged.to_csv(outdir / ("last_15m_open_no_chase_%s.csv" % policy), index=False)
        if not want:
            _smoke(tagged, SMOKE_DEFAULT)
        else:
            _smoke(tagged, tuple(sorted({str(s).upper() for s in trades["stock"]})))
    LOG.info("policies walked in %.1fs", time.perf_counter() - t1)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "last_15m_open_no_chase_summary.csv", index=False)
    elapsed = time.perf_counter() - t0
    LOG.info("wrote %s (wall %.1fs)", outdir, elapsed)
    print("POLICIES=%s" % ",".join(POLICIES))
    print("OUTDIR=%s" % outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
