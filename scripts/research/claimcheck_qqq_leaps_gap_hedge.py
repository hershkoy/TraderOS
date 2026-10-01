#!/usr/bin/env python3
"""Claim-check low-VIX hedges on the QQQ LEAPS gap Theta book.

Requires Theta Terminal on :25503.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_qqq_leaps_gap_hedge.py
  python scripts\\research\\claimcheck_qqq_leaps_gap_hedge.py --hedges H1 --attach entry_day --max-open 10
  python scripts\\research\\claimcheck_qqq_leaps_gap_hedge.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.qqq_leaps_gap import book_stats, prepare_qqq_daily  # noqa: E402
from utils.research.qqq_leaps_gap_hedge import (  # noqa: E402
    ATTACH_MODES,
    HEDGE_IDS,
    apply_max_open,
    combine_leaps_hedge_stats,
    hedge_stats,
    hedges_to_frame,
    leaps_from_csv,
    list_expirations,
    load_vix_prior_close,
    run_hedge_book,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_qqq_leaps_hedge")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_LEAPS_CSV = (
    ROOT
    / "reports"
    / "research"
    / "qqq_leaps_gap"
    / "2026-09-30_theta"
    / "mid_mid_uncapped_combined.csv"
)
DEFAULT_OUT = ROOT / "reports" / "research" / "qqq_leaps_gap" / "2026-10-01_hedge"
WINDOW_START = date(2020, 1, 3)
WINDOW_END = date(2025, 6, 5)


def load_qqq(start: date, end: date):
    warm = (
        date(start.year - 1, start.month, start.day)
        if start.month != 2 or start.day != 29
        else date(start.year - 1, 2, 28)
    )
    frames = load_ohlcv_many(
        ["QQQ"],
        timeframe="1d",
        provider="IB",
        start=datetime(warm.year, warm.month, warm.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
        fallback_provider="ALPACA",
        merge_mode="prefix",
    )
    raw = frames.get("QQQ")
    if raw is None or raw.empty:
        raise RuntimeError("No QQQ daily bars")
    return prepare_qqq_daily(raw)


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _max_open_label(v: Optional[int]) -> str:
    return "uncapped" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="QQQ LEAPS gap hedge claim-check")
    ap.add_argument("--leaps-csv", type=Path, default=DEFAULT_LEAPS_CSV)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", default=WINDOW_START.isoformat())
    ap.add_argument("--end", default=WINDOW_END.isoformat())
    ap.add_argument("--hedges", default="H1,H2,H3")
    ap.add_argument("--attach", default="while_open,continuous,entry_day")
    ap.add_argument("--mark-modes", default="mid_mid")
    ap.add_argument(
        "--max-open",
        default="10",
        help="Comma list of caps; use 'none' for uncapped. Default 10.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="H1 + entry_day + mid_mid + max_open=10 only (fast path).",
    )
    args = ap.parse_args()

    if args.smoke:
        hedges = ["H1"]
        attaches = ["entry_day"]
        marks = ["mid_mid"]
        caps: List[Optional[int]] = [10]
    else:
        hedges = [h.upper() for h in _parse_list(args.hedges)]
        attaches = _parse_list(args.attach)
        marks = _parse_list(args.mark_modes)
        caps = []
        for x in _parse_list(args.max_open):
            if x.lower() in ("none", "uncapped", "null"):
                caps.append(None)
            else:
                caps.append(int(x))
        for h in hedges:
            if h not in HEDGE_IDS:
                raise SystemExit("unknown hedge %s" % h)
        for a in attaches:
            if a not in ATTACH_MODES:
                raise SystemExit("unknown attach %s" % a)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not reachable on :25503")
        return 1

    t0 = time.time()
    leaps_all = leaps_from_csv(args.leaps_csv)
    LOG.info("Loaded %d filled LEAPS from %s", len(leaps_all), args.leaps_csv)

    panel = load_qqq(start, end)
    vix_prior = load_vix_prior_close(start, end)
    LOG.info("VIX prior sessions=%d", len(vix_prior.dropna()))

    qqq_exps = list_expirations(client, "QQQ")
    vix_exps = list_expirations(client, "VIX")
    LOG.info("Expirations QQQ=%d VIX=%d", len(qqq_exps), len(vix_exps))
    eod_cache: Dict = {}

    matrix: List[Dict[str, Any]] = []
    for cap in caps:
        filled = apply_max_open(leaps_all, cap)
        leaps_stats = book_stats(filled)
        LOG.info(
            "LEAPS max_open=%s n=%s PnL=$%s PF=%s MDD=$%s",
            _max_open_label(cap),
            leaps_stats.get("n"),
            leaps_stats.get("total_pnl"),
            leaps_stats.get("pf"),
            leaps_stats.get("max_dd"),
        )
        for structure in hedges:
            for attach in attaches:
                for mark in marks:
                    key = "%s_%s_%s_cap%s" % (
                        structure,
                        attach,
                        mark,
                        _max_open_label(cap),
                    )
                    LOG.info("=== RUN %s ===", key)
                    t1 = time.time()
                    book = run_hedge_book(
                        structure=structure,
                        attach=attach,
                        mark_mode=mark,
                        leaps=leaps_all,
                        panel=panel,
                        vix_prior=vix_prior,
                        max_open=cap,
                        client=client,
                        qqq_exps=qqq_exps,
                        vix_exps=vix_exps,
                        eod_cache=eod_cache,
                        panel_end=end,
                    )
                    elapsed = time.time() - t1
                    comb = combine_leaps_hedge_stats(filled, book.hedges)
                    hs = hedge_stats(book.hedges)
                    row = {
                        "key": key,
                        "structure": structure,
                        "attach": attach,
                        "mark_mode": mark,
                        "max_open": cap,
                        "leaps_n": leaps_stats.get("n"),
                        "leaps_pnl": leaps_stats.get("total_pnl"),
                        "leaps_pf": leaps_stats.get("pf"),
                        "leaps_mdd": leaps_stats.get("max_dd"),
                        "hedge_n": hs.get("n"),
                        "hedge_pnl": hs.get("total_pnl"),
                        "hedge_pf": hs.get("pf"),
                        "hedge_mdd": hs.get("max_dd"),
                        "hedge_skipped": book.hedge_skipped,
                        "combined_pnl": comb.get("combined_pnl"),
                        "combined_pf": comb.get("combined_pf"),
                        "combined_mdd": comb.get("combined_max_dd"),
                        "mdd_lift": (
                            None
                            if leaps_stats.get("max_dd") is None
                            or comb.get("combined_max_dd") is None
                            else round(
                                float(leaps_stats["max_dd"])
                                - float(comb["combined_max_dd"]),
                                2,
                            )
                        ),
                        "elapsed_s": round(elapsed, 1),
                        "notes": book.notes,
                    }
                    matrix.append(row)
                    hedges_to_frame(book.hedges).to_csv(
                        outdir / ("%s_hedges.csv" % key), index=False
                    )
                    (outdir / ("%s_summary.json" % key)).write_text(
                        json.dumps(
                            {"row": row, "leaps": leaps_stats, "combined": comb},
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    LOG.info(
                        "%s hedge_n=%s hedge_pnl=$%s combined_pnl=$%s "
                        "combined_mdd=$%s leaps_mdd=$%s wall=%.0fs",
                        key,
                        row["hedge_n"],
                        row["hedge_pnl"],
                        row["combined_pnl"],
                        row["combined_mdd"],
                        row["leaps_mdd"],
                        elapsed,
                    )

    wall = time.time() - t0
    summary = {
        "window": [start.isoformat(), end.isoformat()],
        "leaps_csv": str(args.leaps_csv),
        "vix_theta": "available",
        "wall_s": round(wall, 1),
        "matrix": matrix,
        "verdict_gate": (
            "Improve MDD/left-tail vs LEAPS-only at max_open=10 without "
            "destroying PF/CAGR; collar that kills LEAPS TP50 = no promote."
        ),
    }
    (outdir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown matrix
    lines = [
        "# QQQ LEAPS gap — low-VIX hedge matrix",
        "",
        f"Window `{start}` -> `{end}`. Wall {wall:.0f}s. Theta VIX options: available.",
        "",
        "| Key | LEAPS n/PnL/MDD | Hedge n/PnL | Combined PnL/PF/MDD | MDD lift |",
        "|-----|----------------:|------------:|--------------------:|---------:|",
    ]
    for r in matrix:
        lines.append(
            "| {key} | {leaps_n}/${leaps_pnl}/${leaps_mdd} | "
            "{hedge_n}/${hedge_pnl} | "
            "${combined_pnl}/{combined_pf}/${combined_mdd} | "
            "{mdd_lift} |".format(**{k: r.get(k) for k in r})
        )
    (outdir / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Wrote %s (wall=%.0fs)", outdir / "matrix_summary.json", wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
