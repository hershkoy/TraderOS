#!/usr/bin/env python3
"""ThetaData listed-mark claim-check for Ravish QQQ LEAPS gap strategies.

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_qqq_leaps_gap_theta.py
  python scripts\\research\\claimcheck_qqq_leaps_gap_theta.py --mark-modes mid_mid,ask_bid --no-max-open
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.qqq_leaps_gap import prepare_qqq_daily  # noqa: E402
from utils.research.qqq_leaps_gap_theta import (  # noqa: E402
    CLAIMED,
    book_stats,
    list_qqq_expirations,
    run_theta_book,
    trades_to_frame,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_qqq_leaps_theta")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "qqq_leaps_gap"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_qqq(start: date, end: date) -> pd.DataFrame:
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


def _merge(a_trades, b_trades):
    return [
        t
        for book in (a_trades, b_trades)
        for t in book
        if not t.skipped and t.pnl_cash is not None
    ]


def _fmt(label: str, s: Dict[str, Any], claim: Optional[Dict[str, Any]] = None) -> str:
    line = (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"PnL=${s.get('total_pnl')} PF={s.get('pf')} "
        f"avgDays={s.get('avg_days')} MDD=${s.get('max_dd')} "
        f"TP%={s.get('tp_pct')} avgDebit=${s.get('avg_entry_premium')}"
    )
    if claim:
        line += (
            f" | claim n={claim.get('n')} WR={claim.get('win_rate_pct')}% "
            f"PnL=${claim.get('total_pnl')} PF={claim.get('pf')}"
        )
    return line


def write_status(
    path: Path,
    *,
    variants: List[Dict[str, Any]],
    elapsed_s: float,
    args: argparse.Namespace,
    verdict: str,
) -> None:
    lines = [
        "# QQQ LEAPS gap — ThetaData listed claim-check (2026-09-30)",
        "",
        f"**Verdict: {verdict}**",
        "",
        "**Source:** Options With Ravish — YouTube [dY2ralsWdQ8](https://www.youtube.com/watch?v=dY2ralsWdQ8).",
        "Prior BS method check: [2026-09-30_qqq_leaps_gap_claimcheck.md](2026-09-30_qqq_leaps_gap_claimcheck.md).",
        "",
        "## Claimed",
        "",
        "| Book | n | WR | PnL | PF |",
        "|------|--:|---:|----:|---:|",
        "| Strat1 | 53 | 96% | $88,400 | 10.48 |",
        "| Strat2 | 27 | 96% | $91,000 | 15 |",
        "| Combined | — | 96% | ~$180k | — |",
        "",
        "## Ours (Theta listed)",
        "",
        f"- Window: `{args.start}` → `{args.end}`",
        f"- Signals: same mechanical gap rules; SMA=`{args.sma_rule}`",
        "- Contract: Jan/Mar/Jun/Sep ~365 DTE; strike ≈ BS 60δ among listed with RTH NBBO",
        "- Entry: first nonzero 1m NBBO after 09:30 ET",
        "- Path: Theta `history/eod` bid/ask (≤365d pulls)",
        f"- Max open: `{None if args.no_max_open else args.max_open}`",
        f"- Wall: {elapsed_s:.1f}s",
        "",
        "Harness: `utils/research/qqq_leaps_gap_theta.py`, `scripts/research/claimcheck_qqq_leaps_gap_theta.py`",
        "",
    ]
    for v in variants:
        lines.append(f"### {v['name']}")
        lines.append("")
        lines.append("| Book | n | WR% | PnL | PF | Avg d | MDD | TP% | avg debit |")
        lines.append("|------|--:|----:|----:|---:|------:|----:|----:|----------:|")
        for row in v["rows"]:
            lines.append(
                f"| {row['book']} | {row['n']} | {row['win_rate_pct']} | "
                f"${row['total_pnl']} | {row['pf']} | {row['avg_days']} | "
                f"${row['max_dd']} | {row['tp_pct']} | ${row['avg_entry_premium']} |"
            )
        lines.append("")
        lines.append(
            f"Meta: s1 signals={v['s1_meta']['signals']} filled={v['s1_meta']['filled']} "
            f"skip_occ={v['s1_meta']['skip_occ']} skip_other={v['s1_meta']['skip_other']}; "
            f"s2 signals={v['s2_meta']['signals']} filled={v['s2_meta']['filled']} "
            f"skip_occ={v['s2_meta']['skip_occ']} skip_other={v['s2_meta']['skip_other']}"
        )
        lines.append("")
        if v.get("year_buckets"):
            lines.append("| Bucket | n | WR% | PnL | PF |")
            lines.append("|--------|--:|----:|----:|---:|")
            for y in v["year_buckets"]:
                lines.append(
                    f"| {y['bucket']} | {y['n']} | {y['win_rate_pct']} | "
                    f"${y['total_pnl']} | {y['pf']} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "1. **mid/mid** is the closest to OptionsKit-style mid marks (video enters at mid on Robinhood).",
            "2. **ask/bid** is the marketable long path (pay ask, exit bid) — haircuts WR/PnL vs mid.",
            "3. Skips (`no_contract` / `no_eod`) mean Theta had no LEAPS-cycle NBBO for that gap day — not silent wins.",
            "4. Even if mid/mid looks claim-like on $PnL, WR≠96 or occupancy/IV path failure still **no promote**; Sharpe>1 vs QQQ not walked until a clean fill book exists.",
            "",
            "## Reproduce",
            "",
            "```bat",
            "venv\\Scripts\\activate",
            "set PYTHONPATH=.",
            "python scripts\\pipeline\\thetadata_terminal.py",
            "python scripts\\research\\claimcheck_qqq_leaps_gap_theta.py",
            "```",
            "",
            f"Artifacts: `{path.parent.as_posix()}/`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-03")
    ap.add_argument("--end", default="2025-06-05")
    ap.add_argument("--sma-rule", choices=("prior", "open"), default="prior")
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--no-max-open", action="store_true")
    ap.add_argument(
        "--mark-modes",
        default="mid_mid,ask_bid",
        help="Comma list: mid_mid, ask_bid",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    max_open = None if args.no_max_open else int(args.max_open)
    mark_modes = [m.strip() for m in args.mark_modes.split(",") if m.strip()]
    outdir = args.outdir or (
        DEFAULT_OUT / f"{date.today().isoformat()}_theta"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not reachable — start thetadata_terminal.py")
        return 1

    t0 = time.time()
    panel = load_qqq(start, end)
    LOG.info("Loading QQQ expirations from Theta…")
    exps = list_qqq_expirations(client)
    LOG.info("QQQ expirations: %d", len(exps))
    eod_cache: Dict = {}

    variants: List[Dict[str, Any]] = []

    # Always run primary mid_mid with configured occupancy, plus uncapped mid if capped
    jobs = []
    for mm in mark_modes:
        jobs.append((f"{mm}_cap{max_open}", mm, max_open))
    if max_open is not None and "mid_mid" in mark_modes:
        jobs.append(("mid_mid_uncapped", "mid_mid", None))

    for name, mm, cap in jobs:
        LOG.info("=== %s ===", name)
        s1 = run_theta_book(
            panel,
            mode="strat1",
            gap_pct=1.0,
            contracts=1,
            start=start,
            end=end,
            mark_mode=mm,
            sma_rule=args.sma_rule,
            max_open=cap,
            client=client,
            exps=exps,
            eod_cache=eod_cache,
        )
        s2 = run_theta_book(
            panel,
            mode="strat2",
            gap_pct=2.0,
            contracts=2,
            start=start,
            end=end,
            mark_mode=mm,
            sma_rule=args.sma_rule,
            max_open=cap,
            client=client,
            exps=exps,
            eod_cache=eod_cache,
        )
        st1 = book_stats(s1.trades)
        st2 = book_stats(s2.trades)
        comb = _merge(s1.trades, s2.trades)
        stc = book_stats(comb)
        LOG.info(_fmt(f"{name} strat1", st1, CLAIMED["strat1"]))
        LOG.info(_fmt(f"{name} strat2", st2, CLAIMED["strat2"]))
        LOG.info(_fmt(f"{name} combined", stc, CLAIMED["combined"]))

        stem = name.replace(" ", "_")
        trades_to_frame(s1.trades).to_csv(outdir / f"{stem}_strat1.csv", index=False)
        trades_to_frame(s2.trades).to_csv(outdir / f"{stem}_strat2.csv", index=False)
        trades_to_frame(comb).to_csv(outdir / f"{stem}_combined.csv", index=False)

        summary = {
            "name": name,
            "mark_mode": mm,
            "max_open": cap,
            "strat1": st1,
            "strat2": st2,
            "combined": stc,
            "s1_meta": {
                "signals": s1.signals_raw,
                "filled": s1.filled,
                "skip_occ": s1.skipped_occ,
                "skip_other": s1.skipped_other,
            },
            "s2_meta": {
                "signals": s2.signals_raw,
                "filled": s2.filled,
                "skip_occ": s2.skipped_occ,
                "skip_other": s2.skipped_other,
            },
            "year_buckets": year_bucket_stats(comb),
            "rows": [
                {"book": "strat1", **st1},
                {"book": "strat2", **st2},
                {"book": "combined", **stc},
            ],
        }
        (outdir / f"{stem}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        variants.append(summary)

    elapsed = time.time() - t0

    # Verdict from primary mid_mid capped if present else first
    primary = next((v for v in variants if v["name"].startswith("mid_mid")), variants[0])
    wr = primary["combined"].get("win_rate_pct")
    n1 = primary["strat1"].get("n") or 0
    fill_rate = 0.0
    if primary["s1_meta"]["signals"]:
        fill_rate = primary["s1_meta"]["filled"] / primary["s1_meta"]["signals"]

    if wr is not None and wr >= 94 and abs(n1 - 53) <= 15 and fill_rate >= 0.7:
        verdict = (
            "no promote (Theta mid can look claim-like on WR; still fat-left-tail / "
            "bull-regime; ask/bid and Sharpe>1 gate not cleared)"
        )
    elif wr is not None and wr >= 85:
        verdict = (
            "no promote (Theta listed WR below claimed 96%; mid may approach claim $PnL "
            "uncapped but occupancy / marketable path degrade)"
        )
    else:
        verdict = (
            "no promote (Theta listed marks do not reprint 96% WR / claimed economics)"
        )

    status_path = outdir / "claimcheck_theta_summary.md"
    write_status(
        status_path,
        variants=variants,
        elapsed_s=elapsed,
        args=args,
        verdict=verdict,
    )
    docs = (
        ROOT
        / "docs"
        / "status_log"
        / "edge_hunt"
        / f"{date.today().isoformat()}_qqq_leaps_gap_theta_claimcheck.md"
    )
    docs.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
    LOG.info("Verdict: %s", verdict)
    LOG.info("Docs: %s", docs)
    LOG.info("Done in %.1fs -> %s", elapsed, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
