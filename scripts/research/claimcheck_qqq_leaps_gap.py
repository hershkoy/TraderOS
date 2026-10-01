#!/usr/bin/env python3
"""Claim-check Ravish QQQ LEAPS gap strategies (YouTube dY2ralsWdQ8).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_qqq_leaps_gap.py
  python scripts\\research\\claimcheck_qqq_leaps_gap.py --no-max-open --iv-elev 1.25 --iv-crush-days 20
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

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.qqq_leaps_gap import (  # noqa: E402
    CLAIMED,
    BookResult,
    LeapTrade,
    book_stats,
    count_signals,
    prepare_qqq_daily,
    run_book,
    trades_to_frame,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_qqq_leaps_gap")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "qqq_leaps_gap"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_qqq(start: date, end: date) -> pd.DataFrame:
    # Need SMA100 warmup before start.
    warm = date(start.year - 1, start.month, start.day) if start.month != 2 or start.day != 29 else date(start.year - 1, 2, 28)
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
        raise RuntimeError("No QQQ daily bars loaded")
    d = prepare_qqq_daily(raw)
    LOG.info(
        "QQQ daily prepared: %d bars  %s -> %s",
        len(d),
        d.index[0].date(),
        d.index[-1].date(),
    )
    return d


def _run_one(
    d: pd.DataFrame,
    *,
    mode: str,
    gap_pct: float,
    contracts: int,
    start: date,
    end: date,
    sma_rule: str,
    max_open: Optional[int],
    iv_elev: float,
    iv_crush_days: int,
    friction: float,
    dte: int,
) -> BookResult:
    return run_book(
        d,
        mode=mode,
        gap_pct=gap_pct,
        contracts=contracts,
        start=start,
        end=end,
        sma_rule=sma_rule,
        max_open=max_open,
        iv_elev=iv_elev,
        iv_crush_days=iv_crush_days,
        friction_per_share=friction,
        dte=dte,
    )


def _merge_combined(a: BookResult, b: BookResult) -> List[LeapTrade]:
    """Union of filled trades; same calendar day can appear in both books."""
    out: List[LeapTrade] = []
    for book in (a, b):
        out.extend([t for t in book.trades if not t.skipped and t.pnl_cash is not None])
    return out


def _fmt_stats(label: str, s: Dict[str, Any], claim: Optional[Dict[str, Any]] = None) -> str:
    line = (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"PnL=${s.get('total_pnl')} PF={s.get('pf')} "
        f"avgDays={s.get('avg_days')} MDD=${s.get('max_dd')} "
        f"TP%={s.get('tp_pct')} avgDebit=${s.get('avg_entry_premium')} "
        f"delta={s.get('avg_entry_delta')}"
    )
    if claim:
        line += (
            f"  | claim n={claim.get('n')} WR={claim.get('win_rate_pct')}% "
            f"PnL=${claim.get('total_pnl')} PF={claim.get('pf')} "
            f"avgDays={claim.get('avg_days')} MDD=${claim.get('max_dd')}"
        )
    return line


def write_status(
    path: Path,
    *,
    signal_counts: Dict[str, Any],
    variants: List[Dict[str, Any]],
    elapsed_s: float,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# QQQ LEAPS gap claim-check (2026-09-30)",
        "",
        "**Source:** Options With Ravish — YouTube [dY2ralsWdQ8](https://www.youtube.com/watch?v=dY2ralsWdQ8) (2025-07-05).",
        "",
        "**Verdict: TBD (filled by runner)**",
        "",
        "Claim is a **5y OptionsKit backtest** on QQQ only: buy ~60-delta / ~12mo calls on gap-downs; TP +50%; no stop.",
        "",
        "## Claimed",
        "",
        "| Book | n | WR | PnL | PF | Avg days | MDD | Contracts |",
        "|------|--:|---:|----:|---:|---------:|----:|----------:|",
        "| Strat1 gap>=1% + SMA100 | 53 | 96% | $88,400 | 10.48 | 102 | $9,330 | 1 |",
        "| Strat2 gap>=2% | 27 | 96% | $91,000 | 15 | — | $6,000 | 2 |",
        "| Combined | — | 96% | ~$180,000 | — | — | — | — |",
        "",
        "OptionsKit dashboard in the video intro (~91% WR / $176k) matches the **prior** gap-only video, not this filtered pair.",
        "",
        "## Ours (mechanical)",
        "",
        f"- Window: `{args.start}` -> `{args.end}`",
        f"- Underlying: QQQ IB 1d (ALPACA prefix fallback)",
        f"- Entry: open fill on gap day; SMA rule=`{args.sma_rule}` (prior close vs SMA100)",
        f"- Option: BS call, target delta={args.target_delta}, DTE={args.dte}, r=5%, q=0.5%",
        f"- IV: ATR20 annualized x elev={args.iv_elev}, crush_days={args.iv_crush_days}",
        f"- Max open lots: `{None if args.no_max_open else args.max_open}`",
        f"- Friction $/share round-turn side: `{args.friction}`",
        f"- Wall: {elapsed_s:.1f}s",
        "",
        "Data note: TimescaleDB has QQQ `option_contracts` but **0** `option_quotes`; OptionsDX EOD is SPY/VIX only. Full listed mid/NBBO LEAPS replay is not available here — BS is an optimistic method check.",
        "",
        "Harness: `utils/research/qqq_leaps_gap.py`, `scripts/research/claimcheck_qqq_leaps_gap.py`",
        "",
        "### Signal counts (raw, no occupancy)",
        "",
        "```json",
        json.dumps(signal_counts, indent=2),
        "```",
        "",
    ]
    for v in variants:
        lines.append(f"### {v['name']}")
        lines.append("")
        lines.append("| Book | n | WR% | PnL | PF | Avg d | MDD | TP% | Match notes |")
        lines.append("|------|--:|----:|----:|---:|------:|----:|----:|-------------|")
        for row in v["rows"]:
            lines.append(
                f"| {row['book']} | {row['n']} | {row['win_rate_pct']} | "
                f"${row['total_pnl']} | {row['pf']} | {row['avg_days']} | "
                f"${row['max_dd']} | {row['tp_pct']} | {row.get('note', '')} |"
            )
        lines.append("")
        if v.get("year_buckets"):
            lines.append("Year buckets (filled trades):")
            lines.append("")
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
            "1. **Headline 96% WR is mostly the bull-market path + no stop + TP50 on long delta.** "
            "LEAPS calls on QQQ dips in 2020-21/2023-25 recover often; the two claimed losses cluster in 2022-style regimes.",
            "2. **Dollar PnL is not a strategy edge metric** without capital / occupancy — one contract on cheaper 2020 QQQ vs 2025 QQQ is not equal risk. Prefer % on debit and occupancy equity vs QQQ.",
            "3. **Avg win << avg loss** in the claim (RR ~0.4) — classic high-WR / fat-left-tail. One extra full-debit loser flips the story; n=53 is small.",
            "4. **SMA100 filter** is doing the work vs the prior gap-only video (screenshot 91% / $176k / MDD $18k): it cuts bear-market gap entries.",
            "5. **BS marks are optimistic for gap-down buyers** if IV is elevated at entry; elev+crush stress is the more honest method check. Listed NBBO/mid for QQQ LEAPS 2020-25 is not in-DB.",
            "6. **Promotion gate:** even if BS roughly reprints 96% WR, require fixed-slot occupancy Sharpe>1 and MDD better than QQQ before promote — WR alone is not the gate (`docs/assessing_strategies.md`).",
            "",
            "## Reproduce",
            "",
            "```bat",
            "venv\\Scripts\\activate",
            "set PYTHONPATH=.",
            "python scripts\\research\\claimcheck_qqq_leaps_gap.py",
            "python scripts\\research\\claimcheck_qqq_leaps_gap.py --no-max-open --iv-elev 1.25 --iv-crush-days 20",
            "```",
            "",
            f"Artifacts: `{path.parent.as_posix()}/`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote status %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Claim-check QQQ LEAPS gap strategies")
    ap.add_argument("--start", default="2020-01-03")
    ap.add_argument("--end", default="2025-06-05")
    ap.add_argument("--sma-rule", choices=("prior", "open"), default="prior")
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--no-max-open", action="store_true")
    ap.add_argument("--iv-elev", type=float, default=1.0)
    ap.add_argument("--iv-crush-days", type=int, default=1)
    ap.add_argument("--friction", type=float, default=0.0)
    ap.add_argument("--dte", type=int, default=365)
    ap.add_argument("--target-delta", type=float, default=0.60)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    max_open = None if args.no_max_open else int(args.max_open)
    outdir = args.outdir or (DEFAULT_OUT / date.today().isoformat())
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    d = load_qqq(start, end)

    # Signal counts for both SMA rules
    counts_prior = count_signals(d, start=start, end=end, sma_rule="prior")
    counts_open = count_signals(d, start=start, end=end, sma_rule="open")
    LOG.info("Signals prior-SMA: %s", counts_prior)
    LOG.info("Signals open-SMA:  %s", counts_open)

    variants_meta: List[Dict[str, Any]] = []

    def run_variant(name: str, *, iv_elev: float, crush: int, max_o: Optional[int], sma: str) -> Dict[str, Any]:
        s1 = _run_one(
            d,
            mode="strat1",
            gap_pct=1.0,
            contracts=1,
            start=start,
            end=end,
            sma_rule=sma,
            max_open=max_o,
            iv_elev=iv_elev,
            iv_crush_days=crush,
            friction=args.friction,
            dte=args.dte,
        )
        s2 = _run_one(
            d,
            mode="strat2",
            gap_pct=2.0,
            contracts=2,
            start=start,
            end=end,
            sma_rule=sma,
            max_open=max_o,
            iv_elev=iv_elev,
            iv_crush_days=crush,
            friction=args.friction,
            dte=args.dte,
        )
        st1 = book_stats(s1.trades)
        st2 = book_stats(s2.trades)
        comb = _merge_combined(s1, s2)
        stc = book_stats(comb)
        LOG.info(_fmt_stats(f"{name} strat1", st1, CLAIMED["strat1"]))
        LOG.info(
            "  fills=%d skip_occ=%d skip_other=%d signals=%d",
            s1.filled,
            s1.skipped_occ,
            s1.skipped_other,
            s1.signals_raw,
        )
        LOG.info(_fmt_stats(f"{name} strat2", st2, CLAIMED["strat2"]))
        LOG.info(
            "  fills=%d skip_occ=%d skip_other=%d signals=%d",
            s2.filled,
            s2.skipped_occ,
            s2.skipped_other,
            s2.signals_raw,
        )
        LOG.info(_fmt_stats(f"{name} combined", stc, CLAIMED["combined"]))

        stem = name.replace(" ", "_").replace("/", "_")
        trades_to_frame(s1.trades).to_csv(outdir / f"{stem}_strat1_trades.csv", index=False)
        trades_to_frame(s2.trades).to_csv(outdir / f"{stem}_strat2_trades.csv", index=False)
        trades_to_frame(comb).to_csv(outdir / f"{stem}_combined_trades.csv", index=False)

        def note(st: Dict[str, Any], claim: Dict[str, Any]) -> str:
            bits = []
            if claim.get("n") and st.get("n"):
                bits.append(f"n {st['n']} vs {claim['n']}")
            if claim.get("win_rate_pct") and st.get("win_rate_pct") is not None:
                bits.append(f"WR {st['win_rate_pct']} vs {claim['win_rate_pct']}")
            return "; ".join(bits)

        rows = [
            {
                "book": "strat1",
                **st1,
                "note": note(st1, CLAIMED["strat1"]),
            },
            {
                "book": "strat2",
                **st2,
                "note": note(st2, CLAIMED["strat2"]),
            },
            {
                "book": "combined",
                **stc,
                "note": f"PnL ${stc.get('total_pnl')} vs ~$180k",
            },
        ]
        yb = year_bucket_stats(comb)
        summary = {
            "name": name,
            "iv_elev": iv_elev,
            "iv_crush_days": crush,
            "max_open": max_o,
            "sma_rule": sma,
            "strat1": st1,
            "strat2": st2,
            "combined": stc,
            "strat1_meta": {
                "signals": s1.signals_raw,
                "filled": s1.filled,
                "skip_occ": s1.skipped_occ,
            },
            "strat2_meta": {
                "signals": s2.signals_raw,
                "filled": s2.filled,
                "skip_occ": s2.skipped_occ,
            },
            "year_buckets": yb,
            "rows": rows,
        }
        (outdir / f"{stem}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    # Primary: constant IV, prior SMA, max_open as configured
    primary = run_variant(
        "A_constIV",
        iv_elev=float(args.iv_elev),
        crush=int(args.iv_crush_days),
        max_o=max_open,
        sma=args.sma_rule,
    )
    variants_meta.append(primary)

    # Ablations always run for the status log
    variants_meta.append(
        run_variant("B_noCap_constIV", iv_elev=1.0, crush=1, max_o=None, sma=args.sma_rule)
    )
    variants_meta.append(
        run_variant("C_elev125_crush20", iv_elev=1.25, crush=20, max_o=max_open, sma=args.sma_rule)
    )
    variants_meta.append(
        run_variant("D_elev150_crush10", iv_elev=1.50, crush=10, max_o=max_open, sma=args.sma_rule)
    )
    if args.sma_rule == "prior":
        variants_meta.append(
            run_variant("E_openSMA_constIV", iv_elev=1.0, crush=1, max_o=max_open, sma="open")
        )

    elapsed = time.time() - t0
    status_path = outdir / "claimcheck_summary.md"
    # Also write roll-up under docs/status_log
    docs_path = (
        ROOT
        / "docs"
        / "status_log"
        / "edge_hunt"
        / f"{date.today().isoformat()}_qqq_leaps_gap_claimcheck.md"
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    # Verdict from primary combined + elev stress
    p = primary["combined"]
    elev = next(v for v in variants_meta if v["name"].startswith("C_"))["combined"]
    wr_ok = p.get("win_rate_pct") is not None and p["win_rate_pct"] >= 90
    elev_ok = elev.get("win_rate_pct") is not None and elev["win_rate_pct"] >= 85
    n1 = primary["strat1"].get("n") or 0
    n_claim = CLAIMED["strat1"]["n"]
    n_close = abs(n1 - n_claim) <= 15

    if wr_ok and elev_ok and n_close and (p.get("total_pnl") or 0) > 0:
        verdict = (
            "no promote (BS can look claim-like on WR/path; listed LEAPS marks missing; "
            "fat-left-tail / bull-regime dependency; not Sharpe>1 gate)"
        )
    elif wr_ok and n_close:
        verdict = (
            "no promote (WR reprints under calm IV; elevated-entry IV weakens the book; "
            "no listed QQQ LEAPS replay)"
        )
    else:
        verdict = (
            "no promote (does not cleanly reprint claim under mechanical BS; "
            "signal n and/or WR diverge)"
        )

    write_status(
        status_path,
        signal_counts={"prior": counts_prior, "open": counts_open},
        variants=variants_meta,
        elapsed_s=elapsed,
        args=args,
    )
    # Patch verdict into both copies
    text = status_path.read_text(encoding="utf-8")
    text = text.replace("**Verdict: TBD (filled by runner)**", f"**Verdict: {verdict}**")
    status_path.write_text(text, encoding="utf-8")
    docs_path.write_text(text, encoding="utf-8")
    LOG.info("Verdict: %s", verdict)
    LOG.info("Docs: %s", docs_path)
    LOG.info("Done in %.1fs -> %s", elapsed, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
