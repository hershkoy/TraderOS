#!/usr/bin/env python3
"""Claim-check Quantified Strategies weekly strategies.

1) Free SPY down-week rule
2) Strategy #69 SPY/EFA/GLD/TLT rotation reconstructions (ROC lookbacks)

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\claimcheck_qs_weekly.py
  python scripts\\research\\claimcheck_qs_weekly.py --refresh-yahoo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.qs_weekly import (  # noqa: E402
    CLAIMED_69_MONTHLY,
    CLAIMED_69_WEEKLY,
    CLAIMED_DOWN_WEEK,
    ETFS_69,
    backtest_rotation69,
    backtest_spy_down_week,
    best_rotation_rows,
    sweep_rotation69,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimcheck_qs_weekly")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "qs_weekly"
YAHOO_DIR = DEFAULT_OUTDIR / "yahoo"


def _yahoo_url(symbol: str) -> str:
    # period1 ~ 1993-01-29 for SPY; Yahoo will clip earlier for younger ETFs.
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/%s"
        "?period1=727876800&period2=9999999999&interval=1d&events=history" % symbol
    )


def download_yahoo(symbol: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        _yahoo_url(symbol),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    logger.info("Downloaded Yahoo %s -> %s (%d bytes)", symbol, path, len(data))
    return path


def load_yahoo(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    path = YAHOO_DIR / ("%s.json" % symbol.lower())
    if refresh or not path.exists() or path.stat().st_size < 5_000:
        download_yahoo(symbol, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="s", utc=True),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    )
    df = (
        df.dropna(subset=["close", "open"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    logger.info(
        "Yahoo %s: %d bars  %s -> %s",
        symbol,
        len(df),
        df["ts"].iloc[0].date(),
        df["ts"].iloc[-1].date(),
    )
    return df


def load_all(symbols: tuple, *, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    return {s: load_yahoo(s, refresh=refresh) for s in symbols}


def _fmt_row(r: Dict[str, Any]) -> str:
    return (
        f"{r.get('name', '?'):28s}  n={r.get('n', 0):4d}  "
        f"WR={r.get('win_rate_pct')}%  PF={r.get('profit_factor')}  "
        f"E={r.get('avg_gain_pct')}%  CAGR={r.get('cagr_pct')}%  "
        f"Sharpe={r.get('sharpe')}  MDD={r.get('mdd_daily_pct')}%  "
        f"gate={r.get('gate')}"
    )


def write_status_md(
    out_path: Path,
    *,
    down_claim: Dict[str, Any],
    down_honest: Dict[str, Any],
    sweep: list,
    elapsed_s: float,
) -> None:
    best = best_rotation_rows(sweep, top_n=10)
    lines = [
        "# QS weekly strategies claim-check",
        "",
        f"- Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"- Wall clock: {elapsed_s:.1f}s",
        "- Sources: [weekly trading strategy](https://www.quantifiedstrategies.com/weekly-trading-strategy/), "
        "[SPY down-week](https://www.quantifiedstrategies.com/sp500-down-week-trading-strategy/), "
        "[#69 shop](https://www.quantifiedstrategies.com/product/strategy69/)",
        "",
        "## 1) SPY down-week (free rules)",
        "",
        "Rules: if Friday close < prior Friday close, long for the next week.",
        "",
        "| Book | n | WR | PF | Avg% | CAGR | Sharpe | MDD | Exposure | Gate |",
        "|------|---|----|----|------|------|--------|-----|----------|------|",
    ]
    claimed = CLAIMED_DOWN_WEEK
    lines.append(
        f"| Claimed | — | — | — | {claimed['avg_gain_pct']} | {claimed['cagr_pct']}% | — | "
        f"{claimed['mdd_pct']}% | {claimed['exposure_pct']}% | CLAIM |"
    )
    for r in (down_claim, down_honest):
        lines.append(
            f"| {r['name']} | {r['n']} | {r['win_rate_pct']}% | {r['profit_factor']} | "
            f"{r['avg_gain_pct']} | {r['cagr_pct']}% | {r['sharpe']} | {r['mdd_daily_pct']}% | "
            f"{r.get('exposure_pct')}% | {r['gate']} |"
        )
    lines.extend(
        [
            "",
            f"SPY buy&hold on same window (claim fill book): CAGR {down_claim['spy_cagr_pct']}% "
            f"Sharpe {down_claim['spy_sharpe']} MDD {down_claim['spy_mdd_pct']}%.",
            "",
            "## 2) Strategy #69 reconstructions (paywalled — ROC proxy)",
            "",
            "Universe SPY/EFA/GLD/TLT. Rank by N-period ROC; buy lowest (Substack teaser) or highest "
            "(classic momentum); hold one period. Not the exact #69 variable.",
            "",
            "### Claimed",
            "",
            f"- Weekly article: n={CLAIMED_69_WEEKLY['n']} CAGR {CLAIMED_69_WEEKLY['cagr_pct']}% "
            f"avg {CLAIMED_69_WEEKLY['avg_gain_pct']}% MDD {CLAIMED_69_WEEKLY['mdd_pct']}%",
            f"- Monthly shop: n={CLAIMED_69_MONTHLY['n']} CAGR {CLAIMED_69_MONTHLY['cagr_pct']}% "
            f"avg {CLAIMED_69_MONTHLY['avg_gain_pct']}% WR {CLAIMED_69_MONTHLY['win_rate_pct']}% "
            f"PF {CLAIMED_69_MONTHLY['profit_factor']} MDD {CLAIMED_69_MONTHLY['mdd_pct']}%",
            "",
            "### Top by Sharpe (n>=30)",
            "",
            "| name | n | WR | PF | Avg% | CAGR | Sharpe | MDD | keep% | Gate |",
            "|------|---|----|----|------|------|--------|-----|-------|------|",
        ]
    )
    for r in best:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['win_rate_pct']}% | {r['profit_factor']} | "
            f"{r['avg_gain_pct']} | {r['cagr_pct']}% | {r['sharpe']} | {r['mdd_daily_pct']}% | "
            f"{r.get('keep_same_pct')} | {r['gate']} |"
        )

    any_pass = any("PASS" in str(r.get("gate", "")) for r in [down_claim, down_honest] + list(sweep))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
        ]
    )
    if any_pass:
        lines.append("**At least one book cleared the Phase-5 gate — review before promote.**")
    else:
        lines.append(
            "**No promote.** Down-week may roughly match claim direction (lower exposure / milder MDD vs "
            "buy&hold) but fails Sharpe>1 vs SPY. #69 reconstructions do not clear the gate on these ROC "
            "proxies; exact paid variable remains unverified."
        )
    lines.extend(
        [
            "",
            f"Artifacts: `{DEFAULT_OUTDIR.as_posix()}/`",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote status %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--friction", type=float, default=0.0, help="Round-trip friction percent points")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    t0 = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = load_all(ETFS_69, refresh=args.refresh_yahoo)
    spy = frames["SPY"]

    down_claim = backtest_spy_down_week(spy, fill_mode="claim_close", friction_pct=args.friction)
    down_honest = backtest_spy_down_week(spy, fill_mode="next_open", friction_pct=args.friction)

    logger.info("--- SPY down-week ---")
    logger.info(
        "Claimed: avg=%s%% CAGR=%s%% MDD=%s%% exposure=%s%%",
        CLAIMED_DOWN_WEEK["avg_gain_pct"],
        CLAIMED_DOWN_WEEK["cagr_pct"],
        CLAIMED_DOWN_WEEK["mdd_pct"],
        CLAIMED_DOWN_WEEK["exposure_pct"],
    )
    logger.info(_fmt_row(down_claim))
    logger.info(_fmt_row(down_honest))

    logger.info("--- Strategy #69 ROC sweep ---")
    sweep = sweep_rotation69(frames, friction_pct=args.friction)
    for r in best_rotation_rows(sweep, top_n=12):
        logger.info(_fmt_row(r))

    # Also log a few headline reconstructions matching teaser shape.
    for pick in ("lowest", "highest"):
        for lb in (1, 4, 12):
            r = backtest_rotation69(
                frames, lookback=lb, pick=pick, cadence="weekly", friction_pct=args.friction
            )
            logger.info("headline %s", _fmt_row(r))

    # Persist
    (args.outdir / "down_week_claim.json").write_text(
        json.dumps({k: v for k, v in down_claim.items() if k != "trades"}, indent=2),
        encoding="utf-8",
    )
    (args.outdir / "down_week_honest.json").write_text(
        json.dumps({k: v for k, v in down_honest.items() if k != "trades"}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(down_claim["trades"]).to_csv(args.outdir / "down_week_claim_trades.csv", index=False)
    pd.DataFrame(down_honest["trades"]).to_csv(args.outdir / "down_week_honest_trades.csv", index=False)
    pd.DataFrame(sweep).to_csv(args.outdir / "rotation69_sweep.csv", index=False)

    elapsed = time.time() - t0
    status = (
        ROOT
        / "docs"
        / "status_log"
        / "edge_hunt"
        / ("%s_qs_weekly_claimcheck.md" % pd.Timestamp.now().strftime("%Y-%m-%d"))
    )
    write_status_md(
        status,
        down_claim=down_claim,
        down_honest=down_honest,
        sweep=sweep,
        elapsed_s=elapsed,
    )
    logger.info("Done in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
