#!/usr/bin/env python3
"""Claim-check Gozlan weekly Precision Candle (PCE) on QQQ/SPY.

Weekly PCE = one ETF closes green (close>open) while the other closes red.
Long the green leg for one week. This is the HTF signal book only — no 4H
FVG/OB entry (QQQ has no IB 15m/5m in TimescaleDB; no NQ/ES futures).

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\claimcheck_pce_weekly.py
  python scripts\\research\\claimcheck_pce_weekly.py --refresh-yahoo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.pce_weekly import (  # noqa: E402
    SOURCE_URL,
    align_weekly_pair,
    run_pce_weekly_suite,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimcheck_pce_weekly")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "pce_weekly"
YAHOO_DIR = DEFAULT_OUTDIR / "yahoo"
STATUS_PATH = ROOT / "docs" / "status_log" / "edge_hunt" / "2026-10-07_pce_weekly_claimcheck.md"
PAIR = ("QQQ", "SPY")


def _yahoo_url(symbol: str) -> str:
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


def write_status_md(
    out_path: Path,
    *,
    books: List[Dict[str, Any]],
    n_pce: int,
    n_weeks: int,
    elapsed_s: float,
) -> None:
    lines = [
        "# Gozlan PCE weekly claim-check",
        "",
        f"- Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"- Wall clock: {elapsed_s:.1f}s",
        f"- Source: [Gozlan PCE model]({SOURCE_URL})",
        "- Pair: QQQ vs SPY (ETF proxy for NQ vs ES from the video)",
        "- Scope: **weekly signal book only** — long the green leg for one week after a Precision Candle",
        "  (one closes green, the other red). No 4H FVG/OB entry layer (no QQQ intraday / no NQ-ES futures in DB).",
        "",
        "## Coverage",
        "",
        f"- Aligned Friday weeks: **{n_weeks}**",
        f"- Weekly PCE events: **{n_pce}** ({100.0 * n_pce / n_weeks:.1f}% of weeks)" if n_weeks else "- No weeks",
        "",
        "## Books",
        "",
        "| Book | n | WR | PF | Avg% | CAGR | Sharpe | MDD | Exposure | Gate |",
        "|------|---|----|----|------|------|--------|-----|----------|------|",
    ]
    for r in books:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['win_rate_pct']}% | {r['profit_factor']} | "
            f"{r['avg_gain_pct']} | {r['cagr_pct']}% | {r['sharpe']} | {r['mdd_daily_pct']}% | "
            f"{r.get('exposure_pct')}% | {r['gate']} |"
        )
    if books:
        spy = books[0]
        lines.extend(
            [
                "",
                f"SPY buy&hold on same window: CAGR {spy['spy_cagr_pct']}% "
                f"Sharpe {spy['spy_sharpe']} MDD {spy['spy_mdd_pct']}%.",
            ]
        )

    any_pass = any("PASS" in str(r.get("gate", "")) for r in books)
    honest = next((r for r in books if r["name"] == "long_green_next_open"), books[0] if books else None)
    lines.extend(["", "## Verdict", ""])
    if any_pass:
        lines.append("**At least one book cleared the Phase-5 gate — review before promote.**")
    else:
        lines.append(
            "**No promote.** Weekly PCE as an HTF long-the-green signal on QQQ/SPY does not clear "
            "Sharpe>1 / Sharpe>=SPY / MDD<SPY. This does **not** validate the full video model "
            "(True Opens bias + 4H FVG/OB entry + London/NY session filter)."
        )
    if honest is not None:
        lines.append(
            f"Honest next-open book: n={honest['n']} E {honest['avg_gain_pct']}% "
            f"PF {honest['profit_factor']} Sharpe {honest['sharpe']} ({honest['gate']})."
        )
    lines.extend(
        [
            "",
            "### Data gaps vs video",
            "",
            "- No NQ/ES futures OHLCV in TimescaleDB (used QQQ/SPY Yahoo daily).",
            "- QQQ has no IB 15m/5m — cannot resample a causal 4H entry book on the Nasdaq leg.",
            "- DIA / IWM not in TimescaleDB; Dow proxy skipped.",
            "- True London/Daily Open discount/premium filter not applied (needs intraday opens).",
            "",
            f"Artifacts: `{DEFAULT_OUTDIR.as_posix()}/`; harness "
            "`utils/research/pce_weekly.py` / `scripts/research/claimcheck_pce_weekly.py`.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote status %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--friction", type=float, default=0.10, help="Round-trip friction percent points")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--status", type=Path, default=STATUS_PATH)
    args = ap.parse_args()

    t0 = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)

    a_name, b_name = PAIR
    a_daily = load_yahoo(a_name, refresh=args.refresh_yahoo)
    b_daily = load_yahoo(b_name, refresh=args.refresh_yahoo)

    weekly = align_weekly_pair(a_daily, b_daily, a_name=a_name, b_name=b_name)
    n_pce = int(weekly["pce"].sum())
    n_weeks = int(len(weekly))
    logger.info("Weekly PCE: %d / %d weeks (%.1f%%)", n_pce, n_weeks, 100.0 * n_pce / max(n_weeks, 1))

    books = run_pce_weekly_suite(
        a_daily,
        b_daily,
        a_name=a_name,
        b_name=b_name,
        friction_pct=args.friction,
    )
    for r in books:
        logger.info(
            "%-32s n=%4d WR=%5.1f%% PF=%5.2f E=%6.2f%% Sharpe=%5.2f gate=%s",
            r["name"],
            r["n"],
            r["win_rate_pct"],
            r["profit_factor"],
            r["avg_gain_pct"],
            r["sharpe"],
            r["gate"],
        )

    # Persist trades + summary JSON
    summary = {
        "source": SOURCE_URL,
        "pair": "%s/%s" % PAIR,
        "n_weeks": n_weeks,
        "n_pce_weeks": n_pce,
        "books": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in books
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in books:
        trades = r.get("trades") or []
        if trades:
            pd.DataFrame(trades).to_csv(args.outdir / ("%s_trades.csv" % r["name"]), index=False)

    elapsed = time.time() - t0
    write_status_md(args.status, books=books, n_pce=n_pce, n_weeks=n_weeks, elapsed_s=elapsed)
    logger.info("Done in %.1fs -> %s", elapsed, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
