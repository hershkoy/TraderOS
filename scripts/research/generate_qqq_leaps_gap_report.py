#!/usr/bin/env python3
"""HTML report for QQQ LEAPS gap Theta claim-check (capital + SPY BH).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_qqq_leaps_gap_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.qqq_leaps_gap import LeapTrade, book_stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("generate_qqq_leaps_gap_report")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_THETA = ROOT / "reports" / "research" / "qqq_leaps_gap" / "2026-09-30_theta"
DEFAULT_ASKBID = (
    ROOT / "reports" / "research" / "qqq_leaps_gap" / "2026-09-30_theta_askbid"
)
DEFAULT_ASKBID_UNC = (
    ROOT
    / "reports"
    / "research"
    / "qqq_leaps_gap"
    / "2026-09-30_theta_askbid_uncapped"
)
DEFAULT_OUTDIR = (
    ROOT / "reports" / "research" / "qqq_leaps_gap" / "current_best"
)
TEMPLATE = Path(__file__).with_name("qqq_leaps_gap_report_template.html")

WINDOW_START = date(2020, 1, 3)
WINDOW_END = date(2025, 6, 5)

CLAIMED_COMBINED = {
    "n": 80,
    "win_rate_pct": 96.0,
    "total_pnl": 180000.0,
    "pf": None,
    "avg_days": 102.0,
    "max_dd": 9330.0,
}

DEFAULTS = {
    "capital": 100_000.0,
    "max_open": 3,
    "require_cash": True,
}

RULES = [
    "Asset: QQQ only. Buy ~60-delta calls with ~12 months to expiration (LEAPS).",
    "Strat1: open gaps down >= 1% AND prior close above SMA100; buy 1 contract.",
    "Strat2: open gaps down >= 2%; buy 2 contracts.",
    "Exit: take profit at +50% of premium paid; no stop-loss.",
    "Theta marks: entry = first nonzero 1m NBBO after 09:30 ET; path = EOD bid/ask.",
    "mid/mid uses mid marks both sides; ask/bid pays ask and exits on bid (TP needs bid >= 1.5x ask).",
    "Realistic book: account size + max concurrent + cash check (debit = premium x 100 x contracts).",
    "Equity marks opens at cost; steps on exits. Orange line = SPY buy-and-hold with same starting capital.",
    "Profit factor = gross winning $ / gross losing $ on kept trades (not equity-curve PF).",
    "Video risk note: cap concurrent opens at 3. Source: Options With Ravish YouTube dY2ralsWdQ8.",
]


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _num(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_trades_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "skipped" in df.columns:
        df = df.loc[~df["skipped"].astype(bool)]
    df = df.dropna(subset=["pnl_cash", "entry_date", "exit_date"])
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        prem = _num(r.get("entry_premium")) or 0.0
        ctr = int(_num(r.get("contracts")) or 1)
        debit = round(prem * 100.0 * ctr, 2)
        rows.append(
            {
                "mode": str(r.get("mode") or ""),
                "entry_date": str(r["entry_date"])[:10],
                "exit_date": str(r["exit_date"])[:10],
                "contracts": ctr,
                "entry_px": _num(r.get("entry_px")),
                "strike": _num(r.get("strike")),
                "entry_premium": prem,
                "exit_premium": _num(r.get("exit_premium")),
                "entry_delta": _num(r.get("entry_delta")),
                "dte": int(_num(r.get("dte")) or 0),
                "hold_days": int(_num(r.get("hold_days")) or 0),
                "exit_reason": str(r.get("exit_reason") or ""),
                "pnl_cash": _num(r.get("pnl_cash")),
                "gain_pct": _num(r.get("gain_pct")),
                "debit_cash": debit,
            }
        )
    return rows


def summary_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    fake = [
        LeapTrade(
            mode=t["mode"],
            entry_date=datetime.strptime(t["entry_date"], "%Y-%m-%d").date(),
            exit_date=datetime.strptime(t["exit_date"], "%Y-%m-%d").date(),
            contracts=t["contracts"],
            entry_px=t["entry_px"] or 0.0,
            strike=t["strike"] or 0.0,
            entry_premium=t["entry_premium"] or 0.0,
            exit_premium=t["exit_premium"],
            entry_delta=t["entry_delta"] or 0.0,
            entry_iv=0.0,
            dte=t["dte"],
            hold_days=t["hold_days"],
            exit_reason=t["exit_reason"],
            pnl_cash=t["pnl_cash"],
            gain_pct=t["gain_pct"],
            skipped=False,
        )
        for t in trades
    ]
    return book_stats(fake)


def load_spy_bh(start: date, end: date) -> Dict[str, Any]:
    """Daily SPY closes + buy-and-hold equity ratio (base=1 at first bar)."""
    frames = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="ALPACA",
        start=datetime(start.year, start.month, start.day),
        end=datetime(end.year, end.month, end.day),
        use_cache=True,
        workers=1,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    raw = frames.get("SPY")
    if raw is None or raw.empty:
        raise RuntimeError("No SPY daily bars for buy-and-hold overlay")
    df = raw.copy()
    if "close" not in df.columns:
        raise RuntimeError("SPY frame missing close")
    # Normalize index to date strings
    idx = pd.to_datetime(df.index)
    closes = pd.to_numeric(df["close"], errors="coerce").astype(float)
    closes.index = idx
    closes = closes.dropna().sort_index()
    # Clip to window
    closes = closes.loc[
        (closes.index.date >= start) & (closes.index.date <= end)
    ]
    if closes.empty:
        raise RuntimeError("SPY closes empty after window clip")
    base = float(closes.iloc[0])
    daily = []
    for ts, px in closes.items():
        d = pd.Timestamp(ts).strftime("%Y-%m-%d")
        daily.append(
            {
                "date": d,
                "close": round(float(px), 4),
                "equity": round(float(px) / base, 6),
            }
        )
    # Month-start samples + final bar for lighter SPY overlay alignment
    monthly: List[Dict[str, Any]] = []
    last_m = None
    for row in daily:
        m = row["date"][:7]
        if m != last_m:
            monthly.append(row)
            last_m = m
    if daily and monthly and monthly[-1]["date"] != daily[-1]["date"]:
        monthly.append(daily[-1])

    LOG.info(
        "SPY BH: %d daily bars %s -> %s  base=%.2f last_eq=%.3f",
        len(daily),
        daily[0]["date"],
        daily[-1]["date"],
        base,
        daily[-1]["equity"],
    )
    return {
        "symbol": "SPY",
        "start": daily[0]["date"],
        "end": daily[-1]["date"],
        "base_close": round(base, 4),
        "final_equity": daily[-1]["equity"],
        "daily": daily,
        "monthly": monthly,
    }


def build_payload(
    books: Dict[str, Dict[str, Any]],
    *,
    default_book: str,
    spy_bh: Dict[str, Any],
    capital: float,
    max_open: int,
) -> Dict[str, Any]:
    return {
        "claimed_combined": CLAIMED_COMBINED,
        "defaults": {
            "capital": capital,
            "max_open": max_open,
            "require_cash": DEFAULTS["require_cash"],
            "book": default_book,
            "mode": "all",
        },
        "default_book": default_book,
        "books": books,
        "spy_bh": spy_bh,
        "rules": RULES,
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window": f"{WINDOW_START.isoformat()} -> {WINDOW_END.isoformat()}",
            "underlying": "QQQ",
            "benchmark": "SPY buy & hold",
            "marks": "ThetaData Standard NBBO + EOD",
            "sizing": "debit = premium x 100 x contracts; skip if cash or max-open blocks",
            "pf_def": "gross winning $ / gross losing $ on kept trades",
            "verdict": "no promote",
            "status": "docs/status_log/edge_hunt/2026-09-30_qqq_leaps_gap_theta_claimcheck.md",
            "harness": "scripts/research/claimcheck_qqq_leaps_gap_theta.py",
            "report": "scripts/research/generate_qqq_leaps_gap_report.py",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="QQQ LEAPS gap HTML report")
    ap.add_argument("--theta-dir", type=Path, default=DEFAULT_THETA)
    ap.add_argument("--askbid-dir", type=Path, default=DEFAULT_ASKBID)
    ap.add_argument("--askbid-uncapped-dir", type=Path, default=DEFAULT_ASKBID_UNC)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--capital", type=float, default=DEFAULTS["capital"])
    ap.add_argument("--max-open", type=int, default=DEFAULTS["max_open"])
    ap.add_argument(
        "--title",
        default="QQQ LEAPS Gap (Theta) — Capital vs SPY buy&hold",
    )
    args = ap.parse_args()

    sources = [
        (
            "mid_mid_uncapped",
            "Mid/mid uncapped pool (apply max-open / cash in UI)",
            args.theta_dir / "mid_mid_uncapped_combined.csv",
        ),
        (
            "mid_mid_cap3",
            "Mid/mid pre-capped max-open 3",
            args.theta_dir / "mid_mid_cap3_combined.csv",
        ),
        (
            "ask_bid_uncapped",
            "Ask/bid uncapped pool (marketable)",
            args.askbid_uncapped_dir / "ask_bid_capNone_combined.csv",
        ),
        (
            "ask_bid_cap3",
            "Ask/bid pre-capped max-open 3",
            args.askbid_dir / "ask_bid_cap3_combined.csv",
        ),
    ]

    books: Dict[str, Dict[str, Any]] = {}
    for key, label, path in sources:
        if not path.exists():
            LOG.warning("Missing %s — skip book %s", path, key)
            continue
        trades = load_trades_csv(path)
        summary = summary_from_trades(trades)
        books[key] = {
            "label": label,
            "trades": trades,
            "summary": summary,
            "source": str(path),
        }
        LOG.info(
            "%s: n=%s WR=%s PnL=%s from %s",
            key,
            summary.get("n"),
            summary.get("win_rate_pct"),
            summary.get("total_pnl"),
            path.name,
        )

    if not books:
        LOG.error("No trade CSVs found")
        return 1

    default_book = (
        "mid_mid_uncapped" if "mid_mid_uncapped" in books else next(iter(books))
    )
    spy_bh = load_spy_bh(WINDOW_START, WINDOW_END)
    payload = build_payload(
        books,
        default_book=default_book,
        spy_bh=spy_bh,
        capital=float(args.capital),
        max_open=int(args.max_open),
    )
    raw = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__TITLE__", _esc(args.title))
    html = html.replace("__DATA_JSON__", raw)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "qqq_leaps_gap_theta.html"
    out.write_text(html, encoding="utf-8")
    LOG.info("Wrote %s (%.1f KB)", out, out.stat().st_size / 1024)

    dated = args.theta_dir / "qqq_leaps_gap_theta.html"
    dated.write_text(html, encoding="utf-8")
    LOG.info("Also wrote %s", dated)

    readme = outdir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# QQQ LEAPS gap — current_best HTML",
                "",
                f"- Report: [`qqq_leaps_gap_theta.html`](qqq_leaps_gap_theta.html)",
                "- Defaults: **$100k** account, **max-open 3**, cash debit check, SPY buy&hold overlay.",
                "- Default trade pool: mid/mid uncapped (UI applies concurrency / cash).",
                "- Verdict: **no promote**.",
                "- Status: `docs/status_log/edge_hunt/2026-09-30_qqq_leaps_gap_theta_claimcheck.md`",
                "",
                "Regenerate:",
                "",
                "```bat",
                "venv\\Scripts\\activate",
                "set PYTHONPATH=.",
                "python scripts\\research\\generate_qqq_leaps_gap_report.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
