#!/usr/bin/env python3
"""Deep Pullback SPY claim-check HTML report (Quantified Strategies rules).

Entry (buy at close) when all true:
  1) close is lowest close of last 15 sessions
  2) 1-day %% return is lowest of last 10 sessions
  3) close > SMA200
Exit (QS): sell at close when close > prior day's high.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_deep_pullback_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_deep_pullback_report")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "deep_pullback"
YAHOO_CACHE = ROOT / "reports" / "research" / "spy_yahoo.json"

CLAIMED: Dict[str, Any] = {
    "n": 128,
    "win_rate_pct": 78.0,
    "profit_factor": 3.0,
    "avg_gain_pct": 0.8,
    "mdd_pct": 14.0,
    "cagr_pct": 3.0,
    "exposure_pct": 5.0,
    "risk_adj_return_pct": 52.0,
    "source": "Quantified Strategies video eV8YhMUpk3Q / deep-pullback article",
}


def download_yahoo(path: Path = YAHOO_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
        "?period1=727876800&period2=9999999999&interval=1d&events=history"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    logger.info("Downloaded Yahoo SPY -> %s (%d bytes)", path, len(data))
    return path


def load_yahoo(path: Path = YAHOO_CACHE, refresh: bool = False) -> pd.DataFrame:
    if refresh or not path.exists() or path.stat().st_size < 10_000:
        download_yahoo(path)
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
        df.dropna(subset=["close", "high"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    logger.info(
        "Yahoo SPY: %d bars  %s -> %s",
        len(df),
        df["ts"].iloc[0].date(),
        df["ts"].iloc[-1].date(),
    )
    return df


def backtest(df: pd.DataFrame, friction_pct: float = 0.0) -> Dict[str, Any]:
    d = df.copy()
    d["ret1"] = d["close"].pct_change()
    d["sma200"] = d["close"].rolling(200, min_periods=200).mean()
    d["min_close_15"] = d["close"].rolling(15, min_periods=15).min()
    d["min_ret_10"] = d["ret1"].rolling(10, min_periods=10).min()
    d["prior_high"] = d["high"].shift(1)

    cond1 = d["close"] <= d["min_close_15"]
    cond2 = d["ret1"] <= d["min_ret_10"]
    cond3 = d["close"] > d["sma200"]
    d["entry_signal"] = cond1 & cond2 & cond3 & d["sma200"].notna()

    trades: List[Dict[str, Any]] = []
    in_pos = False
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None

    for i in range(len(d)):
        row = d.iloc[i]
        if not in_pos:
            if bool(row["entry_signal"]):
                in_pos = True
                entry_i = i
                entry_px = float(row["close"])
        else:
            assert entry_i is not None and entry_px is not None
            if pd.notna(row["prior_high"]) and float(row["close"]) > float(row["prior_high"]):
                exit_px = float(row["close"])
                ret = (exit_px / entry_px - 1.0) * 100.0 - friction_pct
                trades.append(
                    {
                        "symbol": "SPY",
                        "entry_date": str(d.iloc[entry_i]["ts"].date()),
                        "exit_date": str(row["ts"].date()),
                        "entry_px": round(entry_px, 4),
                        "exit_px": round(exit_px, 4),
                        "ret_pct": round(ret, 4),
                        "hold_days": int(i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_px = None

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    if n == 0:
        return {
            "n": 0,
            "trades": [],
            "by_year": [],
            "equity_monthly": [],
            "friction_pct": friction_pct,
        }

    gains = tdf["ret_pct"].to_numpy(dtype=float)
    wins = gains[gains > 0]
    losses = gains[gains <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = [1.0]
    for r in gains:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    eq = np.array(equity)

    hold_sum = int(tdf["hold_days"].sum())
    start_i = 199
    calendar_days = max(len(d) - start_i, 1)
    exposure = hold_sum / calendar_days

    full_start = d.iloc[start_i]["ts"]
    full_end = d.iloc[-1]["ts"]
    full_years = max((full_end - full_start).days / 365.25, 1e-9)
    cagr_full = (eq[-1] ** (1.0 / full_years) - 1.0) * 100.0
    risk_adj = cagr_full / exposure if exposure > 0 else None

    bh0 = float(d.iloc[start_i]["close"])
    bh1 = float(d.iloc[-1]["close"])
    bh_cagr = ((bh1 / bh0) ** (1.0 / full_years) - 1.0) * 100.0

    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    by_year: List[Dict[str, Any]] = []
    for y, g in tdf.groupby("year"):
        gy = g["ret_pct"].to_numpy(dtype=float)
        w = float(gy[gy > 0].sum())
        l = float(-gy[gy <= 0].sum())
        by_year.append(
            {
                "year": int(y),
                "n": int(len(g)),
                "E": round(float(gy.mean()), 3),
                "WR": round(float((gy > 0).mean() * 100), 1),
                "PF": None if l <= 0 else round(w / l, 2),
                "sum_ret": round(float(gy.sum()), 2),
            }
        )

    # Monthly MTM equity for chart
    equity_monthly: List[Dict[str, Any]] = []
    pos = False
    e_px: Optional[float] = None
    eq_mtm = 1.0
    peak_mtm = 1.0
    mdd_mtm = 0.0
    last_month = None
    for i in range(len(d)):
        row = d.iloc[i]
        if not pos and bool(row["entry_signal"]):
            pos = True
            e_px = float(row["close"])
        elif (
            pos
            and e_px is not None
            and pd.notna(row["prior_high"])
            and float(row["close"]) > float(row["prior_high"])
        ):
            r = (float(row["close"]) / e_px - 1.0) - friction_pct / 100.0
            eq_mtm *= 1.0 + r
            pos = False
            e_px = None
        mtm = eq_mtm * (float(row["close"]) / e_px) if (pos and e_px) else eq_mtm
        peak_mtm = max(peak_mtm, mtm)
        mdd_mtm = min(mdd_mtm, mtm / peak_mtm - 1.0)
        month = (int(row["ts"].year), int(row["ts"].month))
        if i >= start_i and (month != last_month or i == len(d) - 1):
            equity_monthly.append(
                {"date": str(row["ts"].date()), "equity": round(float(mtm), 6)}
            )
            last_month = month

    # Buy & hold index aligned to monthly sample (from warmup)
    bh_monthly: List[Dict[str, Any]] = []
    last_month = None
    for i in range(start_i, len(d)):
        row = d.iloc[i]
        month = (int(row["ts"].year), int(row["ts"].month))
        if month != last_month or i == len(d) - 1:
            bh_monthly.append(
                {
                    "date": str(row["ts"].date()),
                    "equity": round(float(row["close"]) / bh0, 6),
                }
            )
            last_month = month

    return {
        "n": n,
        "n_entry_signals": int(d["entry_signal"].sum()),
        "win_rate_pct": round(float((gains > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3),
        "avg_gain_pct": round(float(gains.mean()), 3),
        "median_gain_pct": round(float(np.median(gains)), 3),
        "mdd_mtm_pct": round(float(mdd_mtm * 100), 2),
        "cagr_full_window_pct": round(float(cagr_full), 2),
        "spy_bh_cagr_pct": round(float(bh_cagr), 2),
        "exposure_pct": round(exposure * 100, 2),
        "risk_adj_return_pct": None if risk_adj is None else round(float(risk_adj), 1),
        "avg_hold_days": round(float(tdf["hold_days"].mean()), 2),
        "final_equity": round(float(eq[-1]), 4),
        "first_bar": str(d["ts"].iloc[0].date()),
        "last_bar": str(d["ts"].iloc[-1].date()),
        "first_trade": tdf["entry_date"].iloc[0],
        "last_trade": tdf["exit_date"].iloc[-1],
        "friction_pct": friction_pct,
        "by_year": by_year,
        "trades": trades,
        "equity_monthly": equity_monthly,
        "bh_monthly": bh_monthly,
    }


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(
    *,
    primary: Dict[str, Any],
    friction_rows: List[Dict[str, Any]],
    claimed: Dict[str, Any],
    title: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> str:
    if defaults is None:
        defaults = {
            "capital": 100_000.0,
            "sizeMode": "fixed",
            "sizeVal": 100_000.0,
            "friction": 0.0,
            "maxPerDay": 0,
            "maxOpen": 0,
            "winCap": 0,
            "excludeSym": "",
        }

    trades_js: List[Dict[str, Any]] = []
    for i, t in enumerate(primary["trades"]):
        trades_js.append(
            {
                "ord": i,
                "symbol": t.get("symbol", "SPY"),
                "buy": t["entry_date"],
                "sell": t["exit_date"],
                "buy_at": t["entry_date"],
                "sell_at": t["exit_date"],
                "gain": t["ret_pct"],
                "entry_px": t["entry_px"],
                "exit_px": t["exit_px"],
                "bars": t["hold_days"],
                "hold_days": t["hold_days"],
            }
        )

    payload = {
        "claimed": claimed,
        "primary": {k: v for k, v in primary.items() if k not in ("trades",)},
        "trades": trades_js,
        "defaults": defaults,
        "friction": friction_rows,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rules": [
            "Buy at close when close is lowest of last 15 sessions",
            "AND 1-day % return is lowest of last 10 sessions",
            "AND close > SMA200",
            "Sell at close when close > prior day's high (QS exit)",
        ],
    }
    raw = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    template_path = Path(__file__).with_name("deep_pullback_report_template.html")
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__TITLE__", _esc(title))
    html = html.replace("__DATA_JSON__", raw)
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep Pullback SPY HTML report")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--start", default="1993-01-01", help="Inclusive start date filter")
    ap.add_argument("--friction-pct", type=float, default=0.0)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument(
        "--size-mode",
        choices=("fixed", "pct_initial", "pct_equity"),
        default="fixed",
    )
    ap.add_argument("--notional", type=float, default=100_000.0)
    ap.add_argument("--size-pct", type=float, default=100.0)
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    yahoo = load_yahoo(refresh=args.refresh_yahoo)
    df = yahoo[yahoo["ts"] >= args.start].reset_index(drop=True)

    primary = backtest(df, friction_pct=0.0)
    frictions = [0.0, 0.10, 0.25]
    friction_results = [backtest(df, friction_pct=f) for f in frictions]
    friction_rows = [
        {
            "friction_pct": r["friction_pct"],
            "n": r["n"],
            "avg_gain_pct": r["avg_gain_pct"],
            "win_rate_pct": r["win_rate_pct"],
            "profit_factor": r["profit_factor"],
            "cagr_full_window_pct": r["cagr_full_window_pct"],
            "mdd_mtm_pct": r["mdd_mtm_pct"],
            "exposure_pct": r["exposure_pct"],
        }
        for r in friction_results
    ]

    size_val = (
        float(args.notional)
        if args.size_mode == "fixed"
        else float(args.size_pct)
    )
    defaults = {
        "capital": float(args.initial_capital),
        "sizeMode": args.size_mode,
        "sizeVal": size_val,
        "friction": float(args.friction_pct),
        "maxPerDay": 0,
        "maxOpen": 0,
        "winCap": 0,
        "excludeSym": "",
    }

    title = "Deep Pullback Strategy — SPY claim check"
    html = render_html(
        primary=primary,
        friction_rows=friction_rows,
        claimed=CLAIMED,
        title=title,
        defaults=defaults,
    )
    html_path = outdir / "deep_pullback_spy.html"
    html_path.write_text(html, encoding="utf-8")

    trades_path = outdir / "deep_pullback_spy_trades.csv"
    pd.DataFrame(primary["trades"]).to_csv(trades_path, index=False)

    summary = {
        "claimed": CLAIMED,
        "yahoo_full": {k: v for k, v in primary.items() if k not in ("trades",)},
        "friction": friction_rows,
        "defaults": defaults,
        "html": str(html_path),
        "trades_csv": str(trades_path),
    }
    summary["yahoo_full"]["n_trades"] = primary["n"]
    json_path = outdir / "deep_pullback_spy_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    full_path = outdir / "deep_pullback_spy_full.json"
    full_path.write_text(
        json.dumps(
            {
                "claimed": CLAIMED,
                "primary": primary,
                "friction": friction_rows,
                "defaults": defaults,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "n=%s WR=%.2f PF=%.3f E=%.3f MDD=%.2f CAGR=%.2f exp=%.2f%%",
        primary["n"],
        primary["win_rate_pct"],
        primary["profit_factor"],
        primary["avg_gain_pct"],
        primary["mdd_mtm_pct"],
        primary["cagr_full_window_pct"],
        primary["exposure_pct"],
    )
    logger.info("Wrote %s", html_path)
    logger.info("Wrote %s", trades_path)
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
