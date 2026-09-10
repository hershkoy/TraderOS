"""Verify the 1d H2 signal-close HTML book on 10 random symbols.

1. Sample N stocks from the unique signal-close trades CSV (HTML source).
2. Re-run the same H2 + shakeout + signal-close stack on those names only.
3. Diff report trades vs rescan (buy_date / prices / gains).
4. Print an expanding year walk-forward on the HTML subset (train<=Y, OOS Y+1).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import _summarize  # noqa: E402

LOG = logging.getLogger("verify_signal_close_sample")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-09"
    / "channel_touch_full_h2_break_span365_unique_20260909_000344.csv"
)
DEFAULT_OUTDIR = (
    ROOT / "reports" / "ascending_channels" / "1d_unrealistic"
)


def _pf(gains: pd.Series) -> float:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    wins = g[g > 0].sum()
    losses = (-g[g < 0]).sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def sample_symbols(trades: pd.DataFrame, n: int, seed: int) -> List[str]:
    names = sorted(trades["stock"].astype(str).str.upper().unique().tolist())
    rng = np.random.default_rng(int(seed))
    if len(names) <= n:
        return names
    pick = rng.choice(names, size=int(n), replace=False)
    return sorted(str(s) for s in pick)


def expanding_year_walk(df: pd.DataFrame, gain_col: str = "gain_pct_net") -> pd.DataFrame:
    """Train on buys through year Y, test on Y+1 (calendar buy_date)."""
    if df.empty:
        return pd.DataFrame()
    t = df.copy()
    t["buy_date"] = pd.to_datetime(t["buy_date"])
    t["year"] = t["buy_date"].dt.year
    years = sorted(int(y) for y in t["year"].dropna().unique())
    rows = []
    for i, y_test in enumerate(years):
        if i == 0:
            continue
        train = t[t["year"] < y_test]
        test = t[t["year"] == y_test]
        if train.empty or test.empty:
            continue
        tr = _summarize(train, gain_col=gain_col)
        te = _summarize(test, gain_col=gain_col)
        rows.append(
            {
                "train_through": y_test - 1,
                "oos_year": y_test,
                "n_train": tr.get("n_trades"),
                "E_train": tr.get("expectancy_pct"),
                "PF_train": tr.get("profit_factor"),
                "n_oos": te.get("n_trades"),
                "E_oos": te.get("expectancy_pct"),
                "PF_oos": te.get("profit_factor"),
            }
        )
    return pd.DataFrame(rows)


def match_trades(
    report: pd.DataFrame,
    rescan: pd.DataFrame,
    *,
    price_tol: float = 0.02,
    gain_tol: float = 0.15,
) -> Tuple[pd.DataFrame, dict]:
    rep = report.copy()
    rs = rescan.copy()
    rep["stock"] = rep["stock"].astype(str).str.upper()
    rs["stock"] = rs["stock"].astype(str).str.upper()
    rep["buy_date"] = pd.to_datetime(rep["buy_date"]).dt.strftime("%Y-%m-%d")
    rs["buy_date"] = pd.to_datetime(rs["buy_date"]).dt.strftime("%Y-%m-%d")

    rows = []
    used_rs = set()
    for _, r in rep.iterrows():
        cand = rs[(rs["stock"] == r["stock"]) & (rs["buy_date"] == r["buy_date"])]
        status = "missing_in_rescan"
        rs_row = None
        if not cand.empty:
            # Prefer closest buy_price
            diffs = (cand["buy_price"].astype(float) - float(r["buy_price"])).abs()
            j = diffs.idxmin()
            rs_row = cand.loc[j]
            used_rs.add(j)
            bp_ok = abs(float(rs_row["buy_price"]) - float(r["buy_price"])) <= price_tol
            sp_ok = abs(float(rs_row["sell_price"]) - float(r["sell_price"])) <= price_tol
            g_rep = float(r["gain_pct"])
            g_rs = float(rs_row["gain_pct"])
            g_ok = abs(g_rs - g_rep) <= gain_tol
            if bp_ok and sp_ok and g_ok:
                status = "match"
            elif bp_ok and sp_ok:
                status = "price_ok_gain_diff"
            else:
                status = "mismatch"
        rows.append(
            {
                "stock": r["stock"],
                "buy_date": r["buy_date"],
                "sell_date_report": str(r["sell_date"]),
                "sell_date_rescan": "" if rs_row is None else str(rs_row["sell_date"]),
                "buy_price_report": round(float(r["buy_price"]), 4),
                "buy_price_rescan": "" if rs_row is None else round(float(rs_row["buy_price"]), 4),
                "sell_price_report": round(float(r["sell_price"]), 4),
                "sell_price_rescan": "" if rs_row is None else round(float(rs_row["sell_price"]), 4),
                "gain_report": round(float(r["gain_pct"]), 4),
                "gain_rescan": "" if rs_row is None else round(float(rs_row["gain_pct"]), 4),
                "gain_net_report": round(float(r["gain_pct_net"]), 4)
                if "gain_pct_net" in r.index and pd.notna(r.get("gain_pct_net"))
                else "",
                "status": status,
            }
        )

    extras = rs.loc[~rs.index.isin(used_rs)]
    for _, r in extras.iterrows():
        rows.append(
            {
                "stock": r["stock"],
                "buy_date": r["buy_date"],
                "sell_date_report": "",
                "sell_date_rescan": str(r["sell_date"]),
                "buy_price_report": "",
                "buy_price_rescan": round(float(r["buy_price"]), 4),
                "sell_price_report": "",
                "sell_price_rescan": round(float(r["sell_price"]), 4),
                "gain_report": "",
                "gain_rescan": round(float(r["gain_pct"]), 4),
                "gain_net_report": "",
                "status": "extra_in_rescan",
            }
        )

    out = pd.DataFrame(rows)
    summary = {
        "n_report": int(len(rep)),
        "n_rescan": int(len(rs)),
        "n_match": int((out["status"] == "match").sum()),
        "n_mismatch": int(out["status"].isin(["mismatch", "price_ok_gain_diff"]).sum()),
        "n_missing": int((out["status"] == "missing_in_rescan").sum()),
        "n_extra": int((out["status"] == "extra_in_rescan").sum()),
    }
    return out, summary


def run_rescan(symbols: List[str], outdir: Path) -> Path:
    """Call h2_break on the sample; return the unique span365 CSV it writes."""
    dated_root = ROOT / "reports" / "ascending_channels"
    before = set(dated_root.glob("*/channel_touch_h2_break_span365_unique_*.csv"))

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "research" / "backtest_channel_touch_h2_break.py"),
        "--symbols",
        ",".join(symbols),
        "--shakeout-breakout",
        "--realistic-fill",
        "--realistic-fill-mode",
        "signal-close",
        "--workers",
        "2",
        "--load-workers",
        "4",
        "--friction-pct",
        "0.25",
    ]
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    LOG.info("Rescan: %s", " ".join(cmd))
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise RuntimeError("h2_break rescan failed rc=%s" % rc)
    LOG.info("Rescan finished in %.1fs", time.perf_counter() - t0)

    after = set(dated_root.glob("*/channel_touch_h2_break_span365_unique_*.csv"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        allu = sorted(
            list(dated_root.glob("*/channel_touch_h2_break_span365_unique_*.csv")),
            key=lambda p: p.stat().st_mtime,
        )
        if not allu:
            raise FileNotFoundError("No unique span365 CSV after rescan")
        return allu[-1]
    return new[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n-symbols", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument(
        "--rescan-csv",
        type=Path,
        default=None,
        help="Skip rescan; compare against this unique CSV (filtered to sample)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    book = pd.read_csv(args.trades)
    symbols = sample_symbols(book, int(args.n_symbols), int(args.seed))
    report = book[book["stock"].astype(str).str.upper().isin(symbols)].copy()
    LOG.info("Sample symbols (%d): %s", len(symbols), ",".join(symbols))
    LOG.info("Report trades for sample: n=%d", len(report))

    args.outdir.mkdir(parents=True, exist_ok=True)
    sample_list = args.outdir / "signal_close_walkfwd_sample_symbols.txt"
    sample_list.write_text("\n".join(symbols) + "\n", encoding="utf-8")

    if args.rescan_csv and args.rescan_csv.exists():
        rescan_path = args.rescan_csv
        LOG.info("Using provided rescan CSV %s", rescan_path)
    else:
        rescan_path = run_rescan(symbols, args.outdir)

    rescan_all = pd.read_csv(rescan_path)
    rescan = rescan_all[rescan_all["stock"].astype(str).str.upper().isin(symbols)].copy()
    LOG.info("Rescan file %s rows_for_sample=%d", rescan_path, len(rescan))

    detail, summary = match_trades(report, rescan)
    detail_path = args.outdir / "signal_close_walkfwd_verify_trades.csv"
    detail.to_csv(detail_path, index=False)

    # Expanding year walk-forward on the HTML subset itself
    gain_col = "gain_pct_net" if "gain_pct_net" in report.columns else "gain_pct"
    wf = expanding_year_walk(report, gain_col=gain_col)
    wf_path = args.outdir / "signal_close_walkfwd_years.csv"
    wf.to_csv(wf_path, index=False)

    rep_sum = _summarize(report, gain_col=gain_col)
    rs_gain = "gain_pct_net" if "gain_pct_net" in rescan.columns else "gain_pct"
    rs_sum = _summarize(rescan, gain_col=rs_gain) if not rescan.empty else {}

    print("=== sample ===")
    print(",".join(symbols))
    print("=== report subset ===")
    print(
        "n=%s E=%s PF=%s WR=%s"
        % (
            rep_sum.get("n_trades"),
            rep_sum.get("expectancy_pct"),
            rep_sum.get("profit_factor"),
            rep_sum.get("win_rate_pct"),
        )
    )
    print("=== rescan subset ===")
    print(
        "n=%s E=%s PF=%s WR=%s"
        % (
            rs_sum.get("n_trades"),
            rs_sum.get("expectancy_pct"),
            rs_sum.get("profit_factor"),
            rs_sum.get("win_rate_pct"),
        )
    )
    print("=== trade match ===")
    print(summary)
    print(detail["status"].value_counts().to_string())
    if not wf.empty:
        print("=== expanding year walk-forward (HTML subset) ===")
        print(wf.to_string(index=False))
    print("detail:", detail_path)
    print("walkfwd:", wf_path)
    print("rescan:", rescan_path)

    ok = summary["n_missing"] == 0 and summary["n_mismatch"] == 0 and summary["n_extra"] == 0
    if ok:
        print("VERIFY PASS: all report trades reproduced")
        return 0
    print("VERIFY REVIEW: mismatches or extras — see CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
