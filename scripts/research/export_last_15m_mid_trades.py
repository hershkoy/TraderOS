"""Build trades CSV + summary from last_15m_open_mid_compare.csv for the TV HTML report."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("export_last_15m_mid_trades")
ET = ZoneInfo("America/New_York")

COMPARE = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "last_15m_open_mid_compare.csv"
)
SOURCE = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-09"
    / "channel_touch_full_h2_break_span365_unique_20260909_000344.csv"
)
TRADES_NAME = "channel_touch_h2_last_15m_open_mid_span365.csv"


def _et_to_utc_clock(text: object) -> str:
    s = str(text or "").strip()
    if not s or s.lower() == "nan":
        return ""
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize(ET)
    else:
        t = t.tz_convert(ET)
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def _profit_factor(gains: pd.Series) -> float:
    wins = gains[gains > 0].sum()
    losses = (-gains[gains < 0]).sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", type=Path, default=COMPARE)
    ap.add_argument("--before", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    cmp = pd.read_csv(args.compare)
    src = pd.read_csv(args.before)
    matched = cmp[cmp["row_kind"] == "matched"].copy()
    skipped = cmp[cmp["row_kind"] == "before_skipped"]
    src["stock"] = src["stock"].astype(str).str.upper()
    src["buy_date"] = src["buy_date"].astype(str)
    src["sell_date"] = src["sell_date"].astype(str)
    src["buy_price_r"] = pd.to_numeric(src["buy_price"], errors="coerce").round(4)
    src["sell_price_r"] = pd.to_numeric(src["sell_price"], errors="coerce").round(4)
    m = matched.copy()
    m["stock"] = m["stock"].astype(str).str.upper()
    m["buy_date"] = m["buy_date"].astype(str)
    m["sell_date"] = m["sell_datetime"].astype(str).str.slice(0, 10)
    m["buy_price_r"] = pd.to_numeric(m["buy_price_before"], errors="coerce").round(4)
    m["sell_price_r"] = pd.to_numeric(m["sell_price"], errors="coerce").round(4)
    merged = m.merge(
        src,
        on=["stock", "buy_date", "buy_price_r", "sell_date", "sell_price_r"],
        how="left",
        suffixes=("", "_src"),
    )
    rows = []
    feat_cols = (
        "last_15m_open",
        "open_above_rail_pct",
        "atr_15m",
        "atr_15m_pct",
        "atr_1d_pct",
        "signal_x",
    )
    for _, row in merged.iterrows():
        rec = {c: row[c] for c in src.columns if c in row.index and not str(c).endswith("_r")}
        rec["stock"] = str(row["stock"]).upper()
        rec["buy_date"] = str(row["buy_date"])
        rec["buy_price"] = float(row["buy_price_after"])
        rec["sell_date"] = str(row["sell_date"])
        rec["sell_price"] = float(row["sell_price"])
        rec["gain_pct"] = float(row["gain_after"])
        rec["buy_time"] = _et_to_utc_clock(row.get("buy_datetime_after"))
        for col in feat_cols:
            if col in row.index and pd.notna(row[col]):
                rec[col] = row[col]
        rows.append(rec)
    trades = pd.DataFrame(rows)
    dated = Path(args.outdir) if args.outdir is not None else dated_outdir()
    dated.mkdir(parents=True, exist_ok=True)
    trades_path = dated / TRADES_NAME
    trades.to_csv(trades_path, index=False)
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    summary_path = dated / (Path(TRADES_NAME).stem + "_summary.txt")
    lines = [
        "H2 resistance-break unique-symbol/day (span<=365)",
        "realistic_fill=True",
        "fill=last RTH 15m mid if that bar opened above the rail",
        "Note: Before book is unique signal-close (lookback). Sells kept.",
        "Note: Open of last 15m known 15:45 ET; mid known 16:00 with the daily close.",
        "Note: Occupancy not re-walked.",
        "",
        "n_trades=%d" % len(trades),
        "n_skipped=%d" % len(skipped),
        "win_rate_pct=%.2f" % float((g > 0).mean() * 100.0),
        "avg_gain_pct=%.2f" % float(g.mean()),
        "median_gain_pct=%.2f" % float(g.median()),
        "expectancy_pct=%.2f" % float(g.mean()),
        "profit_factor=%.3f" % _profit_factor(g),
        "",
        "Exit: sells from the signal-close unique book (unchanged)",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Wrote %s n=%d", trades_path, len(trades))
    print("trades:", trades_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
