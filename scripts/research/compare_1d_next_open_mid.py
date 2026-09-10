"""Compare 1d_unrealistic next-mid fills vs next-session 09:30 ET 15m mid.

Before book: reports/ascending_channels/1d_unrealistic/delay_15m_mid_compare.csv
(next-mid lookback buy_price_before / gain_before).

After: EOD daily close-above-resist is known at 16:00 ET; buy the **mid** of
the next RTH session's first 15m (09:30). Sell kept from before when matched.

This fill is live-executable. Same columns as signal_close_compare.csv.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from compare_1d_unrealistic_signal_close import (  # noqa: E402
    attach_buy_date,
    build_session_index,
    _fmt_ts,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.realistic_purchaser import (  # noqa: E402
    _as_session_date,
    as_et,
    purchase_next_session_open_mid,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("next_open_mid_compare")

DEFAULT_BEFORE = (
    ROOT / "reports" / "ascending_channels" / "1d_unrealistic" / "delay_15m_mid_compare.csv"
)
DEFAULT_SOURCE = ROOT / "reports" / "ascending_channels" / "channel_touch_h2_break_span365.csv"
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "next_open_mid_compare.csv"
)
COMPARE_COLS = [
    "row_kind",
    "stock",
    "buy_date",
    "buy_datetime_before",
    "buy_price_before",
    "buy_datetime_after",
    "buy_price_after",
    "sell_datetime",
    "sell_price",
    "gain_before",
    "gain_after",
    "gain_diff",
    "signal_x",
    "session_date",
    "status",
    "skip_reason",
    "in_full_signal_close",
]


def _utc_clock(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M")


def compare_before_rows(
    before: pd.DataFrame,
    indexed: Dict[str, Dict],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, tr in before.iterrows():
        sym = str(tr["stock"]).upper()
        buy_day = _as_session_date(tr.get("buy_date"), naive_tz="UTC")
        buy_before = float(tr["buy_price_before"])
        sell_px = float(tr["sell_price"])
        gain_before = float(tr["gain_before"])
        base = {
            "row_kind": "matched_attempt",
            "stock": sym,
            "buy_date": str(tr.get("buy_date") or ""),
            "buy_datetime_before": tr.get("buy_datetime_before") or "",
            "buy_price_before": round(buy_before, 4),
            "buy_datetime_after": "",
            "buy_price_after": "",
            "sell_datetime": str(tr["sell_datetime"]),
            "sell_price": round(sell_px, 4),
            "gain_before": round(gain_before, 4),
            "gain_after": "",
            "gain_diff": "",
            "signal_x": "",
            "session_date": "",
            "status": "",
            "skip_reason": "",
            "in_full_signal_close": False,
        }
        by_day = indexed.get(sym)
        if not by_day or buy_day is None:
            base["row_kind"] = "before_skipped"
            base["status"] = "skipped"
            base["skip_reason"] = "no_15m" if buy_day is not None else "no_buy_date"
            rows.append(base)
            continue

        got = purchase_next_session_open_mid(
            None,
            signal_session_date=buy_day,
            session_index=by_day,
        )
        if not got.filled or got.fill_px is None:
            base["row_kind"] = "before_skipped"
            base["status"] = "skipped"
            base["skip_reason"] = got.reason or "no_next_open"
            rows.append(base)
            continue

        px = float(got.fill_px)
        gain_after = (sell_px / px - 1.0) * 100.0
        sess = as_et(got.exec_bar_ts).date() if got.exec_bar_ts is not None else None
        base.update(
            {
                "row_kind": "matched",
                "buy_datetime_after": _fmt_ts(got.exec_bar_ts) if got.exec_bar_ts else "",
                "buy_price_after": round(px, 4),
                "gain_after": round(gain_after, 4),
                "gain_diff": round(gain_after - gain_before, 4),
                "session_date": str(sess) if sess is not None else "",
                "status": "ok",
                "skip_reason": "",
                "_buy_time_utc": _utc_clock(got.exec_bar_ts) if got.exec_bar_ts else "",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _profit_factor(gains: pd.Series) -> float:
    wins = gains[gains > 0].sum()
    losses = (-gains[gains < 0]).sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def trades_from_matched(matched: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Source rails + next-open-mid fill; sells kept from the before book."""
    src = source.copy()
    src["stock"] = src["stock"].astype(str).str.upper()
    src["buy_date"] = src["buy_date"].astype(str)
    src["sell_datetime"] = src["sell_date"].astype(str)
    src["buy_price_r"] = pd.to_numeric(src["buy_price"], errors="coerce").round(4)
    src["sell_price_r"] = pd.to_numeric(src["sell_price"], errors="coerce").round(4)

    m = matched.copy()
    m["stock"] = m["stock"].astype(str).str.upper()
    m["buy_date"] = m["buy_date"].astype(str)
    m["buy_price_r"] = pd.to_numeric(m["buy_price_before"], errors="coerce").round(4)
    m["sell_price_r"] = pd.to_numeric(m["sell_price"], errors="coerce").round(4)
    m["sell_datetime"] = m["sell_datetime"].astype(str)

    merged = m.merge(
        src,
        on=["stock", "buy_price_r", "sell_datetime", "sell_price_r"],
        how="left",
        suffixes=("", "_src"),
    )
    out = src.iloc[0:0].copy()
    rows = []
    for _, row in merged.iterrows():
        rec = {c: row[c] for c in src.columns if c in row.index}
        rec["stock"] = str(row["stock"]).upper()
        rec["buy_date"] = str(row.get("session_date") or row["buy_date"])
        rec["buy_price"] = float(row["buy_price_after"])
        rec["sell_date"] = str(row["sell_datetime"])[:10]
        rec["sell_price"] = float(row["sell_price"])
        rec["gain_pct"] = float(row["gain_after"])
        rec["buy_time"] = str(row.get("_buy_time_utc") or "")
        rec["sell_time"] = str(row["sell_datetime"])
        try:
            bd = pd.Timestamp(rec["buy_date"])
            sd = pd.Timestamp(rec["sell_date"])
            rec["hold_days"] = int((sd - bd).days)
        except (TypeError, ValueError):
            pass
        rows.append(rec)
    if not rows:
        return out
    return pd.DataFrame(rows)


def write_summary(path: Path, matched: pd.DataFrame, skipped: pd.DataFrame) -> None:
    n = len(matched)
    g = pd.to_numeric(matched["gain_after"], errors="coerce") if n else pd.Series(dtype=float)
    lines = [
        "H2 resistance-break unique-symbol/day (span<=365)",
        "realistic_fill=True",
        "realistic_fill_mode=next-open-mid",
        "Note: EOD daily close-above-resist, buy next session 09:30 ET 15m mid",
        "Note: Compare sells kept from the next-mid before book (not a full occupancy re-walk)",
        "Note: Occupancy: keep_one_per_symbol_day (earliest fill per name per calendar day)",
        "",
        "n_trades=%d" % n,
        "n_skipped=%d" % len(skipped),
        "win_rate_pct=%.2f" % (float((g > 0).mean() * 100.0) if n else 0.0),
        "avg_gain_pct=%.2f" % (float(g.mean()) if n else 0.0),
        "median_gain_pct=%.2f" % (float(g.median()) if n else 0.0),
        "expectancy_pct=%.2f" % (float(g.mean()) if n else 0.0),
        "profit_factor=%.3f" % (_profit_factor(g) if n else float("nan")),
        "",
        "Exit: sells from the before next-mid book (unchanged)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT.parent)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT.name)
    ap.add_argument("--trades-name", type=str, default="channel_touch_h2_next_open_mid_span365.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-days", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    before_raw = pd.read_csv(args.before)
    source = pd.read_csv(args.source)
    before = attach_buy_date(before_raw, source)
    n_dated = int(before["buy_date"].notna().sum()) if "buy_date" in before.columns else 0
    LOG.info("Before rows=%d with buy_date=%d/%d", len(before), n_dated, len(before))

    symbols = sorted({str(s).upper() for s in before["stock"].tolist()})
    buy_min = pd.to_datetime(before["buy_date"], errors="coerce").min()
    buy_max = pd.to_datetime(before["buy_date"], errors="coerce").max()
    if pd.isna(buy_min) or pd.isna(buy_max):
        buy_min = pd.to_datetime(source["buy_date"]).min()
        buy_max = pd.to_datetime(source["buy_date"]).max()
    buy_min = buy_min - timedelta(days=int(args.pad_days))
    buy_max = buy_max + timedelta(days=int(args.pad_days))

    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min,
        end=buy_max,
        workers=int(args.workers),
        use_cache=True,
    )
    LOG.info(
        "Loaded IB 15m %d/%d in %.1fs",
        sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty),
        len(symbols),
        time.perf_counter() - t0,
    )

    indexed = build_session_index(panels, before, pad_days=int(args.pad_days))
    out = compare_before_rows(before, indexed)
    cols = [c for c in COMPARE_COLS if c in out.columns]
    extra = [c for c in out.columns if c not in cols and not str(c).startswith("_")]
    out_write = out[cols + extra]

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.out_name
    out_write.to_csv(out_path, index=False)

    matched = out[out["row_kind"] == "matched"]
    skipped = out[out["row_kind"] == "before_skipped"]
    LOG.info(
        "Wrote %s matched=%d skipped=%d",
        out_path,
        len(matched),
        len(skipped),
    )
    if not matched.empty:
        LOG.info(
            "matched E_before=%.3f E_after=%.3f mean_diff=%.3f",
            float(matched["gain_before"].mean()),
            float(matched["gain_after"].mean()),
            float(matched["gain_diff"].mean()),
        )

    dated = dated_outdir()
    trades = trades_from_matched(matched, source)
    trades_path = dated / args.trades_name
    trades.to_csv(trades_path, index=False)
    summary_path = dated / (Path(args.trades_name).stem + "_summary.txt")
    write_summary(summary_path, matched, skipped)
    LOG.info("Wrote trades %s n=%d", trades_path, len(trades))

    print(out["row_kind"].value_counts().to_string())
    if not skipped.empty:
        print("skip_reason:")
        print(skipped["skip_reason"].value_counts().head(10).to_string())
    print("compare:", out_path)
    print("trades:", trades_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
