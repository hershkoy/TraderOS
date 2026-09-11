"""Compare 1d H2 signal-close unique fills vs last-RTH 15m mid if that bar opened above rail.

Before: reports/ascending_channels/2026-09-09 unique signal-close book
(the trades behind
channel_touch_tv_report_interactive_fric0.25_1d_h2_signal_close_span365_shakeout_20260909_013839.html).

After: same session's last RTH 15m (15:45 ET). Fill mid only if **open > rail**.
Sells kept from before when matched.

Also records ATR(14) on IB 15m at that bar and (open-rail)/rail of the last 15m open.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from compare_1d_unrealistic_signal_close import (  # noqa: E402
    build_session_index,
    _fmt_ts,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.realistic_purchaser import (  # noqa: E402
    _as_session_date,
    as_et,
    purchase_last_rth_open_above_mid,
)

LOG = logging.getLogger("last_15m_mid_compare")

DEFAULT_BEFORE = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-09"
    / "channel_touch_full_h2_break_span365_unique_20260909_000344.csv"
)
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "last_15m_open_mid_compare.csv"
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
    "last_15m_open",
    "open_above_rail_pct",
    "atr_15m",
    "atr_15m_pct",
    "atr_1d_pct",
]
ATR_LEN = 14


def _rail_from_row(tr: pd.Series) -> Optional[float]:
    for col in ("touch_price", "signal_x"):
        if col in tr.index:
            try:
                px = float(tr[col])
            except (TypeError, ValueError):
                continue
            if px == px and px > 0:
                return px
    return None


def _atr_at_ts(
    by_day: Dict[date, List[Dict[str, Any]]],
    exec_ts: Any,
    *,
    length: int = ATR_LEN,
) -> Optional[float]:
    bars: List[Dict[str, Any]] = []
    for d in sorted(by_day.keys()):
        bars.extend(by_day[d])
    if not bars:
        return None
    bars.sort(key=lambda b: b["ts"])
    want = as_et(exec_ts) if exec_ts is not None else None
    idx = None
    if want is not None:
        for i, b in enumerate(bars):
            if as_et(b["ts"]) == want:
                idx = i
                break
    if idx is None:
        idx = len(bars) - 1
    if idx + 1 < length:
        return None
    high = [float(b["high"]) for b in bars]
    low = [float(b["low"]) for b in bars]
    close = [float(b["close"]) for b in bars]
    tr: List[float] = [high[0] - low[0]]
    for i in range(1, idx + 1):
        pc = close[i - 1]
        tr.append(
            max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
        )
    if len(tr) < length:
        return None
    atr = sum(tr[:length]) / float(length)
    alpha = 1.0 / float(length)
    for i in range(length, len(tr)):
        atr = atr * (1.0 - alpha) + tr[i] * alpha
    if atr != atr or atr <= 0:
        return None
    return float(atr)


def compare_before_rows(
    before: pd.DataFrame,
    indexed: Dict[str, Dict],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, tr in before.iterrows():
        sym = str(tr["stock"]).upper()
        buy_day = _as_session_date(tr.get("buy_date"), naive_tz="UTC")
        buy_before = float(tr["buy_price"])
        sell_px = float(tr["sell_price"])
        gain_before = float(tr["gain_pct"])
        rail = _rail_from_row(tr)
        atr_1d = None
        if "atr_pct" in tr.index:
            try:
                atr_1d = float(tr["atr_pct"])
            except (TypeError, ValueError):
                atr_1d = None
            if atr_1d is not None and atr_1d != atr_1d:
                atr_1d = None
        base = {
            "row_kind": "matched_attempt",
            "stock": sym,
            "buy_date": str(tr.get("buy_date") or ""),
            "buy_datetime_before": "",
            "buy_price_before": round(buy_before, 4),
            "buy_datetime_after": "",
            "buy_price_after": "",
            "sell_datetime": str(tr.get("sell_date") or ""),
            "sell_price": round(sell_px, 4),
            "gain_before": round(gain_before, 4),
            "gain_after": "",
            "gain_diff": "",
            "signal_x": round(float(rail), 4) if rail is not None else "",
            "session_date": str(buy_day) if buy_day is not None else "",
            "status": "",
            "skip_reason": "",
            "in_full_signal_close": True,
            "last_15m_open": "",
            "open_above_rail_pct": "",
            "atr_15m": "",
            "atr_15m_pct": "",
            "atr_1d_pct": round(float(atr_1d), 4) if atr_1d is not None else "",
        }
        by_day = indexed.get(sym)
        if not by_day or buy_day is None or rail is None:
            base["row_kind"] = "before_skipped"
            base["status"] = "skipped"
            if rail is None:
                base["skip_reason"] = "no_rail"
            elif buy_day is None:
                base["skip_reason"] = "no_buy_date"
            else:
                base["skip_reason"] = "no_15m"
            rows.append(base)
            continue

        got = purchase_last_rth_open_above_mid(
            None,
            signal_session_date=buy_day,
            rail=rail,
            session_index=by_day,
        )
        opened = got.bar_open
        if opened is not None:
            base["last_15m_open"] = round(float(opened), 4)
            base["open_above_rail_pct"] = round(
                (float(opened) - float(rail)) / float(rail) * 100.0, 4
            )
        atr_15 = _atr_at_ts(by_day, got.exec_bar_ts or got.hit_bar_ts)
        if atr_15 is not None:
            base["atr_15m"] = round(atr_15, 6)
            ref = float(opened) if opened else None
            if ref and ref > 0:
                base["atr_15m_pct"] = round(atr_15 / ref * 100.0, 4)
        if not got.filled or got.fill_px is None:
            base["row_kind"] = "before_skipped"
            base["status"] = "skipped"
            base["skip_reason"] = got.reason or "no_last_15m"
            if got.exec_bar_ts is not None:
                base["buy_datetime_after"] = _fmt_ts(got.exec_bar_ts)
            rows.append(base)
            continue

        px = float(got.fill_px)
        gain_after = (sell_px / px - 1.0) * 100.0
        base.update(
            {
                "row_kind": "matched",
                "buy_datetime_after": _fmt_ts(got.exec_bar_ts) if got.exec_bar_ts else "",
                "buy_price_after": round(px, 4),
                "gain_after": round(gain_after, 4),
                "gain_diff": round(gain_after - gain_before, 4),
                "status": "ok",
                "skip_reason": "",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT.parent)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT.name)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-days", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    before = pd.read_csv(args.before)
    before["stock"] = before["stock"].astype(str).str.upper()
    LOG.info("Before unique signal-close n=%d", len(before))

    symbols = sorted({str(s).upper() for s in before["stock"].tolist()})
    buy_min = pd.to_datetime(before["buy_date"], errors="coerce").min()
    buy_max = pd.to_datetime(before["buy_date"], errors="coerce").max()
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
    extra = [c for c in out.columns if c not in cols]
    out_write = out[cols + extra]

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.out_name
    out_write.to_csv(out_path, index=False)

    matched = out[out["row_kind"] == "matched"]
    skipped = out[out["row_kind"] == "before_skipped"]
    LOG.info("Wrote %s matched=%d skipped=%d", out_path, len(matched), len(skipped))
    if not matched.empty:
        LOG.info(
            "matched E_before=%.3f E_after=%.3f mean_diff=%.3f mean_open_above_rail_pct=%.3f",
            float(matched["gain_before"].mean()),
            float(matched["gain_after"].mean()),
            float(matched["gain_diff"].mean()),
            float(pd.to_numeric(matched["open_above_rail_pct"], errors="coerce").mean()),
        )
    print(out["row_kind"].value_counts().to_string())
    if not skipped.empty:
        print("skip_reason:")
        print(skipped["skip_reason"].value_counts().head(10).to_string())
    print("out:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
