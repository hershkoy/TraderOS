"""Compare 1d_unrealistic next-mid fills vs 15m signal-close confirmation.

Before book: reports/ascending_channels/1d_unrealistic/delay_15m_mid_compare.csv
(next-mid buy_price_before / gain_before).

After: same reconstructed print bar, fill at that bar's **close** (wait for 15m
close confirmation). Sell kept from before when matched.

Optional --signal-close-trades CSV from a full H2 signal-close backtest adds rows
that are new (not in the before book) or documents backtest skip overlap.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.realistic_purchaser import (  # noqa: E402
    DEFAULT_MAX_LOW_TO_MID_PCT,
    _as_session_date,
    _coerce_15m_bars,
    as_et,
    bar_contains_price,
    bar_mid,
    is_rth_15m_bar_start,
    low_to_mid_pct,
)

LOG = logging.getLogger("signal_close_compare")

DEFAULT_BEFORE = (
    ROOT / "reports" / "ascending_channels" / "1d_unrealistic" / "delay_15m_mid_compare.csv"
)
DEFAULT_SOURCE = ROOT / "reports" / "ascending_channels" / "channel_touch_h2_break_span365.csv"
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "signal_close_compare.csv"
)


def _fmt_ts(ts: Any) -> str:
    return as_et(ts).strftime("%Y-%m-%d %H:%M")


def _buggy_session_day(buy_day: date) -> date:
    return buy_day - timedelta(days=1)


def _reconstruct_next_mid(
    bars: List[Dict[str, Any]],
    buy_price: float,
    *,
    eps: float = 0.02,
) -> Optional[Tuple[int, int, float, float]]:
    """Return (hit_i, exec_i, implied_x, err)."""
    best = None
    for i in range(len(bars) - 1):
        hit = bars[i]
        nxt = bars[i + 1]
        mid = bar_mid(float(nxt["high"]), float(nxt["low"]))
        x = 2.0 * float(buy_price) - mid
        if not bar_contains_price(float(hit["high"]), float(hit["low"]), x):
            continue
        if low_to_mid_pct(float(nxt["high"]), float(nxt["low"])) > float(
            DEFAULT_MAX_LOW_TO_MID_PCT
        ) + 1e-12:
            continue
        err = abs((x + mid) / 2.0 - float(buy_price))
        if err > eps:
            continue
        if best is None or err < best[3]:
            best = (i, i + 1, float(x), float(err))
    return best


def _session_candidates(buy_day: date) -> List[date]:
    out: List[date] = []
    for d in (_buggy_session_day(buy_day), buy_day):
        if d not in out:
            out.append(d)
    for delta in range(-5, 6):
        d = buy_day + timedelta(days=delta)
        if d not in out:
            out.append(d)
    return out


def build_session_index(
    panels_15m: Dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    *,
    pad_days: int = 7,
) -> Dict[str, Dict[date, List[Dict[str, Any]]]]:
    buy_by_sym: Dict[str, List[pd.Timestamp]] = {}
    for _, tr in trades.iterrows():
        sym = str(tr["stock"]).upper()
        buy_by_sym.setdefault(sym, []).append(pd.Timestamp(tr["buy_date"]))

    out: Dict[str, Dict[date, List[Dict[str, Any]]]] = {}
    for sym, buys in buy_by_sym.items():
        panel = panels_15m.get(sym)
        if panel is None or panel.empty:
            continue
        lo = min(buys) - timedelta(days=int(pad_days))
        hi = max(buys) + timedelta(days=int(pad_days))
        idx = panel.index
        if getattr(idx, "tz", None) is not None:
            lo_ts = pd.Timestamp(lo).tz_localize("UTC")
            hi_ts = pd.Timestamp(hi).tz_localize("UTC") + timedelta(days=1)
        else:
            lo_ts = pd.Timestamp(lo)
            hi_ts = pd.Timestamp(hi) + timedelta(days=1)
        sliced = panel.loc[(idx >= lo_ts) & (idx < hi_ts)]
        if sliced.empty:
            continue
        by_day: Dict[date, List[Dict[str, Any]]] = {}
        for b in _coerce_15m_bars(sliced, naive_tz="UTC"):
            if not is_rth_15m_bar_start(b["ts"]):
                continue
            d = as_et(b["ts"]).date()
            by_day.setdefault(d, []).append(b)
        out[sym] = by_day
    return out


def attach_buy_date(before: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Map delay-compare rows back to calendar buy_date via source next-mid book."""
    src = source.copy()
    src["stock"] = src["stock"].astype(str).str.upper()
    src["buy_price_r"] = pd.to_numeric(src["buy_price"], errors="coerce").round(4)
    src["sell_price_r"] = pd.to_numeric(src["sell_price"], errors="coerce").round(4)
    src["sell_datetime"] = src["sell_date"].astype(str)
    src = src[
        [
            "stock",
            "buy_date",
            "buy_price_r",
            "sell_datetime",
            "sell_price_r",
            "channel_start",
            "h2_time",
            "wait_bars",
            "shakeout_breakout",
        ]
    ]

    out = before.copy()
    out["stock"] = out["stock"].astype(str).str.upper()
    out["buy_price_r"] = pd.to_numeric(out["buy_price_before"], errors="coerce").round(4)
    out["sell_price_r"] = pd.to_numeric(out["sell_price"], errors="coerce").round(4)
    out["sell_datetime"] = out["sell_datetime"].astype(str)

    merged = out.merge(
        src,
        on=["stock", "buy_price_r", "sell_datetime", "sell_price_r"],
        how="left",
        suffixes=("", "_src"),
    )
    # Fallback: stock + sell_datetime + buy_price only
    miss = merged["buy_date"].isna()
    if miss.any():
        fb = out.loc[miss].merge(
            src.drop(columns=["sell_price_r"]),
            on=["stock", "buy_price_r", "sell_datetime"],
            how="left",
            suffixes=("", "_fb"),
        )
        for col in ("buy_date", "channel_start", "h2_time", "wait_bars", "shakeout_breakout"):
            if col in fb.columns:
                merged.loc[miss, col] = fb[col].values
    return merged


def signal_close_for_before_row(
    by_day: Dict[date, List[Dict[str, Any]]],
    buy_day: date,
    buy_price_before: float,
) -> Dict[str, Any]:
    """Reconstruct next-mid print bar, then fill at that same bar's close."""
    for rank, d in enumerate(_session_candidates(buy_day)):
        bars = by_day.get(d) or []
        if len(bars) < 2:
            continue
        got = _reconstruct_next_mid(bars, buy_price_before)
        if got is None:
            continue
        hit_i, _exec_i, x, _err = got
        hit = bars[hit_i]
        close_px = float(hit["close"])
        if close_px != close_px or close_px <= 0:
            return {
                "buy_datetime_after": "",
                "buy_price_after": "",
                "signal_x": round(float(x), 4),
                "session_date": str(d),
                "status": "skipped",
                "skip_reason": "bad_ohlc",
                "match_rank": rank,
            }
        return {
            "buy_datetime_after": _fmt_ts(hit["ts"]),
            "buy_price_after": round(close_px, 4),
            "signal_x": round(float(x), 4),
            "session_date": str(d),
            "status": "ok",
            "skip_reason": "",
            "match_rank": rank,
        }

    return {
        "buy_datetime_after": "",
        "buy_price_after": "",
        "signal_x": "",
        "session_date": "",
        "status": "skipped",
        "skip_reason": "reconstruct_fail",
        "match_rank": -1,
    }


def compare_before_rows(
    before: pd.DataFrame,
    indexed: Dict[str, Dict[date, List[Dict[str, Any]]]],
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
        }
        by_day = indexed.get(sym)
        if not by_day or buy_day is None:
            base["status"] = "skipped"
            base["skip_reason"] = "no_15m" if buy_day is not None else "no_buy_date"
            rows.append(base)
            continue

        sc = signal_close_for_before_row(by_day, buy_day, buy_before)
        base.update(
            {
                "buy_datetime_after": sc["buy_datetime_after"],
                "buy_price_after": sc["buy_price_after"],
                "signal_x": sc["signal_x"],
                "session_date": sc["session_date"],
                "status": sc["status"],
                "skip_reason": sc["skip_reason"],
            }
        )
        if sc["status"].startswith("ok") and sc["buy_price_after"] not in ("", None):
            px = float(sc["buy_price_after"])
            gain_after = (sell_px / px - 1.0) * 100.0
            base["gain_after"] = round(gain_after, 4)
            base["gain_diff"] = round(gain_after - gain_before, 4)
            base["row_kind"] = "matched"
        else:
            base["row_kind"] = "before_skipped"
            base["status"] = "skipped"
        rows.append(base)
    return pd.DataFrame(rows)


def append_new_trades(
    matched: pd.DataFrame,
    signal_close_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Append signal-close backtest trades whose (stock, buy_date) are not in before."""
    if signal_close_trades is None or signal_close_trades.empty:
        return matched

    sc = signal_close_trades.copy()
    sc["stock"] = sc["stock"].astype(str).str.upper()
    sc["buy_date"] = sc["buy_date"].astype(str)
    before_keys = set(
        zip(
            matched["stock"].astype(str).str.upper(),
            matched["buy_date"].astype(str),
        )
    )
    new_rows: List[Dict[str, Any]] = []
    for _, tr in sc.iterrows():
        key = (str(tr["stock"]).upper(), str(tr["buy_date"]))
        if key in before_keys:
            continue
        buy_px = float(tr["buy_price"])
        sell_px = float(tr["sell_price"])
        gain = float(tr["gain_pct"])
        buy_time = ""
        if "buy_time" in tr.index and pd.notna(tr.get("buy_time")):
            buy_time = str(tr["buy_time"])
        new_rows.append(
            {
                "row_kind": "new",
                "stock": key[0],
                "buy_date": key[1],
                "buy_datetime_before": "",
                "buy_price_before": "",
                "buy_datetime_after": buy_time,
                "buy_price_after": round(buy_px, 4),
                "sell_datetime": str(tr["sell_date"]),
                "sell_price": round(sell_px, 4),
                "gain_before": "",
                "gain_after": round(gain, 4),
                "gain_diff": "",
                "signal_x": "",
                "session_date": key[1],
                "status": "new",
                "skip_reason": "",
            }
        )
    if not new_rows:
        return matched
    return pd.concat([matched, pd.DataFrame(new_rows)], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument(
        "--signal-close-trades",
        type=Path,
        default=None,
        help="Optional unique signal-close backtest CSV to append new trades",
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT.parent)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT.name)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-days", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    before_raw = pd.read_csv(args.before)
    source = pd.read_csv(args.source)
    before = attach_buy_date(before_raw, source)
    n_dated = int(before["buy_date"].notna().sum()) if "buy_date" in before.columns else 0
    LOG.info("Before rows=%d with buy_date=%d/%d", len(before), n_dated, len(before))

    symbols = sorted({str(s).upper() for s in before["stock"].tolist()})
    # Also load symbols from signal-close file for new-trade panels if needed later
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

    if args.signal_close_trades and Path(args.signal_close_trades).exists():
        sc = pd.read_csv(args.signal_close_trades)
        LOG.info("Appending new trades from %s n=%d", args.signal_close_trades, len(sc))
        out = append_new_trades(out, sc)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.out_name
    out.to_csv(out_path, index=False)

    matched = out[out["row_kind"] == "matched"]
    skipped = out[out["row_kind"] == "before_skipped"]
    new = out[out["row_kind"] == "new"]
    LOG.info(
        "Wrote %s matched=%d skipped=%d new=%d",
        out_path,
        len(matched),
        len(skipped),
        len(new),
    )
    if not matched.empty:
        LOG.info(
            "matched E_before=%.3f E_after=%.3f mean_diff=%.3f",
            float(matched["gain_before"].mean()),
            float(matched["gain_after"].mean()),
            float(matched["gain_diff"].mean()),
        )
    print(out["row_kind"].value_counts().to_string())
    if not skipped.empty:
        print("skip_reason:")
        print(skipped["skip_reason"].value_counts().head(10).to_string())
    print("out:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
