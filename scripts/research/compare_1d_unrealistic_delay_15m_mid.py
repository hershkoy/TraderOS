"""Compare 1d_unrealistic next-mid buys vs the same exits delayed +1 15m bar (mid fill).

Source book: reports/ascending_channels/channel_touch_h2_break_span365.csv
(the trade set behind reports/ascending_channels/1d_unrealistic/).

That HTML/CSV is the pre-_as_session_date next-mid overlay: buy_date is the
daily signal day, but the 15m join often used the prior US session. This script
reconstructs that exec bar from the stored buy_price, shifts one more 15m bar,
fills at mid, and keeps sell_date/sell_price unchanged.
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

LOG = logging.getLogger("delay_15m_mid_compare")

DEFAULT_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_h2_break_span365.csv"
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "delay_15m_mid_compare.csv"
)


def _fmt_ts(ts: Any) -> str:
    """RTH clock (America/New_York), matching channel-touch report convention."""
    return as_et(ts).strftime("%Y-%m-%d %H:%M")


def _buggy_session_day(buy_day: date) -> date:
    """Pre-fix next-mid: midnight UTC stamped buy_date becomes prior ET calendar day."""
    return buy_day - timedelta(days=1)


def _reconstruct_on_bars(
    bars: List[Dict[str, Any]],
    buy_price: float,
    *,
    eps: float = 0.02,
) -> Optional[Tuple[int, float, float, float]]:
    """Return (exec_i, implied_x, exec_mid, err) for a stored next-mid fill."""
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
            best = (i + 1, float(x), float(mid), float(err))
    return best


def _delay_mid(
    bars: List[Dict[str, Any]],
    exec_i: int,
) -> Tuple[Optional[float], Optional[Any], str]:
    delay_i = int(exec_i) + 1
    if delay_i >= len(bars):
        return None, None, "no_next_bar"
    bar = bars[delay_i]
    if as_et(bars[exec_i]["ts"]).date() != as_et(bar["ts"]).date():
        return None, None, "overnight"
    mid = bar_mid(float(bar["high"]), float(bar["low"]))
    return float(mid), bar["ts"], "ok"


def build_session_index(
    panels_15m: Dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    *,
    pad_days: int = 7,
) -> Dict[str, Dict[date, List[Dict[str, Any]]]]:
    """Coerce only bars near each symbol's buy dates; bucket by ET session date."""
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
        # Panels are naive UTC; keep a wide clock pad so ET sessions are intact.
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


def reconstruct_exec_indexed(
    by_day: Dict[date, List[Dict[str, Any]]],
    buy_day: date,
    buy_price: float,
) -> Optional[Tuple[List[Dict[str, Any]], int, date]]:
    candidates: List[date] = []
    for d in (_buggy_session_day(buy_day), buy_day):
        if d not in candidates:
            candidates.append(d)
    for delta in range(-5, 6):
        d = buy_day + timedelta(days=delta)
        if d not in candidates:
            candidates.append(d)

    best = None
    for rank, d in enumerate(candidates):
        bars = by_day.get(d) or []
        if len(bars) < 2:
            continue
        got = _reconstruct_on_bars(bars, buy_price)
        if got is None:
            continue
        exec_i, _x, _mid, err = got
        key = (rank, err)
        if best is None or key < best[0]:
            best = (key, bars, exec_i, d)
    if best is None:
        return None
    _key, bars, exec_i, d = best
    return bars, exec_i, d


def compare_trades(
    trades: pd.DataFrame,
    panels_15m: Dict[str, pd.DataFrame],
    *,
    pad_days: int = 7,
) -> pd.DataFrame:
    t_idx = time.perf_counter()
    indexed = build_session_index(panels_15m, trades, pad_days=pad_days)
    LOG.info("Indexed 15m sessions for %d symbols in %.1fs", len(indexed), time.perf_counter() - t_idx)

    rows: List[Dict[str, Any]] = []
    for _, tr in trades.iterrows():
        sym = str(tr["stock"]).upper()
        buy_day = _as_session_date(tr["buy_date"], naive_tz="UTC")
        buy_px = float(tr["buy_price"])
        sell_date = tr["sell_date"]
        sell_px = float(tr["sell_price"])
        gain_before = float(tr["gain_pct"])

        base = {
            "stock": sym,
            "buy_datetime_before": "",
            "buy_price_before": round(buy_px, 4),
            "buy_datetime_after": "",
            "buy_price_after": "",
            "sell_datetime": str(sell_date),
            "sell_price": round(sell_px, 4),
            "gain_before": round(gain_before, 4),
            "gain_after": "",
            "status": "",
        }

        by_day = indexed.get(sym)
        if not by_day or buy_day is None:
            base["status"] = "no_15m"
            rows.append(base)
            continue

        got = reconstruct_exec_indexed(by_day, buy_day, buy_px)
        if got is None:
            base["status"] = "reconstruct_fail"
            rows.append(base)
            continue

        bars, exec_i, _sess = got
        base["buy_datetime_before"] = _fmt_ts(bars[exec_i]["ts"])
        delay_px, delay_ts, status = _delay_mid(bars, exec_i)
        if delay_px is None or delay_ts is None:
            base["status"] = status
            rows.append(base)
            continue

        gain_after = (sell_px / delay_px - 1.0) * 100.0
        base["buy_datetime_after"] = _fmt_ts(delay_ts)
        base["buy_price_after"] = round(float(delay_px), 4)
        base["gain_after"] = round(float(gain_after), 4)
        base["status"] = "ok"
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT.parent)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT.name)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-days", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trades = pd.read_csv(args.trades)
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    buy_min = pd.to_datetime(trades["buy_date"]).min() - timedelta(days=int(args.pad_days))
    buy_max = pd.to_datetime(trades["buy_date"]).max() + timedelta(days=int(args.pad_days))
    LOG.info(
        "Trades n=%d symbols=%d 15m window %s -> %s",
        len(trades),
        len(symbols),
        buy_min.strftime("%Y-%m-%d"),
        buy_max.strftime("%Y-%m-%d"),
    )

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
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    LOG.info("Loaded IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)

    out = compare_trades(trades, panels, pad_days=int(args.pad_days))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.out_name
    out.to_csv(out_path, index=False)

    ok = out[out["status"] == "ok"].copy()
    LOG.info("Wrote %s rows=%d ok=%d", out_path, len(out), len(ok))
    if not ok.empty:
        LOG.info(
            "ok E_before=%.3f E_after=%.3f delta=%.3f",
            float(ok["gain_before"].mean()),
            float(ok["gain_after"].mean()),
            float(ok["gain_after"].mean()) - float(ok["gain_before"].mean()),
        )
    print(out["status"].value_counts().to_string())
    print("out:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
