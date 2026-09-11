"""Reprice last-15m-open-mid sells with realistic 15m / next-open-mid exits.

Before: last-RTH 15m mid buys with daily occupancy sells kept (same-bar stop
clip). WVE 2023-12-06 buy 6.85 then sell 2023-12-07 @ 6.0254 is that clip.

After (two modes, occupancy not re-walked):
  1) 15m-next-mid: ATR k=2 + 10% trail on RTH 15m. Decision on bar N, fill N+1 mid.
  2) daily-close-next-open-mid: same stop on the session; decide at the close,
     fill next session 09:30 ET 15m mid.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from compare_1d_unrealistic_signal_close import _fmt_ts  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.realistic_exits import (  # noqa: E402
    EXIT_MODE_15M_NEXT_MID,
    EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID,
    EXIT_MODES,
    atr_dollars,
    normalize_exit_mode,
    simulate_exit,
)
from utils.research.realistic_purchaser import (  # noqa: E402
    _as_session_date,
    as_et,
    index_rth_15m_by_session,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("last_15m_realistic_sells")
ET = ZoneInfo("America/New_York")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_span365.csv"
)
DEFAULT_COMPARE = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "1d_unrealistic"
    / "last_15m_open_mid_compare.csv"
)
DEFAULT_OUT_NAME = "last_15m_realistic_sells_compare.csv"
FRICTION_PCT = 0.25


def _profit_factor(gains: pd.Series) -> float:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    wins = g[g > 0].sum()
    losses = (-g[g < 0]).sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _orig_buy_px(row: pd.Series) -> Optional[float]:
    for col in ("buy_price_before", "orig_buy_price"):
        if col in row.index:
            try:
                px = float(row[col])
            except (TypeError, ValueError):
                continue
            if px == px and px > 0:
                return px
    return None


def _atr_for_row(row: pd.Series, entry_px: float) -> Optional[float]:
    atr_pct = None
    for col in ("atr_1d_pct", "atr_pct"):
        if col in row.index:
            try:
                atr_pct = float(row[col])
            except (TypeError, ValueError):
                atr_pct = None
            if atr_pct is not None and atr_pct == atr_pct and atr_pct > 0:
                break
            atr_pct = None
    ref = _orig_buy_px(row)
    if ref is None:
        ref = entry_px
    return atr_dollars(atr_pct=atr_pct, ref_px=ref)


def _session_day(ts: Any) -> str:
    if ts is None:
        return ""
    try:
        return as_et(ts).date().isoformat()
    except (TypeError, ValueError):
        return ""


def _et_clock_to_utc(text: object) -> str:
    s = str(text or "").strip()
    if not s or s.lower() == "nan":
        return ""
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize(ET)
    else:
        t = t.tz_convert(ET)
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def build_hold_session_index(
    panels_15m: Dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    *,
    pad_before: int = 5,
    pad_after: int = 21,
) -> Dict[str, Dict[date, List[Dict[str, Any]]]]:
    windows: Dict[str, List[pd.Timestamp]] = {}
    for _, tr in trades.iterrows():
        sym = str(tr["stock"]).upper()
        buy = pd.Timestamp(tr["buy_date"])
        sell_raw = tr["sell_date"] if "sell_date" in tr.index else tr.get("sell_datetime")
        sell = pd.Timestamp(sell_raw) if pd.notna(sell_raw) else buy
        pair = windows.setdefault(sym, [buy, sell])
        pair[0] = min(pair[0], buy)
        pair[1] = max(pair[1], sell)

    out: Dict[str, Dict[date, List[Dict[str, Any]]]] = {}
    for sym, (lo, hi) in windows.items():
        panel = panels_15m.get(sym)
        if panel is None or panel.empty:
            continue
        lo_d = lo - timedelta(days=int(pad_before))
        hi_d = hi + timedelta(days=int(pad_after))
        idx = panel.index
        if getattr(idx, "tz", None) is not None:
            lo_ts = pd.Timestamp(lo_d).tz_localize("UTC")
            hi_ts = pd.Timestamp(hi_d).tz_localize("UTC") + timedelta(days=1)
        else:
            lo_ts = pd.Timestamp(lo_d)
            hi_ts = pd.Timestamp(hi_d) + timedelta(days=1)
        sliced = panel.loc[(idx >= lo_ts) & (idx < hi_ts)]
        if sliced.empty:
            continue
        out[sym] = index_rth_15m_by_session(sliced, naive_tz="UTC")
    return out


def apply_realistic_sells(
    trades: pd.DataFrame,
    indexed: Dict[str, Dict[date, List[Dict[str, Any]]]],
    *,
    mode: str,
) -> pd.DataFrame:
    mode_n = normalize_exit_mode(mode)
    rows: List[Dict[str, Any]] = []
    for _, tr in trades.iterrows():
        sym = str(tr["stock"]).upper()
        entry_px = float(tr["buy_price"])
        sell_before = float(tr["sell_price"])
        gain_before = float(tr["gain_pct"])
        buy_ts = tr.get("buy_time")
        if buy_ts is None or (isinstance(buy_ts, float) and pd.isna(buy_ts)):
            buy_ts = None
        base = {
            "mode": mode_n,
            "stock": sym,
            "buy_date": str(tr.get("buy_date") or ""),
            "buy_time": str(tr.get("buy_time") or ""),
            "buy_price": round(entry_px, 4),
            "sell_datetime_before": str(tr.get("sell_date") or ""),
            "sell_price_before": round(sell_before, 4),
            "gain_before": round(gain_before, 4),
            "sell_datetime_after": "",
            "sell_time_after": "",
            "sell_price_after": "",
            "gain_after": "",
            "gain_diff": "",
            "exit_reason_before": str(tr.get("exit_reason") or ""),
            "exit_reason_after": "",
            "decision_datetime": "",
            "hard_stop_after": "",
            "peak_after": "",
            "status": "",
            "skip_reason": "",
        }
        by_day = indexed.get(sym)
        if not by_day or buy_ts in (None, ""):
            base["status"] = "skipped"
            base["skip_reason"] = "no_15m" if not by_day else "no_buy_time"
            rows.append(base)
            continue
        atr = _atr_for_row(tr, entry_px)
        got = simulate_exit(
            by_day,
            entry_ts=buy_ts,
            entry_px=entry_px,
            mode=mode_n,
            atr_at_entry=atr,
        )
        if not got.filled or got.sell_px is None:
            base["status"] = "skipped"
            base["skip_reason"] = got.reason or "no_fill"
            if got.decision_ts is not None:
                base["decision_datetime"] = _fmt_ts(got.decision_ts)
            if got.hard_stop is not None:
                base["hard_stop_after"] = round(float(got.hard_stop), 4)
            rows.append(base)
            continue
        px = float(got.sell_px)
        gain_after = (px / entry_px - 1.0) * 100.0
        base.update(
            {
                "status": "ok",
                "skip_reason": "",
                "sell_datetime_after": _session_day(got.exec_bar_ts),
                "sell_time_after": _fmt_ts(got.exec_bar_ts) if got.exec_bar_ts else "",
                "sell_price_after": round(px, 4),
                "gain_after": round(gain_after, 4),
                "gain_diff": round(gain_after - gain_before, 4),
                "exit_reason_after": str(got.exit_reason or ""),
                "decision_datetime": _fmt_ts(got.decision_ts) if got.decision_ts else "",
                "hard_stop_after": round(float(got.hard_stop), 4)
                if got.hard_stop is not None
                else "",
                "peak_after": round(float(got.peak_px), 4)
                if got.peak_px is not None
                else "",
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def trades_from_compare(src: pd.DataFrame, cmp_ok: pd.DataFrame) -> pd.DataFrame:
    src = src.copy()
    src["stock"] = src["stock"].astype(str).str.upper()
    src["buy_date"] = src["buy_date"].astype(str)
    m = cmp_ok.copy()
    m["stock"] = m["stock"].astype(str).str.upper()
    m["buy_date"] = m["buy_date"].astype(str)
    merged = src.merge(
        m[
            [
                "stock",
                "buy_date",
                "sell_datetime_after",
                "sell_time_after",
                "sell_price_after",
                "gain_after",
                "exit_reason_after",
                "hard_stop_after",
                "peak_after",
            ]
        ],
        on=["stock", "buy_date"],
        how="inner",
        suffixes=("", "_cmp"),
    )
    rows = []
    for _, row in merged.iterrows():
        rec = {c: row[c] for c in src.columns if c in row.index}
        rec["stock"] = str(row["stock"]).upper()
        rec["buy_date"] = str(row["buy_date"])
        rec["buy_price"] = float(row["buy_price"])
        rec["sell_date"] = str(row["sell_datetime_after"])
        rec["sell_price"] = float(row["sell_price_after"])
        rec["gain_pct"] = float(row["gain_after"])
        rec["exit_reason"] = str(row["exit_reason_after"] or rec.get("exit_reason") or "")
        rec["sell_time"] = _et_clock_to_utc(row.get("sell_time_after"))
        if pd.notna(row.get("hard_stop_after")) and str(row.get("hard_stop_after")) != "":
            rec["hard_stop_price"] = float(row["hard_stop_after"])
        if pd.notna(row.get("peak_after")) and str(row.get("peak_after")) != "":
            rec["peak_price"] = float(row["peak_after"])
        buy_d = _as_session_date(rec["buy_date"], naive_tz="UTC")
        sell_d = _as_session_date(rec["sell_date"], naive_tz="UTC")
        if buy_d is not None and sell_d is not None:
            rec["hold_days"] = int((sell_d - buy_d).days)
        rows.append(rec)
    return pd.DataFrame(rows)


def _log_book(label: str, ok: pd.DataFrame) -> None:
    if ok.empty:
        LOG.info("%s n=0", label)
        return
    g = pd.to_numeric(ok["gain_after"], errors="coerce")
    g_net = g - FRICTION_PCT
    LOG.info(
        "%s n=%d E_gross=%.3f PF_gross=%.3f E_net0.25=%.3f PF_net0.25=%.3f vs kept-sell E=%.3f",
        label,
        len(ok),
        float(g.mean()),
        _profit_factor(g),
        float(g_net.mean()),
        _profit_factor(g_net),
        float(pd.to_numeric(ok["gain_before"], errors="coerce").mean()),
    )


def _write_summary(path: Path, *, mode: str, trades: pd.DataFrame, skipped: int) -> None:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    g_net = g - FRICTION_PCT
    lines = [
        "H2 resistance-break unique-symbol/day (span<=365)",
        "realistic_fill=True",
        "fill=last RTH 15m mid if that bar opened above the rail",
        "realistic_exit=%s" % mode,
        "Note: Before book kept daily occupancy sells (unrealistic same-bar clip).",
        "Note: Occupancy not re-walked. ATR k=2 clamp 1.5-6% + 10% trail.",
        "",
        "n_trades=%d" % len(trades),
        "n_skipped=%d" % skipped,
        "win_rate_pct=%.2f" % float((g > 0).mean() * 100.0),
        "avg_gain_pct=%.2f" % float(g.mean()),
        "median_gain_pct=%.2f" % float(g.median()),
        "expectancy_pct=%.2f" % float(g.mean()),
        "profit_factor=%.3f" % _profit_factor(g),
        "expectancy_net_0.25=%.2f" % float(g_net.mean()),
        "profit_factor_net_0.25=%.3f" % _profit_factor(g_net),
        "",
        "Exit: %s" % mode,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


STEM_MAP = {
    EXIT_MODE_15M_NEXT_MID: "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365",
    EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID: (
        "channel_touch_h2_last_15m_open_mid_sell_eod_next_open_mid_span365"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--compare-in", type=Path, default=DEFAULT_COMPARE)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME)
    ap.add_argument(
        "--mode",
        default="both",
        help="15m-next-mid, daily-close-next-open-mid, or both",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=5)
    ap.add_argument("--pad-after", type=int, default=21)
    ap.add_argument("--symbols", type=str, default="", help="Optional comma list")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    if args.compare_in.exists():
        cmp = pd.read_csv(args.compare_in)
        matched = cmp[cmp.get("row_kind", "matched") == "matched"].copy()
        if "buy_price_before" in matched.columns:
            matched["stock"] = matched["stock"].astype(str).str.upper()
            matched["buy_date"] = matched["buy_date"].astype(str)
            trades["buy_date"] = trades["buy_date"].astype(str)
            trades = trades.merge(
                matched[["stock", "buy_date", "buy_price_before"]],
                on=["stock", "buy_date"],
                how="left",
            )
    want = str(args.symbols or "").strip().upper()
    if want:
        keep = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(keep)].copy()
        LOG.info("Restricted to %s n=%d", sorted(keep), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    modes: Sequence[str]
    raw_mode = str(args.mode or "both").strip().lower()
    if raw_mode in ("both", "all"):
        modes = EXIT_MODES
    else:
        modes = (normalize_exit_mode(args.mode),)

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
        days=int(args.pad_before)
    )
    sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
        days=int(args.pad_after)
    )
    LOG.info(
        "Trades n=%d symbols=%d 15m window %s -> %s",
        len(trades),
        len(symbols),
        buy_min.strftime("%Y-%m-%d"),
        sell_max.strftime("%Y-%m-%d"),
    )

    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min,
        end=sell_max,
        workers=int(args.workers),
        use_cache=True,
    )
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    LOG.info("Loaded IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)

    indexed = build_hold_session_index(
        panels,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )
    LOG.info("Indexed 15m sessions for %d symbols", len(indexed))

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for mode in modes:
        part = apply_realistic_sells(trades, indexed, mode=mode)
        pieces.append(part)
        ok = part[part["status"] == "ok"]
        skipped = int((part["status"] != "ok").sum())
        _log_book(mode, ok)
        bad = part.loc[part["status"] != "ok", "skip_reason"]
        if not bad.empty:
            print("%s skip_reason:" % mode)
            print(bad.value_counts().head(10).to_string())
        if ok.empty:
            continue
        book = trades_from_compare(trades, ok)
        trades_path = outdir / (STEM_MAP[mode] + ".csv")
        book.to_csv(trades_path, index=False)
        _write_summary(
            outdir / (STEM_MAP[mode] + "_summary.txt"),
            mode=mode,
            trades=book,
            skipped=skipped,
        )
        LOG.info("Wrote %s n=%d skipped=%d", trades_path, len(book), skipped)
        wve = ok[ok["stock"] == "WVE"]
        if not wve.empty:
            r = wve.iloc[0]
            LOG.info(
                "WVE %s buy=%s sell_before=%s @ %s -> sell_after=%s @ %s gain %.3f -> %.3f",
                mode,
                r["buy_price"],
                r["sell_price_before"],
                r["sell_datetime_before"],
                r["sell_price_after"],
                r["sell_time_after"],
                r["gain_before"],
                r["gain_after"],
            )

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out_path = outdir / args.out_name
    out.to_csv(out_path, index=False)
    print("out:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
