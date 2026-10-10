"""FCFS occupancy + equal-slot equity for last-15m wheel vs SPY.

Reads filled claimcheck CSVs, walks max_open in {1,4,8,15}, reports
CAGR/Sharpe/MDD vs SPY buy-hold and Phase-5 gates.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_wheel_occupancy.py
  python scripts\\research\\overlay_last_15m_wheel_occupancy.py --trades-mid path.csv --trades-askbid path.csv
  python scripts\\research\\overlay_last_15m_wheel_occupancy.py --liq-presets
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_wheel import (  # noqa: E402
    filled_trades_to_occ_rows,
    filter_wheel_by_liquidity,
    occupancy_fcfs_wheel,
    spy_buyhold_equity,
    wheel_slot_equity,
)
from utils.research.metrics import (  # noqa: E402
    passes_phase5_gates,
    perf_stats,
    rebase_equity,
    window_equity,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_wheel_occupancy")

DEFAULT_MID = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-10-09"
    / "last_15m_wheel_mid_20261009_133405.csv"
)
DEFAULT_ASKBID = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-10-09"
    / "last_15m_wheel_ask_bid_20261009_133643.csv"
)
MAX_OPENS = (1, 4, 8, 15)
WINDOWS = (
    ("FULL", None, None),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2022-23", "2022-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", "2026-12-31"),
)


def _stats_dict(st) -> Dict[str, Any]:
    return {
        "cagr": round(float(st.cagr), 4),
        "total_return": round(float(st.total_return), 4),
        "sharpe": round(float(st.sharpe), 3),
        "max_drawdown": round(float(st.max_drawdown), 4),
        "pct_time_invested": round(float(st.pct_time_invested), 3),
        "n_trades": int(st.n_trades),
        "win_rate": round(float(st.win_rate), 4),
        "profit_factor": round(float(st.profit_factor), 3),
        "expectancy": round(float(st.expectancy), 4),
        "start": st.start,
        "end": st.end,
    }


def _score_window(
    label: str,
    eq: pd.Series,
    inv: pd.Series,
    spy_eq: pd.Series,
    trades: List[Dict[str, Any]],
    *,
    start: Optional[str],
    end: Optional[str],
    window: Optional[str] = None,
) -> Dict[str, Any]:
    if start and end:
        eq_w, inv_w = window_equity(eq, inv, start, end)
        spy_w, _ = window_equity(spy_eq, None, start, end)
    else:
        eq_w = rebase_equity(eq)
        inv_w = inv.reindex(eq_w.index).fillna(0.0)
        spy_w = rebase_equity(spy_eq.reindex(eq_w.index).ffill().bfill())

    common = eq_w.index.intersection(spy_w.index)
    if len(common) < 2:
        return {
            "label": label,
            "gate_pass": False,
            "gate_reason": "insufficient equity",
        }
    eq_w = eq_w.reindex(common).ffill()
    inv_w = inv_w.reindex(common).fillna(0.0)
    spy_w = spy_w.reindex(common).ffill()
    eq_w = eq_w / float(eq_w.iloc[0])
    spy_w = spy_w / float(spy_w.iloc[0])

    if start and end:
        t_win = [
            t
            for t in trades
            if start <= pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d") <= end
        ]
    else:
        t_win = list(trades)
    trade_rets = [float(t["gain_pct"]) / 100.0 for t in t_win]

    spy_st = perf_stats("SPY_buy_hold", spy_w, notes="price NAV", trade_rets=[])
    sleeve_st = perf_stats(
        label,
        eq_w,
        inv_w,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes="wheel equal-slot collateral, mark-at-cost",
        trade_rets=trade_rets,
    )
    ok, reason = passes_phase5_gates(sleeve_st, spy_st, window=window)
    LOG.info(
        "%s n=%s E=%.3f PF=%.2f | Sharpe=%.2f CAGR=%.1f%% MDD=%.1f%% inv=%.0f%% | "
        "SPY Sharpe=%.2f CAGR=%.1f%% MDD=%.1f%% | %s",
        label,
        sleeve_st.n_trades,
        100.0 * sleeve_st.expectancy,
        sleeve_st.profit_factor,
        sleeve_st.sharpe,
        100.0 * sleeve_st.cagr,
        100.0 * sleeve_st.max_drawdown,
        100.0 * sleeve_st.pct_time_invested,
        spy_st.sharpe,
        100.0 * spy_st.cagr,
        100.0 * spy_st.max_drawdown,
        reason,
    )
    return {
        "label": label,
        "gate_pass": bool(ok),
        "gate_reason": reason,
        "sleeve": _stats_dict(sleeve_st),
        "spy": _stats_dict(spy_st),
        "excess_sharpe": round(sleeve_st.sharpe - spy_st.sharpe, 3),
        "excess_cagr": round(sleeve_st.cagr - spy_st.cagr, 4),
        "occ_n": len(t_win),
    }


def run_book(
    name: str,
    trades_raw: List[Dict[str, Any]],
    spy_close: pd.Series,
    calendar: pd.DatetimeIndex,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "book": name,
        "n_raw": len(trades_raw),
        "max_open": {},
    }
    for mo in MAX_OPENS:
        kept, meta = occupancy_fcfs_wheel(
            trades_raw, max_open=int(mo), unique_symbol=True
        )
        eq, inv = wheel_slot_equity(kept, calendar, max_open=int(mo))
        spy_eq = spy_buyhold_equity(spy_close, calendar)
        windows = {}
        for wlabel, wstart, wend in WINDOWS:
            windows[wlabel] = _score_window(
                "%s_occ%s_%s" % (name, mo, wlabel),
                eq,
                inv,
                spy_eq,
                kept,
                start=wstart,
                end=wend,
                window=wlabel,
            )
        out["max_open"][str(mo)] = {
            "occupancy": meta,
            "windows": windows,
        }
        full = windows["FULL"]
        LOG.info(
            "---- %s max_open=%s kept=%s skip=%s FULL Sharpe=%.2f vs SPY %.2f | %s",
            name,
            mo,
            meta["n_kept"],
            meta["n_skipped"],
            full.get("sleeve", {}).get("sharpe") or float("nan"),
            full.get("spy", {}).get("sharpe") or float("nan"),
            full.get("gate_reason"),
        )
    return out


LIQ_PRESETS: List[Tuple[str, Optional[float], Optional[float]]] = [
    # name, min_adv, max_half_pct_of_mid
    ("half15", None, 15.0),
    ("adv25_half15", 25e6, 15.0),
]


def _load_books(
    mid_path: Path,
    askbid_path: Path,
    *,
    min_adv: Optional[float] = None,
    max_half_pct: Optional[float] = None,
    prefix: str = "",
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], Dict[str, Any]]:
    """Load mid + ask_bid CSVs, optionally liquidity-filter, return occ rows."""
    mid_df = pd.read_csv(mid_path)
    ab_df = pd.read_csv(askbid_path)
    meta: Dict[str, Any] = {"filter": None}
    if min_adv is not None or max_half_pct is not None:
        mid_df, ab_df, meta = filter_wheel_by_liquidity(
            mid_df,
            ab_df,
            min_adv=min_adv,
            max_half_pct_of_mid=max_half_pct,
        )
        meta["filter"] = {
            "min_adv": min_adv,
            "max_half_pct_of_mid": max_half_pct,
            **{k: v for k, v in meta.items() if k != "filter"},
        }
        LOG.info(
            "liquidity filter min_adv=%s max_half%%=%s matched=%s kept=%s med_half%%=%s",
            min_adv,
            max_half_pct,
            meta.get("n_matched"),
            meta.get("n_kept"),
            meta.get("median_half_pct_kept"),
        )
    books: List[Tuple[str, List[Dict[str, Any]]]] = []
    for label, df in (("mid", mid_df), ("ask_bid", ab_df)):
        name = ("%s_%s" % (prefix, label)) if prefix else label
        rows = filled_trades_to_occ_rows(df)
        LOG.info("%s filled rows for occ: %s", name, len(rows))
        books.append((name, rows))
    return books, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-mid", type=Path, default=DEFAULT_MID)
    ap.add_argument("--trades-askbid", type=Path, default=DEFAULT_ASKBID)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--min-adv", type=float, default=None)
    ap.add_argument(
        "--max-half-pct-of-mid",
        type=float,
        default=None,
        help="Max put half-spread as %% of mid credit (e.g. 15)",
    )
    ap.add_argument(
        "--liq-presets",
        action="store_true",
        help="Run half15 + adv25_half15 books (skips unfiltered)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not Path(args.trades_mid).exists() or not Path(args.trades_askbid).exists():
        LOG.error("need both mid and ask_bid CSVs")
        return 2

    books: List[Tuple[str, List[Dict[str, Any]]]] = []
    filter_meta: Dict[str, Any] = {}
    if args.liq_presets:
        for pname, min_adv, max_half in LIQ_PRESETS:
            b, meta = _load_books(
                args.trades_mid,
                args.trades_askbid,
                min_adv=min_adv,
                max_half_pct=max_half,
                prefix=pname,
            )
            books.extend(b)
            filter_meta[pname] = meta
    else:
        b, meta = _load_books(
            args.trades_mid,
            args.trades_askbid,
            min_adv=args.min_adv,
            max_half_pct=args.max_half_pct_of_mid,
            prefix="",
        )
        books.extend(b)
        if meta.get("filter"):
            filter_meta["custom"] = meta

    if not books or all(len(rs) == 0 for _, rs in books):
        LOG.error("no trade rows after load/filter")
        return 2

    # Calendar from first entry through last exit
    all_rows = [r for _, rs in books for r in rs]
    start = min(pd.Timestamp(r["entry_date"]) for r in all_rows) - pd.Timedelta(days=5)
    end = max(pd.Timestamp(r["exit_date"]) for r in all_rows) + pd.Timedelta(days=5)
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        start=start.date(),
        end=end.date(),
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        LOG.error("SPY panel missing")
        return 2
    close = spy["close"].astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    calendar = close.index[
        (close.index >= start.normalize()) & (close.index <= end.normalize())
    ]

    t0 = time.time()
    summary: Dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "max_opens": list(MAX_OPENS),
        "model": "equal-slot cash-secured collateral; mark-at-cost until exit",
        "liquidity_filters": filter_meta,
        "books": {},
    }
    for label, rows in books:
        summary["books"][label] = run_book(label, rows, close, calendar)
    summary["wall_s"] = round(time.time() - t0, 1)

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "last_15m_wheel_occupancy"
    if args.liq_presets:
        stem = "last_15m_wheel_occupancy_liq"
    elif args.min_adv is not None or args.max_half_pct_of_mid is not None:
        stem = "last_15m_wheel_occupancy_liq"
    path = outdir / ("%s_%s.json" % (stem, stamp))
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("wrote %s", path)

    # Compact CSV of FULL window by book x max_open
    flat = []
    for book, blob in summary["books"].items():
        for mo, block in blob["max_open"].items():
            w = block["windows"]["FULL"]
            flat.append(
                {
                    "book": book,
                    "max_open": int(mo),
                    "n_kept": block["occupancy"]["n_kept"],
                    "n_skipped": block["occupancy"]["n_skipped"],
                    "sharpe": (w.get("sleeve") or {}).get("sharpe"),
                    "cagr": (w.get("sleeve") or {}).get("cagr"),
                    "mdd": (w.get("sleeve") or {}).get("max_drawdown"),
                    "spy_sharpe": (w.get("spy") or {}).get("sharpe"),
                    "spy_cagr": (w.get("spy") or {}).get("cagr"),
                    "spy_mdd": (w.get("spy") or {}).get("max_drawdown"),
                    "gate": w.get("gate_reason"),
                    "pass": w.get("gate_pass"),
                }
            )
            for ykey in ("2022", "2023", "2022-23"):
                yw = block["windows"].get(ykey) or {}
                flat[-1]["sharpe_%s" % ykey] = (yw.get("sleeve") or {}).get("sharpe")
                flat[-1]["cagr_%s" % ykey] = (yw.get("sleeve") or {}).get("cagr")
                flat[-1]["mdd_%s" % ykey] = (yw.get("sleeve") or {}).get("max_drawdown")
                flat[-1]["pass_%s" % ykey] = yw.get("gate_pass")
                flat[-1]["gate_%s" % ykey] = yw.get("gate_reason")
    csv_path = outdir / ("%s_%s.csv" % (stem, stamp))
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    LOG.info("wrote %s", csv_path)
    print(pd.DataFrame(flat).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
