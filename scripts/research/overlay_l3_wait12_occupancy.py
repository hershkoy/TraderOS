"""FCFS occupancy + equal-slot equity for 15m L3 wait-12 vs SPY.

Phase-5 gates: Sharpe>1, Sharpe>=SPY, MDD better than SPY (2022 waives Sharpe>1).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_l3_wait12_occupancy.py
  python scripts\\research\\overlay_l3_wait12_occupancy.py --trades path.csv --max-opens 1,3,5,7,8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from channel_touch_robustness import apply_max_open, concurrent_open_stats  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_wheel import spy_buyhold_equity, wheel_slot_equity  # noqa: E402
from utils.research.metrics import (  # noqa: E402
    passes_phase5_gates,
    perf_stats,
    rebase_equity,
    window_equity,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_l3_wait12_occupancy")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"
)
MAX_OPENS = (1, 3, 5, 7, 8, 15)
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


def _load_trades(path: Path, *, friction_pct: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["stock"] = df["stock"].astype(str).str.upper()
    if "gain_pct_net" not in df.columns:
        g = pd.to_numeric(df["gain_pct"], errors="coerce")
        df["gain_pct_net"] = g - 2.0 * float(friction_pct)
    else:
        df["gain_pct_net"] = pd.to_numeric(df["gain_pct_net"], errors="coerce")
    df["gain_pct"] = pd.to_numeric(df["gain_pct"], errors="coerce")
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["sell_date"] = pd.to_datetime(df["sell_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "buy_time" in df.columns:
        df["buy_time"] = pd.to_datetime(df["buy_time"], errors="coerce")
    if "sell_time" in df.columns:
        df["sell_time"] = pd.to_datetime(df["sell_time"], errors="coerce")
    df = df.dropna(subset=["buy_date", "sell_date", "gain_pct_net", "stock"])
    return df.reset_index(drop=True)


def _to_occ_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "symbol": str(r["stock"]).upper(),
                "entry_date": str(r["buy_date"])[:10],
                "exit_date": str(r["sell_date"])[:10],
                "gain_pct": float(r["gain_pct_net"]),
            }
        )
    return rows


def _score_window(
    label: str,
    eq: pd.Series,
    inv: pd.Series,
    spy_eq: pd.Series,
    trades: Sequence[Dict[str, Any]],
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
        return {"label": label, "gate_pass": False, "gate_reason": "insufficient equity"}
    eq_w = eq_w.reindex(common).ffill()
    inv_w = inv_w.reindex(common).fillna(0.0)
    spy_w = spy_w.reindex(common).ffill()
    eq_w = eq_w / float(eq_w.iloc[0])
    spy_w = spy_w / float(spy_w.iloc[0])

    if start and end:
        t_win = [
            t
            for t in trades
            if start <= str(t["entry_date"])[:10] <= end
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
        notes="L3 wait-12 equal-slot mark-at-cost",
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


def _pooled(df: pd.DataFrame) -> Dict[str, Any]:
    g = pd.to_numeric(df["gain_pct_net"], errors="coerce").dropna()
    if g.empty:
        return {"n": 0, "E": None, "PF": None}
    wins = float(g[g > 0].sum())
    losses = float((-g[g <= 0]).sum())
    pf = wins / losses if losses > 0 else float("inf")
    return {
        "n": int(len(g)),
        "E": round(float(g.mean()), 4),
        "PF": round(pf, 3),
        "WR": round(float((g > 0).mean() * 100.0), 2),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="L3 wait-12 occupancy vs SPY Phase-5")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--friction-pct", type=float, default=0.10)
    ap.add_argument(
        "--max-opens",
        default=",".join(str(x) for x in MAX_OPENS),
        help="Comma list of max_open caps",
    )
    ap.add_argument(
        "--tie-break",
        default="fifo",
        choices=("fifo", "wait", "rs"),
        help="Same-day order for greedy max_open (default fifo=symbol)",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not args.trades.is_file():
        LOG.error("Trades CSV not found: %s", args.trades)
        return 1
    df = _load_trades(args.trades, friction_pct=float(args.friction_pct))
    caps = [int(x.strip()) for x in str(args.max_opens).split(",") if x.strip()]
    LOG.info(
        "Loaded %d L3 trades from %s (fric=%.2f)",
        len(df),
        args.trades,
        float(args.friction_pct),
    )
    conc = concurrent_open_stats(df)
    LOG.info("Unconstrained concurrent: %s", conc)
    base = _pooled(df)
    LOG.info("Pooled net: n=%s E=%s PF=%s WR=%s", base["n"], base["E"], base["PF"], base["WR"])

    panels = load_ohlcv_many(
        ["SPY"],
        start=date(2018, 1, 1),
        end=date.today(),
        timeframe="1d",
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        LOG.error("Failed to load SPY")
        return 1
    spy_close = spy["close"].astype(float).copy()
    spy_close.index = pd.DatetimeIndex(pd.to_datetime(spy_close.index)).tz_localize(None).normalize()
    spy_close = spy_close[~spy_close.index.duplicated(keep="last")].sort_index()

    buy0 = pd.to_datetime(df["buy_date"]).min()
    sell1 = pd.to_datetime(df["sell_date"]).max()
    calendar = spy_close.index[(spy_close.index >= buy0) & (spy_close.index <= sell1)]
    spy_eq = spy_buyhold_equity(spy_close, calendar)

    outdir = dated_outdir(args.outdir or (ROOT / "reports" / "ascending_channels"))
    summary_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "trades": str(args.trades),
        "n_raw": int(len(df)),
        "friction_pct": float(args.friction_pct),
        "tie_break": str(args.tie_break),
        "concurrent": conc,
        "pooled_net": base,
        "max_open": {},
        "elapsed_sec": None,
    }

    for mo in caps:
        kept = apply_max_open(df, int(mo), tie_break=str(args.tie_break))
        pooled = _pooled(kept)
        rows = _to_occ_rows(kept)
        eq, inv = wheel_slot_equity(rows, calendar, max_open=int(mo))
        book: Dict[str, Any] = {
            "max_open": int(mo),
            "n_kept": int(len(kept)),
            "n_skipped": int(len(df) - len(kept)),
            "pooled_net": pooled,
            "windows": {},
        }
        for label, start, end in WINDOWS:
            win_key = label if label != "FULL" else None
            scored = _score_window(
                "L3_occ%d_%s" % (mo, label),
                eq,
                inv,
                spy_eq,
                rows,
                start=start,
                end=end,
                window=win_key,
            )
            book["windows"][label] = scored
            summary_rows.append(
                {
                    "max_open": int(mo),
                    "window": label,
                    "n_kept": int(len(kept)) if label == "FULL" else scored.get("occ_n"),
                    "pooled_E": pooled.get("E"),
                    "pooled_PF": pooled.get("PF"),
                    "gate_pass": scored.get("gate_pass"),
                    "gate_reason": scored.get("gate_reason"),
                    "sharpe": (scored.get("sleeve") or {}).get("sharpe"),
                    "cagr": (scored.get("sleeve") or {}).get("cagr"),
                    "mdd": (scored.get("sleeve") or {}).get("max_drawdown"),
                    "spy_sharpe": (scored.get("spy") or {}).get("sharpe"),
                    "spy_mdd": (scored.get("spy") or {}).get("max_drawdown"),
                    "inv_pct": (scored.get("sleeve") or {}).get("pct_time_invested"),
                }
            )
        payload["max_open"][str(mo)] = book

    payload["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = outdir / ("l3_wait12_occupancy_summary_%s.csv" % stamp)
    json_path = outdir / ("l3_wait12_occupancy_%s.json" % stamp)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    LOG.info("Summary -> %s", summary_csv)
    LOG.info("JSON -> %s", json_path)
    print("\nOccupancy summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nUnconstrained concurrent:", conc)
    print("Pooled net:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
