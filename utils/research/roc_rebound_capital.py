"""Capital-constrained ROC rebound: ranking, regime, velocity, N-slot + Rf equity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.roc_rebound import (
    backtest_roc_rebound,
    closes_panel_from_frames,
    roc_series,
    _naive_index,
)


def spy_features(spy_close: pd.Series, *, roc_length: int = 7) -> pd.DataFrame:
    s = spy_close.dropna().astype(float).sort_index()
    s.index = _naive_index(pd.DatetimeIndex(s.index)).normalize()
    s = s[~s.index.duplicated(keep="last")]
    out = pd.DataFrame({"close": s})
    out["sma200"] = out["close"].rolling(200, min_periods=200).mean()
    out["roc"] = roc_series(out["close"], roc_length)
    return out


def filter_trades_spy_regime(
    trades: List[Dict[str, Any]],
    spy_feat: pd.DataFrame,
    *,
    require_spy_sma200: bool = False,
    require_rel_weak: bool = False,
) -> List[Dict[str, Any]]:
    """Post-filter on signal day (= calendar day before entry, approx prior session).

    Uses entry_date's prior available spy bar as the signal as-of (conservative).
    """
    if not trades:
        return []
    if not require_spy_sma200 and not require_rel_weak:
        return list(trades)
    idx = spy_feat.index
    kept: List[Dict[str, Any]] = []
    for tr in trades:
        entry = pd.Timestamp(tr["entry_date"]).normalize()
        # Signal is prior session; use last spy bar strictly before entry date.
        prior = idx[idx < entry]
        if len(prior) == 0:
            continue
        day = prior[-1]
        row = spy_feat.loc[day]
        if require_spy_sma200:
            if pd.isna(row["sma200"]) or float(row["close"]) <= float(row["sma200"]):
                continue
        if require_rel_weak:
            er = tr.get("entry_roc")
            sr = row["roc"]
            if er is None or pd.isna(sr) or float(er) >= float(sr):
                continue
        out = dict(tr)
        out["spy_roc"] = None if pd.isna(row["roc"]) else round(float(row["roc"]), 3)
        kept.append(out)
    return kept


def occupancy_ranked(
    trades: List[Dict[str, Any]],
    *,
    max_open: int = 1,
    unique_symbol: bool = True,
    rank_by: str = "deepest_roc",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Daily batch ranking into free slots (not arrival-time FCFS).

    Each entry date: sort candidates by rank_by, then fill remaining slots.
    rank_by:
      deepest_roc — lowest entry_roc first
      vol_stretch — highest vol_stretch first (needs feature on trade)
    """
    if not trades:
        return [], {"n_kept": 0, "n_skipped": 0, "max_open": int(max_open), "rank_by": rank_by}

    rows: List[Dict[str, Any]] = []
    for i, tr in enumerate(trades):
        buy = pd.Timestamp(tr["entry_date"])
        sell = pd.Timestamp(tr["exit_date"])
        if buy.tzinfo is not None:
            buy = buy.tz_convert("UTC").tz_localize(None)
        if sell.tzinfo is not None:
            sell = sell.tz_convert("UTC").tz_localize(None)
        roc = tr.get("entry_roc")
        try:
            roc_f = float(roc) if roc is not None and pd.notna(roc) else 0.0
        except (TypeError, ValueError):
            roc_f = 0.0
        vs = tr.get("vol_stretch")
        try:
            vs_f = float(vs) if vs is not None and pd.notna(vs) else -1e9
        except (TypeError, ValueError):
            vs_f = -1e9
        rows.append(
            {
                **tr,
                "_i": i,
                "_buy": buy.normalize(),
                "_sell": sell.normalize(),
                "_sym": str(tr.get("symbol") or "").upper(),
                "_roc": roc_f,
                "_vs": vs_f,
            }
        )

    by_day: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for row in rows:
        by_day.setdefault(row["_buy"], []).append(row)

    open_pos: List[Tuple[pd.Timestamp, str]] = []
    kept: List[Dict[str, Any]] = []
    n_skip = 0

    for day in sorted(by_day.keys()):
        open_pos = [(s, sym) for s, sym in open_pos if s > day]
        cand = by_day[day]
        if rank_by == "vol_stretch":
            cand = sorted(cand, key=lambda r: (-r["_vs"], r["_roc"], r["_sym"], r["_i"]))
        else:
            # deepest_roc
            cand = sorted(cand, key=lambda r: (r["_roc"], r["_sym"], r["_i"]))
        for row in cand:
            if unique_symbol and any(sym == row["_sym"] for _, sym in open_pos):
                n_skip += 1
                continue
            if int(max_open) > 0 and len(open_pos) >= int(max_open):
                n_skip += 1
                continue
            kept.append({k: v for k, v in row.items() if not str(k).startswith("_")})
            open_pos.append((row["_sell"], row["_sym"]))

    return kept, {
        "n_kept": len(kept),
        "n_skipped": n_skip,
        "max_open": int(max_open),
        "rank_by": rank_by,
    }


def n_slot_equity(
    trades: List[Dict[str, Any]],
    close_panel: pd.DataFrame,
    *,
    max_open: int = 1,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
    rf_annual: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """Equal 1/max_open of mark-to-market equity per slot; Rf on idle cash.

    ``rf_annual`` is simple daily credit: cash *= (1 + rf_annual/252) each day
    while uninvested fraction sits in cash.
    """
    if close_panel is None or close_panel.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    panel = close_panel.copy()
    panel.index = _naive_index(pd.DatetimeIndex(panel.index)).normalize()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()

    s = pd.Timestamp(eval_start) if eval_start else panel.index[0]
    e = pd.Timestamp(eval_end) if eval_end else panel.index[-1]
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    calendar = panel.index[(panel.index >= s.normalize()) & (panel.index <= e.normalize())]
    if calendar.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    slots = max(1, int(max_open))
    alloc = 1.0 / float(slots)
    daily_rf = float(rf_annual) / 252.0

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    if tdf.empty:
        eq = pd.Series(1.0, index=calendar)
        if daily_rf != 0.0:
            # 100% cash earns rf
            for i in range(1, len(eq)):
                eq.iloc[i] = eq.iloc[i - 1] * (1.0 + daily_rf)
            eq = eq / float(eq.iloc[0])
        inv = pd.Series(0.0, index=calendar)
        return eq, inv

    tdf = tdf.copy()
    buy = pd.to_datetime(tdf["entry_date"], errors="coerce")
    sell = pd.to_datetime(tdf["exit_date"], errors="coerce")
    if getattr(buy.dt, "tz", None) is not None:
        buy = buy.dt.tz_convert("UTC").dt.tz_localize(None)
    if getattr(sell.dt, "tz", None) is not None:
        sell = sell.dt.tz_convert("UTC").dt.tz_localize(None)
    tdf["_buy"] = buy.dt.normalize()
    tdf["_sell"] = sell.dt.normalize()
    tdf["_sym"] = tdf["symbol"].astype(str).str.upper()
    tdf["entry_px"] = pd.to_numeric(tdf["entry_px"], errors="coerce")
    tdf["exit_px"] = pd.to_numeric(tdf["exit_px"], errors="coerce")
    tdf = tdf.dropna(subset=["_buy", "_sell", "entry_px", "exit_px"])
    # Prefer deepest roc within day for equity fill order (matches ranked occ).
    if "entry_roc" in tdf.columns:
        tdf["_roc"] = pd.to_numeric(tdf["entry_roc"], errors="coerce").fillna(0.0)
    else:
        tdf["_roc"] = 0.0
    tdf = tdf.sort_values(["_buy", "_roc", "_sym"]).reset_index(drop=True)

    by_buy: Dict[pd.Timestamp, List[int]] = {}
    for idx, row in tdf.iterrows():
        by_buy.setdefault(pd.Timestamp(row["_buy"]), []).append(int(idx))

    cash = 1.0
    positions: Dict[str, Dict[str, Any]] = {}
    eq_pts: List[float] = []
    inv_pts: List[float] = []

    def _mark() -> float:
        eq = cash
        for pos in positions.values():
            eq += pos["shares"] * pos["last_px"]
        return eq

    for dt in calendar:
        # Credit Rf on cash before trading activity.
        if daily_rf != 0.0 and cash > 0:
            cash *= 1.0 + daily_rf

        # Exits first.
        to_close = [
            sym
            for sym, pos in positions.items()
            if pd.Timestamp(pos["sell_date"]) <= dt
        ]
        for sym in to_close:
            pos = positions.pop(sym)
            cash += pos["shares"] * float(pos["exit_px"])

        if dt in by_buy:
            for idx in by_buy[dt]:
                if len(positions) >= slots:
                    break
                row = tdf.loc[idx]
                sym = str(row["_sym"])
                if sym in positions:
                    continue
                entry_px = float(row["entry_px"])
                if entry_px <= 0:
                    continue
                eq_now = _mark()
                spend = min(eq_now * alloc, cash)
                if spend <= 0:
                    continue
                shares = spend / entry_px
                cash -= spend
                last_px = entry_px
                if dt in panel.index and sym in panel.columns and pd.notna(panel.loc[dt, sym]):
                    last_px = float(panel.loc[dt, sym])
                positions[sym] = {
                    "shares": shares,
                    "last_px": last_px,
                    "sell_date": pd.Timestamp(row["_sell"]),
                    "exit_px": float(row["exit_px"]),
                }

        for sym, pos in positions.items():
            if dt in panel.index and sym in panel.columns and pd.notna(panel.loc[dt, sym]):
                pos["last_px"] = float(panel.loc[dt, sym])

        eq_pts.append(_mark())
        inv_pts.append(len(positions) / float(slots))

    if positions:
        for pos in list(positions.values()):
            cash += pos["shares"] * float(pos["exit_px"])
        positions.clear()
        eq_pts[-1] = cash

    eq = pd.Series(eq_pts, index=calendar)
    inv = pd.Series(inv_pts, index=calendar)
    if float(eq.iloc[0]) > 0:
        eq = eq / float(eq.iloc[0])
    return eq, inv


def apply_time_stop_to_trades(
    trades: List[Dict[str, Any]],
    frames: Dict[str, pd.DataFrame],
    *,
    time_stop_bars: int,
    friction_pct: float,
) -> List[Dict[str, Any]]:
    """Clip open holds to time_stop by exiting at the open of bar entry+N.

    Recomputes ret_pct from prices; used when base book had no time stop.
    """
    if not trades or time_stop_bars is None:
        return list(trades)
    out: List[Dict[str, Any]] = []
    for tr in trades:
        sym = str(tr["symbol"]).upper()
        df = frames.get(sym)
        if df is None or df.empty:
            out.append(tr)
            continue
        d = df.copy()
        if "ts" not in d.columns and isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={d.index.name or "index": "ts"})
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts", "open"]).sort_values("ts").drop_duplicates("ts")
        d["_day"] = d["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        entry = pd.Timestamp(tr["entry_date"]).normalize()
        exit_d = pd.Timestamp(tr["exit_date"]).normalize()
        days = list(d["_day"])
        if entry not in days:
            out.append(tr)
            continue
        i_entry = days.index(entry)
        i_stop = min(i_entry + int(time_stop_bars), len(days) - 1)
        stop_day = days[i_stop]
        # Exit at earlier of original exit and time-stop open.
        if stop_day < exit_d:
            stop_px = float(d.iloc[i_stop]["open"])
            entry_px = float(tr["entry_px"])
            ret = (stop_px / entry_px - 1.0) * 100.0 - float(friction_pct)
            # Remove friction double-count: original ret already had friction once;
            # recompute clean from prices with single friction.
            newt = dict(tr)
            newt["exit_date"] = str(stop_day.date())
            newt["exit_px"] = round(stop_px, 4)
            newt["ret_pct"] = round(ret, 4)
            newt["hold_days"] = int(i_stop - i_entry)
            newt["exit_reason"] = "time_stop"
            out.append(newt)
        else:
            out.append(tr)
    return out


def collect_variant_trades(
    frames: Dict[str, pd.DataFrame],
    *,
    friction_pct: float,
    overlay: Dict[str, Any],
    spy_close: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """Run per-symbol backtest; optional SPY filters applied after."""
    require_spy = bool(overlay.get("require_spy_sma200"))
    require_rw = bool(overlay.get("require_rel_weak"))
    min_entry_roc = overlay.get("min_entry_roc")  # e.g. -5.0 = require deeper dip
    allowed = {
        "require_sma200",
        "vol_scale",
        "vol_lookback",
        "vol_k",
        "time_stop_bars",
        "entry_thresh_pct",
        "exit_thresh_pct",
        "roc_length",
    }
    bt_kwargs = {k: v for k, v in overlay.items() if k in allowed}
    trades: List[Dict[str, Any]] = []
    for sym, raw in sorted(frames.items()):
        if raw is None or raw.empty:
            continue
        df = raw
        if "ts" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        res = backtest_roc_rebound(
            df,
            symbol=sym,
            friction_pct=friction_pct,
            **bt_kwargs,
        )
        trades.extend(res.get("trades") or [])

    if min_entry_roc is not None:
        thr = float(min_entry_roc)
        trades = [
            t
            for t in trades
            if t.get("entry_roc") is not None and float(t["entry_roc"]) <= thr
        ]

    if spy_close is not None and (require_spy or require_rw):
        feat = spy_features(spy_close)
        trades = filter_trades_spy_regime(
            trades,
            feat,
            require_spy_sma200=require_spy,
            require_rel_weak=require_rw,
        )
    return trades
