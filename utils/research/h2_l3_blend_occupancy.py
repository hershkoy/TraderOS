"""Timestamp occupancy for a last-15m H2 + 15m L3 blend.

Unified FCFS uses one max_open across both sleeves (H2 duration can starve L3).
Dedicated sleeves reserve L3 slots so the short book can still fire when H2 is
full. Unique-symbol is the single-account rule (one lot per name). Independent
sleeves walk each book with its own cap and may double a name.

Sells free a slot when sell_ts <= next buy_ts. Occupancy is greedy on the
existing fill/exit times (not a re-detect).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SLEEVE_H2 = "last15m_h2"
SLEEVE_L3 = "l3_wait12"
SKIP_CAP = "cap_full"
SKIP_SLEEVE = "sleeve_full"
SKIP_SYMBOL = "same_symbol"


def _naive_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def prepare_blend(
    h2: pd.DataFrame,
    l3: pd.DataFrame,
    *,
    gain_col: str = "gain_pct",
) -> pd.DataFrame:
    """Stack books with buy/sell timestamps and a sleeve tag."""
    parts: List[pd.DataFrame] = []
    for sleeve, df in ((SLEEVE_H2, h2), (SLEEVE_L3, l3)):
        if df is None or df.empty:
            continue
        out = df.copy()
        out["sleeve"] = sleeve
        buy_src = "buy_time" if "buy_time" in out.columns else "buy_date"
        sell_src = "sell_time" if "sell_time" in out.columns else "sell_date"
        out["_buy"] = _naive_ts(out[buy_src])
        out["_sell"] = _naive_ts(out[sell_src])
        out["_stock"] = out["stock"].astype(str).str.upper()
        if gain_col in out.columns:
            out["_gain"] = pd.to_numeric(out[gain_col], errors="coerce")
        elif "gain_pct" in out.columns:
            out["_gain"] = pd.to_numeric(out["gain_pct"], errors="coerce")
        else:
            out["_gain"] = pd.NA
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    stacked = pd.concat(parts, ignore_index=True, sort=False)
    stacked = stacked.dropna(subset=["_buy", "_sell", "_stock"])
    stacked = stacked.loc[stacked["_sell"] >= stacked["_buy"]].copy()
    return stacked.sort_values(["_buy", "_stock", "sleeve"], kind="mergesort").reset_index(drop=True)


def occupancy_unified_fcfs(
    trades: pd.DataFrame,
    *,
    max_open: int,
    unique_symbol: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One capital pool. First buy_time wins until a slot frees.

    ``max_open <= 0`` keeps unique-symbol (if set) but no size cap.
    """
    return _walk(
        trades,
        mode="unified",
        max_open=int(max_open),
        h2_cap=0,
        l3_cap=0,
        unique_symbol=bool(unique_symbol),
        unique_across=bool(unique_symbol),
    )


def occupancy_dedicated_sleeves(
    trades: pd.DataFrame,
    *,
    h2_cap: int,
    l3_cap: int,
    unique_across: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reserved slot budgets. L3 can take while H2 is at cap if L3 slots are free.

    ``unique_across=True``: one lot per name in the account.
    ``unique_across=False``: independent books (same name may be in both).
    Each sleeve still unique-symbol versus itself.
    """
    return _walk(
        trades,
        mode="dedicated",
        max_open=0,
        h2_cap=int(h2_cap),
        l3_cap=int(l3_cap),
        unique_symbol=True,
        unique_across=bool(unique_across),
    )


def _walk(
    trades: pd.DataFrame,
    *,
    mode: str,
    max_open: int,
    h2_cap: int,
    l3_cap: int,
    unique_symbol: bool,
    unique_across: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or trades.empty:
        empty = trades.iloc[0:0].copy() if trades is not None else pd.DataFrame()
        return empty, pd.DataFrame(columns=["reason", "sleeve"])

    ordered = trades.sort_values(["_buy", "_stock", "sleeve"], kind="mergesort")
    open_h2: List[Tuple[Any, str]] = []
    open_l3: List[Tuple[Any, str]] = []
    open_all: List[Tuple[Any, str]] = []
    keep_idx: List[Any] = []
    skips: List[dict] = []

    for idx, row in ordered.iterrows():
        buy_ts = row["_buy"]
        stock = str(row["_stock"])
        sleeve = str(row.get("sleeve") or "")
        open_h2 = [p for p in open_h2 if p[0] > buy_ts]
        open_l3 = [p for p in open_l3 if p[0] > buy_ts]
        open_all = [p for p in open_all if p[0] > buy_ts]

        if unique_symbol:
            if unique_across:
                held = any(s == stock for _, s in open_all)
            elif sleeve == SLEEVE_H2:
                held = any(s == stock for _, s in open_h2)
            else:
                held = any(s == stock for _, s in open_l3)
            if held:
                skips.append({"reason": SKIP_SYMBOL, "sleeve": sleeve})
                continue

        if mode == "unified":
            if max_open > 0 and len(open_all) >= max_open:
                skips.append({"reason": SKIP_CAP, "sleeve": sleeve})
                continue
            keep_idx.append(idx)
            open_all.append((row["_sell"], stock))
            if sleeve == SLEEVE_H2:
                open_h2.append((row["_sell"], stock))
            else:
                open_l3.append((row["_sell"], stock))
            continue

        cap = h2_cap if sleeve == SLEEVE_H2 else l3_cap
        cur = open_h2 if sleeve == SLEEVE_H2 else open_l3
        if cap <= 0 or len(cur) >= cap:
            skips.append({"reason": SKIP_SLEEVE, "sleeve": sleeve})
            continue
        keep_idx.append(idx)
        cur.append((row["_sell"], stock))
        open_all.append((row["_sell"], stock))
        if sleeve == SLEEVE_H2:
            open_h2 = cur
        else:
            open_l3 = cur

    kept = ordered.loc[keep_idx].copy()
    skip_df = pd.DataFrame(skips)
    return kept, skip_df


def skip_counts(skip_df: pd.DataFrame) -> Dict[str, int]:
    if skip_df is None or skip_df.empty:
        return {}
    out: Dict[str, int] = {}
    for (reason, sleeve), n in skip_df.groupby(["reason", "sleeve"]).size().items():
        out["%s_%s" % (reason, sleeve)] = int(n)
    out["n_skipped"] = int(len(skip_df))
    return out


def sleeve_counts(kept: pd.DataFrame) -> Dict[str, int]:
    if kept is None or kept.empty or "sleeve" not in kept.columns:
        return {SLEEVE_H2: 0, SLEEVE_L3: 0, "n": 0}
    vc = kept["sleeve"].astype(str).value_counts()
    return {
        SLEEVE_H2: int(vc.get(SLEEVE_H2, 0)),
        SLEEVE_L3: int(vc.get(SLEEVE_L3, 0)),
        "n": int(len(kept)),
    }


def year_sleeve_counts(kept: pd.DataFrame, year: int) -> Dict[str, int]:
    if kept is None or kept.empty:
        return {SLEEVE_H2: 0, SLEEVE_L3: 0}
    d = pd.to_datetime(kept["buy_date"], errors="coerce")
    part = kept.loc[d.dt.year == int(year)]
    return sleeve_counts(part)
