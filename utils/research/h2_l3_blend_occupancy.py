"""Timestamp occupancy for a two-sleeve trade blend.

Unified FCFS uses one max_open across both sleeves. Dedicated sleeves reserve
per-sleeve caps so the second book can still fire when the first is full.
Unique-symbol is the single-account rule (one lot per name). Independent
sleeves walk each book with its own cap and may double a name.

Default sleeve tags keep the last-15m H2 + 15m L3 names for the original
overlay. Short fade uses ``SLEEVE_SHORT``.

Sells free a slot when sell_ts <= next buy_ts. Occupancy is greedy on the
existing fill/exit times (not a re-detect).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

SLEEVE_H2 = "last15m_h2"
SLEEVE_L3 = "l3_wait12"
SLEEVE_SHORT = "resist_short_h3"
SKIP_CAP = "cap_full"
SKIP_SLEEVE = "sleeve_full"
SKIP_SYMBOL = "same_symbol"


def _naive_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def prepare_blend(
    book_a: pd.DataFrame,
    book_b: pd.DataFrame,
    *,
    gain_col: str = "gain_pct",
    sleeve_a: str = SLEEVE_H2,
    sleeve_b: str = SLEEVE_L3,
) -> pd.DataFrame:
    """Stack two books with buy/sell timestamps and sleeve tags.

    ``book_a`` / ``book_b`` were historically ``h2`` / ``l3``; any two trade
    frames work as long as they have stock + buy/sell clocks and a gain column.
    """
    parts: List[pd.DataFrame] = []
    for sleeve, df in ((str(sleeve_a), book_a), (str(sleeve_b), book_b)):
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
    sleeve_a: str = SLEEVE_H2,
    sleeve_b: str = SLEEVE_L3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One capital pool. First buy_time wins until a slot frees.

    ``max_open <= 0`` keeps unique-symbol (if set) but no size cap.
    """
    return _walk(
        trades,
        mode="unified",
        max_open=int(max_open),
        a_cap=0,
        b_cap=0,
        unique_symbol=bool(unique_symbol),
        unique_across=bool(unique_symbol),
        sleeve_a=str(sleeve_a),
        sleeve_b=str(sleeve_b),
    )


def occupancy_dedicated_sleeves(
    trades: pd.DataFrame,
    *,
    h2_cap: Optional[int] = None,
    l3_cap: Optional[int] = None,
    a_cap: Optional[int] = None,
    b_cap: Optional[int] = None,
    unique_across: bool = True,
    sleeve_a: str = SLEEVE_H2,
    sleeve_b: str = SLEEVE_L3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reserved slot budgets. Sleeve B can take while A is at cap if B slots free.

    ``unique_across=True``: one lot per name in the account.
    ``unique_across=False``: independent books (same name may be in both).
    Each sleeve still unique-symbol versus itself.

    Prefer ``a_cap`` / ``b_cap``. Legacy ``h2_cap`` / ``l3_cap`` still work.
    """
    cap_a = int(a_cap if a_cap is not None else (h2_cap if h2_cap is not None else 0))
    cap_b = int(b_cap if b_cap is not None else (l3_cap if l3_cap is not None else 0))
    return _walk(
        trades,
        mode="dedicated",
        max_open=0,
        a_cap=cap_a,
        b_cap=cap_b,
        unique_symbol=True,
        unique_across=bool(unique_across),
        sleeve_a=str(sleeve_a),
        sleeve_b=str(sleeve_b),
    )


def _walk(
    trades: pd.DataFrame,
    *,
    mode: str,
    max_open: int,
    a_cap: int,
    b_cap: int,
    unique_symbol: bool,
    unique_across: bool,
    sleeve_a: str,
    sleeve_b: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or trades.empty:
        empty = trades.iloc[0:0].copy() if trades is not None else pd.DataFrame()
        return empty, pd.DataFrame(columns=["reason", "sleeve"])

    ordered = trades.sort_values(["_buy", "_stock", "sleeve"], kind="mergesort")
    open_a: List[Tuple[Any, str]] = []
    open_b: List[Tuple[Any, str]] = []
    open_all: List[Tuple[Any, str]] = []
    keep_idx: List[Any] = []
    skips: List[dict] = []

    for idx, row in ordered.iterrows():
        buy_ts = row["_buy"]
        stock = str(row["_stock"])
        sleeve = str(row.get("sleeve") or "")
        open_a = [p for p in open_a if p[0] > buy_ts]
        open_b = [p for p in open_b if p[0] > buy_ts]
        open_all = [p for p in open_all if p[0] > buy_ts]

        if unique_symbol:
            if unique_across:
                held = any(s == stock for _, s in open_all)
            elif sleeve == sleeve_a:
                held = any(s == stock for _, s in open_a)
            else:
                held = any(s == stock for _, s in open_b)
            if held:
                skips.append({"reason": SKIP_SYMBOL, "sleeve": sleeve})
                continue

        if mode == "unified":
            if max_open > 0 and len(open_all) >= max_open:
                skips.append({"reason": SKIP_CAP, "sleeve": sleeve})
                continue
            keep_idx.append(idx)
            open_all.append((row["_sell"], stock))
            if sleeve == sleeve_a:
                open_a.append((row["_sell"], stock))
            else:
                open_b.append((row["_sell"], stock))
            continue

        cap = a_cap if sleeve == sleeve_a else b_cap
        cur = open_a if sleeve == sleeve_a else open_b
        if cap <= 0 or len(cur) >= cap:
            skips.append({"reason": SKIP_SLEEVE, "sleeve": sleeve})
            continue
        keep_idx.append(idx)
        cur.append((row["_sell"], stock))
        open_all.append((row["_sell"], stock))
        if sleeve == sleeve_a:
            open_a = cur
        else:
            open_b = cur

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


def sleeve_counts(
    kept: pd.DataFrame,
    *,
    sleeve_a: str = SLEEVE_H2,
    sleeve_b: str = SLEEVE_L3,
) -> Dict[str, int]:
    if kept is None or kept.empty or "sleeve" not in kept.columns:
        return {sleeve_a: 0, sleeve_b: 0, "n": 0}
    vc = kept["sleeve"].astype(str).value_counts()
    return {
        sleeve_a: int(vc.get(sleeve_a, 0)),
        sleeve_b: int(vc.get(sleeve_b, 0)),
        "n": int(len(kept)),
    }


def year_sleeve_counts(
    kept: pd.DataFrame,
    year: int,
    *,
    sleeve_a: str = SLEEVE_H2,
    sleeve_b: str = SLEEVE_L3,
) -> Dict[str, int]:
    if kept is None or kept.empty:
        return {sleeve_a: 0, sleeve_b: 0}
    d = pd.to_datetime(kept["buy_date"], errors="coerce")
    part = kept.loc[d.dt.year == int(year)]
    return sleeve_counts(part, sleeve_a=sleeve_a, sleeve_b=sleeve_b)
