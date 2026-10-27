"""As-of-filed fundamentals join for channel-touch trade CSVs.

Uses TimescaleDB ``fundamentals_quarterly``. Causality is ``filed <= buy
session date`` (not period_end). Junk future period_end rows are dropped.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.sec_fundamentals import FundamentalRow, passes_four_filters

FUND_RATIO_COLS: Sequence[str] = (
    "ocf_to_debt",
    "capex_to_ocf",
    "net_buyback_pct",
    "fcf_roa",
)

FUND_FEATURE_COLS: Sequence[str] = (
    "has_fund",
    "fund_passes_screen",
    "fund_n_filters_pass",
    "f1_ocf_debt",
    "f2_capex_ocf",
    "f3_buybacks",
    "f4_fcf_roa",
) + tuple(FUND_RATIO_COLS)


def _as_date(v: object) -> Optional[date]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    ts = pd.Timestamp(v)
    if pd.isna(ts):
        return None
    return ts.date()


def load_fundamentals_panel(
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load quarterly fund rows (exclude_sector filtered, sane period_end)."""
    db = get_timescaledb_client()
    if not db.connect():
        raise RuntimeError("TimescaleDB connection failed")

    sql = """
    SELECT symbol, cik, period_end, filed, fy, fp,
           ocf, capex, fcf, total_debt, assets, shares_out,
           buybacks_ttm, issuance_ttm, net_buybacks_ttm, mkt_cap,
           ocf_to_debt, capex_to_ocf, net_buyback_pct, fcf_roa,
           sic, sic_desc, exclude_sector, debt_zero
    FROM fundamentals_quarterly
    WHERE exclude_sector IS NOT TRUE
      AND period_end <= CURRENT_DATE
      AND filed IS NOT NULL
      AND fp LIKE 'Q%%'
    """
    params: List[Any] = []
    if symbols:
        syms = sorted({str(s).upper() for s in symbols})
        sql += " AND symbol = ANY(%s)"
        params.append(syms)
    sql += " ORDER BY symbol, filed, period_end"

    rows = db.execute_query(sql, tuple(params) if params else None)
    cols = [
        "symbol",
        "cik",
        "period_end",
        "filed",
        "fy",
        "fp",
        "ocf",
        "capex",
        "fcf",
        "total_debt",
        "assets",
        "shares_out",
        "buybacks_ttm",
        "issuance_ttm",
        "net_buybacks_ttm",
        "mkt_cap",
        "ocf_to_debt",
        "capex_to_ocf",
        "net_buyback_pct",
        "fcf_roa",
        "sic",
        "sic_desc",
        "exclude_sector",
        "debt_zero",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    return df


def _row_to_fundamental(r: pd.Series) -> FundamentalRow:
    fr = FundamentalRow(
        symbol=str(r.get("symbol") or ""),
        cik=str(r.get("cik") or ""),
        period_end=_as_date(r.get("period_end")) or date(1970, 1, 1),
        filed=_as_date(r.get("filed")) or date(1970, 1, 1),
        fy=int(r["fy"]) if pd.notna(r.get("fy")) else None,
        fp=str(r.get("fp") or ""),
        ocf=float(r["ocf"]) if pd.notna(r.get("ocf")) else None,
        capex=float(r["capex"]) if pd.notna(r.get("capex")) else None,
        fcf=float(r["fcf"]) if pd.notna(r.get("fcf")) else None,
        total_debt=float(r["total_debt"]) if pd.notna(r.get("total_debt")) else None,
        assets=float(r["assets"]) if pd.notna(r.get("assets")) else None,
        shares_out=float(r["shares_out"]) if pd.notna(r.get("shares_out")) else None,
        buybacks_ttm=float(r["buybacks_ttm"]) if pd.notna(r.get("buybacks_ttm")) else None,
        issuance_ttm=float(r["issuance_ttm"]) if pd.notna(r.get("issuance_ttm")) else None,
        net_buybacks_ttm=(
            float(r["net_buybacks_ttm"]) if pd.notna(r.get("net_buybacks_ttm")) else None
        ),
        mkt_cap=float(r["mkt_cap"]) if pd.notna(r.get("mkt_cap")) else None,
        ocf_to_debt=float(r["ocf_to_debt"]) if pd.notna(r.get("ocf_to_debt")) else None,
        capex_to_ocf=float(r["capex_to_ocf"]) if pd.notna(r.get("capex_to_ocf")) else None,
        net_buyback_pct=(
            float(r["net_buyback_pct"]) if pd.notna(r.get("net_buyback_pct")) else None
        ),
        fcf_roa=float(r["fcf_roa"]) if pd.notna(r.get("fcf_roa")) else None,
        debt_zero=bool(r["debt_zero"]) if pd.notna(r.get("debt_zero")) else False,
        exclude_sector=bool(r["exclude_sector"]) if pd.notna(r.get("exclude_sector")) else False,
    )
    return fr


def _flags_from_row(r: pd.Series) -> Dict[str, Optional[bool]]:
    return passes_four_filters(_row_to_fundamental(r))


def attach_fundamentals_asof(
    trades: pd.DataFrame,
    fund: Optional[pd.DataFrame] = None,
    *,
    symbol_col: str = "stock",
    date_col: str = "buy_date",
) -> pd.DataFrame:
    """Left-join latest fund row with filed <= buy session date per trade.

    Adds ``has_fund``, four-filter flags, ``fund_passes_screen``,
    ``fund_n_filters_pass``, and ratio columns. Missing fund stays in the
    book with ``has_fund=0``.
    """
    out = trades.copy()
    for c in FUND_FEATURE_COLS:
        if c in ("has_fund", "fund_passes_screen", "fund_n_filters_pass"):
            out[c] = 0
        elif c.startswith("f"):
            out[c] = np.nan
        else:
            out[c] = np.nan

    if out.empty:
        return out

    syms = sorted({str(s).upper() for s in out[symbol_col].astype(str)})
    if fund is None:
        fund = load_fundamentals_panel(syms)
    else:
        fund = fund.copy()
        fund["symbol"] = fund["symbol"].astype(str).str.upper()
        fund["filed"] = pd.to_datetime(fund["filed"]).dt.date
        fund["period_end"] = pd.to_datetime(fund["period_end"]).dt.date

    if fund.empty:
        return out

    # Per-symbol arrays sorted by filed for searchsorted
    by_sym: Dict[str, pd.DataFrame] = {}
    for sym, g in fund.groupby("symbol", sort=False):
        gg = g.sort_values(["filed", "period_end"]).reset_index(drop=True)
        by_sym[str(sym).upper()] = gg

    buy_dates = pd.to_datetime(out[date_col], errors="coerce")
    stocks = out[symbol_col].astype(str).str.upper()

    has_fund = np.zeros(len(out), dtype=int)
    passes = np.zeros(len(out), dtype=int)
    n_pass = np.zeros(len(out), dtype=int)
    f1 = np.full(len(out), np.nan)
    f2 = np.full(len(out), np.nan)
    f3 = np.full(len(out), np.nan)
    f4 = np.full(len(out), np.nan)
    ocf_d = np.full(len(out), np.nan)
    capex = np.full(len(out), np.nan)
    buyback = np.full(len(out), np.nan)
    fcf = np.full(len(out), np.nan)

    for i in range(len(out)):
        bd = buy_dates.iloc[i]
        if pd.isna(bd):
            continue
        asof = bd.date() if hasattr(bd, "date") else _as_date(bd)
        if asof is None:
            continue
        sym = stocks.iloc[i]
        g = by_sym.get(sym)
        if g is None or g.empty:
            continue
        filed_arr = g["filed"].to_numpy()
        # last index with filed <= asof
        pos = int(np.searchsorted(filed_arr, asof, side="right") - 1)
        if pos < 0:
            continue
        row = g.iloc[pos]
        has_fund[i] = 1
        flags = _flags_from_row(row)
        vals = [
            flags.get("f1_ocf_debt"),
            flags.get("f2_capex_ocf"),
            flags.get("f3_buybacks"),
            flags.get("f4_fcf_roa"),
        ]
        f1[i] = 1.0 if vals[0] is True else (0.0 if vals[0] is False else np.nan)
        f2[i] = 1.0 if vals[1] is True else (0.0 if vals[1] is False else np.nan)
        f3[i] = 1.0 if vals[2] is True else (0.0 if vals[2] is False else np.nan)
        f4[i] = 1.0 if vals[3] is True else (0.0 if vals[3] is False else np.nan)
        known = [v for v in vals if v is not None]
        n_pass[i] = int(sum(1 for v in known if v is True))
        if known and all(v is True for v in known) and len(known) == 4:
            passes[i] = 1
        if pd.notna(row.get("ocf_to_debt")):
            ocf_d[i] = float(row["ocf_to_debt"])
        if pd.notna(row.get("capex_to_ocf")):
            capex[i] = float(row["capex_to_ocf"])
        if pd.notna(row.get("net_buyback_pct")):
            buyback[i] = float(row["net_buyback_pct"])
        if pd.notna(row.get("fcf_roa")):
            fcf[i] = float(row["fcf_roa"])

    out["has_fund"] = has_fund
    out["fund_passes_screen"] = passes
    out["fund_n_filters_pass"] = n_pass
    out["f1_ocf_debt"] = f1
    out["f2_capex_ocf"] = f2
    out["f3_buybacks"] = f3
    out["f4_fcf_roa"] = f4
    out["ocf_to_debt"] = ocf_d
    out["capex_to_ocf"] = capex
    out["net_buyback_pct"] = buyback
    out["fcf_roa"] = fcf
    return out
