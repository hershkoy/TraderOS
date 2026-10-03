"""Cash-flow screen v3: top-N cross-section + relative 12-1 mom + CapEx exception + portfolio.

vs v2:
  - Soften absolute gates into a broad eligibility pool, then rank
  - Composite = rank(FCF yield) + rank(cash OP ROA) + rank(12-1 mom)
  - Keep top_n (default 30) per filed calendar year
  - Relative momentum: stock 12-1 return > SPY 12-1 return
  - CapEx/OCF < 35% OR (CapEx high AND cash_op_roa > 15%)
  - Equal-weight annual portfolio equity vs SPY and RSP
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.cashflow_four_filters import (
    _as_date,
    _close_asof,
    _close_on_or_before,
    book_stats,
    build_trades,
    dedupe_symbol_year,
    load_closes_many,
)
from utils.research.cashflow_four_filters_v2 import (
    CASH_OP_ROA_MIN,
    FCF_YIELD_MIN,
    SH_YIELD_MIN,
    enrich_from_companyfacts,
)

CAPEX_OCF_MAX = 0.35
CASH_OP_HIGH_REINVEST = 0.15  # allow CapEx>35% if cash OP ROA above this
TOP_N = 30
HOLD_YEARS = 1
SKIP_RECENT_DAYS = 21  # 12-1 momentum skip month


def load_base_rows_v3(
    *,
    fp_mode: str = "Q",
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
) -> pd.DataFrame:
    """Eligibility pool without hard CapEx gate (applied later with exception)."""
    from utils.db.timescaledb_client import get_timescaledb_client

    db = get_timescaledb_client()
    if not db.connect():
        raise RuntimeError("TimescaleDB connection failed")

    if fp_mode.upper() == "FY":
        fp_clause = "fp = 'FY'"
    elif fp_mode.upper() == "BOTH":
        fp_clause = "(fp LIKE 'Q%%' OR fp = 'FY')"
    else:
        fp_clause = "fp LIKE 'Q%%'"

    sql = f"""
    SELECT symbol, cik, period_end, filed, fy, fp,
           ocf, capex, fcf, total_debt, assets, shares_out,
           buybacks_ttm, issuance_ttm, net_buybacks_ttm, mkt_cap,
           capex_to_ocf, sic, exclude_sector, debt_zero
    FROM fundamentals_quarterly
    WHERE exclude_sector IS NOT TRUE
      AND period_end <= CURRENT_DATE
      AND {fp_clause}
      AND ocf IS NOT NULL
      AND fcf IS NOT NULL
      AND assets IS NOT NULL AND assets > 0
      AND shares_out IS NOT NULL AND shares_out > 0
      AND capex_to_ocf IS NOT NULL
    """
    params: List[Any] = []
    if min_filed is not None:
        sql += " AND filed >= %s"
        params.append(min_filed)
    if max_filed is not None:
        sql += " AND filed <= %s"
        params.append(max_filed)
    sql += " ORDER BY filed, symbol"
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
        "capex_to_ocf",
        "sic",
        "exclude_sector",
        "debt_zero",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    return df


def mom_12_1(series: pd.Series, asof: date, skip_days: int = SKIP_RECENT_DAYS) -> Optional[float]:
    """12-1 momentum: return from asof-365 to asof-skip_days."""
    end_d = asof - timedelta(days=int(skip_days))
    start_d = asof - timedelta(days=365)
    end_px = _close_on_or_before(series, end_d)
    start_px = _close_on_or_before(series, start_d)
    if end_px is None or start_px is None or start_px[1] <= 0:
        return None
    return end_px[1] / start_px[1] - 1.0


def build_eligible_pool(
    df: pd.DataFrame,
    closes: Dict[str, pd.Series],
    spy: pd.Series,
    *,
    fcf_yield_min: float = FCF_YIELD_MIN,
    sh_yield_min: float = SH_YIELD_MIN,
    cash_op_roa_min: float = CASH_OP_ROA_MIN,
    require_rel_momentum: bool = True,
) -> pd.DataFrame:
    """Broad fundamental eligibility + CapEx exception + relative 12-1 mom."""
    if df.empty:
        return df.copy()
    rows_out = []
    for rec in df.to_dict(orient="records"):
        sym = rec["symbol"]
        filed = rec["filed"]
        shares = rec.get("shares_out")
        series = closes.get(sym, pd.Series(dtype=float))
        px = _close_on_or_before(series, filed)
        if px is None or shares is None or float(shares) <= 0:
            continue
        mkt = float(shares) * float(px[1])
        if mkt <= 0:
            continue

        fcf = rec.get("fcf")
        debt = float(rec["total_debt"]) if rec.get("total_debt") is not None else 0.0
        cash = float(rec["cash"]) if rec.get("cash") is not None else 0.0
        ev = mkt + max(debt, 0.0) - max(cash, 0.0)
        if ev < mkt * 0.5:
            ev = mkt
        if fcf is None or ev <= 0:
            continue
        fcf_yield = float(fcf) / float(ev)
        if fcf_yield <= fcf_yield_min:
            continue

        net_bb = float(rec["net_buybacks_ttm"]) if rec.get("net_buybacks_ttm") is not None else 0.0
        divs = float(rec["dividends_ttm"]) if rec.get("dividends_ttm") is not None else 0.0
        sh_yield = (net_bb + divs) / mkt
        if sh_yield <= sh_yield_min:
            continue

        ocf = rec.get("ocf")
        op = rec.get("op_income_ttm")
        ni = rec.get("ni_ttm")
        assets = rec.get("assets")
        if ocf is None or assets is None or float(assets) <= 0:
            continue
        if op is not None and ni is not None:
            cash_op = float(op) - (float(ni) - float(ocf))
        else:
            cash_op = float(ocf)
        cash_op_roa = cash_op / float(assets)
        if cash_op_roa <= cash_op_roa_min:
            continue

        capex_ratio = rec.get("capex_to_ocf")
        if capex_ratio is None:
            continue
        capex_ok = float(capex_ratio) < CAPEX_OCF_MAX or cash_op_roa > CASH_OP_HIGH_REINVEST
        if not capex_ok:
            continue

        mom_stk = mom_12_1(series, filed)
        mom_spy = mom_12_1(spy, filed) if spy is not None and not spy.empty else None
        if require_rel_momentum:
            if mom_stk is None or mom_spy is None or mom_stk <= mom_spy:
                continue

        rec = dict(rec)
        rec["mkt_cap"] = mkt
        rec["ev"] = ev
        rec["fcf_yield"] = fcf_yield
        rec["sh_yield"] = sh_yield
        rec["cash_op"] = cash_op
        rec["cash_op_roa"] = cash_op_roa
        rec["mom_12_1"] = mom_stk
        rec["spy_mom_12_1"] = mom_spy
        rec["rel_mom"] = None if mom_stk is None or mom_spy is None else mom_stk - mom_spy
        rec["capex_exception"] = bool(float(capex_ratio) >= CAPEX_OCF_MAX)
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


def select_topn_by_year(
    eligible: pd.DataFrame,
    *,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Per filed-year: unique symbol (earliest filed), rank composite, keep top_n."""
    if eligible.empty:
        return eligible.copy()
    uniq = dedupe_symbol_year(eligible)
    uniq = uniq.copy()
    uniq["filed_year"] = uniq["filed"].map(lambda d: d.year)
    parts = []
    for y, g in uniq.groupby("filed_year"):
        gg = g.copy()
        # higher is better for all three
        gg["r_fcf"] = gg["fcf_yield"].rank(method="average", ascending=True, pct=True)
        gg["r_cop"] = gg["cash_op_roa"].rank(method="average", ascending=True, pct=True)
        # prefer relative mom if present else 12-1
        mom_col = "rel_mom" if gg["rel_mom"].notna().any() else "mom_12_1"
        gg["r_mom"] = gg[mom_col].rank(method="average", ascending=True, pct=True)
        gg["composite"] = gg["r_fcf"] + gg["r_cop"] + gg["r_mom"]
        gg = gg.sort_values(["composite", "fcf_yield"], ascending=False)
        parts.append(gg.head(int(top_n)))
    if not parts:
        return eligible.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _summarize_portfolio_slice(
    years_df: pd.DataFrame,
    benchmarks: Optional[Dict[str, pd.Series]],
) -> Dict[str, Any]:
    """Compound CAGR / excess on a year-table slice (already filtered)."""
    if years_df.empty:
        return {"n_years": 0}
    port = years_df["port_ret"].astype(float)
    equity = float((1.0 + port).prod())
    n = int(len(years_df))
    summary: Dict[str, Any] = {
        "n_years": n,
        "cagr_pct": round((equity ** (1.0 / n) - 1.0) * 100.0, 2),
        "total_return_pct": round((equity - 1.0) * 100.0, 2),
        "mean_year_ret_pct": round(100.0 * float(port.mean()), 2),
        "median_year_ret_pct": round(100.0 * float(port.median()), 2),
        "win_year_pct": round(100.0 * float((port > 0).mean()), 2),
        "final_equity": round(equity, 4),
    }
    for bname in (benchmarks or {}).keys():
        col = f"{bname}_ret"
        if col in years_df.columns and years_df[col].notna().any():
            br = years_df[col].dropna().astype(float)
            beq = float((1.0 + br).prod())
            summary[f"{bname}_cagr_pct"] = round(
                (beq ** (1.0 / len(br)) - 1.0) * 100.0, 2
            )
            summary[f"{bname}_total_return_pct"] = round((beq - 1.0) * 100.0, 2)
            summary[f"{bname}_median_year_ret_pct"] = round(
                100.0 * float(br.median()), 2
            )
            summary[f"excess_cagr_vs_{bname}_pct"] = round(
                summary["cagr_pct"] - summary[f"{bname}_cagr_pct"], 2
            )
    return summary


def portfolio_annual_rebalance(
    selected: pd.DataFrame,
    closes: Dict[str, pd.Series],
    *,
    hold_years: int = 1,
    benchmarks: Optional[Dict[str, pd.Series]] = None,
    min_names: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Equal-weight portfolio: each year buy top-N at next close after cohort max filed in year.

    Cohort year Y: names selected with filed in calendar year Y.
    Entry = next session on/after max(filed) among that year's selected names
            (simultaneous annual rebalance date), fallback per-name entry if needed.
    Exit = entry + hold_years.

    Headline CAGR uses years with n_names >= min_names (default 10) so single-name
    cohorts cannot inflate; raw years stay in the returned table.
    """
    if selected.empty:
        return pd.DataFrame(), {"n_years": 0}

    work = selected.copy()
    work["filed_year"] = work["filed"].map(lambda d: d.year)
    year_rows = []
    equity_all = 1.0

    for y in sorted(work["filed_year"].unique()):
        cohort = work[work["filed_year"] == y]
        rebal_signal = max(cohort["filed"])
        name_rets = []
        entry_dates = []
        exit_dates = []
        for rec in cohort.to_dict(orient="records"):
            sym = rec["symbol"]
            series = closes.get(sym)
            if series is None or series.empty:
                continue
            entry = _close_asof(series, rebal_signal)
            if entry is None:
                continue
            entry_d, entry_px = entry
            exit_target = entry_d + timedelta(days=int(round(365.25 * hold_years)))
            exit_hit = _close_asof(series, exit_target)
            if exit_hit is None:
                continue
            exit_d, exit_px = exit_hit
            name_rets.append(exit_px / entry_px - 1.0)
            entry_dates.append(entry_d)
            exit_dates.append(exit_d)
        if not name_rets:
            continue
        port_ret = float(np.mean(name_rets))
        equity_all *= 1.0 + port_ret
        row: Dict[str, Any] = {
            "year": int(y),
            "n_names": len(name_rets),
            "rebal_signal": rebal_signal,
            "entry_date": min(entry_dates),
            "exit_date": max(exit_dates),
            "port_ret": port_ret,
            "port_ret_pct": port_ret * 100.0,
            "equity_all": equity_all,
        }
        e0 = min(entry_dates)
        x0 = max(exit_dates)
        for bname, bseries in (benchmarks or {}).items():
            if bseries is None or bseries.empty:
                row[f"{bname}_ret"] = None
                continue
            be = _close_asof(bseries, e0)
            bx = _close_asof(bseries, x0)
            if be and bx:
                row[f"{bname}_ret"] = bx[1] / be[1] - 1.0
                row[f"{bname}_ret_pct"] = row[f"{bname}_ret"] * 100.0
            else:
                row[f"{bname}_ret"] = None
        year_rows.append(row)

    years_df = pd.DataFrame(year_rows)
    if years_df.empty:
        return years_df, {"n_years": 0}

    thick = years_df[years_df["n_names"] >= int(min_names)].copy()
    if thick.empty:
        summary = _summarize_portfolio_slice(years_df, benchmarks)
        summary["min_names"] = int(min_names)
        summary["n_years_all"] = int(len(years_df))
        summary["note"] = f"no years with n_names>={min_names}; reporting all"
    else:
        eq = 1.0
        eq_col = []
        for r in thick["port_ret"].astype(float):
            eq *= 1.0 + float(r)
            eq_col.append(eq)
        thick = thick.copy()
        thick["equity"] = eq_col
        summary = _summarize_portfolio_slice(thick, benchmarks)
        summary["min_names"] = int(min_names)
        summary["n_years_all"] = int(len(years_df))
        all_sum = _summarize_portfolio_slice(years_df, benchmarks)
        summary["cagr_all_years_pct"] = all_sum.get("cagr_pct")
        summary["excess_cagr_all_vs_SPY_pct"] = all_sum.get("excess_cagr_vs_SPY_pct")
        years_df = years_df.merge(thick[["year", "equity"]], on="year", how="left")

    return years_df, summary


def load_yahoo_daily(symbol: str, cache_path: Path, *, refresh: bool = False) -> pd.Series:
    """Yahoo chart API -> close Series indexed by date (for RSP etc.)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists() or cache_path.stat().st_size < 1000:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1=946684800&period2=9999999999&interval=1d&events=history"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            cache_path.write_bytes(resp.read())
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    pairs = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        d = datetime.utcfromtimestamp(t).date()
        pairs.append((d, float(c)))
    s = pd.Series({d: px for d, px in pairs}).sort_index()
    return s[~s.index.duplicated(keep="last")]


@dataclass
class V3Result:
    eligible_n: int
    selected_n: int
    trades: pd.DataFrame
    trade_stats: Dict[str, Any]
    portfolio_years: pd.DataFrame
    portfolio_stats: Dict[str, Any]
    params: Dict[str, Any]


def run_claimcheck_v3(
    *,
    fp_mode: str = "Q",
    top_n: int = TOP_N,
    hold_years: int = HOLD_YEARS,
    require_full_hold: bool = True,
    require_rel_momentum: bool = True,
    fcf_yield_min: float = FCF_YIELD_MIN,
    sh_yield_min: float = SH_YIELD_MIN,
    cash_op_roa_min: float = CASH_OP_ROA_MIN,
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
    cache_root: str = "data/sec",
    yahoo_dir: Optional[Path] = None,
    refresh_yahoo: bool = False,
) -> V3Result:
    base = load_base_rows_v3(fp_mode=fp_mode, min_filed=min_filed, max_filed=max_filed)
    base = enrich_from_companyfacts(base, cache_root=cache_root)
    symbols = sorted(set(base["symbol"].tolist()) | {"SPY"})
    closes = load_closes_many(symbols)
    spy = closes.get("SPY", pd.Series(dtype=float))

    # RSP from Yahoo (not in TimescaleDB)
    ydir = yahoo_dir or Path("reports/research/cashflow_four_filters_v3/yahoo")
    rsp = load_yahoo_daily("RSP", ydir / "rsp.json", refresh=refresh_yahoo)

    eligible = build_eligible_pool(
        base,
        closes,
        spy,
        fcf_yield_min=fcf_yield_min,
        sh_yield_min=sh_yield_min,
        cash_op_roa_min=cash_op_roa_min,
        require_rel_momentum=require_rel_momentum,
    )
    selected = select_topn_by_year(eligible, top_n=top_n)

    if require_full_hold and not spy.empty:
        last_px = _as_date(spy.index[-1])
        max_entry = last_px - timedelta(days=int(round(365.25 * hold_years)))
        selected = selected[selected["filed"] <= max_entry].copy()

    trades = build_trades(
        selected, closes, hold_years=hold_years, require_full_hold=require_full_hold, spy=spy
    )
    # attach RSP return per trade
    if not trades.empty and not rsp.empty:
        rsp_rets = []
        for rec in trades.to_dict(orient="records"):
            be = _close_asof(rsp, rec["buy_date"])
            bx = _close_asof(rsp, rec["sell_date"])
            if be and bx:
                rsp_rets.append(bx[1] / be[1] - 1.0)
            else:
                rsp_rets.append(None)
        trades = trades.copy()
        trades["rsp_ret"] = rsp_rets
        trades["rsp_ret_pct"] = [
            None if x is None else x * 100.0 for x in rsp_rets
        ]

    tstats = book_stats(trades)
    if not trades.empty and "rsp_ret" in trades.columns:
        rr = trades["rsp_ret"].dropna().astype(float)
        if not rr.empty:
            tstats["rsp_median_ret_pct"] = round(100.0 * float(rr.median()), 2)
            tstats["rsp_mean_ret_pct"] = round(100.0 * float(rr.mean()), 2)
            tstats["excess_median_vs_rsp_pct"] = round(
                float(trades["ret"].median() * 100.0) - tstats["rsp_median_ret_pct"], 2
            )

    port_years, port_stats = portfolio_annual_rebalance(
        selected,
        closes,
        hold_years=hold_years,
        benchmarks={"SPY": spy, "RSP": rsp},
    )

    params = {
        "top_n": top_n,
        "hold_years": hold_years,
        "fcf_yield_min": fcf_yield_min,
        "sh_yield_min": sh_yield_min,
        "cash_op_roa_min": cash_op_roa_min,
        "capex_ocf_max": CAPEX_OCF_MAX,
        "cash_op_high_reinvest": CASH_OP_HIGH_REINVEST,
        "require_rel_momentum": require_rel_momentum,
        "composite": "rank(fcf_yield)+rank(cash_op_roa)+rank(rel_mom)",
    }
    return V3Result(
        eligible_n=int(len(eligible)),
        selected_n=int(len(selected)),
        trades=trades,
        trade_stats=tstats,
        portfolio_years=port_years,
        portfolio_stats=port_stats,
        params=params,
    )
