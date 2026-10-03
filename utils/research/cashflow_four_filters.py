"""Cash-flow four-filter stock screen claim-check (YouTube video screen).

Filters (all required):
  1. OCF / total_debt > 0.30  (debt=0 and OCF>0 passes)
  2. CapEx / OCF < 0.35
  3. Net buybacks TTM / mkt_cap > 0.02
  4. FCF / assets > 0.12

Sector exclusions via SIC (financials/energy/utilities/materials/RE).
Entry: next 1d close on/after ``filed``. Hold ``hold_years`` calendar years.
Claim: median ~86% 5y total return vs market ~66%.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.sec_fundamentals import attach_mkt_caps, FundamentalRow, passes_four_filters

CLAIMED = {
    "median_5y_pct": 86.0,
    "market_median_5y_pct": 66.0,
    "hold_years": 5,
}

OCF_DEBT_MIN = 0.30
CAPEX_OCF_MAX = 0.35
BUYBACK_PCT_MIN = 0.02
FCF_ROA_MIN = 0.12


def _as_date(d: Any) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def load_candidate_rows(
    *,
    fp_mode: str = "Q",
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
) -> pd.DataFrame:
    """Load rows that pass f1/f2/f4 and have buybacks+shares (f3 applied after mkt_cap)."""
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
           ocf_to_debt, capex_to_ocf, net_buyback_pct, fcf_roa,
           sic, sic_desc, exclude_sector, debt_zero
    FROM fundamentals_quarterly
    WHERE exclude_sector IS NOT TRUE
      AND period_end <= CURRENT_DATE
      AND {fp_clause}
      AND (
            (debt_zero IS TRUE AND ocf > 0)
            OR (ocf_to_debt IS NOT NULL AND ocf_to_debt > %s)
          )
      AND capex_to_ocf IS NOT NULL AND capex_to_ocf < %s
      AND fcf_roa IS NOT NULL AND fcf_roa > %s
      AND net_buybacks_ttm IS NOT NULL
      AND shares_out IS NOT NULL AND shares_out > 0
    """
    params: List[Any] = [OCF_DEBT_MIN, CAPEX_OCF_MAX, FCF_ROA_MIN]
    if min_filed is not None:
        sql += " AND filed >= %s"
        params.append(min_filed)
    if max_filed is not None:
        sql += " AND filed <= %s"
        params.append(max_filed)
    sql += " ORDER BY filed, symbol"

    rows = db.execute_query(sql, tuple(params))
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


def load_closes_many(symbols: Sequence[str]) -> Dict[str, pd.Series]:
    """ALPACA 1d closes keyed by symbol -> Series indexed by date."""
    db = get_timescaledb_client()
    if not db.connect():
        raise RuntimeError("TimescaleDB connection failed")
    syms = sorted({str(s).upper() for s in symbols})
    out: Dict[str, pd.Series] = {}
    bs = 100
    for i in range(0, len(syms), bs):
        batch = syms[i : i + bs]
        rows = db.execute_query(
            """
            SELECT symbol, (ts AT TIME ZONE 'UTC')::date AS d, close
            FROM market_data
            WHERE symbol = ANY(%s)
              AND timeframe = '1d'
              AND provider = 'ALPACA'
              AND close IS NOT NULL AND close > 0
            ORDER BY symbol, ts
            """,
            (batch,),
        )
        by: Dict[str, List[Tuple[date, float]]] = {}
        for r in rows or []:
            sym = str(r[0]).upper()
            by.setdefault(sym, []).append((_as_date(r[1]), float(r[2])))
        missing = [s for s in batch if s not in by]
        if missing:
            rows2 = db.execute_query(
                """
                SELECT symbol, (ts AT TIME ZONE 'UTC')::date AS d, close
                FROM market_data
                WHERE symbol = ANY(%s)
                  AND timeframe = '1d'
                  AND close IS NOT NULL AND close > 0
                ORDER BY symbol, ts
                """,
                (missing,),
            )
            for r in rows2 or []:
                sym = str(r[0]).upper()
                by.setdefault(sym, []).append((_as_date(r[1]), float(r[2])))
        for sym, pairs in by.items():
            s = pd.Series({d: px for d, px in pairs}).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            out[sym] = s
    return out


def _close_asof(series: pd.Series, d: date) -> Optional[Tuple[date, float]]:
    if series is None or series.empty:
        return None
    # first date >= d
    idx = series.index
    pos = idx.searchsorted(d)
    if pos >= len(idx):
        return None
    dd = idx[pos]
    return _as_date(dd), float(series.iloc[pos])


def _close_on_or_before(series: pd.Series, d: date) -> Optional[Tuple[date, float]]:
    if series is None or series.empty:
        return None
    idx = series.index
    pos = idx.searchsorted(d, side="right") - 1
    if pos < 0:
        return None
    dd = idx[pos]
    return _as_date(dd), float(series.iloc[pos])


def apply_filter3_with_prices(df: pd.DataFrame, closes: Dict[str, pd.Series]) -> pd.DataFrame:
    """Recompute mkt_cap / net_buyback_pct from shares x close asof filed; keep f3 passers."""
    if df.empty:
        return df.copy()
    rows_out = []
    for rec in df.to_dict(orient="records"):
        sym = rec["symbol"]
        filed = rec["filed"]
        shares = rec.get("shares_out")
        net_bb = rec.get("net_buybacks_ttm")
        mkt = rec.get("mkt_cap")
        px_hit = _close_on_or_before(closes.get(sym, pd.Series(dtype=float)), filed)
        if px_hit is not None and shares is not None and float(shares) > 0:
            mkt = float(shares) * float(px_hit[1])
        bb_pct = None
        if mkt is not None and mkt > 0 and net_bb is not None:
            bb_pct = float(net_bb) / float(mkt)
        if bb_pct is None or bb_pct <= BUYBACK_PCT_MIN:
            continue
        rec = dict(rec)
        rec["mkt_cap"] = mkt
        rec["net_buyback_pct"] = bb_pct
        # verify all four via dataclass path
        fr = FundamentalRow(
            symbol=sym,
            cik=str(rec.get("cik") or ""),
            period_end=rec["period_end"],
            filed=filed,
            fy=int(rec["fy"]) if rec.get("fy") is not None and not pd.isna(rec.get("fy")) else None,
            fp=str(rec["fp"]),
            ocf=rec.get("ocf"),
            capex=rec.get("capex"),
            fcf=rec.get("fcf"),
            total_debt=rec.get("total_debt"),
            assets=rec.get("assets"),
            shares_out=shares,
            net_buybacks_ttm=net_bb,
            mkt_cap=mkt,
            ocf_to_debt=rec.get("ocf_to_debt"),
            capex_to_ocf=rec.get("capex_to_ocf"),
            net_buyback_pct=bb_pct,
            fcf_roa=rec.get("fcf_roa"),
            debt_zero=bool(rec.get("debt_zero")),
            exclude_sector=bool(rec.get("exclude_sector")),
        )
        flags = passes_four_filters(fr)
        if not all(flags.get(k) is True for k in ("f1_ocf_debt", "f2_capex_ocf", "f3_buybacks", "f4_fcf_roa")):
            continue
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


def dedupe_symbol_year(df: pd.DataFrame) -> pd.DataFrame:
    """Keep earliest filed pass per symbol-calendar-year."""
    if df.empty:
        return df.copy()
    work = df.copy()
    work["filed_year"] = work["filed"].map(lambda d: d.year)
    work = work.sort_values(["symbol", "filed_year", "filed"])
    return work.drop_duplicates(["symbol", "filed_year"], keep="first").reset_index(drop=True)


def build_trades(
    screen: pd.DataFrame,
    closes: Dict[str, pd.Series],
    *,
    hold_years: int = 5,
    require_full_hold: bool = True,
    spy: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Buy next close on/after filed; sell close on/after filed+hold_years."""
    records = []
    hold_delta = timedelta(days=int(round(365.25 * hold_years)))
    for rec in screen.to_dict(orient="records"):
        sym = rec["symbol"]
        filed = rec["filed"]
        series = closes.get(sym)
        if series is None or series.empty:
            continue
        entry = _close_asof(series, filed)
        if entry is None:
            continue
        entry_d, entry_px = entry
        exit_target = filed + hold_delta
        exit_hit = _close_asof(series, exit_target)
        if exit_hit is None:
            if require_full_hold:
                continue
            # use last available
            last_d = _as_date(series.index[-1])
            last_px = float(series.iloc[-1])
            exit_d, exit_px = last_d, last_px
            full = False
        else:
            exit_d, exit_px = exit_hit
            full = True
            if require_full_hold and (exit_d - entry_d).days < int(365.25 * hold_years) - 30:
                # got a print but too early (data hole) — still accept if within ~1m
                pass

        ret = exit_px / entry_px - 1.0
        spy_ret = None
        if spy is not None and not spy.empty:
            se = _close_asof(spy, entry_d)
            sx = _close_asof(spy, exit_d) if full or not require_full_hold else _close_asof(spy, exit_target)
            if se and sx:
                spy_ret = sx[1] / se[1] - 1.0

        records.append(
            {
                "symbol": sym,
                "filed": filed,
                "period_end": rec["period_end"],
                "fp": rec["fp"],
                "buy_date": entry_d,
                "sell_date": exit_d,
                "buy_px": entry_px,
                "sell_px": exit_px,
                "ret": ret,
                "ret_pct": ret * 100.0,
                "spy_ret": spy_ret,
                "spy_ret_pct": None if spy_ret is None else spy_ret * 100.0,
                "full_hold": full,
                "hold_days": (exit_d - entry_d).days,
                "ocf_to_debt": rec.get("ocf_to_debt"),
                "capex_to_ocf": rec.get("capex_to_ocf"),
                "net_buyback_pct": rec.get("net_buyback_pct"),
                "fcf_roa": rec.get("fcf_roa"),
                "mkt_cap": rec.get("mkt_cap"),
                "sic": rec.get("sic"),
            }
        )
    cols = [
        "symbol",
        "filed",
        "period_end",
        "fp",
        "buy_date",
        "sell_date",
        "buy_px",
        "sell_px",
        "ret",
        "ret_pct",
        "spy_ret",
        "spy_ret_pct",
        "full_hold",
        "hold_days",
        "ocf_to_debt",
        "capex_to_ocf",
        "net_buyback_pct",
        "fcf_roa",
        "mkt_cap",
        "sic",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).sort_values(["buy_date", "symbol"]).reset_index(drop=True)


def book_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {"n": 0}
    r = trades["ret"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else None)
    spy = trades["spy_ret"].dropna().astype(float) if "spy_ret" in trades.columns else pd.Series(dtype=float)
    out = {
        "n": int(len(trades)),
        "n_symbols": int(trades["symbol"].nunique()),
        "win_rate_pct": round(100.0 * float((r > 0).mean()), 2),
        "mean_ret_pct": round(100.0 * float(r.mean()), 2),
        "median_ret_pct": round(100.0 * float(r.median()), 2),
        "p25_ret_pct": round(100.0 * float(r.quantile(0.25)), 2),
        "p75_ret_pct": round(100.0 * float(r.quantile(0.75)), 2),
        "expectancy_pct": round(100.0 * float(r.mean()), 2),
        "pf": None if pf is None or pf == float("inf") else round(float(pf), 3),
        "avg_hold_days": round(float(trades["hold_days"].mean()), 1),
        "buy_from": str(trades["buy_date"].min()),
        "buy_to": str(trades["buy_date"].max()),
        "spy_median_ret_pct": None
        if spy.empty
        else round(100.0 * float(spy.median()), 2),
        "spy_mean_ret_pct": None if spy.empty else round(100.0 * float(spy.mean()), 2),
        "excess_median_pct": None
        if spy.empty
        else round(100.0 * float(r.median() - spy.median()), 2),
    }
    # year of buy buckets
    by_year = {}
    years = pd.to_datetime(trades["buy_date"]).dt.year
    for y, g in trades.groupby(years):
        rr = g["ret"].astype(float)
        by_year[int(y)] = {
            "n": int(len(g)),
            "median_ret_pct": round(100.0 * float(rr.median()), 2),
            "mean_ret_pct": round(100.0 * float(rr.mean()), 2),
        }
    out["by_buy_year"] = by_year
    return out


@dataclass
class ClaimResult:
    screen_n: int
    trades: pd.DataFrame
    stats: Dict[str, Any]
    stats_unique_year: Dict[str, Any]
    claimed: Dict[str, float]


def run_claimcheck(
    *,
    fp_mode: str = "Q",
    hold_years: int = 5,
    require_full_hold: bool = True,
    dedupe_year: bool = True,
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
) -> ClaimResult:
    cands = load_candidate_rows(fp_mode=fp_mode, min_filed=min_filed, max_filed=max_filed)
    symbols = sorted(set(cands["symbol"].tolist()) | {"SPY"})
    closes = load_closes_many(symbols)
    spy = closes.get("SPY", pd.Series(dtype=float))

    screened = apply_filter3_with_prices(cands, closes)
    if dedupe_year:
        screened_u = dedupe_symbol_year(screened)
    else:
        screened_u = screened

    # Cap entry so full hold can complete: last price date - hold_years
    if require_full_hold and not spy.empty:
        last_px = _as_date(spy.index[-1])
        max_entry = last_px - timedelta(days=int(round(365.25 * hold_years)))
        screened = screened[screened["filed"] <= max_entry].copy()
        screened_u = screened_u[screened_u["filed"] <= max_entry].copy()

    trades_all = build_trades(
        screened, closes, hold_years=hold_years, require_full_hold=require_full_hold, spy=spy
    )
    trades_u = build_trades(
        screened_u, closes, hold_years=hold_years, require_full_hold=require_full_hold, spy=spy
    )
    return ClaimResult(
        screen_n=int(len(screened)),
        trades=trades_u if dedupe_year else trades_all,
        stats=book_stats(trades_all),
        stats_unique_year=book_stats(trades_u),
        claimed=dict(CLAIMED),
    )
