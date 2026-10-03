"""Cash-flow screen v2: 1y hold + FCF yield + shareholder yield + cash OP + momentum.

Changes vs v1 (video four-filter):
  1. Hold 1 year (annual rebalance via unique symbol-year)
  2. FCF yield = FCF / EV  (EV = mkt_cap + debt - cash) instead of FCF/Assets
  3. Total shareholder yield = (net buybacks + dividends) / mkt_cap  (hurdle 1%)
  4. Cash-based operating profitability = (OpInc - (NI - OCF)) / Assets
     replaces the OCF/debt gate
  5. Keep CapEx/OCF < 35%
  6. Overlay: 12-month price momentum > 0 at entry
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.api.sec.companyfacts_client import CompanyfactsClient
from utils.research.cashflow_four_filters import (
    _as_date,
    _close_on_or_before,
    book_stats,
    build_trades,
    dedupe_symbol_year,
    load_closes_many,
)
from utils.research.sec_fundamentals import (
    QUARTER_FPS,
    first_tag_series,
    index_by_end_fp,
    parse_date,
    ttm_sum,
)

# Thresholds (locked)
CAPEX_OCF_MAX = 0.35
FCF_YIELD_MIN = 0.05
SH_YIELD_MIN = 0.01
CASH_OP_ROA_MIN = 0.06
MOM_12M_MIN = 0.0
HOLD_YEARS = 1

DIV_TAGS = (
    "PaymentsOfDividendsCommonStock",
    "PaymentsOfDividends",
    "PaymentsOfOrdinaryDividends",
)
OP_TAGS = ("OperatingIncomeLoss",)
NI_TAGS = ("NetIncomeLoss",)
CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
)


def load_base_rows(
    *,
    fp_mode: str = "Q",
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
) -> pd.DataFrame:
    """Broad candidates: sector-ok, CapEx gate, have OCF/FCF/assets/shares."""
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
      AND capex_to_ocf IS NOT NULL AND capex_to_ocf < %s
    """
    params: List[Any] = [CAPEX_OCF_MAX]
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


def _ttm_at(facts: dict, tags: Sequence[str], end: date, *, abs_out: bool = False) -> Optional[float]:
    tag, rows = first_tag_series(facts, tags)
    if not tag:
        return None
    idx = index_by_end_fp(rows)
    q_vals: List[Tuple[date, Optional[float]]] = []
    for (end_s, fp), row in idx.items():
        if fp not in QUARTER_FPS:
            continue
        d = parse_date(end_s)
        if not d:
            continue
        v = row.get("val")
        try:
            fv = float(v) if v is not None else None
        except (TypeError, ValueError):
            fv = None
        if abs_out and fv is not None:
            fv = abs(fv)
        q_vals.append((d, fv))
    q_vals.sort(key=lambda x: x[0])
    return ttm_sum(q_vals, end, 4)


def _point_at(facts: dict, tags: Sequence[str], end: date) -> Optional[float]:
    tag, rows = first_tag_series(facts, tags)
    if not tag:
        return None
    idx = index_by_end_fp(rows)
    # prefer exact end match any fp
    best = None
    best_filed = ""
    end_s = end.isoformat()
    for (e, fp), row in idx.items():
        if e != end_s:
            continue
        filed = str(row.get("filed") or "")
        if best is None or filed >= best_filed:
            try:
                best = float(row["val"]) if row.get("val") is not None else None
                best_filed = filed
            except (TypeError, ValueError):
                continue
    if best is not None:
        return best
    # nearest prior balance-sheet point
    prior = None
    for (e, fp), row in idx.items():
        d = parse_date(e)
        if not d or d > end:
            continue
        if prior is None or d > prior[0]:
            try:
                prior = (d, float(row["val"]))
            except (TypeError, ValueError):
                continue
    return None if prior is None else prior[1]


def enrich_from_companyfacts(df: pd.DataFrame, cache_root: str = "data/sec") -> pd.DataFrame:
    """Add dividends_ttm, op_income_ttm, ni_ttm, cash from local companyfacts cache."""
    if df.empty:
        return df.copy()
    client = CompanyfactsClient(cache_root=cache_root, sleep_s=0)
    # cache per CIK
    facts_by_cik: Dict[str, Optional[dict]] = {}
    divs: List[Optional[float]] = []
    ops: List[Optional[float]] = []
    nis: List[Optional[float]] = []
    cashs: List[Optional[float]] = []

    for rec in df.itertuples(index=False):
        cik = str(rec.cik).zfill(10)
        if cik not in facts_by_cik:
            facts_by_cik[cik] = client.load_cached_companyfacts(cik)
        facts = facts_by_cik[cik]
        end = rec.period_end
        if not facts:
            divs.append(None)
            ops.append(None)
            nis.append(None)
            cashs.append(None)
            continue
        if str(rec.fp) == "FY":
            # use FY point for income; still TTM divs from quarters when possible
            op = _point_at(facts, OP_TAGS, end)
            ni = _point_at(facts, NI_TAGS, end)
            if op is None:
                op = _ttm_at(facts, OP_TAGS, end)
            if ni is None:
                ni = _ttm_at(facts, NI_TAGS, end)
        else:
            op = _ttm_at(facts, OP_TAGS, end)
            ni = _ttm_at(facts, NI_TAGS, end)
        div = _ttm_at(facts, DIV_TAGS, end, abs_out=True)
        cash = _point_at(facts, CASH_TAGS, end)
        divs.append(div)
        ops.append(op)
        nis.append(ni)
        cashs.append(abs(cash) if cash is not None else None)

    out = df.copy()
    out["dividends_ttm"] = divs
    out["op_income_ttm"] = ops
    out["ni_ttm"] = nis
    out["cash"] = cashs
    return out


def mom_12m(series: pd.Series, asof: date, lookback_days: int = 365) -> Optional[float]:
    """Price momentum: close_asof / close_(asof-lookback) - 1."""
    now = _close_on_or_before(series, asof)
    if now is None:
        return None
    past = _close_on_or_before(series, asof - timedelta(days=lookback_days))
    if past is None or past[1] <= 0:
        return None
    return now[1] / past[1] - 1.0


def apply_v2_screen(
    df: pd.DataFrame,
    closes: Dict[str, pd.Series],
    *,
    fcf_yield_min: float = FCF_YIELD_MIN,
    sh_yield_min: float = SH_YIELD_MIN,
    cash_op_roa_min: float = CASH_OP_ROA_MIN,
    mom_min: float = MOM_12M_MIN,
    require_momentum: bool = True,
) -> pd.DataFrame:
    """Apply v2 fundamental gates + optional 12m momentum."""
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
            ev = mkt  # avoid pathological EV
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
        # cash-based OP = OpInc - accruals; accruals ≈ NI - OCF
        if op is not None and ni is not None:
            cash_op = float(op) - (float(ni) - float(ocf))
        else:
            # fallback: OCF itself when income statement tags missing
            cash_op = float(ocf)
        cash_op_roa = cash_op / float(assets)
        if cash_op_roa <= cash_op_roa_min:
            continue

        capex_ratio = rec.get("capex_to_ocf")
        if capex_ratio is None or float(capex_ratio) >= CAPEX_OCF_MAX:
            continue

        mom = mom_12m(series, filed)
        if require_momentum:
            if mom is None or mom <= mom_min:
                continue

        rec = dict(rec)
        rec["mkt_cap"] = mkt
        rec["ev"] = ev
        rec["fcf_yield"] = fcf_yield
        rec["sh_yield"] = sh_yield
        rec["cash_op"] = cash_op
        rec["cash_op_roa"] = cash_op_roa
        rec["mom_12m"] = mom
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


@dataclass
class V2Result:
    screen_n: int
    trades: pd.DataFrame
    stats: Dict[str, Any]
    stats_unique_year: Dict[str, Any]
    params: Dict[str, Any]


def run_claimcheck_v2(
    *,
    fp_mode: str = "Q",
    hold_years: int = HOLD_YEARS,
    require_full_hold: bool = True,
    dedupe_year: bool = True,
    require_momentum: bool = True,
    fcf_yield_min: float = FCF_YIELD_MIN,
    sh_yield_min: float = SH_YIELD_MIN,
    cash_op_roa_min: float = CASH_OP_ROA_MIN,
    mom_min: float = MOM_12M_MIN,
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
    cache_root: str = "data/sec",
) -> V2Result:
    base = load_base_rows(fp_mode=fp_mode, min_filed=min_filed, max_filed=max_filed)
    base = enrich_from_companyfacts(base, cache_root=cache_root)
    symbols = sorted(set(base["symbol"].tolist()) | {"SPY"})
    closes = load_closes_many(symbols)
    spy = closes.get("SPY", pd.Series(dtype=float))

    screened = apply_v2_screen(
        base,
        closes,
        fcf_yield_min=fcf_yield_min,
        sh_yield_min=sh_yield_min,
        cash_op_roa_min=cash_op_roa_min,
        mom_min=mom_min,
        require_momentum=require_momentum,
    )
    screened_u = dedupe_symbol_year(screened) if dedupe_year else screened

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
    # attach v2 metrics onto trades for CSV
    if not trades_u.empty and not screened_u.empty:
        meta = screened_u[
            ["symbol", "filed", "fcf_yield", "sh_yield", "cash_op_roa", "mom_12m", "ev"]
        ].drop_duplicates(["symbol", "filed"])
        trades_u = trades_u.merge(meta, on=["symbol", "filed"], how="left")
        trades_all = trades_all.merge(meta, on=["symbol", "filed"], how="left")

    params = {
        "hold_years": hold_years,
        "fcf_yield_min": fcf_yield_min,
        "sh_yield_min": sh_yield_min,
        "cash_op_roa_min": cash_op_roa_min,
        "mom_12m_min": mom_min,
        "require_momentum": require_momentum,
        "capex_ocf_max": CAPEX_OCF_MAX,
    }
    return V2Result(
        screen_n=int(len(screened)),
        trades=trades_u if dedupe_year else trades_all,
        stats=book_stats(trades_all),
        stats_unique_year=book_stats(trades_u),
        params=params,
    )
