"""Tachun long-only screen + 20y DCF equal-weight book (estimate-free).

Survivor-biased (current listings only) and estimate-free (no forward EPS).
As-of for statement gates is ``filed``. Quarterly rows use TTM flows from the
SEC parser. Five-year growth / uptrend use FY history filed on or before the
evaluation date.

Portfolio: equal-weight up to ``max_names`` among currently passing names with
price at or below ``trim_mult`` x optimistic DCF value. Cash for unused weight.
One failed screen while held is allowed; two consecutive failed filings exit.
No stop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.metrics import perf_stats, passes_phase5_gates
from utils.research.panel import load_spy_close, month_ends, to_naive_index

logger = logging.getLogger(__name__)

# Trailing gates (course Finviz-style, no forward EPS).
MKT_CAP_MIN = 2_000_000_000.0
EPS_YOY_MIN = 0.05
ROE_MIN = 0.15
CURRENT_RATIO_MIN = 1.0
GROSS_MARGIN_MIN = 0.40
OPERATING_MARGIN_MIN = 0.20
NET_MARGIN_MIN = 0.10
DEBT_TO_EBITDA_MAX = 4.0
INTEREST_TO_OCF_MAX = 0.40

# DCF defaults.
ERP = 0.05  # equity risk premium over DGS10
GROWTH_CAP = 0.15
TERMINAL_GROWTH = 0.02
DCF_YEARS = 20
BETA_FLOOR = 0.50
BETA_CEIL = 3.00
BETA_OPT_CAP = 1.50
TRIM_MULT = 1.50
MAX_NAMES = 15
RF_FALLBACK = 0.04
FRED_DGS10_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
DGS10_CACHE = Path("data/cache/fred/dgs10.csv")


@dataclass
class ScreenResult:
    symbol: str
    filed: date
    period_end: date
    fp: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    mkt_cap: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    ocf: Optional[float] = None
    fcf: Optional[float] = None
    diluted_eps: Optional[float] = None
    diluted_shares: Optional[float] = None
    shares_out: Optional[float] = None
    sales_cagr_5y: Optional[float] = None
    eps_cagr_5y: Optional[float] = None
    eps_yoy: Optional[float] = None
    roe: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    interest_to_ocf: Optional[float] = None
    debt_zero: bool = False


@dataclass
class Valuation:
    symbol: str
    asof: date
    fcf0: float
    g0: float
    rf: float
    beta_raw: float
    beta_opt: float
    beta_pess: float
    v_opt: float
    v_pess: float
    shares: float


@dataclass
class TachunBacktestResult:
    equity: pd.Series
    invested: pd.Series
    spy_equity: pd.Series
    holdings: pd.DataFrame
    screen_events: pd.DataFrame
    valuations: pd.DataFrame
    stats: Dict[str, Any]
    notes: str = ""


def _as_date(d: Any) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _finite(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def cagr(first: float, last: float, years: float) -> Optional[float]:
    if first is None or last is None or years is None:
        return None
    if first <= 0 or last <= 0 or years <= 0:
        return None
    return float((last / first) ** (1.0 / years) - 1.0)


def years_between(a: date, b: date) -> float:
    return max((b - a).days / 365.25, 1e-9)


def uptrend_ok(values: Sequence[Tuple[date, float]], *, min_points: int = 2) -> bool:
    """Uptrend as last > first over the provided points (callers pass the FY window ends)."""
    pts = [(d, float(v)) for d, v in values if v is not None and np.isfinite(v)]
    if len(pts) < min_points:
        return False
    span = years_between(pts[0][0], pts[-1][0])
    if span < 3.5:
        return False
    return pts[-1][1] > pts[0][1]


def fade_growth_path(g0: float, years: int = DCF_YEARS, g_term: float = TERMINAL_GROWTH) -> List[float]:
    g0 = float(np.clip(g0, -0.05, GROWTH_CAP))
    if years <= 1:
        return [g0]
    return [g0 + (g_term - g0) * (t / float(years)) for t in range(1, years + 1)]


def dcf_equity_value(
    fcf0: float,
    g0: float,
    discount: float,
    *,
    years: int = DCF_YEARS,
    g_term: float = TERMINAL_GROWTH,
) -> Optional[float]:
    """20y fade DCF on a cash-flow level; returns total equity value (not per share)."""
    if fcf0 is None or not np.isfinite(fcf0) or fcf0 <= 0:
        return None
    if discount is None or not np.isfinite(discount) or discount <= g_term:
        return None
    gs = fade_growth_path(g0, years=years, g_term=g_term)
    fcf = float(fcf0)
    pv = 0.0
    for t, g in enumerate(gs, start=1):
        fcf *= 1.0 + float(g)
        pv += fcf / ((1.0 + discount) ** t)
    tv = fcf * (1.0 + g_term) / (discount - g_term)
    pv += tv / ((1.0 + discount) ** years)
    return float(pv)


def haircut_betas(beta_raw: float) -> Tuple[float, float, float]:
    """Return (raw_clamped, optimistic, pessimistic). Discretionary haircut is fixed."""
    raw = float(np.clip(beta_raw, BETA_FLOOR, BETA_CEIL))
    opt = float(min(0.5 * raw + 0.5 * 1.0, BETA_OPT_CAP))
    opt = max(opt, BETA_FLOOR)
    pess = float(min(max(raw * 0.85, opt + 0.20), BETA_CEIL))
    return raw, opt, pess


def rolling_beta(
    stock: pd.Series,
    market: pd.Series,
    asof: date,
    *,
    lookback_days: int = 252 * 3,
    min_obs: int = 126,
) -> Optional[float]:
    """Causal daily beta vs market using returns strictly before ``asof``."""
    if stock is None or market is None or stock.empty or market.empty:
        return None
    end = pd.Timestamp(asof) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(lookback_days) + 30)
    s = stock.loc[(stock.index >= start) & (stock.index <= end)].astype(float)
    m = market.reindex(s.index).astype(float)
    df = pd.concat([s, m], axis=1, keys=["s", "m"]).dropna()
    if len(df) < min_obs + 1:
        return None
    rs = df["s"].pct_change().dropna()
    rm = df["m"].pct_change().reindex(rs.index).dropna()
    aligned = pd.concat([rs, rm], axis=1, keys=["s", "m"]).dropna()
    if len(aligned) < min_obs:
        return None
    var_m = float(aligned["m"].var(ddof=0))
    if var_m <= 1e-18:
        return None
    cov = float(aligned["s"].cov(aligned["m"]))
    return cov / var_m


def load_dgs10(
    *,
    cache_path: Path = DGS10_CACHE,
    url: str = FRED_DGS10_CSV,
    force: bool = False,
) -> pd.Series:
    """FRED DGS10 daily percent yield -> decimal series indexed by date."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not cache_path.exists():
        logger.info("Downloading FRED DGS10 -> %s", cache_path)
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path)
    col = "DGS10" if "DGS10" in df.columns else df.columns[-1]
    dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    val = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(val.values, index=dt).dropna()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # FRED stores percent.
    s = s.astype(float) / 100.0
    s = s[s > 0]
    return s


def rf_asof(dgs10: pd.Series, d: date, fallback: float = RF_FALLBACK) -> float:
    if dgs10 is None or dgs10.empty:
        return float(fallback)
    hit = dgs10.loc[dgs10.index <= pd.Timestamp(d)]
    if hit.empty:
        return float(fallback)
    return float(hit.iloc[-1])


def load_fundamentals_panel(
    *,
    min_filed: Optional[date] = None,
    max_filed: Optional[date] = None,
) -> pd.DataFrame:
    db = get_timescaledb_client()
    if not db.connect():
        raise RuntimeError("TimescaleDB connection failed")
    sql = """
    SELECT symbol, cik, period_end, filed, fy, fp,
           ocf, capex, fcf, total_debt, assets, shares_out,
           mkt_cap, revenue, gross_profit, operating_income, net_income,
           pretax_income, interest_expense, interest_income,
           current_assets, current_liabilities, equity,
           diluted_shares, diluted_eps, depreciation, ebitda,
           gross_margin, operating_margin, net_margin, roe, current_ratio,
           debt_to_ebitda, net_interest, interest_to_ocf,
           debt_zero, exclude_sector
    FROM fundamentals_quarterly
    WHERE period_end <= CURRENT_DATE
      AND filed <= CURRENT_DATE
      AND filed >= DATE '2008-01-01'
    """
    params: List[Any] = []
    if min_filed is not None:
        sql += " AND filed >= %s"
        params.append(min_filed)
    if max_filed is not None:
        sql += " AND filed <= %s"
        params.append(max_filed)
    sql += " ORDER BY symbol, filed, period_end, fp"
    rows = db.execute_query(sql, tuple(params) if params else None)
    cols = [
        "symbol", "cik", "period_end", "filed", "fy", "fp",
        "ocf", "capex", "fcf", "total_debt", "assets", "shares_out",
        "mkt_cap", "revenue", "gross_profit", "operating_income", "net_income",
        "pretax_income", "interest_expense", "interest_income",
        "current_assets", "current_liabilities", "equity",
        "diluted_shares", "diluted_eps", "depreciation", "ebitda",
        "gross_margin", "operating_margin", "net_margin", "roe", "current_ratio",
        "debt_to_ebitda", "net_interest", "interest_to_ocf",
        "debt_zero", "exclude_sector",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    for c in df.columns:
        if c in ("symbol", "cik", "period_end", "filed", "fp", "fy", "debt_zero", "exclude_sector"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fy_history(df_sym: pd.DataFrame) -> pd.DataFrame:
    fy = df_sym[df_sym["fp"] == "FY"].copy()
    if fy.empty:
        return fy
    fy = fy.sort_values(["period_end", "filed"])
    fy = fy.drop_duplicates("period_end", keep="last")
    return fy


def _pick_fy_window(fy: pd.DataFrame, asof: date, n: int = 5) -> pd.DataFrame:
    if fy.empty:
        return fy
    sub = fy[fy["filed"] <= asof]
    if len(sub) < n:
        return sub
    return sub.iloc[-n:]


def evaluate_screen_row(
    row: pd.Series,
    fy_hist: pd.DataFrame,
    *,
    mkt_cap_override: Optional[float] = None,
) -> ScreenResult:
    sym = str(row["symbol"]).upper()
    filed = _as_date(row["filed"])
    period_end = _as_date(row["period_end"])
    reasons: List[str] = []

    mkt = _finite(mkt_cap_override if mkt_cap_override is not None else row.get("mkt_cap"))
    roe = _finite(row.get("roe"))
    cr = _finite(row.get("current_ratio"))
    gm = _finite(row.get("gross_margin"))
    om = _finite(row.get("operating_margin"))
    nm = _finite(row.get("net_margin"))
    d2e = _finite(row.get("debt_to_ebitda"))
    i2o = _finite(row.get("interest_to_ocf"))
    debt_zero = bool(row.get("debt_zero"))
    eps = _finite(row.get("diluted_eps"))

    sales_cagr = None
    eps_cagr = None
    eps_yoy = None
    # Prefer FY rows that actually carry revenue so early null XBRL years do not
    # zero out CAGR for names like MSFT/AAPL.
    fy_asof = fy_hist[fy_hist["filed"] <= filed] if not fy_hist.empty else fy_hist
    fy_rev = fy_asof[fy_asof["revenue"].notna()] if not fy_asof.empty else fy_asof
    win = fy_rev.iloc[-5:] if len(fy_rev) >= 5 else fy_rev
    if len(win) >= 5:
        y0 = win.iloc[0]
        y1 = win.iloc[-1]
        yrs = years_between(_as_date(y0["period_end"]), _as_date(y1["period_end"]))
        r0 = _finite(y0.get("revenue"))
        r1 = _finite(y1.get("revenue"))
        sales_cagr = cagr(r0 or 0.0, r1 or 0.0, yrs)
        # Prefer diluted EPS; fall back to NI / diluted shares.
        e0 = _finite(y0.get("diluted_eps"))
        e1 = _finite(y1.get("diluted_eps"))
        if e0 is None or e1 is None:
            sh0 = _finite(y0.get("diluted_shares")) or _finite(y0.get("shares_out"))
            sh1 = _finite(y1.get("diluted_shares")) or _finite(y1.get("shares_out"))
            ni0 = _finite(y0.get("net_income"))
            ni1 = _finite(y1.get("net_income"))
            if sh0 and sh1 and sh0 > 0 and sh1 > 0 and ni0 is not None and ni1 is not None:
                e0 = ni0 / sh0
                e1 = ni1 / sh1
        if e0 is not None and e1 is not None and e0 > 0 and e1 > 0:
            eps_cagr = cagr(e0, e1, yrs)

        def _ends(col: str) -> list:
            pts = []
            for _, r in win.iterrows():
                v = _finite(r.get(col))
                if v is not None:
                    pts.append((_as_date(r["period_end"]), v))
            if len(pts) < 2:
                return []
            return [pts[0], pts[-1]]

        if not uptrend_ok(_ends("revenue"), min_points=2):
            reasons.append("rev_uptrend")
        if not uptrend_ok(_ends("net_income"), min_points=2):
            reasons.append("ni_uptrend")
        if not uptrend_ok(_ends("ocf"), min_points=2):
            reasons.append("ocf_uptrend")
    else:
        reasons.append("fy_hist<5")

    # YoY EPS from the last two revenue-bearing FY rows when available.
    if len(win) >= 2:
        prev = win.iloc[-2]
        cur = win.iloc[-1]
        e_cur = _finite(cur.get("diluted_eps"))
        e_prev = _finite(prev.get("diluted_eps"))
        if e_cur is None or e_prev is None or e_prev == 0:
            sh_c = _finite(cur.get("diluted_shares")) or _finite(cur.get("shares_out"))
            sh_p = _finite(prev.get("diluted_shares")) or _finite(prev.get("shares_out"))
            ni_c = _finite(cur.get("net_income"))
            ni_p = _finite(prev.get("net_income"))
            if sh_c and sh_p and sh_c > 0 and sh_p > 0 and ni_c is not None and ni_p is not None:
                e_cur = ni_c / sh_c
                e_prev = ni_p / sh_p
        if e_cur is not None and e_prev is not None and e_prev > 0:
            eps_yoy = e_cur / e_prev - 1.0

    if mkt is None or mkt < MKT_CAP_MIN:
        reasons.append("mkt_cap")
    if sales_cagr is None or sales_cagr <= 0:
        reasons.append("sales_cagr_5y")
    if eps_cagr is None or eps_cagr <= 0:
        reasons.append("eps_cagr_5y")
    if eps_yoy is None or eps_yoy <= EPS_YOY_MIN:
        reasons.append("eps_yoy")
    if roe is None or roe < ROE_MIN:
        reasons.append("roe")
    if cr is None or cr < CURRENT_RATIO_MIN:
        reasons.append("current_ratio")
    if gm is None or gm < GROSS_MARGIN_MIN:
        reasons.append("gross_margin")
    if om is None or om < OPERATING_MARGIN_MIN:
        reasons.append("operating_margin")
    if nm is None or nm < NET_MARGIN_MIN:
        reasons.append("net_margin")
    if not debt_zero:
        if d2e is None or d2e > DEBT_TO_EBITDA_MAX:
            reasons.append("debt_to_ebitda")
    if i2o is None:
        # Missing interest is treated as 0 burden only when both legs are null/zero-ish.
        ix = _finite(row.get("interest_expense"))
        ii = _finite(row.get("interest_income"))
        ocf = _finite(row.get("ocf"))
        if ix is None and ii is None:
            i2o = 0.0
        elif ocf is not None and ocf > 0:
            net_i = (abs(ix) if ix is not None else 0.0) - (ii if ii is not None else 0.0)
            i2o = net_i / ocf
    if i2o is None or i2o > INTEREST_TO_OCF_MAX:
        reasons.append("interest_to_ocf")

    return ScreenResult(
        symbol=sym,
        filed=filed,
        period_end=period_end,
        fp=str(row.get("fp")),
        passed=len(reasons) == 0,
        reasons=reasons,
        mkt_cap=mkt,
        revenue=_finite(row.get("revenue")),
        net_income=_finite(row.get("net_income")),
        ocf=_finite(row.get("ocf")),
        fcf=_finite(row.get("fcf")),
        diluted_eps=eps,
        diluted_shares=_finite(row.get("diluted_shares")),
        shares_out=_finite(row.get("shares_out")),
        sales_cagr_5y=sales_cagr,
        eps_cagr_5y=eps_cagr,
        eps_yoy=eps_yoy,
        roe=roe,
        current_ratio=cr,
        gross_margin=gm,
        operating_margin=om,
        net_margin=nm,
        debt_to_ebitda=d2e,
        interest_to_ocf=i2o,
        debt_zero=debt_zero,
    )


def build_screen_events(
    fundamentals: pd.DataFrame,
    closes: Optional[Dict[str, pd.Series]] = None,
    *,
    fp_mode: str = "Q",
) -> pd.DataFrame:
    """Evaluate trailing screen on each filing in ``fp_mode`` (Q/FY/BOTH)."""
    if fundamentals.empty:
        return pd.DataFrame()
    mode = fp_mode.upper()
    if mode == "FY":
        mask = fundamentals["fp"] == "FY"
    elif mode == "BOTH":
        mask = fundamentals["fp"].astype(str).str.startswith("Q") | (fundamentals["fp"] == "FY")
    else:
        mask = fundamentals["fp"].astype(str).str.startswith("Q")
    events = fundamentals.loc[mask].copy()
    out: List[Dict[str, Any]] = []
    for sym, g in fundamentals.groupby("symbol", sort=False):
        fy = _fy_history(g)
        eg = events[events["symbol"] == sym]
        px = None if closes is None else closes.get(str(sym).upper())
        for _, row in eg.iterrows():
            mkt_override = None
            shares = _finite(row.get("shares_out")) or _finite(row.get("diluted_shares"))
            if px is not None and shares and shares > 0:
                hit = px.loc[px.index <= pd.Timestamp(row["filed"])]
                if not hit.empty:
                    mkt_override = float(shares) * float(hit.iloc[-1])
            sr = evaluate_screen_row(row, fy, mkt_cap_override=mkt_override)
            out.append(
                {
                    "symbol": sr.symbol,
                    "filed": sr.filed,
                    "period_end": sr.period_end,
                    "fp": sr.fp,
                    "passed": sr.passed,
                    "reasons": ",".join(sr.reasons),
                    "mkt_cap": sr.mkt_cap,
                    "revenue": sr.revenue,
                    "net_income": sr.net_income,
                    "ocf": sr.ocf,
                    "fcf": sr.fcf,
                    "diluted_eps": sr.diluted_eps,
                    "diluted_shares": sr.diluted_shares,
                    "shares_out": sr.shares_out,
                    "sales_cagr_5y": sr.sales_cagr_5y,
                    "eps_cagr_5y": sr.eps_cagr_5y,
                    "eps_yoy": sr.eps_yoy,
                    "roe": sr.roe,
                    "current_ratio": sr.current_ratio,
                    "gross_margin": sr.gross_margin,
                    "operating_margin": sr.operating_margin,
                    "net_margin": sr.net_margin,
                    "debt_to_ebitda": sr.debt_to_ebitda,
                    "interest_to_ocf": sr.interest_to_ocf,
                    "debt_zero": sr.debt_zero,
                }
            )
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df = df.sort_values(["filed", "symbol"]).reset_index(drop=True)
    return df


def value_passer(
    event: Dict[str, Any] | pd.Series,
    *,
    stock_px: pd.Series,
    spy_px: pd.Series,
    dgs10: pd.Series,
    g0_override: Optional[float] = None,
) -> Optional[Valuation]:
    filed = _as_date(event["filed"])
    fcf0 = _finite(event.get("fcf"))
    if fcf0 is None or fcf0 <= 0:
        fcf0 = _finite(event.get("net_income"))
    if fcf0 is None or fcf0 <= 0:
        return None
    shares = _finite(event.get("diluted_shares")) or _finite(event.get("shares_out"))
    if shares is None or shares <= 0:
        return None
    g0 = g0_override
    if g0 is None:
        g0 = _finite(event.get("sales_cagr_5y"))
    if g0 is None:
        g0 = 0.05
    g0 = float(np.clip(g0, 0.0, GROWTH_CAP))
    beta = rolling_beta(stock_px, spy_px, filed)
    if beta is None:
        beta = 1.0
    raw, opt_b, pess_b = haircut_betas(beta)
    rf = rf_asof(dgs10, filed)
    r_opt = rf + opt_b * ERP
    r_pess = rf + pess_b * ERP
    v_opt_eq = dcf_equity_value(fcf0, g0, r_opt)
    v_pess_eq = dcf_equity_value(fcf0, g0, r_pess)
    if v_opt_eq is None or v_pess_eq is None:
        return None
    return Valuation(
        symbol=str(event["symbol"]).upper(),
        asof=filed,
        fcf0=float(fcf0),
        g0=g0,
        rf=rf,
        beta_raw=raw,
        beta_opt=opt_b,
        beta_pess=pess_b,
        v_opt=float(v_opt_eq / shares),
        v_pess=float(v_pess_eq / shares),
        shares=float(shares),
    )


def load_closes_prefix(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    workers: int = 4,
) -> Dict[str, pd.Series]:
    """ALPACA daily with IB prefix stitch (complete history path)."""
    frames = load_ohlcv_many(
        list(symbols),
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=workers,
        chunk_size=50,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    out: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = to_naive_index(df["close"].astype(float))
        out[str(sym).upper()] = s
    return out


def _month_end_dates(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    ends = month_ends(calendar)
    return list(pd.DatetimeIndex(ends))


def simulate_ew_book(
    screen_events: pd.DataFrame,
    valuations: pd.DataFrame,
    closes: Dict[str, pd.Series],
    spy: pd.Series,
    *,
    start: date,
    end: date,
    max_names: int = MAX_NAMES,
    trim_mult: float = TRIM_MULT,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Month-end equal-weight book with one-fail allowance and DCF trim."""
    if screen_events.empty:
        idx = spy.loc[(spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))].index
        z = pd.Series(1.0, index=idx)
        return z, pd.Series(0.0, index=idx), pd.DataFrame()

    # Latest screen state per symbol over time via event list.
    events = screen_events.sort_values(["filed", "symbol"]).reset_index(drop=True)
    val_map: Dict[Tuple[str, date], Valuation] = {}
    for _, r in valuations.iterrows():
        val_map[(str(r["symbol"]).upper(), _as_date(r["asof"]))] = Valuation(
            symbol=str(r["symbol"]).upper(),
            asof=_as_date(r["asof"]),
            fcf0=float(r["fcf0"]),
            g0=float(r["g0"]),
            rf=float(r["rf"]),
            beta_raw=float(r["beta_raw"]),
            beta_opt=float(r["beta_opt"]),
            beta_pess=float(r["beta_pess"]),
            v_opt=float(r["v_opt"]),
            v_pess=float(r["v_pess"]),
            shares=float(r["shares"]),
        )

    # Build calendar from SPY.
    cal = spy.loc[(spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))].index
    if len(cal) < 2:
        raise RuntimeError("SPY calendar too short")
    rebalance_days = _month_end_dates(pd.DatetimeIndex(cal))

    # State
    fail_streak: Dict[str, int] = {}
    last_pass: Dict[str, bool] = {}
    last_val: Dict[str, Valuation] = {}
    held: Dict[str, float] = {}  # symbol -> weight
    event_i = 0
    n_events = len(events)

    equity = 1.0
    eq_rows: List[Tuple[pd.Timestamp, float]] = []
    inv_rows: List[Tuple[pd.Timestamp, float]] = []
    hold_rows: List[Dict[str, Any]] = []
    prev_day: Optional[pd.Timestamp] = None

    rebalance_set = set(rebalance_days)

    for day in cal:
        # Apply filings dated on this day (filed <= day).
        while event_i < n_events and pd.Timestamp(events.iloc[event_i]["filed"]) <= day:
            ev = events.iloc[event_i]
            sym = str(ev["symbol"]).upper()
            passed = bool(ev["passed"])
            last_pass[sym] = passed
            if passed:
                fail_streak[sym] = 0
                key = (sym, _as_date(ev["filed"]))
                if key in val_map:
                    last_val[sym] = val_map[key]
            else:
                fail_streak[sym] = int(fail_streak.get(sym, 0)) + 1
            event_i += 1

        # Daily mark-to-market of prior weights.
        if prev_day is not None and held:
            port_ret = 0.0
            wsum = 0.0
            for sym, w in list(held.items()):
                px = closes.get(sym)
                if px is None or prev_day not in px.index or day not in px.index:
                    continue
                p0 = float(px.loc[prev_day])
                p1 = float(px.loc[day])
                if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
                    port_ret += w * (p1 / p0 - 1.0)
                    wsum += w
            # Cash sleeve earns 0.
            equity *= 1.0 + port_ret
        eq_rows.append((day, equity))
        inv_rows.append((day, float(sum(held.values()))))

        if day in rebalance_set:
            candidates: List[Tuple[float, str, Valuation, float]] = []
            for sym, passed in last_pass.items():
                streak = int(fail_streak.get(sym, 0))
                # Eligible to hold: currently passing, or still inside one-fail allowance
                # while already held.
                allowed = (passed and streak == 0) or (sym in held and streak == 1 and not passed)
                if not allowed:
                    continue
                if streak >= 2:
                    continue
                val = last_val.get(sym)
                if val is None or val.v_opt <= 0:
                    continue
                px = closes.get(sym)
                if px is None:
                    continue
                hit = px.loc[px.index <= day]
                if hit.empty:
                    continue
                price = float(hit.iloc[-1])
                if not np.isfinite(price) or price <= 0:
                    continue
                # Trim / avoid far above optimistic value.
                if price > trim_mult * val.v_opt:
                    continue
                # Prefer names not above optimistic; allow under pessimistic (add zone).
                if price > val.v_opt and sym not in held:
                    continue
                score = price / val.v_opt  # lower = deeper discount
                candidates.append((score, sym, val, price))
            candidates.sort(key=lambda x: (x[0], x[1]))
            picked = candidates[: int(max_names)]
            new_held: Dict[str, float] = {}
            if picked:
                w = 1.0 / float(len(picked))
                for score, sym, val, price in picked:
                    new_held[sym] = w
                    hold_rows.append(
                        {
                            "date": day.date(),
                            "symbol": sym,
                            "weight": w,
                            "price": price,
                            "v_opt": val.v_opt,
                            "v_pess": val.v_pess,
                            "score": score,
                            "fail_streak": int(fail_streak.get(sym, 0)),
                            "passed": bool(last_pass.get(sym, False)),
                        }
                    )
            held = new_held
        prev_day = day

    equity_s = pd.Series({d: v for d, v in eq_rows}).sort_index()
    invested_s = pd.Series({d: v for d, v in inv_rows}).sort_index()
    holdings = pd.DataFrame(hold_rows)
    return equity_s, invested_s, holdings


def run_tachun_backtest(
    *,
    start: date = date(2009, 1, 1),
    end: Optional[date] = None,
    max_names: int = MAX_NAMES,
    trim_mult: float = TRIM_MULT,
    fp_mode: str = "FY",
    max_symbols: int = 0,
    workers: int = 4,
) -> TachunBacktestResult:
    """
    End-to-end free Tachun book.

    Default ``fp_mode='FY'`` matches the annual Finviz-style checklist and gives
    clean YoY EPS / 5y windows. Use ``Q`` to re-score every 10-Q.
    """
    end = end or date.today()
    t0 = datetime.now()
    logger.info("Loading fundamentals panel...")
    fundamentals = load_fundamentals_panel(max_filed=end)
    if max_symbols and max_symbols > 0:
        keep = sorted(fundamentals["symbol"].unique())[: int(max_symbols)]
        fundamentals = fundamentals[fundamentals["symbol"].isin(keep)].copy()
    symbols = sorted(fundamentals["symbol"].unique())
    logger.info("Fundamentals symbols=%d rows=%d", len(symbols), len(fundamentals))

    logger.info("Loading DGS10 + SPY...")
    dgs10 = load_dgs10()
    spy = load_spy_close(datetime(2006, 1, 1), datetime(end.year, end.month, end.day))
    spy = to_naive_index(spy)

    # First screen pass with stored mkt_cap to shrink the price-load set.
    logger.info("Screen pass (stored mkt_cap)...")
    events0 = build_screen_events(fundamentals, closes=None, fp_mode=fp_mode)
    # Load prices for names that clear hard statement/ratio gates, even if
    # mkt_cap or growth windows still fail (price can fix mkt_cap).
    hard_fail = {
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roe",
        "current_ratio",
        "debt_to_ebitda",
        "interest_to_ocf",
    }

    def _maybe(row: pd.Series) -> bool:
        reasons = set(str(row.get("reasons") or "").split(",")) - {""}
        return bool(row["passed"]) or reasons.isdisjoint(hard_fail)

    near_syms = sorted(events0[events0.apply(_maybe, axis=1)]["symbol"].unique())
    if not near_syms:
        near_syms = symbols
    logger.info("Loading closes for %d candidate symbols (ALPACA+IB prefix)...", len(near_syms))
    closes = load_closes_prefix(
        near_syms,
        start=datetime(2006, 1, 1),
        end=datetime(end.year, end.month, end.day),
        workers=workers,
    )
    closes["SPY"] = spy

    logger.info("Screen pass (price mkt_cap)...")
    # Restrict fundamentals to candidates for speed.
    fund2 = fundamentals[fundamentals["symbol"].isin(near_syms)].copy()
    events = build_screen_events(fund2, closes=closes, fp_mode=fp_mode)
    n_pass = int(events["passed"].sum()) if not events.empty else 0
    logger.info("Screen events=%d passers=%d unique_pass_syms=%d",
                len(events), n_pass,
                events.loc[events["passed"], "symbol"].nunique() if n_pass else 0)

    # Value each passing event.
    val_rows: List[Dict[str, Any]] = []
    pass_ev = events[events["passed"]].copy() if not events.empty else events
    for _, ev in pass_ev.iterrows():
        sym = str(ev["symbol"]).upper()
        px = closes.get(sym)
        if px is None:
            continue
        val = value_passer(ev, stock_px=px, spy_px=spy, dgs10=dgs10)
        if val is None:
            continue
        val_rows.append(
            {
                "symbol": val.symbol,
                "asof": val.asof,
                "fcf0": val.fcf0,
                "g0": val.g0,
                "rf": val.rf,
                "beta_raw": val.beta_raw,
                "beta_opt": val.beta_opt,
                "beta_pess": val.beta_pess,
                "v_opt": val.v_opt,
                "v_pess": val.v_pess,
                "shares": val.shares,
            }
        )
    valuations = pd.DataFrame(val_rows)
    logger.info("Valuations=%d", len(valuations))

    equity, invested, holdings = simulate_ew_book(
        events,
        valuations,
        closes,
        spy,
        start=start,
        end=end,
        max_names=max_names,
        trim_mult=trim_mult,
    )
    spy_w = spy.reindex(equity.index).ffill()
    spy_eq = (spy_w / float(spy_w.iloc[0])).astype(float)

    st = perf_stats("tachun_ew15", equity, invested=invested)
    spy_st = perf_stats("SPY", spy_eq, invested=pd.Series(1.0, index=spy_eq.index))
    st.excess_cagr_vs_spy = st.cagr - spy_st.cagr
    st.excess_sharpe_vs_spy = st.sharpe - spy_st.sharpe
    ok, reason = passes_phase5_gates(st, spy_st)
    elapsed = (datetime.now() - t0).total_seconds()
    notes = (
        f"survivor-biased estimate-free; fp_mode={fp_mode}; "
        f"max_names={max_names}; trim_mult={trim_mult}; "
        f"pass_events={n_pass}; valuations={len(valuations)}; "
        f"phase5={reason}; elapsed_s={elapsed:.1f}"
    )
    stats = {
        "sleeve": st.as_dict(),
        "spy": spy_st.as_dict(),
        "phase5_ok": ok,
        "phase5_reason": reason,
        "n_screen_events": int(len(events)),
        "n_pass_events": n_pass,
        "n_pass_symbols": int(events.loc[events["passed"], "symbol"].nunique()) if n_pass else 0,
        "n_valuations": int(len(valuations)),
        "elapsed_s": elapsed,
        "label": "survivor-biased and estimate-free",
    }
    return TachunBacktestResult(
        equity=equity,
        invested=invested,
        spy_equity=spy_eq,
        holdings=holdings,
        screen_events=events,
        valuations=valuations,
        stats=stats,
        notes=notes,
    )
