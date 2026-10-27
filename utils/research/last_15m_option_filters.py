"""Prior-session option filters for the last-15m stock book.

Market and sector features come from listed SPY and sector-ETF chains.
Name features are the same ATM IV and 25-delta skew on liquid names.
A feature is the prior completed session only. Missing marks keep the trade.

The first pass is a quintile table. A skip locks only when the pre-registered
tail separates losers from the right tail: not a winner-dollar clip, and the
2022 leftover is not worse than the unfiltered book.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.options.greeks import calculate_greeks
from utils.research.cup_handle import sector_etf_for
from utils.research.last_15m_volume_geometry import winner_cut_skip

MARKET_SYMBOL = "SPY"
SECTOR_ETFS: Tuple[str, ...] = (
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLRE",
    "XLB",
    "XLC",
)
UNDERLIERS: Tuple[str, ...] = (MARKET_SYMBOL,) + SECTOR_ETFS

PCT_WINDOW = 252
PCT_MIN_PERIODS = 126
SKIP_TOP = 0.80
SKIP_BOTTOM = 0.20
MIN_DROPPED = 30
MIN_SECTORS_FOR_DISPERSION = 6
SKEW_CHANGE_SESSIONS = 5
# Name pull still runs when a locked ETF skip leaves 2022 this red.
NAME_LAYER_2022_E_FLOOR = -1.0
ATM_DELTA = -0.50
PUT25_DELTA = -0.25
DELTA_TOL_ATM = 0.15
DELTA_TOL_25 = 0.10

# role -> target DTE, window, target delta
ROLE_SPECS: Dict[str, Tuple[int, int, int, float]] = {
    "atm30": (30, 18, 45, ATM_DELTA),
    "put25": (30, 18, 45, PUT25_DELTA),
    "atm14": (14, 7, 21, ATM_DELTA),
    "atm45": (45, 30, 70, ATM_DELTA),
}

MARKET_RULES: Tuple[Tuple[str, str, float], ...] = (
    ("spy_skew_pct", "top", SKIP_TOP),
    ("spy_skew_chg5_pct", "top", SKIP_TOP),
    ("dispersion_pct", "top", SKIP_TOP),
)
SECTOR_RULES: Tuple[Tuple[str, str, float], ...] = (
    ("sector_iv_pct", "top", SKIP_TOP),
    ("sector_skew_pct", "top", SKIP_TOP),
)
NAME_RULES: Tuple[Tuple[str, str, float], ...] = (
    ("name_iv_pct", "top", SKIP_TOP),
    ("name_iv_pct", "bottom", SKIP_BOTTOM),
    ("name_skew_pct", "top", SKIP_TOP),
    ("name_vs_sector_pct", "top", SKIP_TOP),
)

QUINTILE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
QUINTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def snap_strike(price: float) -> float:
    """Nearest listed-style strike. $1 above $25, $0.50 below."""
    px = float(price)
    if not math.isfinite(px) or px <= 0:
        return float("nan")
    step = 1.0 if px >= 25.0 else 0.5
    return round(round(px / step) * step, 3)


def choose_expiration(
    expirations: Sequence[date],
    day: date,
    target_dte: int,
    dte_lo: int,
    dte_hi: int,
) -> Optional[date]:
    """Expiry closest to target_dte inside [dte_lo, dte_hi]."""
    best: Optional[date] = None
    best_key: Optional[Tuple[int, date]] = None
    for exp in expirations:
        dte = (exp - day).days
        if dte < int(dte_lo) or dte > int(dte_hi):
            continue
        key = (abs(dte - int(target_dte)), exp)
        if best_key is None or key < best_key:
            best_key = key
            best = exp
    return best


def put_strike_for_delta(
    spot: float,
    dte_days: int,
    iv: float,
    target_delta: float = PUT25_DELTA,
    r: float = 0.03,
) -> float:
    """Listed strike whose Black-Scholes put delta is nearest target_delta."""
    if (
        not math.isfinite(spot)
        or spot <= 0
        or not math.isfinite(iv)
        or iv <= 0.01
        or int(dte_days) <= 0
    ):
        return float("nan")
    t_years = max(int(dte_days), 1) / 365.0
    sigma = min(max(float(iv), 0.05), 1.50)
    lo = spot * 0.40
    hi = spot * 1.05
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        delta = float(calculate_greeks(spot, mid, t_years, r, sigma, "P")["delta"])
        if delta < float(target_delta):
            hi = mid
        else:
            lo = mid
    return snap_strike(0.5 * (lo + hi))


def _as_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).date()


def parse_greeks_csv(text: str) -> pd.DataFrame:
    """Theta greeks/eod CSV -> session, strike, delta, iv, volume."""
    empty = pd.DataFrame(
        columns=["session", "expiration", "strike", "delta", "iv", "volume"]
    )
    if not text or not str(text).strip():
        return empty
    raw = pd.read_csv(StringIO(text))
    if raw.empty:
        return empty
    iv_col = next(
        (c for c in ("implied_vol", "iv", "implied_volatility") if c in raw.columns),
        None,
    )
    if iv_col is None or "delta" not in raw.columns:
        return empty
    if "timestamp" in raw.columns:
        sessions = pd.to_datetime(raw["timestamp"], errors="coerce")
    elif "date" in raw.columns:
        sessions = pd.to_datetime(raw["date"], errors="coerce")
    else:
        return empty
    out = pd.DataFrame(
        {
            "session": sessions.dt.date,
            "expiration": (
                pd.to_datetime(raw["expiration"], errors="coerce").dt.date
                if "expiration" in raw.columns
                else pd.NaT
            ),
            "strike": pd.to_numeric(raw.get("strike"), errors="coerce"),
            "delta": pd.to_numeric(raw["delta"], errors="coerce"),
            "iv": pd.to_numeric(raw[iv_col], errors="coerce"),
            "volume": (
                pd.to_numeric(raw["volume"], errors="coerce")
                if "volume" in raw.columns
                else np.nan
            ),
        }
    )
    out = out.dropna(subset=["session", "delta", "iv"])
    if out.empty:
        return empty
    med = float(out["iv"].median())
    if math.isfinite(med) and med > 3.0:
        out["iv"] = out["iv"] / 100.0
    out = out[(out["iv"] > 0.01) & (out["iv"] < 3.0)]
    if out.empty:
        return empty
    out = out.drop_duplicates(subset=["session", "strike"], keep="last")
    return out.reset_index(drop=True)


def coalesce_strike_days(
    rows: Sequence[Mapping],
    *,
    max_gap_days: int = 14,
) -> List[dict]:
    """Merge same symbol/expiry/strike/role sessions into date ranges."""
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r["symbol"]),
            r["expiration"],
            float(r["strike"]),
            str(r["role"]),
            r["session"],
        ),
    )
    out: List[dict] = []
    cur: Optional[dict] = None
    for row in ordered:
        if cur is None:
            cur = _range_from_row(row)
            continue
        same = (
            cur["symbol"] == row["symbol"]
            and cur["expiration"] == row["expiration"]
            and abs(float(cur["strike"]) - float(row["strike"])) < 1e-6
            and cur["role"] == row["role"]
        )
        gap = (row["session"] - cur["end"]).days
        if same and 0 <= gap <= int(max_gap_days):
            cur["end"] = row["session"]
            cur["n_sessions"] = int(cur["n_sessions"]) + 1
            continue
        out.append(cur)
        cur = _range_from_row(row)
    if cur is not None:
        out.append(cur)
    return out


def _range_from_row(row: Mapping) -> dict:
    return {
        "symbol": str(row["symbol"]).upper(),
        "expiration": row["expiration"],
        "strike": float(row["strike"]),
        "role": str(row["role"]),
        "start": row["session"],
        "end": row["session"],
        "n_sessions": 1,
    }


def plan_strike_ranges(
    symbol: str,
    close: pd.Series,
    expirations: Sequence[date],
    sessions: Sequence[date],
    roles: Sequence[str],
    *,
    iv_by_session: Optional[Mapping[date, float]] = None,
    max_gap_days: int = 14,
) -> List[dict]:
    """One range per contiguous run of (expiry, strike, role)."""
    sym = str(symbol).upper()
    px = _close_by_date(close)
    iv_map = iv_by_session or {}
    days: List[dict] = []
    for session in sessions:
        spot = px.get(session)
        if spot is None or not math.isfinite(spot) or spot <= 0:
            continue
        for role in roles:
            spec = ROLE_SPECS.get(str(role))
            if spec is None:
                continue
            target, lo, hi, _delta = spec
            exp = choose_expiration(expirations, session, target, lo, hi)
            if exp is None:
                continue
            if role == "put25":
                iv = iv_map.get(session)
                if iv is None or not math.isfinite(float(iv)):
                    iv = 0.22
                strike = put_strike_for_delta(float(spot), (exp - session).days, float(iv))
            else:
                strike = snap_strike(float(spot))
            if not math.isfinite(strike):
                continue
            days.append(
                {
                    "symbol": sym,
                    "session": session,
                    "expiration": exp,
                    "strike": float(strike),
                    "role": str(role),
                }
            )
    return coalesce_strike_days(days, max_gap_days=max_gap_days)


def _close_by_date(close: pd.Series) -> Dict[date, float]:
    s = pd.to_numeric(close, errors="coerce")
    idx = pd.to_datetime(s.index, errors="coerce")
    out: Dict[date, float] = {}
    for ts, val in zip(idx, s.to_numpy()):
        if pd.isna(ts) or not math.isfinite(float(val)):
            continue
        out[pd.Timestamp(ts).date()] = float(val)
    return out


def quotes_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "session", "expiration", "strike", "delta", "iv", "role"]
        )
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["session"] = df["session"].map(_as_date)
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    return df.dropna(subset=["session", "iv", "delta"])


def _pick_role_rows(quotes: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol/session/role, nearest the role's target delta."""
    if quotes.empty:
        return quotes
    pieces: List[pd.DataFrame] = []
    for role, part in quotes.groupby("role", sort=False):
        spec = ROLE_SPECS.get(str(role))
        if spec is None:
            continue
        target = float(spec[3])
        tol = DELTA_TOL_25 if role == "put25" else DELTA_TOL_ATM
        work = part.copy()
        work["_err"] = (work["delta"] - target).abs()
        work = work[work["_err"] <= tol]
        if work.empty:
            continue
        work = work.sort_values("_err").drop_duplicates(
            subset=["symbol", "session"], keep="first"
        )
        pieces.append(work.drop(columns=["_err"]))
    if not pieces:
        return quotes.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def daily_surface(quotes: pd.DataFrame) -> pd.DataFrame:
    """Wide per symbol-session ATM IV, skew, and term slope."""
    empty = pd.DataFrame(
        columns=["symbol", "session", "atm_iv", "skew", "term", "put25_iv", "iv_14", "iv_45"]
    )
    if quotes is None or len(quotes) == 0:
        return empty
    picked = _pick_role_rows(quotes_frame(quotes.to_dict("records")))
    if picked.empty:
        return pd.DataFrame(
            columns=["symbol", "session", "atm_iv", "skew", "term", "put25_iv", "iv_14", "iv_45"]
        )
    wide_parts: List[pd.DataFrame] = []
    for role, col in (
        ("atm30", "atm_iv"),
        ("put25", "put25_iv"),
        ("atm14", "iv_14"),
        ("atm45", "iv_45"),
    ):
        sub = picked.loc[picked["role"] == role, ["symbol", "session", "iv"]].rename(
            columns={"iv": col}
        )
        wide_parts.append(sub)
    out = wide_parts[0]
    for part in wide_parts[1:]:
        out = out.merge(part, on=["symbol", "session"], how="outer")
    out["skew"] = out["put25_iv"] - out["atm_iv"]
    out["term"] = out["iv_14"] - out["iv_45"]
    return out.sort_values(["symbol", "session"]).reset_index(drop=True)


def trailing_percentile(
    values: pd.Series,
    *,
    window: int = PCT_WINDOW,
    min_periods: int = PCT_MIN_PERIODS,
) -> pd.Series:
    """Causal percentile of the current point inside the trailing window."""
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    w = int(window)
    need = int(min_periods)
    for i, cur in enumerate(vals):
        if not math.isfinite(float(cur)):
            continue
        lo = 0 if i + 1 < w else i + 1 - w
        chunk = vals[lo : i + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < need:
            continue
        out[i] = float(np.mean(finite <= cur))
    return pd.Series(out, index=values.index)


def _session_index(dates: Sequence[date]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()


def market_sector_panel(
    surface: pd.DataFrame,
    sessions: Sequence[date],
) -> pd.DataFrame:
    """SPY and sector features on the session calendar, with trailing percentiles."""
    idx = _session_index(sessions)
    panel = pd.DataFrame(index=idx)
    panel.index.name = "session"
    surf = surface.copy()
    if surf.empty:
        return _with_dispersion(panel)
    surf["session"] = pd.to_datetime(surf["session"]).dt.normalize()
    for sym in UNDERLIERS:
        sub = surf.loc[surf["symbol"] == sym].drop_duplicates("session", keep="last")
        sub = sub.set_index("session").reindex(idx)
        prefix = "spy" if sym == MARKET_SYMBOL else sym
        atm = pd.to_numeric(sub.get("atm_iv"), errors="coerce")
        skew = pd.to_numeric(sub.get("skew"), errors="coerce")
        term = pd.to_numeric(sub.get("term"), errors="coerce")
        panel["%s_atm_iv" % prefix] = atm
        panel["%s_skew" % prefix] = skew
        panel["%s_term" % prefix] = term
        if prefix == "spy":
            panel["spy_skew_pct"] = trailing_percentile(skew)
            chg = skew.diff(SKEW_CHANGE_SESSIONS)
            panel["spy_skew_chg5"] = chg
            panel["spy_skew_chg5_pct"] = trailing_percentile(chg)
            panel["spy_term_pct"] = trailing_percentile(term)
        else:
            panel["%s_iv_pct" % prefix] = trailing_percentile(atm)
            panel["%s_skew_pct" % prefix] = trailing_percentile(skew)
            panel["%s_term_pct" % prefix] = trailing_percentile(term)
    return _with_dispersion(panel)


def _with_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["%s_atm_iv" % etf for etf in SECTOR_ETFS if "%s_atm_iv" % etf in panel.columns]
    out = panel.copy()
    if "spy_atm_iv" not in out.columns or len(cols) < MIN_SECTORS_FOR_DISPERSION:
        out["dispersion"] = np.nan
        out["dispersion_pct"] = np.nan
        return out
    sector_mean = out[cols].mean(axis=1, skipna=True)
    n_ok = out[cols].notna().sum(axis=1)
    disp = out["spy_atm_iv"] - sector_mean
    disp = disp.where(n_ok >= MIN_SECTORS_FOR_DISPERSION)
    out["dispersion"] = disp
    out["dispersion_pct"] = trailing_percentile(disp)
    return out


def prior_session_values(
    panel: pd.DataFrame,
    buy_dates: Sequence,
    column: str,
) -> np.ndarray:
    """Value of column on the last session strictly before each buy date."""
    out = np.full(len(list(buy_dates)), np.nan)
    if panel.empty or column not in panel.columns:
        return out
    sessions = pd.to_datetime(panel.index).normalize().to_numpy(dtype="datetime64[D]")
    order = np.argsort(sessions)
    sessions = sessions[order]
    values = pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)[order]
    buys = pd.to_datetime(pd.Series(list(buy_dates)), errors="coerce").to_numpy(
        dtype="datetime64[D]"
    )
    idx = np.searchsorted(sessions, buys, side="left") - 1
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def attach_etf_features(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    sectors: Mapping[str, str],
) -> pd.DataFrame:
    """Join prior-session SPY and sector-ETF features onto stock fills."""
    out = trades.copy()
    out["stock"] = out["stock"].astype(str).str.upper()
    out["sector"] = out["stock"].map(lambda s: str(sectors.get(s, "") or ""))
    out["sector_etf"] = out["sector"].map(resolve_sector_etf)
    for col in (
        "spy_skew_pct",
        "spy_skew_chg5_pct",
        "dispersion_pct",
        "spy_atm_iv",
        "spy_skew",
        "dispersion",
    ):
        out[col] = prior_session_values(panel, out["buy_date"], col)
    sector_iv = np.full(len(out), np.nan)
    sector_skew = np.full(len(out), np.nan)
    sector_atm = np.full(len(out), np.nan)
    for etf, idx in out.groupby("sector_etf").groups.items():
        if not etf:
            continue
        positions = out.index.get_indexer(idx)
        sector_iv[positions] = prior_session_values(
            panel, out.loc[idx, "buy_date"], "%s_iv_pct" % etf
        )
        sector_skew[positions] = prior_session_values(
            panel, out.loc[idx, "buy_date"], "%s_skew_pct" % etf
        )
        sector_atm[positions] = prior_session_values(
            panel, out.loc[idx, "buy_date"], "%s_atm_iv" % etf
        )
    out["sector_iv_pct"] = sector_iv
    out["sector_skew_pct"] = sector_skew
    out["sector_atm_iv"] = sector_atm
    return out


def name_feature_frame(
    surface: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    min_adv: float,
) -> pd.DataFrame:
    """Trailing percentiles of name IV, skew, and name-minus-sector IV.

    Only trades with adv_20 >= min_adv receive a name feature. The history
    used for the percentile is that symbol's own surface.
    """
    out = trades.copy()
    for col in ("name_iv_pct", "name_skew_pct", "name_vs_sector_pct", "name_atm_iv"):
        out[col] = np.nan
    if surface is None or surface.empty:
        return out
    adv = pd.to_numeric(out.get("adv_20"), errors="coerce")
    liquid_syms = set(out.loc[adv >= float(min_adv), "stock"].astype(str).str.upper())
    surf = surface.copy()
    surf["symbol"] = surf["symbol"].astype(str).str.upper()
    surf["session"] = pd.to_datetime(surf["session"]).dt.normalize()
    for sym, part in surf.groupby("symbol", sort=False):
        if sym not in liquid_syms:
            continue
        hist = part.drop_duplicates("session", keep="last").set_index("session").sort_index()
        iv_pct = trailing_percentile(pd.to_numeric(hist.get("atm_iv"), errors="coerce"))
        skew_pct = trailing_percentile(pd.to_numeric(hist.get("skew"), errors="coerce"))
        spread = pd.to_numeric(hist.get("atm_iv"), errors="coerce") - pd.to_numeric(
            hist.get("sector_atm_iv"), errors="coerce"
        )
        spread_pct = trailing_percentile(spread)
        panel = pd.DataFrame(
            {
                "name_atm_iv": pd.to_numeric(hist.get("atm_iv"), errors="coerce"),
                "name_iv_pct": iv_pct,
                "name_skew_pct": skew_pct,
                "name_vs_sector_pct": spread_pct,
            }
        )
        mask = out["stock"].astype(str).str.upper().eq(sym) & (adv >= float(min_adv))
        if not mask.any():
            continue
        idx = out.index[mask]
        for col in panel.columns:
            out.loc[idx, col] = prior_session_values(panel, out.loc[idx, "buy_date"], col)
    return out


def quintile_table(
    trades: pd.DataFrame,
    feature: str,
    *,
    gain_col: str = "gain_pct",
) -> pd.DataFrame:
    """Fixed 20-point buckets of a trailing percentile, plus the 2022 slice."""
    if feature not in trades.columns:
        return pd.DataFrame()
    work = trades.copy()
    work["_f"] = pd.to_numeric(work[feature], errors="coerce")
    work["_g"] = pd.to_numeric(work[gain_col], errors="coerce")
    work["_y"] = pd.to_datetime(work["buy_date"], errors="coerce").dt.year
    known = work["_f"].notna() & work["_g"].notna()
    rows: List[dict] = []
    rows.append(_bucket_row("NA", work.loc[~known]))
    buckets = pd.cut(
        work.loc[known, "_f"],
        bins=QUINTILE_BINS,
        labels=QUINTILE_LABELS,
        right=False,
        include_lowest=True,
    )
    tagged = work.loc[known].copy()
    tagged["_q"] = buckets.to_numpy()
    for label in QUINTILE_LABELS:
        part = tagged.loc[tagged["_q"] == label]
        rows.append(_bucket_row(label, part))
    return pd.DataFrame(rows)


def _bucket_row(label: str, part: pd.DataFrame) -> dict:
    g = pd.to_numeric(part.get("_g"), errors="coerce") if "_g" in part.columns else pd.Series(dtype=float)
    if g.empty and "gain_pct" in part.columns:
        g = pd.to_numeric(part["gain_pct"], errors="coerce")
    g = g.dropna()
    y22 = part.loc[part["_y"] == 2022] if "_y" in part.columns else part.iloc[0:0]
    g22 = pd.to_numeric(y22.get("_g"), errors="coerce").dropna() if len(y22) else pd.Series(dtype=float)
    wins = float(g[g > 0].sum()) if len(g) else 0.0
    losses = float((-g[g <= 0]).sum()) if len(g) else 0.0
    pf: Optional[float]
    if not len(g):
        pf = None
    elif losses <= 0:
        pf = None if wins <= 0 else float("inf")
    else:
        pf = wins / losses
    wins22 = float(g22[g22 > 0].sum()) if len(g22) else 0.0
    losses22 = float((-g22[g22 <= 0]).sum()) if len(g22) else 0.0
    if not len(g22):
        pf22 = None
    elif losses22 <= 0:
        pf22 = None if wins22 <= 0 else float("inf")
    else:
        pf22 = wins22 / losses22
    return {
        "bucket": label,
        "n": int(len(g)),
        "expectancy_pct": None if not len(g) else round(float(g.mean()), 4),
        "profit_factor": None if pf is None or not math.isfinite(pf) else round(float(pf), 3),
        "loser_rate": None if not len(g) else round(float((g <= 0).mean()), 4),
        "median_pct": None if not len(g) else round(float(g.median()), 4),
        "win_sum": round(wins, 4),
        "lose_sum": round(float(g[g <= 0].sum()) if len(g) else 0.0, 4),
        "n_2022": int(len(g22)),
        "e_2022": None if not len(g22) else round(float(g22.mean()), 4),
        "pf_2022": None if pf22 is None or not math.isfinite(pf22) else round(float(pf22), 3),
    }


def skip_mask(feature: pd.Series, *, direction: str = "top", threshold: float = SKIP_TOP) -> pd.Series:
    """True means keep the trade. A missing feature keeps the trade."""
    f = pd.to_numeric(feature, errors="coerce")
    if direction == "bottom":
        drop = f.notna() & (f <= float(threshold))
    else:
        drop = f.notna() & (f >= float(threshold))
    return ~drop


def evaluate_skip(
    trades: pd.DataFrame,
    feature: str,
    *,
    direction: str = "top",
    threshold: float = SKIP_TOP,
    gain_col: str = "gain_pct",
) -> dict:
    """Does this pre-registered tail separate losers without clipping the right tail?"""
    g = pd.to_numeric(trades[gain_col], errors="coerce")
    if feature not in trades.columns:
        return {"feature": feature, "separates": False, "reason": "missing_column"}
    keep = skip_mask(trades[feature], direction=direction, threshold=threshold).reindex(trades.index)
    keep = keep.fillna(True).astype(bool)
    drop = ~keep
    years = pd.to_datetime(trades["buy_date"], errors="coerce").dt.year
    y22 = years == 2022
    e_base = float(g[y22].mean()) if y22.any() and g[y22].notna().any() else float("nan")
    kept_22 = y22 & keep & g.notna()
    e_kept = float(g[kept_22].mean()) if kept_22.any() else float("nan")
    dropped_g = g[drop & g.notna()]
    kept_g = g[keep & g.notna()]
    cut = winner_cut_skip(g, keep)
    dropped_mean = float(dropped_g.mean()) if len(dropped_g) else float("nan")
    kept_mean = float(kept_g.mean()) if len(kept_g) else float("nan")
    win_sum = float(cut["winner_drop_gain_sum"] or 0.0)
    lose_sum = float(cut["loser_drop_gain_sum"] or 0.0)
    saved = -lose_sum
    separates = bool(
        int(drop.sum()) >= MIN_DROPPED
        and not bool(cut["blunt"])
        and math.isfinite(dropped_mean)
        and math.isfinite(kept_mean)
        and dropped_mean < kept_mean
        and math.isfinite(e_base)
        and math.isfinite(e_kept)
        and e_kept + 1e-9 >= e_base
        and saved > 1e-9
        and win_sum < saved
    )
    return {
        "feature": feature,
        "direction": direction,
        "threshold": float(threshold),
        "separates": separates,
        "n_dropped": int(drop.sum()),
        "n_feature": int(pd.to_numeric(trades[feature], errors="coerce").notna().sum()),
        "dropped_mean": None if not math.isfinite(dropped_mean) else round(dropped_mean, 4),
        "kept_mean": None if not math.isfinite(kept_mean) else round(kept_mean, 4),
        "e_2022_base": None if not math.isfinite(e_base) else round(e_base, 4),
        "e_2022_kept": None if not math.isfinite(e_kept) else round(e_kept, 4),
        "blunt": bool(cut["blunt"]),
        "winner_drop_gain_sum": cut["winner_drop_gain_sum"],
        "loser_drop_gain_sum": cut["loser_drop_gain_sum"],
        "n_kept": int(keep.sum()),
    }


def lock_first_rule(
    trades: pd.DataFrame,
    rules: Sequence[Tuple[str, str, float]],
    *,
    gain_col: str = "gain_pct",
) -> Tuple[Optional[Tuple[str, str, float]], List[dict]]:
    """Walk rules in order. Lock the first tail that separates. Do not pick the best."""
    scored: List[dict] = []
    locked: Optional[Tuple[str, str, float]] = None
    for feature, direction, threshold in rules:
        ev = evaluate_skip(
            trades,
            feature,
            direction=direction,
            threshold=threshold,
            gain_col=gain_col,
        )
        scored.append(ev)
        if locked is None and ev.get("separates"):
            locked = (feature, direction, float(threshold))
    return locked, scored


def etf_layer_wants_names(n_locked: int, e_2022_kept: Optional[float]) -> bool:
    """Name chains run when the ETF layer locks nothing, or 2022 is still deeply red."""
    if int(n_locked) <= 0:
        return True
    if e_2022_kept is None or not math.isfinite(float(e_2022_kept)):
        return True
    return float(e_2022_kept) < float(NAME_LAYER_2022_E_FLOOR)


def combine_keep(trades: pd.DataFrame, rules: Sequence[Tuple[str, str, float]]) -> pd.Series:
    """Intersection of locked skip masks. Unknown rules are ignored."""
    keep = pd.Series(True, index=trades.index)
    for feature, direction, threshold in rules:
        if feature not in trades.columns:
            continue
        keep = keep & skip_mask(trades[feature], direction=direction, threshold=threshold)
    return keep.astype(bool)


def stamp_sector_atm(
    surface: pd.DataFrame,
    panel: pd.DataFrame,
    sectors: Mapping[str, str],
) -> pd.DataFrame:
    """Copy that symbol's sector-ETF ATM IV onto each name-session row."""
    out = surface.copy()
    if out.empty:
        out["sector_atm_iv"] = []
        return out
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["sector_etf"] = out["symbol"].map(
        lambda s: resolve_sector_etf(str(sectors.get(s, "") or ""))
    )
    session = pd.to_datetime(out["session"]).dt.normalize()
    sector_atm = pd.Series(np.nan, index=out.index, dtype=float)
    for etf, idx in out.groupby("sector_etf").groups.items():
        col = "%s_atm_iv" % etf
        if not etf or col not in panel.columns:
            continue
        series = pd.to_numeric(panel[col], errors="coerce")
        series.index = pd.to_datetime(series.index).normalize()
        mapped = session.loc[idx].map(series)
        sector_atm.loc[idx] = pd.to_numeric(mapped, errors="coerce").to_numpy()
    out["sector_atm_iv"] = sector_atm.to_numpy()
    return out


def etf_for_label(sector: str) -> Optional[str]:
    """GICS-style label to a sector ETF. Custom/Clean are not sectors."""
    if not sector:
        return None
    label = str(sector).strip()
    if label in ("", "Custom", "Clean", "Unknown"):
        return None
    extra = {
        "Information Technology": "XLK",
        "Materials": "XLB",
    }
    if label in extra:
        return extra[label]
    if label in SECTOR_ETFS:
        return label
    return sector_etf_for(label)


def sic_to_sector_etf(sic) -> Optional[str]:
    """Coarse SIC division to a sector ETF. Unknown codes stay unmapped."""
    try:
        code = int(str(sic).strip()[:4])
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    if 1310 <= code <= 1389 or 2910 <= code <= 2999:
        return "XLE"
    if 2830 <= code <= 2836 or 3840 <= code <= 3851 or 8000 <= code <= 8099:
        return "XLV"
    if 1000 <= code <= 1499 or 2400 <= code <= 2699 or 2800 <= code <= 2899 or 3300 <= code <= 3399:
        return "XLB"
    if 3570 <= code <= 3579 or 3600 <= code <= 3699 or 7370 <= code <= 7379:
        return "XLK"
    if 4800 <= code <= 4899 or 7800 <= code <= 7949:
        return "XLC"
    if 4900 <= code <= 4949:
        return "XLU"
    if 6500 <= code <= 6553:
        return "XLRE"
    if 6000 <= code <= 6411 or 6700 <= code <= 6799:
        return "XLF"
    if 2000 <= code <= 2111 or 5400 <= code <= 5499:
        return "XLP"
    if 5200 <= code <= 5999 or 7000 <= code <= 7299 or 7500 <= code <= 7599:
        return "XLY"
    if (
        1500 <= code <= 1799
        or 3400 <= code <= 3569
        or 3700 <= code <= 3799
        or 4000 <= code <= 4799
        or 8700 <= code <= 8748
    ):
        return "XLI"
    return None


def resolve_sector_etf(value: str) -> str:
    etf = etf_for_label(value)
    return etf or ""


def repair_strikes(spot: float, role: str) -> List[float]:
    """Neighbor strikes when the snapped strike's delta is too far from target."""
    step = 1.0 if float(spot) >= 25.0 else 0.5
    base = snap_strike(spot)
    if not math.isfinite(base):
        return []
    if role == "put25":
        return [round(base - step, 3), round(base - 2 * step, 3), round(base + step, 3)]
    return [round(base - step, 3), round(base + step, 3)]


def sessions_between(start: date, end: date, calendar: Sequence[date]) -> List[date]:
    return [d for d in calendar if start <= d <= end]


def add_calendar_days(day: date, n: int) -> date:
    return day + timedelta(days=int(n))
