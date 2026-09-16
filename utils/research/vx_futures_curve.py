"""VIX futures-curve slope for last-15m / 1d keep-skip.

Pine ``CBOE:VX1!`` / ``VX2!`` slope is ``((vx2 / vx1) - 1) * 100`` computed
in Python from separate TV daily dumps (not a Pine plot). Last-15m fills
are 15:45 ET; CFE daily closes print after that, so joins use prior-session
``completed_asof``. Missing slope does not skip.

Cash VIX/VIX3M helpers remain as labeled analogs when futures dumps are absent.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from utils.research.causal_regime_hmm import lookup_asof
from utils.research.channel_touch_entry_features import completed_asof
from utils.research.vix_curve import align_series

CBOE_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
USER_SLOPE_MIN = 4.0
PINE_BANDS = (0.0, 5.0, 8.0)


def _as_naive_close(s: pd.Series, name: str) -> pd.Series:
    x = s.astype(float).copy()
    x.index = pd.DatetimeIndex(x.index)
    if x.index.tz is not None:
        x.index = x.index.tz_convert(None)
    x.index = pd.DatetimeIndex(x.index).normalize()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    x.name = name
    return x


def load_cboe_index_close(symbol: str, *, timeout: float = 30.0) -> pd.Series:
    """Daily CLOSE from CBOE's public index CSV (DATE M/D/YYYY)."""
    sym = str(symbol).upper().strip()
    url = CBOE_INDEX_URL.format(sym=sym)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (research; backTraderTest)",
            "Referer": "https://www.cboe.com/",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    df = pd.read_csv(BytesIO(raw))
    cols = {c.strip().upper(): c for c in df.columns}
    if "DATE" not in cols or "CLOSE" not in cols:
        raise RuntimeError("CBOE %s CSV missing DATE/CLOSE: %s" % (sym, list(df.columns)))
    idx = pd.to_datetime(df[cols["DATE"]], format="%m/%d/%Y", errors="coerce")
    close = pd.to_numeric(df[cols["CLOSE"]], errors="coerce")
    out = pd.Series(close.values, index=idx, name=sym)
    out = out[out.index.notna()].sort_index()
    return _as_naive_close(out, sym)


def load_tv_dump_close(path: Path, *, name: str = "close", tz: str = "America/Chicago") -> pd.Series:
    """Daily close from ``dump_tv_bars.mjs`` JSON. Session date is CFE (Chicago)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bars = payload.get("bars") or []
    if not bars:
        raise RuntimeError("no bars in %s" % path)
    unix = [row[0] for row in bars]
    close = [row[4] for row in bars]
    idx = pd.to_datetime(unix, unit="s", utc=True)
    out = pd.Series(close, index=idx, name=name)
    out.index = out.index.tz_convert(tz).tz_localize(None).normalize()
    return _as_naive_close(out, name)


def curve_slope_pct(front: pd.Series, back: pd.Series) -> pd.Series:
    """Pine curve: ((back / front) - 1) * 100. Contango positive."""
    a, b = align_series(_as_naive_close(front, "front"), _as_naive_close(back, "back"))
    ratio = (b / a.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return ((ratio - 1.0) * 100.0).rename("curve_slope_pct")


def interpolate_60d_close(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """Linear 60d constant-maturity index, halfway from 30d VIX to 90d VIX3M.

    This is a price-curve analog of VX2 vs VX1 (not the 30-60 forward vol).
    """
    vx, vx3 = align_series(_as_naive_close(vix, "vix"), _as_naive_close(vix3m, "vix3m"))
    return (0.5 * (vx + vx3)).rename("vix60")


def vx60_slope_pct(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """30d vs interpolated 60d slope % (closest cash analog of VX2/VX1)."""
    vx = _as_naive_close(vix, "vix")
    cm = interpolate_60d_close(vix, vix3m)
    return curve_slope_pct(vx, cm).rename("vx60_slope_pct")


def attach_prior_session_slope(
    trades: pd.DataFrame,
    slope: pd.Series,
    *,
    col: str = "vx_slope_asof",
) -> pd.DataFrame:
    """Join prior-session slope onto each fill. Fill-day CBOE close is illegal at 15:45 ET."""
    out = trades.copy()
    s = _as_naive_close(slope, "slope")
    asofs = []
    vals = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=True)
        asofs.append(asof)
        vals.append(None if asof is None else lookup_asof(s, asof))
    out["regime_asof"] = asofs
    out[col] = vals
    return out


def keep_unless_slope_below(
    trades: pd.DataFrame,
    *,
    thresh: float = USER_SLOPE_MIN,
    col: str = "vx_slope_asof",
) -> pd.Series:
    """Keep when slope missing or >= thresh. Skip when prior-session slope < thresh."""
    slope = pd.to_numeric(trades[col], errors="coerce")
    skip = slope < float(thresh)
    return (~skip.fillna(False)).astype(bool)


def pine_band_label(slope: pd.Series) -> pd.Series:
    """Pine horizontal bands: <0 red, 0-5 orange, 5-8 yellow, >=8 green."""
    zero, mid, top = PINE_BANDS
    x = pd.to_numeric(slope, errors="coerce")
    out = pd.Series("missing", index=x.index, dtype=object)
    out = out.mask(x < zero, "<0")
    out = out.mask((x >= zero) & (x < mid), "0-5")
    out = out.mask((x >= mid) & (x < top), "5-8")
    out = out.mask(x >= top, ">=8")
    return out
