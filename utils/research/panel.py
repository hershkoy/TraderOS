"""
Wide Date x Symbol panels from chunked TimescaleDB loads.

Avoids full-table aggregations: symbols come from get_available_symbols,
bars from load_ohlcv_many (batch SQL + per-symbol parquet).
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.db.timescaledb_client import get_timescaledb_client

logger = logging.getLogger(__name__)

DEFAULT_PANEL_ROOT = Path("data/cache/panels")
DEFAULT_FIELDS = ("open", "close", "low", "volume")


def _symbol_key(symbols: Sequence[str]) -> str:
    joined = ",".join(sorted({s.upper() for s in symbols}))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"n{len(symbols)}_{digest}"


def to_naive_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if isinstance(out, pd.DataFrame):
            if "ts" in out.columns:
                out = out.set_index("ts")
            elif "datetime" in out.columns:
                out = out.set_index("datetime")
            else:
                out.index = pd.to_datetime(out.index)
        else:
            out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def panel_cache_path(
    root: Path,
    provider: str,
    timeframe: str,
    field: str,
    start: Optional[datetime],
    end: Optional[datetime],
    symbol_key: str = "all",
) -> Path:
    start_s = start.strftime("%Y%m%d") if start else "none"
    end_s = end.strftime("%Y%m%d") if end else "none"
    fname = f"{provider.lower()}_{timeframe}_{start_s}_{end_s}_{symbol_key}_{field}.parquet"
    return root / fname


def frames_to_wide(frames: Dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Pivot symbol -> OHLCV frames into a Date x Symbol matrix for one field."""
    series: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        if df is None or df.empty or field not in df.columns:
            continue
        s = to_naive_index(df[field].astype(float))
        series[str(sym).upper()] = s
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series).sort_index()
    out.columns = [str(c).upper() for c in out.columns]
    return out


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=pd.DatetimeIndex(index))
    return s.groupby(s.index.to_period("M")).tail(1).index


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    c = close.astype(float)
    v = volume.astype(float).reindex_like(c)
    return c * v


def adv_dollar(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """Point-in-time rolling average dollar volume (uses data through t)."""
    dv = dollar_volume(close, volume)
    return dv.rolling(window, min_periods=max(5, window // 2)).mean()


def top_n_liquid_mask(
    adv: pd.DataFrame,
    top_n: int = 500,
    min_adv: float = 1_000_000.0,
    close: Optional[pd.DataFrame] = None,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """
    Boolean Date x Symbol mask: top_n names by PIT ADV, with floors.

    Rank is computed independently on each date (no look-ahead).
    """
    ok = adv >= float(min_adv)
    if close is not None:
        ok = ok & (close.reindex_like(adv) >= float(min_price))
    ranked = adv.where(ok).rank(axis=1, ascending=False, method="first")
    return ranked <= int(top_n)


def list_daily_symbols(
    provider: str = "ALPACA",
    timeframe: str = "1d",
    retries: int = 5,
) -> List[str]:
    """Universe discovery via DISTINCT symbol (filtered by provider/timeframe)."""
    client = get_timescaledb_client()
    for i in range(retries):
        if not client.ensure_connection():
            time.sleep(1.0 + i)
            continue
        syms = client.get_available_symbols(provider=provider, timeframe=timeframe) or []
        if syms:
            return sorted({s.upper() for s in syms})
        time.sleep(1.0 + i)
    logger.warning("list_daily_symbols returned empty after %d retries", retries)
    return []


def load_wide_panels(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    provider: str = "ALPACA",
    timeframe: str = "1d",
    fields: Sequence[str] = DEFAULT_FIELDS,
    use_cache: bool = True,
    cache_root: Path = DEFAULT_PANEL_ROOT,
    workers: int = 4,
    chunk_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    """
    Load aligned wide panels, caching each field as parquet under data/cache/panels/.
    """
    fields_u = tuple(fields)
    skey = _symbol_key(symbols)
    paths = {
        f: panel_cache_path(cache_root, provider, timeframe, f, start, end, skey) for f in fields_u
    }
    if use_cache and all(p.exists() for p in paths.values()):
        logger.info("Wide panel cache hit (%d fields) under %s", len(paths), cache_root)
        out: Dict[str, pd.DataFrame] = {}
        for f, p in paths.items():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert(None)
            df.index = pd.DatetimeIndex(df.index).normalize()
            df.columns = [str(c).upper() for c in df.columns]
            out[f] = df.sort_index()
        return out

    logger.info(
        "Building wide panels for %d symbols (%s %s %s -> %s)",
        len(symbols),
        provider,
        timeframe,
        start.date() if start else "none",
        end.date() if end else "none",
    )
    frames = load_ohlcv_many(
        symbols,
        timeframe=timeframe,
        provider=provider,
        start=start,
        end=end,
        use_cache=True,
        workers=workers,
        chunk_size=chunk_size,
    )
    wide = {f: frames_to_wide(frames, f) for f in fields_u}
    if use_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        for f, df in wide.items():
            path = paths[f]
            try:
                df.to_parquet(path)
            except Exception as exc:
                logger.warning("Failed writing panel cache %s: %s", path, exc)
    return wide


def load_spy_close(
    start: datetime,
    end: datetime,
    preferred_provider: str = "IB",
    fallback_provider: str = "ALPACA",
) -> pd.Series:
    """IB SPY preferred (deeper history); fall back to ALPACA."""
    for provider in (preferred_provider, fallback_provider):
        frames = load_ohlcv_many(
            ["SPY"],
            timeframe="1d",
            provider=provider,
            start=start,
            end=end,
            use_cache=True,
            workers=1,
        )
        df = frames.get("SPY")
        if df is None or df.empty or "close" not in df.columns:
            logger.warning("SPY close missing for provider=%s", provider)
            continue
        s = to_naive_index(df["close"].astype(float))
        logger.info(
            "Loaded SPY %s n=%d %s -> %s",
            provider,
            len(s),
            s.index[0].date(),
            s.index[-1].date(),
        )
        return s
    raise RuntimeError("SPY daily close not found for IB or ALPACA")


def align_spy_regime(
    spy_close: pd.Series,
    calendar: pd.DatetimeIndex,
    sma_period: int = 200,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Align SPY close and SMA to a panel calendar.

    SMA is computed on the full spy series first so a short/gapped panel
    does not delay the 200-day regime until 200 panel rows exist.
    """
    spy = to_naive_index(spy_close.astype(float))
    sma = spy.rolling(int(sma_period), min_periods=int(sma_period)).mean()
    cal = pd.DatetimeIndex(calendar)
    if cal.tz is not None:
        cal = cal.tz_convert(None)
    cal = cal.normalize()
    spy_a = spy.reindex(cal).ffill()
    sma_a = sma.reindex(cal).ffill()
    risk_on = (spy_a > sma_a).fillna(False)
    return spy_a, sma_a, risk_on
