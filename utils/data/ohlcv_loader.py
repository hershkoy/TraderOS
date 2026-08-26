"""
Fast multi-symbol OHLCV loading with TimescaleDB batch SQL + optional parquet cache.

Supports Alpaca-primary + IB-prefix fallback for daily research panels:
  load_ohlcv_many(..., provider="ALPACA", fallback_provider="IB", merge_mode="prefix")
uses IB bars strictly before the first Alpaca bar (same calendar date), then Alpaca onward.
Merged frames are cached under provider key ``ALPACA_IB`` (or ``{primary}_{fallback}``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("data/cache/ohlcv")
OHLC_COLS = ("open", "high", "low", "close", "volume")


def _cache_path(
    root: Path,
    provider: str,
    timeframe: str,
    symbol: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Path:
    start_s = start.strftime("%Y%m%d") if start else "none"
    end_s = end.strftime("%Y%m%d") if end else "none"
    return root / provider.upper() / timeframe / f"{symbol.upper()}_{start_s}_{end_s}.parquet"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" in df.columns:
                df = df.set_index("ts")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        cols = [c for c in OHLC_COLS if c in df.columns]
        return df[cols].sort_index()
    except Exception as exc:
        logger.warning("Failed reading cache %s: %s", path, exc)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("Failed writing cache %s: %s", path, exc)


def _normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to naive midnight timestamps keyed by calendar date (for 1d stitch)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    # Collapse to date so IB ET->UTC and Alpaca UTC align on session day
    dates = pd.to_datetime(out.index.date)
    out = out.copy()
    out.index = dates
    out = out[~out.index.duplicated(keep="last")].sort_index()
    cols = [c for c in OHLC_COLS if c in out.columns]
    return out[cols]


def merge_provider_prefix(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    max_jump_pct: float = 15.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Stitch fallback bars strictly before the first primary bar.

    Returns (merged_df, meta) where meta includes jump diagnostics.
    If primary is empty, returns normalized fallback.
    If fallback empty, returns normalized primary.
    """
    meta = {
        "primary_bars": 0,
        "fallback_prefix_bars": 0,
        "jump_pct": None,
        "scaled": False,
        "rejected_jump": False,
    }
    p = _normalize_daily_index(primary) if primary is not None and not primary.empty else None
    f = _normalize_daily_index(fallback) if fallback is not None and not fallback.empty else None

    if p is None or p.empty:
        if f is None or f.empty:
            return pd.DataFrame(columns=list(OHLC_COLS)), meta
        meta["fallback_prefix_bars"] = int(len(f))
        return f, meta
    meta["primary_bars"] = int(len(p))
    if f is None or f.empty:
        return p, meta

    first_p = p.index.min()
    prefix = f.loc[f.index < first_p]
    if prefix.empty:
        return p, meta

    # Level continuity check at the join
    last_fb = float(prefix["close"].iloc[-1])
    first_pr = float(p["close"].iloc[0])
    jump = None
    if last_fb > 0 and first_pr > 0:
        jump = (first_pr / last_fb - 1.0) * 100.0
        meta["jump_pct"] = round(jump, 3)
        # Scale IB prefix onto Alpaca level if jump is large but finite (split mismatch)
        if abs(jump) > float(max_jump_pct):
            scale = first_pr / last_fb
            prefix = prefix.copy()
            for col in ("open", "high", "low", "close"):
                if col in prefix.columns:
                    prefix[col] = prefix[col].astype(float) * scale
            meta["scaled"] = True
            # Recompute jump after scale (should be ~0)
            last_fb2 = float(prefix["close"].iloc[-1])
            meta["jump_pct_after_scale"] = round((first_pr / last_fb2 - 1.0) * 100.0, 3) if last_fb2 else None

    meta["fallback_prefix_bars"] = int(len(prefix))
    merged = pd.concat([prefix, p], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged, meta


def _first_late(df: Optional[pd.DataFrame], start: Optional[datetime], slack_days: int = 5) -> bool:
    """True if series missing or starts more than slack_days after requested start."""
    if df is None or df.empty:
        return True
    if start is None:
        return False
    first = pd.Timestamp(df.index.min()).normalize()
    start_d = pd.Timestamp(start.replace(tzinfo=None) if getattr(start, "tzinfo", None) else start).normalize()
    return first > (start_d + pd.Timedelta(days=int(slack_days)))


def _load_provider_many(
    symbols: Sequence[str],
    *,
    timeframe: str,
    provider: str,
    start: Optional[datetime],
    end: Optional[datetime],
    use_cache: bool,
    cache_root: Path,
    chunk_size: int,
    workers: int,
) -> Dict[str, pd.DataFrame]:
    """Single-provider load (existing behavior)."""
    symbols_u = [s.upper() for s in symbols]
    out: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    if use_cache:
        for sym in symbols_u:
            cached = _read_cache(_cache_path(cache_root, provider, timeframe, sym, start, end))
            if cached is not None and not cached.empty:
                out[sym] = cached
            else:
                missing.append(sym)
    else:
        missing = list(symbols_u)

    if not missing:
        return out

    logger.info(
        "Loading %d/%d symbols from TimescaleDB provider=%s (cache hits=%d, chunk=%d, workers=%d)",
        len(missing),
        len(symbols_u),
        provider.upper(),
        len(symbols_u) - len(missing),
        chunk_size,
        workers,
    )

    chunks = [
        missing[i : i + max(1, chunk_size)]
        for i in range(0, len(missing), max(1, chunk_size))
    ]

    def _fetch_chunk(chunk: List[str]) -> Dict[str, pd.DataFrame]:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            raise RuntimeError("Cannot connect to TimescaleDB")
        return client.get_ohlcv_batch(
            chunk,
            timeframe=timeframe,
            provider=provider,
            start_time=start,
            end_time=end,
            chunk_size=len(chunk),
        )

    fetched: Dict[str, pd.DataFrame] = {}
    if workers and workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_chunk, c): c for c in chunks}
            for fut in as_completed(futs):
                chunk = futs[fut]
                try:
                    fetched.update(fut.result())
                except Exception as exc:
                    logger.error("Chunk fetch failed (%s...): %s", chunk[0], exc)
    else:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            raise RuntimeError("Cannot connect to TimescaleDB")
        fetched = client.get_ohlcv_batch(
            missing,
            timeframe=timeframe,
            provider=provider,
            start_time=start,
            end_time=end,
            chunk_size=chunk_size,
        )

    for sym, df in fetched.items():
        out[sym] = df
        if use_cache:
            _write_cache(_cache_path(cache_root, provider, timeframe, sym, start, end), df)

    still_missing = [s for s in missing if s not in out]
    if still_missing:
        logger.warning(
            "No OHLCV for %d symbols provider=%s (e.g. %s)",
            len(still_missing),
            provider.upper(),
            still_missing[:5],
        )
    return out


def load_ohlcv_many(
    symbols: Sequence[str],
    timeframe: str = "1d",
    provider: str = "ALPACA",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    *,
    use_cache: bool = True,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    chunk_size: int = 50,
    workers: int = 1,
    fallback_provider: Optional[str] = None,
    merge_mode: Optional[str] = None,
    max_jump_pct: float = 15.0,
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV for many symbols.

    Primary path: chunked batch SQL via TimescaleDBClient.get_ohlcv_batch.
    Optional parquet cache under data/cache/ohlcv/ (gitignored via /data/).

    Fallback (daily research):
      fallback_provider="IB", merge_mode="prefix"
      -> for symbols that are empty or start late vs ``start``, load IB and
         prepend IB bars before the first primary bar. Cache key = ``ALPACA_IB``.
    """
    provider_u = provider.upper()
    mode = (merge_mode or "").strip().lower() or None
    fb = fallback_provider.upper() if fallback_provider else None

    if fb and mode == "prefix" and timeframe == "1d":
        cache_provider = f"{provider_u}_{fb}"
    else:
        cache_provider = provider_u
        fb = None
        mode = None

    symbols_u = [s.upper() for s in symbols]

    # Try merged cache first when fallback enabled
    if fb and use_cache:
        out_cached: Dict[str, pd.DataFrame] = {}
        need: List[str] = []
        for sym in symbols_u:
            cached = _read_cache(_cache_path(cache_root, cache_provider, timeframe, sym, start, end))
            if cached is not None and not cached.empty:
                out_cached[sym] = cached
            else:
                need.append(sym)
        if not need:
            logger.info("OHLCV merged cache hit for all %d symbols (%s)", len(symbols_u), cache_provider)
            return out_cached
        primary = _load_provider_many(
            need,
            timeframe=timeframe,
            provider=provider_u,
            start=start,
            end=end,
            use_cache=use_cache,
            cache_root=cache_root,
            chunk_size=chunk_size,
            workers=workers,
        )
        # Also keep already-cached merged
        out = dict(out_cached)
        need_fb = [s for s in need if _first_late(primary.get(s), start)]
        fb_frames: Dict[str, pd.DataFrame] = {}
        if need_fb:
            logger.info(
                "Prefix-fallback: loading %d/%d symbols from %s",
                len(need_fb),
                len(need),
                fb,
            )
            fb_frames = _load_provider_many(
                need_fb,
                timeframe=timeframe,
                provider=fb,
                start=start,
                end=end,
                use_cache=use_cache,
                cache_root=cache_root,
                chunk_size=chunk_size,
                workers=workers,
            )
        n_prefixed = 0
        n_scaled = 0
        for sym in need:
            p = primary.get(sym)
            f = fb_frames.get(sym) if sym in need_fb else None
            if f is not None and not (f is None or f.empty):
                merged, meta = merge_provider_prefix(p if p is not None else pd.DataFrame(), f, max_jump_pct=max_jump_pct)
                if meta.get("fallback_prefix_bars", 0) > 0:
                    n_prefixed += 1
                if meta.get("scaled"):
                    n_scaled += 1
                out[sym] = merged
            elif p is not None and not p.empty:
                out[sym] = _normalize_daily_index(p) if timeframe == "1d" else p
            if sym in out and use_cache:
                _write_cache(
                    _cache_path(cache_root, cache_provider, timeframe, sym, start, end),
                    out[sym],
                )
        logger.info(
            "Prefix-fallback done: prefixed=%d scaled_for_jump=%d cache=%s",
            n_prefixed,
            n_scaled,
            cache_provider,
        )
        still = [s for s in symbols_u if s not in out]
        if still:
            logger.warning("No OHLCV after merge for %d symbols (e.g. %s)", len(still), still[:5])
        return out

    # Single-provider path (original)
    if use_cache:
        out: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []
        for sym in symbols_u:
            cached = _read_cache(_cache_path(cache_root, cache_provider, timeframe, sym, start, end))
            if cached is not None and not cached.empty:
                out[sym] = cached
            else:
                missing.append(sym)
        if not missing:
            logger.info("OHLCV cache hit for all %d symbols", len(symbols_u))
            return out
    else:
        out = {}
        missing = list(symbols_u)

    loaded = _load_provider_many(
        missing,
        timeframe=timeframe,
        provider=provider_u,
        start=start,
        end=end,
        use_cache=use_cache,
        cache_root=cache_root,
        chunk_size=chunk_size,
        workers=workers,
    )
    out.update(loaded)
    return out
