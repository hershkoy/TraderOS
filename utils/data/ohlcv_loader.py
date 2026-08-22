"""
Fast multi-symbol OHLCV loading with TimescaleDB batch SQL + optional parquet cache.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("data/cache/ohlcv")


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
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
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
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV for many symbols.

    Primary path: chunked batch SQL via TimescaleDBClient.get_ohlcv_batch.
    Optional parquet cache under data/cache/ohlcv/ (gitignored via /data/).
    Optional ThreadPoolExecutor over batch chunks when workers > 1.
    """
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
        logger.info("OHLCV cache hit for all %d symbols", len(symbols_u))
        return out

    logger.info(
        "Loading %d/%d symbols from TimescaleDB (cache hits=%d, chunk=%d, workers=%d)",
        len(missing),
        len(symbols_u),
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
                    part = fut.result()
                    fetched.update(part)
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
        logger.warning("No OHLCV for %d symbols (e.g. %s)", len(still_missing), still_missing[:5])

    return out
