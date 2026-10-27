"""Prior-session option filters on the current_best last-15m stock book.

Pulls Theta greeks/eod for SPY and the 11 sector ETFs (ATM and 25-delta, plus
14/45 DTE for term slope). Joins the prior session onto each fill, writes
quintile tables, and locks at most one market rule and one sector rule.

Name-level IV and skew run only when that ETF layer locks nothing, or the
2022 leftover is still deeply red.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_option_filters.py
  python scripts\\research\\overlay_last_15m_option_filters.py --names yes
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.last_15m_option_filters import (  # noqa: E402
    MARKET_RULES,
    MARKET_SYMBOL,
    NAME_RULES,
    SECTOR_ETFS,
    SECTOR_RULES,
    UNDERLIERS,
    attach_etf_features,
    combine_keep,
    daily_surface,
    etf_layer_wants_names,
    lock_first_rule,
    market_sector_panel,
    name_feature_frame,
    parse_greeks_csv,
    plan_strike_ranges,
    put_strike_for_delta,
    quintile_table,
    repair_strikes,
    snap_strike,
    stamp_sector_atm,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_option_filters")

DEFAULT_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
DEFAULT_CACHE = ROOT / "reports" / "ascending_channels" / "option_filters" / "cache"
HISTORY_START = date(2018, 1, 2)
NAME_ADV = 25_000_000
ATM_ROLES = ("atm30", "atm14", "atm45")
NAME_ROLES_ATM = ("atm30",)


def _cache_file(cache_dir: Path, task: dict) -> Path:
    strike = "%.3f" % float(task["strike"])
    name = "%s_%s_%s_%s_%s_%s.csv" % (
        task["symbol"],
        task["expiration"].isoformat(),
        strike,
        task["start"].isoformat(),
        task["end"].isoformat(),
        task["role"],
    )
    return cache_dir / name


def _fetch_one(task: dict, cache_dir: Path, base_url: str) -> str:
    path = _cache_file(cache_dir, task)
    if path.exists() and path.stat().st_size > 0:
        return "cached"
    params = {
        "symbol": task["symbol"],
        "expiration": task["expiration"].isoformat(),
        "right": "put",
        "start_date": task["start"].isoformat(),
        "end_date": task["end"].isoformat(),
        "strike": "%.3f" % float(task["strike"]),
    }
    session = requests.Session()
    url = base_url.rstrip("/") + "/v3/option/history/greeks/eod"
    last = "fail"
    for attempt in range(4):
        try:
            resp = session.get(url, params=params, timeout=90)
        except requests.RequestException as exc:
            last = "error"
            LOG.debug("theta %s %s: %s", task["symbol"], task["strike"], exc)
            time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code in (429, 478) or resp.status_code >= 500:
            last = "retry"
            time.sleep(2.0 * (attempt + 1))
            continue
        if resp.status_code == 472 or not resp.content:
            path.write_text("symbol,expiration,strike,timestamp,delta,implied_vol\n", encoding="utf-8")
            return "empty"
        if resp.status_code != 200:
            last = "http_%s" % resp.status_code
            if resp.status_code in (403, 404):
                return last
            time.sleep(1.0 * (attempt + 1))
            continue
        path.write_bytes(resp.content)
        return "ok"
    return last


def fetch_tasks(
    tasks: Sequence[dict],
    cache_dir: Path,
    base_url: str,
    *,
    workers: int,
) -> Dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    if not tasks:
        return counts
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_fetch_one, task, cache_dir, base_url) for task in tasks]
        for fut in as_completed(futures):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 100 == 0 or done == len(tasks):
                LOG.info(
                    "theta %d/%d in %.0fs %s",
                    done,
                    len(tasks),
                    time.time() - t0,
                    counts,
                )
    return counts


def _role_from_name(name: str) -> str:
    stem = name[:-4] if name.lower().endswith(".csv") else name
    return stem.rsplit("_", 1)[-1]


def load_cached_quotes(cache_dir: Path, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    wanted = None if symbols is None else {str(s).upper() for s in symbols}
    rows: List[dict] = []
    if not cache_dir.exists():
        return pd.DataFrame()
    for path in cache_dir.glob("*.csv"):
        role = _role_from_name(path.name)
        sym = path.name.split("_", 1)[0].upper()
        if wanted is not None and sym not in wanted:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = parse_greeks_csv(text)
        if parsed.empty:
            continue
        for rec in parsed.to_dict("records"):
            rec["symbol"] = sym
            rec["role"] = role
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_sectors() -> Dict[str, str]:
    """Symbol to sector ETF. Prefer a real GICS label, else the latest SIC."""
    from utils.research.last_15m_option_filters import (  # noqa: WPS433
        etf_for_label,
        sic_to_sector_etf,
    )

    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB is not connected; sector map needs ticker_universe")
    cur = client.connection.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT ON (upper(symbol))
                   upper(symbol) AS symbol,
                   COALESCE(sector, '')
            FROM ticker_universe
            WHERE sector IS NOT NULL
              AND btrim(sector) <> ''
              AND sector NOT IN ('Custom', 'Clean')
            ORDER BY upper(symbol), last_updated DESC NULLS LAST
            """
        )
        labels = [(r[0], r[1]) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT DISTINCT ON (upper(symbol))
                   upper(symbol) AS symbol,
                   sic
            FROM fundamentals_quarterly
            WHERE sic IS NOT NULL
              AND btrim(sic) <> ''
            ORDER BY upper(symbol), period_end DESC NULLS LAST
            """
        )
        sics = [(r[0], r[1]) for r in cur.fetchall()]
    finally:
        cur.close()
    out: Dict[str, str] = {}
    for sym, label in labels:
        etf = etf_for_label(label)
        if etf:
            out[str(sym).upper()] = etf
    for sym, sic in sics:
        key = str(sym).upper()
        if key in out:
            continue
        etf = sic_to_sector_etf(sic)
        if etf:
            out[key] = etf
    return out


def _split_ratios(symbol: str, cache_dir: Path) -> List[Tuple[date, float]]:
    """Yahoo split ratios. Close before a split is scaled so it matches listed strikes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("splits_%s.txt" % str(symbol).upper())
    if path.exists():
        out: List[Tuple[date, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            day_s, ratio_s = line.split(",")
            out.append((date.fromisoformat(day_s), float(ratio_s)))
        return out
    import yfinance as yf

    raw = yf.Ticker(str(symbol).upper()).splits
    out = []
    if raw is not None and len(raw):
        for ts, ratio in raw.items():
            try:
                ratio_f = float(ratio)
            except (TypeError, ValueError):
                continue
            if not ratio_f or ratio_f <= 0:
                continue
            day = pd.Timestamp(ts)
            if day.tzinfo is not None:
                day = day.tz_localize(None)
            out.append((day.date(), ratio_f))
    path.write_text(
        "\n".join("%s,%s" % (d.isoformat(), r) for d, r in out),
        encoding="utf-8",
    )
    return out


def _close_on_strike_scale(close: pd.Series, splits: Sequence[Tuple[date, float]]) -> pd.Series:
    """Multiply pre-split closes by later split ratios so the price matches option strikes."""
    out = pd.to_numeric(close, errors="coerce").astype(float).copy()
    if not splits or out.empty:
        return out
    idx = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index = idx
    for split_day, ratio in splits:
        out.loc[out.index < pd.Timestamp(split_day)] *= float(ratio)
    return out


def _series_from_frame(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close.notna()]
    if close.empty:
        return None
    return close


def _yahoo_closes(symbols: Sequence[str], start: date, end: date) -> Dict[str, pd.Series]:
    """Sector ETFs are not in TimescaleDB. Yahoo supplies the close used to pick strikes."""
    import yfinance as yf

    if not symbols:
        return {}
    raw = yf.download(
        list(symbols),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
    )
    out: Dict[str, pd.Series] = {}
    if raw is None or raw.empty:
        return out
    if len(symbols) == 1:
        sym = str(symbols[0]).upper()
        frame = raw.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        series = _series_from_frame(frame)
        if series is not None:
            out[sym] = series
        return out
    for sym in symbols:
        key = str(sym).upper()
        if isinstance(raw.columns, pd.MultiIndex) and key in raw.columns.get_level_values(0):
            frame = raw[key].copy()
        elif isinstance(raw.columns, pd.MultiIndex) and key in raw.columns.get_level_values(1):
            frame = raw.xs(key, axis=1, level=1).copy()
        else:
            continue
        frame.columns = [str(c).lower() for c in frame.columns]
        series = _series_from_frame(frame)
        if series is not None:
            out[key] = series
    return out


def load_closes(symbols: Sequence[str], start: date, end: date, split_cache: Path) -> Dict[str, pd.Series]:
    frames = load_ohlcv_many(
        list(symbols),
        timeframe="1d",
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    out: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        series = _series_from_frame(df)
        if series is not None:
            out[str(sym).upper()] = series
    missing = [s for s in symbols if str(s).upper() not in out]
    if missing:
        LOG.info("Yahoo closes for %s", ",".join(missing))
        yahoo = _yahoo_closes(missing, start, end)
        out.update(yahoo)
        still = [s for s in missing if str(s).upper() not in out]
        if still:
            LOG.warning("no daily close for %s", ",".join(still))
    scaled: Dict[str, pd.Series] = {}
    for sym, series in out.items():
        splits = _split_ratios(sym, split_cache)
        scaled[sym] = _close_on_strike_scale(series, splits)
        if splits:
            LOG.info("%s strike-scale splits=%s", sym, ",".join(d.isoformat() for d, _r in splits))
    return scaled


def _expirations(client, cache_dir: Path, symbol: str) -> List[date]:
    path = cache_dir / ("expirations_%s.txt" % symbol)
    if path.exists():
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(date.fromisoformat(line))
        if out:
            return out
    exps = list_expirations(client, symbol)
    path.write_text("\n".join(e.isoformat() for e in exps), encoding="utf-8")
    return exps


def _sessions_from_close(close: pd.Series, start: date, end: date) -> List[date]:
    idx = pd.to_datetime(close.index).date
    # .date on DatetimeIndex is an array of datetime.date via python loop
    days = []
    for ts in pd.to_datetime(close.index):
        day = pd.Timestamp(ts).date()
        if start <= day <= end:
            days.append(day)
    return days


def _iv_map_for(symbol: str, quotes: pd.DataFrame) -> Dict[date, float]:
    if quotes.empty:
        return {}
    surf = daily_surface(quotes.loc[quotes["symbol"] == symbol])
    out: Dict[date, float] = {}
    if surf.empty or "atm_iv" not in surf.columns:
        return out
    for row in surf.itertuples(index=False):
        iv = getattr(row, "atm_iv", float("nan"))
        if iv == iv:
            out[row.session] = float(iv)
    return out


def _ranges_for(
    symbols: Sequence[str],
    closes: Dict[str, pd.Series],
    expirations: Dict[str, List[date]],
    sessions: Sequence[date],
    roles: Sequence[str],
    quotes: pd.DataFrame,
    sessions_by_symbol: Optional[Dict[str, Sequence[date]]] = None,
) -> List[dict]:
    tasks: List[dict] = []
    for sym in symbols:
        close = closes.get(sym)
        exps = expirations.get(sym) or []
        if close is None or not exps:
            LOG.warning("no close or expirations for %s", sym)
            continue
        iv_map = _iv_map_for(sym, quotes) if "put25" in roles else None
        use_sessions = list(sessions)
        if sessions_by_symbol is not None and sym in sessions_by_symbol:
            use_sessions = list(sessions_by_symbol[sym])
        tasks.extend(
            plan_strike_ranges(
                sym,
                close,
                exps,
                use_sessions,
                list(roles),
                iv_by_session=iv_map,
            )
        )
    return tasks
    tasks: List[dict] = []
    for sym in symbols:
        close = closes.get(sym)
        exps = expirations.get(sym) or []
        if close is None or not exps:
            LOG.warning("no close or expirations for %s", sym)
            continue
        iv_map = _iv_map_for(sym, quotes) if "put25" in roles else None
        tasks.extend(
            plan_strike_ranges(
                sym,
                close,
                exps,
                list(sessions),
                list(roles),
                iv_by_session=iv_map,
            )
        )
    return tasks


def _repair_tasks(
    symbols: Sequence[str],
    closes: Dict[str, pd.Series],
    expirations: Dict[str, List[date]],
    sessions: Sequence[date],
    quotes: pd.DataFrame,
    role: str,
) -> List[dict]:
    """Single-day neighbor strikes where the snapped strike missed the target delta."""
    surf = daily_surface(quotes) if not quotes.empty else pd.DataFrame()
    have = set()
    if not surf.empty and "atm_iv" in surf.columns and role.startswith("atm"):
        good = surf.loc[surf["atm_iv"].notna()]
        have = set(zip(good["symbol"], good["session"]))
    if role == "put25" and not surf.empty and "skew" in surf.columns:
        good = surf.loc[surf["skew"].notna()]
        have = set(zip(good["symbol"], good["session"]))
    tasks: List[dict] = []
    for sym in symbols:
        close = closes.get(sym)
        exps = expirations.get(sym) or []
        if close is None or not exps:
            continue
        px = {}
        for ts, val in close.items():
            px[pd.Timestamp(ts).date()] = float(val)
        iv_map = _iv_map_for(sym, quotes) if role == "put25" else {}
        for day in sessions:
            if (sym, day) in have:
                continue
            spot = px.get(day)
            if spot is None or spot != spot or spot <= 0:
                continue
            from utils.research.last_15m_option_filters import choose_expiration, ROLE_SPECS

            target, lo, hi, _delta = ROLE_SPECS[role]
            exp = choose_expiration(exps, day, target, lo, hi)
            if exp is None:
                continue
            for strike in repair_strikes(spot, role):
                if strike != strike:
                    continue
                tasks.append(
                    {
                        "symbol": sym,
                        "expiration": exp,
                        "strike": float(strike),
                        "role": role,
                        "start": day,
                        "end": day,
                        "n_sessions": 1,
                    }
                )
            if role == "put25":
                iv = iv_map.get(day, 0.22)
                strike = put_strike_for_delta(spot, (exp - day).days, iv)
                base = snap_strike(spot)
                if strike == strike and abs(strike - base) > 0.1:
                    tasks.append(
                        {
                            "symbol": sym,
                            "expiration": exp,
                            "strike": float(strike),
                            "role": role,
                            "start": day,
                            "end": day,
                            "n_sessions": 1,
                        }
                    )
    # unique
    seen = set()
    out = []
    for task in tasks:
        key = (task["symbol"], task["expiration"], task["strike"], task["start"], task["role"])
        if key in seen:
            continue
        seen.add(key)
        out.append(task)
    return out


def _print_book(label: str, trades: pd.DataFrame) -> dict:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    years = summarize_by_year(trades, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    stats = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = trades.loc[pd.to_datetime(trades["buy_date"]).dt.year == 2022]
    s22 = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s | 2022 n=%s E=%s PF=%s",
        label,
        stats["n"],
        stats["expectancy_pct"],
        stats["profit_factor"],
        None if geo is None else round(float(geo), 3),
        s22["n"],
        s22["expectancy_pct"],
        s22["profit_factor"],
    )
    print("---- %s ----" % label)
    print(years.to_string(index=False))
    return {
        "label": label,
        "n": stats["n"],
        "expectancy_pct": stats["expectancy_pct"],
        "profit_factor": stats["profit_factor"],
        "geo_year_pf": geo,
        "n_2022": s22["n"],
        "e_2022": s22["expectancy_pct"],
        "pf_2022": s22["profit_factor"],
    }


def _write_quintiles(trades: pd.DataFrame, features: Sequence[str], outdir: Path) -> None:
    for feature in features:
        if feature not in trades.columns:
            continue
        table = quintile_table(trades, feature)
        path = outdir / ("quintile_%s.csv" % feature)
        table.to_csv(path, index=False)
        print("---- quintile %s ----" % feature)
        print(table.to_string(index=False))


def _score_locked(
    trades: pd.DataFrame,
    rule: Optional[Tuple[str, str, float]],
    baseline_gain: pd.Series,
) -> Optional[dict]:
    if rule is None:
        return None
    feature, direction, threshold = rule
    keep = combine_keep(trades, [rule])
    part = trades.loc[keep].copy()
    rec = _print_book("locked_%s_%s" % (feature, direction), part)
    cut = winner_cut_skip(baseline_gain, keep)
    rec["winner_cut"] = cut
    print(
        "  winner-cut dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
        % (
            cut["n_dropped"],
            cut["winners_dropped"],
            cut["losers_dropped"],
            cut["winner_drop_gain_sum"],
            cut["loser_drop_gain_sum"],
            cut["blunt"],
        )
    )
    return rec


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def run_pull(
    symbols: Sequence[str],
    closes: Dict[str, pd.Series],
    sessions: Sequence[date],
    cache_dir: Path,
    base_url: str,
    client,
    *,
    workers: int,
    atm_roles: Sequence[str],
    with_put25: bool,
    repair: bool,
    sessions_by_symbol: Optional[Dict[str, Sequence[date]]] = None,
) -> pd.DataFrame:
    expirations = {sym: _expirations(client, cache_dir, sym) for sym in symbols}
    LOG.info("planning ATM ranges for %d symbols", len(symbols))
    tasks = _ranges_for(
        symbols, closes, expirations, sessions, atm_roles, pd.DataFrame(), sessions_by_symbol
    )
    LOG.info("ATM tasks %d", len(tasks))
    fetch_tasks(tasks, cache_dir, base_url, workers=workers)
    quotes = load_cached_quotes(cache_dir, symbols)
    if repair:
        extra = _repair_tasks(symbols, closes, expirations, sessions, quotes, "atm30")
        if extra:
            LOG.info("ATM repair tasks %d", len(extra))
            fetch_tasks(extra, cache_dir, base_url, workers=workers)
            quotes = load_cached_quotes(cache_dir, symbols)
    if with_put25:
        tasks25 = _ranges_for(
            symbols, closes, expirations, sessions, ["put25"], quotes, sessions_by_symbol
        )
        LOG.info("put25 tasks %d", len(tasks25))
        fetch_tasks(tasks25, cache_dir, base_url, workers=workers)
        quotes = load_cached_quotes(cache_dir, symbols)
        if repair:
            extra25 = _repair_tasks(symbols, closes, expirations, sessions, quotes, "put25")
            # put25 repair_strikes are neighbors of spot, not of the 25-delta strike.
            # The explicit BS strike is already in the first put25 pass. Skip the
            # neighbor-of-spot flood; only retry when skew coverage is thin.
            have = 0
            if not quotes.empty:
                surf = daily_surface(quotes)
                have = int(surf["skew"].notna().sum()) if "skew" in surf.columns else 0
            if have < 0.5 * len(sessions) * max(len(symbols), 1) and extra25:
                LOG.info("put25 repair tasks %d (skew rows %d)", len(extra25), have)
                fetch_tasks(extra25, cache_dir, base_url, workers=workers)
                quotes = load_cached_quotes(cache_dir, symbols)
    return quotes


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--names", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    t0 = time.time()
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    trades["buy_date"] = pd.to_datetime(trades["buy_date"])
    LOG.info("trades n=%d %s -> %s", len(trades), trades["buy_date"].min().date(), trades["buy_date"].max().date())

    sectors = load_sectors()
    mapped = trades["stock"].map(lambda s: sectors.get(s, ""))
    LOG.info(
        "sector mapped %d/%d",
        int(mapped.astype(bool).sum()),
        len(trades),
    )

    end = trades["buy_date"].max().date()
    closes = load_closes(UNDERLIERS, HISTORY_START, end + timedelta(days=5), Path(args.cache))
    if MARKET_SYMBOL not in closes:
        raise RuntimeError("SPY daily closes are required")
    sessions = _sessions_from_close(closes[MARKET_SYMBOL], HISTORY_START, end)
    LOG.info("SPY sessions %d (%s -> %s)", len(sessions), sessions[0], sessions[-1])

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = get_thetadata_client()
    base_url = client.base_url
    if not args.skip_fetch:
        if not client.is_ready():
            raise RuntimeError("Theta Terminal is not reachable at %s" % base_url)
        run_pull(
            UNDERLIERS,
            closes,
            sessions,
            cache_dir,
            base_url,
            client,
            workers=args.workers,
            atm_roles=ATM_ROLES,
            with_put25=True,
            repair=True,
        )

    quotes = load_cached_quotes(cache_dir, UNDERLIERS)
    LOG.info("cached quote rows %d", len(quotes))
    surface = daily_surface(quotes)
    panel = market_sector_panel(surface, sessions)
    joined = attach_etf_features(trades, panel, sectors)

    outdir = dated_outdir(args.outdir)
    panel.reset_index().to_csv(outdir / "etf_option_panel.csv", index=False)
    joined.to_csv(outdir / "last15m_option_features.csv", index=False)
    surface.to_csv(outdir / "etf_option_surface.csv", index=False)

    features = [
        "spy_skew_pct",
        "spy_skew_chg5_pct",
        "dispersion_pct",
        "sector_iv_pct",
        "sector_skew_pct",
    ]
    _write_quintiles(joined, features, outdir)
    baseline = _print_book("baseline_last15m", joined)
    baseline_gain = pd.to_numeric(joined["gain_pct"], errors="coerce")

    market_rule, market_scored = lock_first_rule(joined, MARKET_RULES)
    sector_rule, sector_scored = lock_first_rule(joined, SECTOR_RULES)
    LOG.info("locked market=%s sector=%s", market_rule, sector_rule)
    market_rec = _score_locked(joined, market_rule, baseline_gain)
    sector_rec = _score_locked(joined, sector_rule, baseline_gain)

    locked_rules = [r for r in (market_rule, sector_rule) if r is not None]
    if locked_rules:
        both = joined.loc[combine_keep(joined, locked_rules)].copy()
        both_rec = _print_book("locked_market_and_sector", both)
        y22 = both.loc[pd.to_datetime(both["buy_date"]).dt.year == 2022]
        e_2022_kept = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))["expectancy_pct"]
    else:
        both_rec = None
        e_2022_kept = baseline["e_2022"]

    wants_names = etf_layer_wants_names(len(locked_rules), e_2022_kept)
    if args.names == "yes":
        wants_names = True
    elif args.names == "no":
        wants_names = False
    LOG.info("name layer wanted=%s (2022 kept E=%s)", wants_names, e_2022_kept)

    name_rule = None
    name_scored: List[dict] = []
    name_rec = None
    if wants_names:
        adv = pd.to_numeric(joined["adv_20"], errors="coerce")
        liquid = joined.loc[adv >= NAME_ADV]
        symbols = sorted(liquid["stock"].unique())
        LOG.info("liquid names %d trades %d", len(symbols), len(liquid))
        if symbols:
            name_start = (liquid["buy_date"].min() - pd.Timedelta(days=420)).date()
            if name_start < HISTORY_START:
                name_start = HISTORY_START
            sessions_by_symbol: Dict[str, List[date]] = {}
            for sym, part in liquid.groupby("stock"):
                first = (part["buy_date"].min() - pd.Timedelta(days=420)).date()
                last = part["buy_date"].max().date()
                if first < HISTORY_START:
                    first = HISTORY_START
                sessions_by_symbol[str(sym)] = [d for d in sessions if first <= d <= last]
            name_closes = load_closes(symbols, name_start, end + timedelta(days=3), cache_dir)
            run_pull(
                symbols,
                name_closes,
                sessions,
                cache_dir,
                base_url,
                client,
                workers=args.workers,
                atm_roles=NAME_ROLES_ATM,
                with_put25=True,
                repair=False,
                sessions_by_symbol=sessions_by_symbol,
            )
            name_quotes = load_cached_quotes(cache_dir, symbols)
            name_surface = daily_surface(name_quotes)
            name_surface = stamp_sector_atm(name_surface, panel, sectors)
            name_surface.to_csv(outdir / "name_option_surface.csv", index=False)
            joined = name_feature_frame(name_surface, joined, min_adv=NAME_ADV)
            joined.to_csv(outdir / "last15m_option_features.csv", index=False)
            _write_quintiles(
                joined,
                ["name_iv_pct", "name_skew_pct", "name_vs_sector_pct"],
                outdir,
            )
            name_rule, name_scored = lock_first_rule(joined, NAME_RULES)
            LOG.info("locked name=%s", name_rule)
            name_rec = _score_locked(joined, name_rule, baseline_gain)

    summary = {
        "n_trades": int(len(trades)),
        "wall_sec": round(time.time() - t0, 1),
        "baseline": baseline,
        "market_locked": market_rule,
        "sector_locked": sector_rule,
        "market_scored": market_scored,
        "sector_scored": sector_scored,
        "market_book": market_rec,
        "sector_book": sector_rec,
        "combined_book": both_rec,
        "name_layer": wants_names,
        "name_locked": name_rule,
        "name_scored": name_scored,
        "name_book": name_rec,
        "e_2022_after_etf": e_2022_kept,
        "coverage": {
            col: int(pd.to_numeric(joined[col], errors="coerce").notna().sum())
            for col in features + ["name_iv_pct", "name_skew_pct", "name_vs_sector_pct"]
            if col in joined.columns
        },
    }
    (outdir / "option_filter_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    LOG.info("wrote %s wall %.0fs", outdir, time.time() - t0)
    print("outdir", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
