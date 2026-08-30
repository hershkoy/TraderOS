"""TimescaleDB store for 15m channel-touch candidates and dashboard settings.

DB is the dashboard source of truth. JSON watchlist remains a debug sidecar.
Writes use a committed cursor (TimescaleDBClient.execute_query always fetchall).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from utils.scanning.channel_touch_15m import is_hot_proximity

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INIT_SQL_PATHS = (
    ROOT / "init-scripts" / "13-channel-touch-15m-candidates.sql",
    ROOT / "init-scripts" / "14-channel-touch-candidates-timeframe.sql",
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "telegram_on_fill": True,
    "telegram_on_hot": False,
    "proximity_below_pct": 0.0,
    "max_abs_dist_pct": None,
    "sort_key": "abs_dist",
    "sort_dir": "asc",
    "status_filter": "all",
    "search": "",
    "as_of": None,
    "n_universe": 0,
    "stale_warning": None,
    "timeframe_filter": "all",
    "as_of_1d": None,
    "n_universe_1d": 0,
}

CANDIDATE_COLUMNS = (
    "stock",
    "timeframe",
    "status",
    "as_of",
    "h2_time",
    "channel_start",
    "channel_end",
    "channel_span_days",
    "wait_bars",
    "wait_ok",
    "support",
    "resist",
    "last_close",
    "dist_to_resist_pct",
    "volume_rel_20",
    "overshoot_prior",
    "fill_px",
    "overshoot",
    "support_x0",
    "support_y0",
    "support_slope",
    "channel_width",
    "h2_idx",
    "as_of_i",
    "last_price",
    "last_price_ts",
    "dist_live_pct",
    "hot",
    "hot_notified_on",
)

SETTINGS_COLUMNS = (
    "telegram_on_fill",
    "telegram_on_hot",
    "proximity_below_pct",
    "max_abs_dist_pct",
    "sort_key",
    "sort_dir",
    "status_filter",
    "search",
    "as_of",
    "n_universe",
    "stale_warning",
    "timeframe_filter",
    "as_of_1d",
    "n_universe_1d",
)

_BOOL_KEYS = {"telegram_on_fill", "telegram_on_hot", "wait_ok", "hot"}
_INT_KEYS = {"wait_bars", "support_x0", "h2_idx", "as_of_i", "n_universe"}
_FLOAT_KEYS = {
    "proximity_below_pct",
    "max_abs_dist_pct",
    "channel_span_days",
    "support",
    "resist",
    "last_close",
    "dist_to_resist_pct",
    "volume_rel_20",
    "overshoot_prior",
    "fill_px",
    "overshoot",
    "support_y0",
    "support_slope",
    "channel_width",
    "last_price",
    "dist_live_pct",
}


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return default


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_settings(raw: Optional[dict] = None) -> Dict[str, Any]:
    src = dict(DEFAULT_SETTINGS)
    if raw:
        src.update(raw)
    out = dict(DEFAULT_SETTINGS)
    out["telegram_on_fill"] = _as_bool(src.get("telegram_on_fill"), True)
    out["telegram_on_hot"] = _as_bool(src.get("telegram_on_hot"), False)
    below = _as_float(src.get("proximity_below_pct"))
    out["proximity_below_pct"] = 0.0 if below is None else float(below)
    out["max_abs_dist_pct"] = _as_float(src.get("max_abs_dist_pct"))
    sort_key = str(src.get("sort_key") or "abs_dist")
    out["sort_key"] = sort_key if sort_key else "abs_dist"
    sort_dir = str(src.get("sort_dir") or "asc").lower()
    out["sort_dir"] = "desc" if sort_dir == "desc" else "asc"
    status = str(src.get("status_filter") or "all")
    out["status_filter"] = status if status else "all"
    out["search"] = str(src.get("search") or "")
    as_of = src.get("as_of")
    out["as_of"] = None if as_of in (None, "") else str(as_of)
    n_uni = _as_int(src.get("n_universe"))
    out["n_universe"] = 0 if n_uni is None else int(n_uni)
    stale = src.get("stale_warning")
    out["stale_warning"] = None if stale in (None, "") else str(stale)
    tf = str(src.get("timeframe_filter") or "all").lower()
    out["timeframe_filter"] = tf if tf in ("all", "15m", "1d") else "all"
    as_of_1d = src.get("as_of_1d")
    out["as_of_1d"] = None if as_of_1d in (None, "") else str(as_of_1d)
    n_uni_1d = _as_int(src.get("n_universe_1d"))
    out["n_universe_1d"] = 0 if n_uni_1d is None else int(n_uni_1d)
    return out


def apply_settings_patch(current: dict, patch: Optional[dict]) -> Dict[str, Any]:
    merged = dict(current or {})
    if not patch:
        return normalize_settings(merged)
    for key in SETTINGS_COLUMNS:
        if key in patch:
            merged[key] = patch[key]
    return normalize_settings(merged)


def same_setup(prev: dict, new: dict) -> bool:
    return (
        str(prev.get("h2_time") or "") == str(new.get("h2_time") or "")
        and str(prev.get("stock") or "").upper() == str(new.get("stock") or "").upper()
        and str(prev.get("timeframe") or "15m") == str(new.get("timeframe") or "15m")
    )


def _row_key(row: dict) -> str:
    stock = str(row.get("stock") or "").upper()
    tf = str(row.get("timeframe") or "15m")
    return "%s|%s" % (stock, tf)


def merge_preserved_live_fields(
    new_rows: Sequence[dict],
    existing_by_stock: Dict[str, dict],
    *,
    below_pct: float = 0.0,
) -> List[dict]:
    """Keep last price + hot_notified_on when the same symbol+timeframe+h2 is still armed."""
    out: List[dict] = []
    for row in new_rows:
        item = dict(row)
        stock = str(item.get("stock", "")).upper()
        item["stock"] = stock
        item["timeframe"] = str(item.get("timeframe") or "15m")
        prev = existing_by_stock.get(_row_key(item)) or existing_by_stock.get(stock)
        if prev and same_setup(prev, item):
            if item.get("last_price") is None:
                item["last_price"] = prev.get("last_price")
                item["last_price_ts"] = prev.get("last_price_ts")
                item["dist_live_pct"] = prev.get("dist_live_pct")
            item["hot_notified_on"] = prev.get("hot_notified_on")
            px = item.get("last_price")
            resist = item.get("resist")
            if px is not None and resist is not None:
                try:
                    px_f = float(px)
                    lvl = float(resist)
                except (TypeError, ValueError):
                    px_f = None
                    lvl = None
                if px_f is not None and lvl is not None and lvl > 0:
                    item["dist_live_pct"] = round((px_f - lvl) / lvl * 100.0, 4)
                    item["hot"] = is_hot_proximity(px_f, lvl, below_pct=float(below_pct))
        else:
            item.setdefault("last_price", None)
            item.setdefault("last_price_ts", None)
            item.setdefault("dist_live_pct", None)
            item.setdefault("hot", False)
            item["hot_notified_on"] = None
        item["hot"] = bool(item.get("hot"))
        out.append(item)
    return out


def row_to_db_tuple(row: dict) -> tuple:
    item = dict(row)
    item["stock"] = str(item.get("stock", "")).upper()
    item["timeframe"] = str(item.get("timeframe") or "15m")
    item["hot"] = bool(item.get("hot"))
    vals = []
    for col in CANDIDATE_COLUMNS:
        val = item.get(col)
        if col in _BOOL_KEYS:
            val = _as_bool(val, False)
        elif col in _INT_KEYS:
            val = _as_int(val)
        elif col in _FLOAT_KEYS:
            val = _as_float(val)
        elif col == "hot_notified_on" and val not in (None, ""):
            val = str(val)[:10]
        elif val is not None and not isinstance(val, (str, int, float, bool)):
            val = _jsonish(val)
        vals.append(val)
    return tuple(vals)


def dict_from_db_row(raw: dict) -> dict:
    out = {}
    for col in CANDIDATE_COLUMNS:
        out[col] = _jsonish(raw.get(col))
    out["stock"] = str(out.get("stock") or "").upper()
    out["timeframe"] = str(out.get("timeframe") or "15m")
    out["hot"] = _as_bool(out.get("hot"), False)
    out["wait_ok"] = _as_bool(out.get("wait_ok"), False)
    return out


class ChannelTouchCandidatesStore:
    """Load/replace 15m candidates and dashboard settings in TimescaleDB."""

    def __init__(
        self,
        client: Any = None,
        *,
        runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._client = client
        self._runner = runner
        self._ensured = False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from utils.db.timescaledb_client import get_timescaledb_client

        self._client = get_timescaledb_client()
        return self._client

    def _run(self, sql: str, params: Optional[tuple] = None, *, fetch: bool = False) -> Any:
        if self._runner is not None:
            return self._runner(sql, params, fetch=fetch)
        client = self._get_client()
        if not client.ensure_connection():
            raise RuntimeError("TimescaleDB connection failed")
        conn = client.connection
        from psycopg2.extras import RealDictCursor

        cur = conn.cursor(cursor_factory=RealDictCursor) if fetch else conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def ensure_tables(self) -> None:
        if self._ensured:
            return
        chunks: List[str] = []
        for path in INIT_SQL_PATHS:
            if path.exists():
                chunks.append(path.read_text(encoding="utf-8"))
        sql_text = "\n".join(chunks) if chunks else _FALLBACK_DDL
        if self._runner is not None:
            for stmt in _split_sql(sql_text):
                self._runner(stmt, None, fetch=False)
        else:
            client = self._get_client()
            if not client.ensure_connection():
                raise RuntimeError("TimescaleDB connection failed")
            conn = client.connection
            cur = conn.cursor()
            try:
                files = [p for p in INIT_SQL_PATHS if p.exists()]
                if not files:
                    cur.execute(_FALLBACK_DDL)
                    conn.commit()
                else:
                    for path in files:
                        try:
                            cur.execute(path.read_text(encoding="utf-8"))
                            conn.commit()
                        except Exception:
                            conn.rollback()
                            logger.warning("init sql %s failed", path.name, exc_info=True)
            finally:
                cur.close()
        self._migrate_composite_pk()
        self._ensured = True

    def _migrate_composite_pk(self) -> None:
        """Existing DBs had PRIMARY KEY (stock). Switch to (stock, timeframe)."""
        try:
            self._run(
                "ALTER TABLE channel_touch_15m_candidates "
                "DROP CONSTRAINT IF EXISTS channel_touch_15m_candidates_pkey"
            )
            self._run(
                "ALTER TABLE channel_touch_15m_candidates "
                "ADD PRIMARY KEY (stock, timeframe)"
            )
        except Exception:
            logger.debug("composite PK migrate skipped", exc_info=True)

    def load_settings(self) -> Dict[str, Any]:
        self.ensure_tables()
        rows = self._run(
            "SELECT * FROM channel_touch_15m_settings WHERE id = 1",
            fetch=True,
        )
        if not rows:
            return dict(DEFAULT_SETTINGS)
        raw = dict(rows[0])
        raw.pop("id", None)
        raw.pop("updated_at", None)
        return normalize_settings(raw)

    def save_settings(self, patch: Optional[dict] = None, *, current: Optional[dict] = None) -> Dict[str, Any]:
        self.ensure_tables()
        if current is None:
            current = self.load_settings()
        settings = apply_settings_patch(current, patch)
        self._run(
            """
            UPDATE channel_touch_15m_settings
            SET telegram_on_fill = %s,
                telegram_on_hot = %s,
                proximity_below_pct = %s,
                max_abs_dist_pct = %s,
                sort_key = %s,
                sort_dir = %s,
                status_filter = %s,
                search = %s,
                as_of = %s,
                n_universe = %s,
                stale_warning = %s,
                timeframe_filter = %s,
                as_of_1d = %s,
                n_universe_1d = %s,
                updated_at = now()
            WHERE id = 1
            """,
            (
                settings["telegram_on_fill"],
                settings["telegram_on_hot"],
                settings["proximity_below_pct"],
                settings["max_abs_dist_pct"],
                settings["sort_key"],
                settings["sort_dir"],
                settings["status_filter"],
                settings["search"],
                settings["as_of"],
                settings["n_universe"],
                settings["stale_warning"],
                settings["timeframe_filter"],
                settings["as_of_1d"],
                settings["n_universe_1d"],
            ),
        )
        return settings

    def load_rows(self, timeframe: Optional[str] = None) -> List[dict]:
        self.ensure_tables()
        if timeframe:
            rows = self._run(
                "SELECT * FROM channel_touch_15m_candidates WHERE timeframe = %s ORDER BY stock",
                (str(timeframe),),
                fetch=True,
            )
        else:
            rows = self._run(
                "SELECT * FROM channel_touch_15m_candidates ORDER BY stock, timeframe",
                fetch=True,
            )
        if not rows:
            return []
        return [dict_from_db_row(dict(r)) for r in rows]

    def replace_candidates(
        self,
        rows: Sequence[dict],
        *,
        meta: Optional[dict] = None,
        below_pct: Optional[float] = None,
        timeframe: str = "15m",
    ) -> List[dict]:
        self.ensure_tables()
        tf = str(timeframe or "15m")
        settings = self.load_settings()
        if below_pct is None:
            below_pct = float(settings.get("proximity_below_pct") or 0.0)
        tagged: List[dict] = []
        for row in rows:
            item = dict(row)
            item["timeframe"] = tf
            tagged.append(item)
        existing = {_row_key(r): r for r in self.load_rows()}
        merged = merge_preserved_live_fields(tagged, existing, below_pct=float(below_pct))
        new_stocks = [r["stock"] for r in merged]
        if new_stocks:
            placeholders = ",".join(["%s"] * len(new_stocks))
            self._run(
                (
                    "DELETE FROM channel_touch_15m_candidates "
                    "WHERE timeframe = %s AND stock NOT IN (%s)"
                )
                % ("%s", placeholders),
                tuple([tf] + new_stocks),
            )
        else:
            self._run(
                "DELETE FROM channel_touch_15m_candidates WHERE timeframe = %s",
                (tf,),
            )
        if merged:
            cols = ", ".join(CANDIDATE_COLUMNS)
            placeholders = ", ".join(["%s"] * len(CANDIDATE_COLUMNS))
            update_cols = [c for c in CANDIDATE_COLUMNS if c not in ("stock", "timeframe")]
            set_clause = ", ".join("%s = EXCLUDED.%s" % (c, c) for c in update_cols)
            sql = (
                "INSERT INTO channel_touch_15m_candidates (%s) VALUES (%s) "
                "ON CONFLICT (stock, timeframe) DO UPDATE SET %s, updated_at = now()"
            ) % (cols, placeholders, set_clause)
            for row in merged:
                self._run(sql, row_to_db_tuple(row))
        if meta:
            patch = dict(meta)
            if tf == "1d":
                mapped = {}
                if "as_of" in patch:
                    mapped["as_of_1d"] = patch.pop("as_of")
                if "n_universe" in patch:
                    mapped["n_universe_1d"] = patch.pop("n_universe")
                patch.pop("stale_warning", None)
                patch.update(mapped)
            self.save_settings(patch, current=settings)
        return merged

    def update_live_prices(self, rows: Sequence[dict], *, price_ts: Optional[datetime] = None) -> None:
        self.ensure_tables()
        ts = price_ts or datetime.now(timezone.utc)
        for row in rows:
            stock = str(row.get("stock", "")).upper()
            if not stock:
                continue
            self._run(
                """
                UPDATE channel_touch_15m_candidates
                SET last_price = %s,
                    last_price_ts = %s,
                    dist_live_pct = %s,
                    hot = %s,
                    updated_at = now()
                WHERE stock = %s AND timeframe = %s
                """,
                (
                    _as_float(row.get("last_price")),
                    ts,
                    _as_float(row.get("dist_live_pct")),
                    _as_bool(row.get("hot"), False),
                    stock,
                    str(row.get("timeframe") or "15m"),
                ),
            )

    def mark_hot_notified(
        self,
        stocks: Sequence[str],
        day: Optional[str] = None,
        *,
        timeframe: Optional[str] = None,
    ) -> None:
        self.ensure_tables()
        day_s = str(day or date.today().isoformat())[:10]
        tf = str(timeframe) if timeframe else None
        for stock in stocks:
            key = str(stock).upper()
            if not key:
                continue
            if tf:
                self._run(
                    """
                    UPDATE channel_touch_15m_candidates
                    SET hot_notified_on = %s, updated_at = now()
                    WHERE stock = %s AND timeframe = %s
                    """,
                    (day_s, key, tf),
                )
            else:
                self._run(
                    """
                    UPDATE channel_touch_15m_candidates
                    SET hot_notified_on = %s, updated_at = now()
                    WHERE stock = %s
                    """,
                    (day_s, key),
                )


def _split_sql(text: str) -> List[str]:
    stmts = []
    for part in text.split(";"):
        lines = [
            ln for ln in part.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if stmt:
            stmts.append(stmt)
    return stmts


_FALLBACK_DDL = """
CREATE TABLE IF NOT EXISTS channel_touch_15m_settings (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    telegram_on_fill BOOLEAN NOT NULL DEFAULT TRUE,
    telegram_on_hot BOOLEAN NOT NULL DEFAULT FALSE,
    proximity_below_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_abs_dist_pct DOUBLE PRECISION,
    sort_key TEXT NOT NULL DEFAULT 'abs_dist',
    sort_dir TEXT NOT NULL DEFAULT 'asc',
    status_filter TEXT NOT NULL DEFAULT 'all',
    search TEXT NOT NULL DEFAULT '',
    as_of TEXT,
    n_universe INTEGER NOT NULL DEFAULT 0,
    stale_warning TEXT,
    timeframe_filter TEXT NOT NULL DEFAULT 'all',
    as_of_1d TEXT,
    n_universe_1d INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO channel_touch_15m_settings (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
CREATE TABLE IF NOT EXISTS channel_touch_15m_candidates (
    stock TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '15m',
    status TEXT NOT NULL,
    as_of TEXT,
    h2_time TEXT,
    channel_start TEXT,
    channel_end TEXT,
    channel_span_days DOUBLE PRECISION,
    wait_bars INTEGER,
    wait_ok BOOLEAN,
    support DOUBLE PRECISION,
    resist DOUBLE PRECISION,
    last_close DOUBLE PRECISION,
    dist_to_resist_pct DOUBLE PRECISION,
    volume_rel_20 DOUBLE PRECISION,
    overshoot_prior DOUBLE PRECISION,
    fill_px DOUBLE PRECISION,
    overshoot DOUBLE PRECISION,
    support_x0 INTEGER,
    support_y0 DOUBLE PRECISION,
    support_slope DOUBLE PRECISION,
    channel_width DOUBLE PRECISION,
    h2_idx INTEGER,
    as_of_i INTEGER,
    last_price DOUBLE PRECISION,
    last_price_ts TIMESTAMPTZ,
    dist_live_pct DOUBLE PRECISION,
    hot BOOLEAN NOT NULL DEFAULT FALSE,
    hot_notified_on DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stock, timeframe)
);
"""
