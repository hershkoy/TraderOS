"""Parse OptionsDX EOD option-chain CSVs (one row = call + put)."""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .option_utils import build_option_id

ARCHIVE_UNDERLYING_RE = re.compile(r"^(spy|vix)_eod", re.IGNORECASE)
SNAPSHOT_TYPE = "eod"

QUOTE_COLUMNS = [
    "ts",
    "option_id",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "snapshot_type",
]

CONTRACT_COLUMNS = [
    "option_id",
    "underlying",
    "expiration",
    "strike_cents",
    "option_right",
    "multiplier",
    "first_seen",
    "last_seen",
]


def underlying_from_archive_name(name: str) -> str:
    """Map spy_eod_2022q1-....7z / vix_eod_2010-....7z to SPY / VIX."""
    stem = name.split("/")[-1].split("\\")[-1]
    m = ARCHIVE_UNDERLYING_RE.match(stem)
    if not m:
        raise ValueError("Cannot infer underlying from archive name: %s" % name)
    return m.group(1).upper()


def normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").strip("[]").strip().upper().replace(" ", "_")


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in (".", "nan", "NaN", "None"):
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if abs(num) > 1e6:
        return None
    return num


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in (".", "nan", "NaN", "None"):
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def parse_unix_ts(value: Optional[str]) -> datetime:
    if value is None:
        raise ValueError("missing QUOTE_UNIXTIME")
    text = str(value).strip()
    if not text:
        raise ValueError("missing QUOTE_UNIXTIME")
    return datetime.fromtimestamp(int(float(text)), tz=timezone.utc)


def parse_expire_date(value: Optional[str]) -> date:
    if value is None:
        raise ValueError("missing EXPIRE_DATE")
    text = str(value).strip()
    return date.fromisoformat(text[:10])


def _leg_quote(
    row: Dict[str, str],
    prefix: str,
    option_right: str,
    underlying: str,
    expiration: date,
    strike: float,
    ts: datetime,
) -> Tuple[dict, dict]:
    strike_cents = int(round(strike * 100))
    option_id = build_option_id(underlying, expiration, strike, option_right)
    quote = {
        "ts": ts.isoformat(),
        "option_id": option_id,
        "bid": parse_float(row.get("%s_BID" % prefix)),
        "ask": parse_float(row.get("%s_ASK" % prefix)),
        "last": parse_float(row.get("%s_LAST" % prefix)),
        "volume": parse_int(row.get("%s_VOLUME" % prefix)),
        "open_interest": parse_int(
            row.get("%s_OPEN_INTEREST" % prefix) or row.get("%s_OI" % prefix)
        ),
        "iv": parse_float(row.get("%s_IV" % prefix)),
        "delta": parse_float(row.get("%s_DELTA" % prefix)),
        "gamma": parse_float(row.get("%s_GAMMA" % prefix)),
        "theta": parse_float(row.get("%s_THETA" % prefix)),
        "vega": parse_float(row.get("%s_VEGA" % prefix)),
        "snapshot_type": SNAPSHOT_TYPE,
    }
    contract = {
        "option_id": option_id,
        "underlying": underlying,
        "expiration": expiration.isoformat(),
        "strike_cents": strike_cents,
        "option_right": option_right,
        "multiplier": 100,
        "first_seen": ts.isoformat(),
        "last_seen": ts.isoformat(),
    }
    return quote, contract


def quotes_from_row(row: Dict[str, str], underlying: str) -> List[Tuple[dict, dict]]:
    """Return (quote, contract) pairs for the call and put on this chain row."""
    strike = parse_float(row.get("STRIKE"))
    if strike is None:
        return []
    try:
        ts = parse_unix_ts(row.get("QUOTE_UNIXTIME"))
        expiration = parse_expire_date(row.get("EXPIRE_DATE"))
    except (TypeError, ValueError):
        return []
    out = []
    out.append(_leg_quote(row, "C", "C", underlying, expiration, strike, ts))
    out.append(_leg_quote(row, "P", "P", underlying, expiration, strike, ts))
    return out


def iter_csv_rows(text_stream: Iterable[str]) -> Iterator[Dict[str, str]]:
    reader = csv.reader(text_stream)
    try:
        raw_header = next(reader)
    except StopIteration:
        return
    fieldnames = [normalize_header(h) for h in raw_header]
    for parts in reader:
        if not parts or all(not (p or "").strip() for p in parts):
            continue
        row = {}
        for i, name in enumerate(fieldnames):
            row[name] = parts[i] if i < len(parts) else ""
        yield row


def iter_file_quotes(path, underlying: str) -> Iterator[Tuple[dict, dict]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in iter_csv_rows(fh):
            for pair in quotes_from_row(row, underlying):
                yield pair


def parse_csv_text(text: str, underlying: str) -> List[Tuple[dict, dict]]:
    return list(iter_csv_rows_quotes(io.StringIO(text), underlying))


def iter_csv_rows_quotes(text_stream: Iterable[str], underlying: str):
    for row in iter_csv_rows(text_stream):
        for pair in quotes_from_row(row, underlying):
            yield pair
