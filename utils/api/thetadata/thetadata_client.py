"""ThetaData local Terminal REST client (NBBO quotes via OPRA).

Requires Theta Terminal v3 running on localhost (default port 25503).
Launch: python scripts/pipeline/thetadata_terminal.py
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, time
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LOG = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

DateLike = Union[str, date, datetime]


def _ymd(d: DateLike) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _strike_str(strike: float) -> str:
    return f"{float(strike):.3f}"


class ThetaDataClient:
    """Thin HTTP wrapper around Theta Terminal v3 REST (127.0.0.1:25503)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 60.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("THETADATA_BASE_URL", "http://127.0.0.1:25503")
        ).rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()

    def status(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/v3/terminal/status", timeout=10)
        r.raise_for_status()
        return r.json() if r.content else {"ok": True, "status_code": r.status_code}

    def is_ready(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/v3/option/list/expirations", timeout=5)
            # 200 or 4xx both mean HTTP server is up
            return r.status_code < 500
        except Exception:
            return False

    def option_history_quote(
        self,
        *,
        symbol: str,
        expiration: DateLike,
        strike: float,
        right: str = "put",
        day: Optional[DateLike] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        interval: str = "1m",
        start_time: str = "09:30:00",
        end_time: str = "16:00:00",
    ) -> pd.DataFrame:
        """NBBO quotes (interval last quote). Returns bid/ask columns in ET.

        Pass ``day`` for a single session, or ``start_date``/``end_date`` for a
        range (Theta caps multi-day at ~1 month and requires expiration).
        """
        right_s = "put" if str(right).lower().startswith("p") else "call"
        params: Dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "expiration": _ymd(expiration),
            "strike": _strike_str(strike),
            "right": right_s,
            "interval": interval,
            "start_time": start_time,
            "end_time": end_time,
        }
        if day is not None:
            params["date"] = _ymd(day)
        else:
            if start_date is None or end_date is None:
                raise ValueError("option_history_quote needs day or start_date+end_date")
            params["start_date"] = _ymd(start_date)
            params["end_date"] = _ymd(end_date)
        url = f"{self.base_url}/v3/option/history/quote"
        r = self.session.get(url, params=params, timeout=self.timeout)
        # Theta uses 472 for "no data" on a contract/day — treat as empty.
        if r.status_code == 472:
            LOG.debug("Theta 472 no data: %s", r.url)
            return pd.DataFrame()
        r.raise_for_status()
        return self._quotes_to_df(r)

    def morning_minute_quotes(
        self,
        *,
        symbol: str,
        expiration: DateLike,
        strike: float,
        right: str = "put",
        day: DateLike,
        start_time: str = "09:30:00",
        end_time: str = "11:00:00",
    ) -> pd.DataFrame:
        """Entry-window 1m NBBO for one contract/day."""
        return self.option_history_quote(
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            day=day,
            interval="1m",
            start_time=start_time,
            end_time=end_time,
        )

    def daily_clock_quotes(
        self,
        *,
        symbol: str,
        expiration: DateLike,
        strike: float,
        right: str = "put",
        start_date: DateLike,
        end_date: DateLike,
        clock: time = time(15, 45),
        interval: str = "1m",
    ) -> pd.DataFrame:
        """One NBBO row per session at/after ``clock`` (default 15:45 ET).

        Pulls a date-range of 1m quotes around the clock minute, then keeps the
        first usable bid/ask quote at/after clock each calendar day.
        """
        start_s = "%02d:%02d:00" % (int(clock.hour), int(clock.minute))
        # Pull through 16:00 so a missing exact minute can still find a later print
        end_s = "16:00:00"
        raw = self.option_history_quote(
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            start_time=start_s,
            end_time=end_s,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(
                columns=["date", "timestamp", "bid", "ask", "mid", "spread"]
            )
        rows: list[Dict[str, Any]] = []
        raw = raw.copy()
        raw["date"] = raw["timestamp"].dt.date
        for d, grp in raw.groupby("date", sort=True):
            for _, row in grp.sort_values("timestamp").iterrows():
                bid = float(row.get("bid") or 0)
                ask = float(row.get("ask") or 0)
                if bid <= 0 or ask <= 0:
                    continue
                ts = row["timestamp"]
                rows.append(
                    {
                        "date": d,
                        "timestamp": ts,
                        "bid": bid,
                        "ask": ask,
                        "mid": 0.5 * (bid + ask),
                        "spread": ask - bid,
                    }
                )
                break
        return pd.DataFrame(rows)

    def _quotes_to_df(self, r: requests.Response) -> pd.DataFrame:
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" in ctype or (r.content[:1] in (b"{", b"[")):
            payload = r.json()
            rows = payload.get("response") or payload.get("results") or payload
            if isinstance(rows, dict):
                rows = rows.get("data") or []
            if not isinstance(rows, list):
                return pd.DataFrame()
            df = pd.DataFrame(rows)
        else:
            # CSV fallback
            from io import StringIO

            df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return df
        # normalize columns
        rename = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl in ("bid", "bid_price"):
                rename[c] = "bid"
            elif cl in ("ask", "ask_price"):
                rename[c] = "ask"
            elif cl in ("timestamp", "ms_of_day", "datetime", "time"):
                rename[c] = "timestamp"
            elif cl == "bid_size":
                rename[c] = "bid_size"
            elif cl == "ask_size":
                rename[c] = "ask_size"
        df = df.rename(columns=rename)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # if naive, treat as ET
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(ET)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert(ET)
        return df

    def first_rth_nbbo(
        self,
        *,
        symbol: str,
        expiration: DateLike,
        strike: float,
        right: str = "put",
        day: DateLike,
        interval: str = "1m",
    ) -> Optional[Dict[str, Any]]:
        """First RTH (>=09:30 ET) quote with both bid and ask > 0."""
        return self.nbbo_at_or_after(
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            day=day,
            clock=time(9, 30),
            interval=interval,
            end_time="10:30:00",
        )

    def nbbo_at_or_after(
        self,
        *,
        symbol: str,
        expiration: DateLike,
        strike: float,
        right: str = "put",
        day: DateLike,
        clock: time,
        interval: str = "1m",
        end_time: str = "16:00:00",
    ) -> Optional[Dict[str, Any]]:
        """First quote at/after clock ET with both bid and ask > 0."""
        start = "%02d:%02d:00" % (int(clock.hour), int(clock.minute))
        df = self.option_history_quote(
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            day=day,
            interval=interval,
            start_time=start,
            end_time=end_time,
        )
        if df is None or df.empty:
            return None
        for _, row in df.iterrows():
            bid = float(row.get("bid") or 0)
            ask = float(row.get("ask") or 0)
            if bid <= 0 or ask <= 0:
                continue
            ts = row.get("timestamp")
            return {
                "timestamp": ts,
                "bid": bid,
                "ask": ask,
                "mid": 0.5 * (bid + ask),
                "bid_size": row.get("bid_size"),
                "ask_size": row.get("ask_size"),
                "spread": ask - bid,
            }
        return None


_CLIENT: Optional[ThetaDataClient] = None


def get_thetadata_client() -> ThetaDataClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ThetaDataClient()
    return _CLIENT
