"""Parse OCC option tickers and Theta NBBO vertical credits / walks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient
from utils.research.post_earnings_pcs_open_delta import MIN_CREDIT_FRAC

ET = ZoneInfo("America/New_York")

_OCC_RE = re.compile(
    r"^O:(?P<root>[A-Z.]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])(?P<strike>\d{8})$"
)


def parse_occ_ticker(ticker: str) -> Dict[str, Any]:
    m = _OCC_RE.match(str(ticker).upper().strip())
    if not m:
        raise ValueError("bad OCC ticker: %s" % ticker)
    yy = int(m.group("yy"))
    year = 2000 + yy if yy < 80 else 1900 + yy
    strike = int(m.group("strike")) / 1000.0
    return {
        "symbol": m.group("root"),
        "expiration": date(year, int(m.group("mm")), int(m.group("dd"))),
        "right": "put" if m.group("cp") == "P" else "call",
        "strike": strike,
    }


def parse_clock(clock: Union[str, time]) -> time:
    if isinstance(clock, time):
        return clock
    parts = str(clock).strip().split(":")
    return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def vertical_credit_nbbo(
    short_bid: float,
    short_ask: float,
    long_bid: float,
    long_ask: float,
) -> Dict[str, float]:
    """Sell short / buy long fills vs mid."""
    return {
        "credit_bidask": float(short_bid) - float(long_ask),
        "credit_mid": 0.5 * (float(short_bid) + float(short_ask))
        - 0.5 * (float(long_bid) + float(long_ask)),
        "short_mid": 0.5 * (float(short_bid) + float(short_ask)),
        "long_mid": 0.5 * (float(long_bid) + float(long_ask)),
        "short_spread": float(short_ask) - float(short_bid),
        "long_spread": float(long_ask) - float(long_bid),
    }


def marketable_debit(short_ask: float, long_bid: float) -> float:
    """Buy short / sell long to close: pay ask on short, receive bid on long."""
    return max(0.0, float(short_ask) - float(long_bid))


def vertical_nbbo_at_clock(
    client: ThetaDataClient,
    *,
    short_ticker: str,
    long_ticker: str,
    entry_date: date,
    clock: Union[str, time] = "09:30",
    interval: str = "1m",
    end_time: str = "16:00:00",
) -> Dict[str, Any]:
    """NBBO on both legs at/after clock; credit at that mark."""
    clk = parse_clock(clock)
    sp = parse_occ_ticker(short_ticker)
    lp = parse_occ_ticker(long_ticker)
    out: Dict[str, Any] = {
        "short_ticker": short_ticker,
        "long_ticker": long_ticker,
        "entry_date": entry_date.isoformat(),
        "clock": "%02d:%02d" % (clk.hour, clk.minute),
        "status": "pending",
    }
    s = client.nbbo_at_or_after(
        symbol=sp["symbol"],
        expiration=sp["expiration"],
        strike=sp["strike"],
        right=sp["right"],
        day=entry_date,
        clock=clk,
        interval=interval,
        end_time=end_time,
    )
    l = client.nbbo_at_or_after(
        symbol=lp["symbol"],
        expiration=lp["expiration"],
        strike=lp["strike"],
        right=lp["right"],
        day=entry_date,
        clock=clk,
        interval=interval,
        end_time=end_time,
    )
    if s is None:
        out["status"] = "no_short_nbbo"
        return out
    if l is None:
        out["status"] = "no_long_nbbo"
        out["short_bid"] = s["bid"]
        out["short_ask"] = s["ask"]
        return out
    creds = vertical_credit_nbbo(s["bid"], s["ask"], l["bid"], l["ask"])
    out.update(
        {
            "status": "ok",
            "short_ts": str(s["timestamp"]),
            "long_ts": str(l["timestamp"]),
            "short_bid": s["bid"],
            "short_ask": s["ask"],
            "long_bid": l["bid"],
            "long_ask": l["ask"],
            **creds,
        }
    )
    return out


def first_rth_vertical_nbbo(
    client: ThetaDataClient,
    *,
    short_ticker: str,
    long_ticker: str,
    entry_date: date,
    interval: str = "1m",
) -> Dict[str, Any]:
    """First usable RTH NBBO on both legs; credit at that clock."""
    return vertical_nbbo_at_clock(
        client,
        short_ticker=short_ticker,
        long_ticker=long_ticker,
        entry_date=entry_date,
        clock="09:30",
        interval=interval,
        end_time="10:30:00",
    )


def _minute_key(ts: Any) -> Optional[datetime]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(ET)
    else:
        t = t.tz_convert(ET)
    return t.to_pydatetime().replace(second=0, microsecond=0)


def align_minute_legs(
    short_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join 1m NBBO legs on minute timestamp; add marketable credit."""
    if short_df is None or long_df is None or short_df.empty or long_df.empty:
        return pd.DataFrame()
    s = short_df.copy()
    l = long_df.copy()
    s["minute"] = s["timestamp"].map(_minute_key)
    l["minute"] = l["timestamp"].map(_minute_key)
    s = s.dropna(subset=["minute"]).rename(
        columns={"bid": "short_bid", "ask": "short_ask", "timestamp": "short_ts"}
    )
    l = l.dropna(subset=["minute"]).rename(
        columns={"bid": "long_bid", "ask": "long_ask", "timestamp": "long_ts"}
    )
    m = s.merge(l[["minute", "long_bid", "long_ask", "long_ts"]], on="minute", how="inner")
    if m.empty:
        return m
    m["credit_bidask"] = m["short_bid"].astype(float) - m["long_ask"].astype(float)
    m["credit_mid"] = 0.5 * (
        m["short_bid"].astype(float) + m["short_ask"].astype(float)
    ) - 0.5 * (m["long_bid"].astype(float) + m["long_ask"].astype(float))
    return m.sort_values("minute").reset_index(drop=True)


def first_marketable_from_aligned(
    aligned: pd.DataFrame,
    *,
    min_credit: float,
    require_min: bool = True,
) -> Optional[Dict[str, Any]]:
    """First minute row with marketable credit (optionally >= min_credit)."""
    if aligned is None or aligned.empty:
        return None
    for _, row in aligned.iterrows():
        sb = float(row["short_bid"])
        sa = float(row["short_ask"])
        lb = float(row["long_bid"])
        la = float(row["long_ask"])
        if sb <= 0 or sa <= 0 or lb <= 0 or la <= 0:
            continue
        credit = float(row["credit_bidask"])
        if require_min:
            if credit < float(min_credit) - 1e-12:
                continue
        else:
            if credit <= 0:
                continue
        minute = row["minute"]
        wait = None
        if isinstance(minute, datetime):
            open_dt = minute.replace(hour=9, minute=30, second=0, microsecond=0)
            wait = max(0, int((minute - open_dt).total_seconds() // 60))
        return {
            "status": "ok",
            "minute": minute,
            "fill_clock": (
                minute.strftime("%H:%M") if isinstance(minute, datetime) else None
            ),
            "wait_minutes": wait,
            "short_bid": sb,
            "short_ask": sa,
            "long_bid": lb,
            "long_ask": la,
            "credit_bidask": credit,
            "credit_mid": float(row["credit_mid"]),
            "min_credit": float(min_credit),
            "meets_min": bool(credit >= float(min_credit) - 1e-12),
        }
    return None


def first_marketable_vertical(
    client: ThetaDataClient,
    *,
    short_ticker: str,
    long_ticker: str,
    entry_date: date,
    width: float,
    start_time: str = "09:30:00",
    end_time: str = "11:00:00",
    min_credit_frac: float = MIN_CREDIT_FRAC,
) -> Dict[str, Any]:
    """Scan morning 1m NBBO; return first meet-min and first any-positive fills."""
    sp = parse_occ_ticker(short_ticker)
    lp = parse_occ_ticker(long_ticker)
    width_f = float(width)
    min_c = float(min_credit_frac) * max(width_f, 1e-9)
    out: Dict[str, Any] = {
        "short_ticker": short_ticker,
        "long_ticker": long_ticker,
        "entry_date": entry_date.isoformat(),
        "width": width_f,
        "min_credit": min_c,
        "status": "pending",
    }
    try:
        s_df = client.morning_minute_quotes(
            symbol=sp["symbol"],
            expiration=sp["expiration"],
            strike=sp["strike"],
            right=sp["right"],
            day=entry_date,
            start_time=start_time,
            end_time=end_time,
        )
        l_df = client.morning_minute_quotes(
            symbol=lp["symbol"],
            expiration=lp["expiration"],
            strike=lp["strike"],
            right=lp["right"],
            day=entry_date,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:  # noqa: BLE001
        out["status"] = "quote_err"
        out["error"] = str(exc)
        return out

    out["n_short_min"] = int(len(s_df)) if s_df is not None else 0
    out["n_long_min"] = int(len(l_df)) if l_df is not None else 0
    aligned = align_minute_legs(s_df, l_df)
    out["n_aligned_min"] = int(len(aligned))
    if aligned.empty:
        out["status"] = "no_aligned_minutes"
        return out

    meet = first_marketable_from_aligned(aligned, min_credit=min_c, require_min=True)
    any_pos = first_marketable_from_aligned(aligned, min_credit=min_c, require_min=False)
    out["any_positive"] = any_pos
    if meet is None:
        out["status"] = "no_credit"
        if any_pos is not None:
            out["any_positive_credit"] = any_pos.get("credit_bidask")
            out["any_positive_clock"] = any_pos.get("fill_clock")
        return out

    out["status"] = "ok"
    out.update(
        {
            "fill_clock": meet["fill_clock"],
            "wait_minutes": meet["wait_minutes"],
            "short_bid": meet["short_bid"],
            "short_ask": meet["short_ask"],
            "long_bid": meet["long_bid"],
            "long_ask": meet["long_ask"],
            "credit_bidask": meet["credit_bidask"],
            "credit_mid": meet["credit_mid"],
            "meets_min": True,
        }
    )
    if any_pos is not None:
        out["any_positive_credit"] = any_pos.get("credit_bidask")
        out["any_positive_clock"] = any_pos.get("fill_clock")
        out["any_positive_wait"] = any_pos.get("wait_minutes")
    return out


@dataclass
class NbboDebitWalkResult:
    status: str
    entry_date: Optional[date] = None
    exit_date: Optional[date] = None
    credit: Optional[float] = None
    exit_debit: Optional[float] = None
    exit_reason: Optional[str] = None
    hold_days: Optional[int] = None
    pnl: Optional[float] = None
    ror_pct: Optional[float] = None
    n_days: int = 0


def align_daily_debit(
    short_daily: pd.DataFrame,
    long_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily clock quotes; debit = short_ask - long_bid."""
    if short_daily is None or long_daily is None or short_daily.empty or long_daily.empty:
        return pd.DataFrame()
    s = short_daily.rename(
        columns={"bid": "short_bid", "ask": "short_ask", "timestamp": "short_ts"}
    )
    l = long_daily.rename(
        columns={"bid": "long_bid", "ask": "long_ask", "timestamp": "long_ts"}
    )
    m = s.merge(
        l[["date", "long_bid", "long_ask", "long_ts"]],
        on="date",
        how="inner",
    )
    if m.empty:
        return m
    m["debit"] = m.apply(
        lambda r: marketable_debit(float(r["short_ask"]), float(r["long_bid"])),
        axis=1,
    )
    return m.sort_values("date").reset_index(drop=True)


def walk_vertical_nbbo_debit(
    daily: pd.DataFrame,
    *,
    entry_date: date,
    credit: float,
    width: float,
    expiry: date,
    tp_frac: float = 0.50,
    sl_mult: float = 2.0,
    include_entry_day: bool = True,
) -> NbboDebitWalkResult:
    """Walk TP50 / SL2x on marketable debits after (or on) entry day."""
    if daily is None or daily.empty:
        return NbboDebitWalkResult(status="no_bars")
    cred = float(credit)
    if cred <= 1e-12:
        return NbboDebitWalkResult(status="no_credit", entry_date=entry_date, credit=cred)
    width_f = float(width)
    max_risk = max(1e-9, width_f - cred)
    tp_debit = float(tp_frac) * cred
    sl_debit = min(float(sl_mult) * cred, width_f)

    if include_entry_day:
        after = daily[daily["date"] >= entry_date]
    else:
        after = daily[daily["date"] > entry_date]
    after = after[after["date"] <= expiry]
    if after.empty:
        return NbboDebitWalkResult(
            status="no_exit",
            entry_date=entry_date,
            credit=cred,
            n_days=0,
        )

    for _, row in after.iterrows():
        day = row["date"]
        if isinstance(day, pd.Timestamp):
            day = day.date()
        deb = float(row["debit"])
        if day >= expiry:
            pnl = cred - deb
            return NbboDebitWalkResult(
                status="ok",
                entry_date=entry_date,
                exit_date=day,
                credit=cred,
                exit_debit=deb,
                exit_reason="expiry",
                hold_days=(day - entry_date).days,
                pnl=pnl,
                ror_pct=100.0 * pnl / max_risk,
                n_days=int(len(after)),
            )
        if deb <= tp_debit + 1e-12:
            pnl = cred - tp_debit
            return NbboDebitWalkResult(
                status="ok",
                entry_date=entry_date,
                exit_date=day,
                credit=cred,
                exit_debit=tp_debit,
                exit_reason="tp50",
                hold_days=(day - entry_date).days,
                pnl=pnl,
                ror_pct=100.0 * pnl / max_risk,
                n_days=int(len(after)),
            )
        if deb >= sl_debit - 1e-12:
            pnl = cred - sl_debit
            return NbboDebitWalkResult(
                status="ok",
                entry_date=entry_date,
                exit_date=day,
                credit=cred,
                exit_debit=sl_debit,
                exit_reason="sl_2x",
                hold_days=(day - entry_date).days,
                pnl=pnl,
                ror_pct=100.0 * pnl / max_risk,
                n_days=int(len(after)),
            )

    return NbboDebitWalkResult(
        status="no_exit",
        entry_date=entry_date,
        credit=cred,
        n_days=int(len(after)),
    )


def fetch_exit_daily_debits(
    client: ThetaDataClient,
    *,
    short_ticker: str,
    long_ticker: str,
    entry_date: date,
    expiry: date,
    clock: time = time(15, 45),
) -> pd.DataFrame:
    """Daily 15:45 marketable debit series from entry through expiry (+slack)."""
    sp = parse_occ_ticker(short_ticker)
    lp = parse_occ_ticker(long_ticker)
    end = min(expiry + timedelta(days=2), entry_date + timedelta(days=31))
    s = client.daily_clock_quotes(
        symbol=sp["symbol"],
        expiration=sp["expiration"],
        strike=sp["strike"],
        right=sp["right"],
        start_date=entry_date,
        end_date=end,
        clock=clock,
    )
    l = client.daily_clock_quotes(
        symbol=lp["symbol"],
        expiration=lp["expiration"],
        strike=lp["strike"],
        right=lp["right"],
        start_date=entry_date,
        end_date=end,
        clock=clock,
    )
    return align_daily_debit(s, l)
