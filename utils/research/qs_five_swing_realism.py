"""QS five-swing fill realism: oracle MOC vs intraday proxy vs next-open.

Uses settled daily bars for the strategy state machine history. For proxy books,
replaces *today's* OHLC with RTH developing OHLC as of a decision clock
(15:45 / 15:50 / 15:55 ET on 5m, or last RTH 15m), then:
  - arm if entry mask fires on that hybrid row
  - fill at that day's settled close (MOC assumption) when armed

Next-open: arm only when settled daily signal is true; fill next session open.
Oracle: arm+fill on settled close (current claim-check).

Exit for all books: settled close > prior high (isolates entry realism).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.qs_five_swing import (
    ENTRY_MASKS,
    STRATEGY_IDS,
    _metrics_from_trades,
    prepare_frame,
)

# Decision clocks as (hour, minute) America/New_York RTH.
PROXY_CLOCKS_5M: Sequence[Tuple[int, int]] = ((15, 45), (15, 50), (15, 55))


def _session_date_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Calendar date for a bar: keep UTC midnight stamp for Yahoo/IB daily join.

    IB 5m/15m RTH stamps are UTC; America/New_York date of the bar is the session.
    Daily Yahoo bars in this project use UTC midnight of the calendar session date.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    ny = t.tz_convert("America/New_York")
    # Session date = NY calendar date
    return pd.Timestamp(ny.date(), tz="UTC")


def rth_intraday(df_intra: pd.DataFrame) -> pd.DataFrame:
    """Filter to RTH 09:30-16:00 ET inclusive of 09:30 bar, exclusive of after 16:00."""
    d = df_intra.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts")
    ny = d["ts"].dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    # 09:30 = 570, last 5m bar starts 15:55 = 955; 15m last starts 15:45 = 945
    mask = (minutes >= 570) & (minutes <= 955)
    out = d.loc[mask].copy()
    out["session_date"] = [_session_date_utc(t) for t in out["ts"]]
    out["ny_hm"] = list(zip(ny.loc[mask].dt.hour, ny.loc[mask].dt.minute))
    return out.reset_index(drop=True)


def developing_asof(
    rth: pd.DataFrame,
    session_date: pd.Timestamp,
    hour: int,
    minute: int,
) -> Optional[Dict[str, float]]:
    """Session OHLC developing through the bar that starts at hour:minute ET."""
    day = rth[rth["session_date"] == session_date]
    if day.empty:
        return None
    # Bars with start <= decision clock
    target = hour * 60 + minute
    ny_minutes = [h * 60 + m for h, m in day["ny_hm"]]
    use = day.loc[[m <= target for m in ny_minutes]]
    if use.empty:
        return None
    return {
        "open": float(use["open"].iloc[0]),
        "high": float(use["high"].max()),
        "low": float(use["low"].min()),
        "close": float(use["close"].iloc[-1]),
    }


def last_rth_bar_developing(rth: pd.DataFrame, session_date: pd.Timestamp) -> Optional[Dict[str, float]]:
    """Full session developing OHLC through the last available RTH bar that day."""
    day = rth[rth["session_date"] == session_date]
    if day.empty:
        return None
    return {
        "open": float(day["open"].iloc[0]),
        "high": float(day["high"].max()),
        "low": float(day["low"].min()),
        "close": float(day["close"].iloc[-1]),
    }


def hybrid_row_for_decision(
    daily: pd.DataFrame,
    i: int,
    developing: Dict[str, float],
) -> pd.DataFrame:
    """Copy daily frame; replace row i OHLC with developing; recompute features."""
    d = daily.copy()
    d.loc[i, "open"] = developing["open"]
    d.loc[i, "high"] = developing["high"]
    d.loc[i, "low"] = developing["low"]
    d.loc[i, "close"] = developing["close"]
    # Truncate to i inclusive so rolling features don't use future days
    sub = d.iloc[: i + 1].copy()
    return prepare_frame(sub)


def entry_signal_at_index(prepared: pd.DataFrame, i: int, priority: Sequence[str] = STRATEGY_IDS) -> Optional[str]:
    """First firing strategy id at last row (index i in original = last of prepared)."""
    if prepared.empty:
        return None
    last = len(prepared) - 1
    for sid in priority:
        mask = ENTRY_MASKS[sid](prepared)
        if bool(mask.iloc[last]):
            return sid.upper()
    return None


def oracle_signals(daily_prep: pd.DataFrame) -> pd.Series:
    """Boolean series: combined priority entry would fire on settled bar."""
    out = pd.Series(False, index=daily_prep.index)
    for i in range(len(daily_prep)):
        # evaluate on prefix to avoid any accidental future leak in custom cols
        sub = daily_prep.iloc[: i + 1]
        if entry_signal_at_index(sub, i) is not None:
            out.iloc[i] = True
    return out


def _walk_trades(
    daily_prep: pd.DataFrame,
    *,
    entry_armed: Sequence[bool],
    entry_px: Sequence[Optional[float]],
    entry_sid: Sequence[Optional[str]],
    friction_pct: float,
    strategy_id: str,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    """Shared exit: settled close > prior_high; entry when armed with given price."""
    trades: List[Dict[str, Any]] = []
    in_pos = False
    ei: Optional[int] = None
    ep: Optional[float] = None
    es: Optional[str] = None

    for i in range(len(daily_prep)):
        row = daily_prep.iloc[i]
        if not in_pos:
            if entry_armed[i] and entry_px[i] is not None:
                in_pos = True
                ei = i
                ep = float(entry_px[i])
                es = entry_sid[i] or strategy_id
        else:
            assert ei is not None and ep is not None
            if pd.notna(row["prior_high"]) and float(row["close"]) > float(row["prior_high"]):
                exit_px = float(row["close"])
                ret = (exit_px / ep - 1.0) * 100.0 - friction_pct
                trades.append(
                    {
                        "symbol": symbol,
                        "strategy_id": es or strategy_id,
                        "entry_date": str(pd.Timestamp(daily_prep.iloc[ei]["ts"]).date()),
                        "exit_date": str(pd.Timestamp(row["ts"]).date()),
                        "entry_px": round(ep, 4),
                        "exit_px": round(exit_px, 4),
                        "ret_pct": round(ret, 4),
                        "hold_days": int(i - ei),
                    }
                )
                in_pos = False
                ei = None
                ep = None
                es = None

    n_sig = int(sum(1 for a in entry_armed if a))
    return _metrics_from_trades(
        trades,
        daily_prep,
        friction_pct=friction_pct,
        n_entry_signals=n_sig,
        strategy_id=strategy_id,
        symbol=symbol,
        start_i=30,
    )


def _priority_sid_from_flags(
    flags: Dict[str, bool],
    priority: Sequence[str] = STRATEGY_IDS,
) -> Optional[str]:
    for sid in priority:
        if flags.get(sid.upper()) or flags.get(sid):
            return sid.upper()
    return None


def _entry_flags_settled(d: pd.DataFrame, i: int) -> Dict[str, bool]:
    """Causal entry flags on settled daily row i (uses prepare_frame columns)."""
    row = d.iloc[i]
    return {
        "S1": bool(
            pd.notna(row["band_s1"])
            and float(row["close"]) < float(row["band_s1"])
            and float(row["ibs"]) < 0.3
        ),
        "S2": bool(
            int(row["weekday"]) == 0
            and i >= 2
            and float(row["close"]) < float(d.iloc[i - 1]["close"])
            and float(d.iloc[i - 1]["close"]) < float(d.iloc[i - 2]["close"])
        ),
        "S3": bool(
            pd.notna(row["min_low_prior_5"])
            and float(row["close"]) < float(row["min_low_prior_5"])
        ),
        "S4": bool(
            pd.notna(row["min_range_prior_6"])
            and pd.notna(row["adx5"])
            and float(row["range"]) < float(row["min_range_prior_6"])
            and float(row["adx5"]) > 40.0
        ),
        "S5": bool(
            pd.notna(row["max_high_prior_10"])
            and float(row["high"]) > float(row["max_high_prior_10"])
            and float(row["ibs"]) < 0.15
        ),
    }


def _entry_flags_developing(
    d: pd.DataFrame,
    i: int,
    dev: Dict[str, float],
    *,
    ibs_buffer: float = 0.0,
) -> Dict[str, bool]:
    """Entry flags using developing OHLC for *today* only; history from settled daily."""
    prev_close = float(d.iloc[i - 1]["close"]) if i >= 1 else float("nan")
    prev2_close = float(d.iloc[i - 2]["close"]) if i >= 2 else float("nan")
    rng = float(dev["high"] - dev["low"])
    ibs_v = ((dev["close"] - dev["low"]) / rng) if rng > 0 else 0.0

    # Rolling priors from settled bars (exclude today)
    if i >= 1:
        high_10 = float(d["high"].iloc[max(0, i - 9) : i].max())
        high_10 = max(high_10, float(dev["high"]))
        # avg range: prior 24 settled ranges + today
        prior_ranges = d["range"].iloc[max(0, i - 24) : i]
        if len(prior_ranges) >= 24:
            avg_r = float((prior_ranges.sum() + rng) / 25.0)
        else:
            avg_r = float("nan")
        min_low_5 = float(d["low"].iloc[max(0, i - 5) : i].min()) if i >= 5 else float("nan")
        min_rng_6 = float(d["range"].iloc[max(0, i - 6) : i].min()) if i >= 6 else float("nan")
        max_high_10 = float(d["high"].iloc[max(0, i - 10) : i].max()) if i >= 10 else float("nan")
    else:
        high_10 = float(dev["high"])
        avg_r = float("nan")
        min_low_5 = float("nan")
        min_rng_6 = float("nan")
        max_high_10 = float("nan")

    band = high_10 - 2.5 * avg_r if pd.notna(avg_r) else float("nan")
    s1_thr = 0.3 - ibs_buffer
    s5_thr = 0.15 - ibs_buffer

    # ADX: use *prior* settled ADX (causal at decision time; today's ADX needs close).
    adx_ok = (
        i >= 1
        and pd.notna(d.iloc[i - 1]["adx5"])
        and float(d.iloc[i - 1]["adx5"]) > 40.0
    )

    weekday = int(d.iloc[i]["weekday"])
    return {
        "S1": bool(pd.notna(band) and float(dev["close"]) < band and ibs_v < s1_thr),
        "S2": bool(
            weekday == 0
            and pd.notna(prev_close)
            and pd.notna(prev2_close)
            and float(dev["close"]) < prev_close
            and prev_close < prev2_close
        ),
        "S3": bool(pd.notna(min_low_5) and float(dev["close"]) < min_low_5),
        "S4": bool(
            pd.notna(min_rng_6) and rng < min_rng_6 and adx_ok
        ),
        "S5": bool(
            pd.notna(max_high_10)
            and float(dev["high"]) > max_high_10
            and ibs_v < s5_thr
        ),
    }


def backtest_oracle(
    daily: pd.DataFrame,
    *,
    friction_pct: float = 0.0,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    d = prepare_frame(daily)
    armed: List[bool] = []
    px: List[Optional[float]] = []
    sid: List[Optional[str]] = []
    for i in range(len(d)):
        s = _priority_sid_from_flags(_entry_flags_settled(d, i))
        if s is not None:
            armed.append(True)
            px.append(float(d.iloc[i]["close"]))
            sid.append(s)
        else:
            armed.append(False)
            px.append(None)
            sid.append(None)
    out = _walk_trades(
        d,
        entry_armed=armed,
        entry_px=px,
        entry_sid=sid,
        friction_pct=friction_pct,
        strategy_id="ORACLE",
        symbol=symbol,
    )
    out["book"] = "oracle_moc" if abs(friction_pct) < 1e-12 else ("oracle_fric_%.2f" % friction_pct)
    return out


def backtest_next_open(
    daily: pd.DataFrame,
    *,
    friction_pct: float = 0.0,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    """Signal on settled close; fill next session open (only when flat)."""
    d = prepare_frame(daily)
    armed = [False] * len(d)
    px: List[Optional[float]] = [None] * len(d)
    sid: List[Optional[str]] = [None] * len(d)

    in_pos = False
    pending: Optional[str] = None
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None

    for i in range(len(d)):
        row = d.iloc[i]
        if (not in_pos) and pending is not None:
            armed[i] = True
            px[i] = float(row["open"])
            sid[i] = pending
            in_pos = True
            entry_i = i
            entry_px = float(row["open"])
            pending = None
        elif in_pos:
            assert entry_i is not None and entry_px is not None
            if pd.notna(row["prior_high"]) and float(row["close"]) > float(row["prior_high"]):
                in_pos = False
                entry_i = None
                entry_px = None
                s = _priority_sid_from_flags(_entry_flags_settled(d, i))
                if s is not None:
                    pending = s
            continue

        if not in_pos and pending is None:
            s = _priority_sid_from_flags(_entry_flags_settled(d, i))
            if s is not None:
                pending = s

    out = _walk_trades(
        d,
        entry_armed=armed,
        entry_px=px,
        entry_sid=sid,
        friction_pct=friction_pct,
        strategy_id="NEXT_OPEN",
        symbol=symbol,
    )
    out["book"] = "next_open"
    return out


def backtest_proxy_moc(
    daily: pd.DataFrame,
    rth: pd.DataFrame,
    *,
    hour: int,
    minute: int,
    friction_pct: float = 0.0,
    ibs_buffer: float = 0.0,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    """Arm on developing OHLC at clock; fill settled close when armed."""
    d = prepare_frame(daily)
    # Index RTH by session_date for O(1) day lookup
    rth_by = {sd: g for sd, g in rth.groupby("session_date", sort=False)}

    armed: List[bool] = []
    px: List[Optional[float]] = []
    sid: List[Optional[str]] = []
    oracle_hit = 0
    proxy_hit = 0
    both = 0
    proxy_only = 0
    oracle_only = 0
    no_intra = 0

    for i in range(len(d)):
        o_sid = _priority_sid_from_flags(_entry_flags_settled(d, i))
        if o_sid is not None:
            oracle_hit += 1

        sess = _session_date_utc(d.iloc[i]["ts"])
        day = rth_by.get(sess)
        if day is None or day.empty:
            no_intra += 1
            armed.append(False)
            px.append(None)
            sid.append(None)
            if o_sid is not None:
                oracle_only += 1
            continue

        # developing through clock
        target = hour * 60 + minute
        ny_minutes = [h * 60 + m for h, m in day["ny_hm"]]
        use = day.loc[[m <= target for m in ny_minutes]]
        if use.empty:
            no_intra += 1
            armed.append(False)
            px.append(None)
            sid.append(None)
            if o_sid is not None:
                oracle_only += 1
            continue
        dev = {
            "open": float(use["open"].iloc[0]),
            "high": float(use["high"].max()),
            "low": float(use["low"].min()),
            "close": float(use["close"].iloc[-1]),
        }
        p_sid = _priority_sid_from_flags(
            _entry_flags_developing(d, i, dev, ibs_buffer=ibs_buffer)
        )
        if p_sid is not None:
            proxy_hit += 1
            armed.append(True)
            px.append(float(d.iloc[i]["close"]))
            sid.append(p_sid)
            if o_sid is not None:
                both += 1
            else:
                proxy_only += 1
        else:
            armed.append(False)
            px.append(None)
            sid.append(None)
            if o_sid is not None:
                oracle_only += 1

    label = "proxy_%02d%02d" % (hour, minute)
    if ibs_buffer > 0:
        label = label + "_buf"
    out = _walk_trades(
        d,
        entry_armed=armed,
        entry_px=px,
        entry_sid=sid,
        friction_pct=friction_pct,
        strategy_id=label.upper(),
        symbol=symbol,
    )
    out["book"] = label
    out["agreement"] = {
        "oracle_signal_days": oracle_hit,
        "proxy_arm_days": proxy_hit,
        "both": both,
        "proxy_only_false_pos": proxy_only,
        "oracle_only_miss": oracle_only,
        "no_intraday": no_intra,
        "precision": (both / proxy_hit) if proxy_hit else None,
        "recall": (both / oracle_hit) if oracle_hit else None,
    }
    out["ibs_buffer"] = ibs_buffer
    out["decision_clock"] = "%02d:%02d ET" % (hour, minute)
    return out


def run_realism_suite(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    clocks: Sequence[Tuple[int, int]] = PROXY_CLOCKS_5M,
    frictions: Sequence[float] = (0.0, 0.03, 0.10, 0.25),
    ibs_buffer: float = 0.05,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    """Run oracle, next-open, proxy clocks, and oracle friction stress."""
    rth = rth_intraday(intraday)
    books: Dict[str, Any] = {}

    books["oracle_moc"] = backtest_oracle(daily, friction_pct=0.0, symbol=symbol)
    books["next_open"] = backtest_next_open(daily, friction_pct=0.0, symbol=symbol)

    for h, m in clocks:
        key = "proxy_%02d%02d" % (h, m)
        books[key] = backtest_proxy_moc(
            daily, rth, hour=h, minute=m, friction_pct=0.0, ibs_buffer=0.0, symbol=symbol
        )
        if ibs_buffer > 0:
            books[key + "_buf"] = backtest_proxy_moc(
                daily,
                rth,
                hour=h,
                minute=m,
                friction_pct=0.0,
                ibs_buffer=ibs_buffer,
                symbol=symbol,
            )

    for f in frictions:
        if abs(f) < 1e-12:
            continue
        books["oracle_fric_%.2f" % f] = backtest_oracle(
            daily, friction_pct=f, symbol=symbol
        )

    return {"books": books, "rth_bars": len(rth), "daily_bars": len(daily)}
