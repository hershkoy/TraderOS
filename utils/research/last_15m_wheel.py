"""One-cycle wheel (CSP -> assign -> CC) on last-15m H2 breakout fills.

Replaces buying the stock with a single wheel cycle:

  1. Sell ~target_dte / ~target_delta cash-secured put on the signal day
  2. Hold the put to expiry (wheel style; no early TP/SL)
  3. If OTM at expiry: keep the credit; cycle done
  4. If ITM: assigned at put strike; sell a covered call (~same DTE/delta)
  5. At call expiry: called away at call strike, else liquidate shares at close

Marks come from a ``QuoteSource``:

  * ``ThetaEquitySource`` -- greeks/eod chain + eod bid/ask (preferred)
  * ``PolygonDaySource`` -- Friday ladder + day-bar mid; half-spread for ask/bid
  * ``HybridWheelSource`` -- Theta first, Polygon fallback

Cash-secured % = total PnL per share / put_strike * 100 (same notional basis
as the prior last-15m short-put books). Occupancy is not re-walked.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

from utils.api.polygon.polygon_client import PolygonClient, get_polygon_client
from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.research.forward_factor import list_expirations
from utils.research.last_15m_short_put_bs import _as_date, _num, _panel_dates
from utils.research.short_put_live_realism import (
    half_spread,
    nearby_fridays,
    strike_candidates,
)
from utils.research.short_put_polygon_spotcheck import (
    bars_from_aggs_response,
    polygon_option_ticker,
)

# Theta Standard stock option history is sparse / 403 before ~2024.
THETA_EQUITY_START = date(2024, 1, 2)
from utils.research.spy_wheel import entry_credit, pick_contract

LOG = logging.getLogger(__name__)

MULT = 100.0
DEFAULT_DTE = 30
DEFAULT_DTE_TOL = 10
DEFAULT_MIN_DTE = 7
DEFAULT_MIN_CREDIT = 0.05

REASON_OK = "ok"
REASON_NO_ENTRY = "no_entry"
REASON_NO_DAILY = "no_daily"
REASON_NO_PUT_CHAIN = "no_put_chain"
REASON_NO_PUT_QUOTE = "no_put_quote"
REASON_NO_SPOT_EXPIRY = "no_spot_expiry"
REASON_NO_CALL_CHAIN = "no_call_chain"
REASON_NO_CALL_QUOTE = "no_call_quote"
REASON_NO_SPOT_CC = "no_spot_cc"

EXIT_PUT_OTM = "put_expire_otm"
EXIT_PUT_ASSIGN = "put_assign"
EXIT_CALL_ASSIGN = "call_assign"
EXIT_SHARE_LIQ = "share_liquidate"


class QuoteSource(Protocol):
    def chain_on_day(
        self,
        symbol: str,
        day: date,
        right: str,
        dte_lo: int,
        dte_hi: int,
        *,
        spot_hint: Optional[float] = None,
    ) -> pd.DataFrame:
        """Rows: expiration, strike, bid, ask, delta (delta may be NaN)."""

    def quote_on_day(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        day: date,
    ) -> Optional[Dict[str, Any]]:
        """bid/ask/mid on ``day``, or None."""


@dataclass
class WheelLeg:
    right: str
    expiration: date
    strike: float
    credit: float
    entry_date: date
    entry_delta: Optional[float]
    entry_dte: int
    source: str
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class WheelResult:
    filled: bool
    reason: str
    exit_reason: Optional[str] = None
    put: Optional[WheelLeg] = None
    call: Optional[WheelLeg] = None
    put_expiry_spot: Optional[float] = None
    call_expiry_spot: Optional[float] = None
    assigned: bool = False
    called_away: bool = False
    share_exit_px: Optional[float] = None
    pnl_per_share: Optional[float] = None
    gain_cash_secured_pct: Optional[float] = None
    stock_gain_pct: Optional[float] = None
    mark_mode: str = "mid"
    hold_days: Optional[int] = None
    exit_date: Optional[date] = None
    notes: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)


def _spot_on(
    panel_1d: pd.DataFrame, day: date, *, prefer: str = "close"
) -> Optional[float]:
    """Underlying close (or open) on/after ``day`` within a few sessions."""
    if panel_1d is None or panel_1d.empty:
        return None
    work = panel_1d.copy()
    work.index = _panel_dates(work)
    # Exact day first
    exact = work.loc[work.index.date == day]
    if not exact.empty:
        col = prefer if prefer in exact.columns else "close"
        v = float(exact.iloc[0][col])
        return v if math.isfinite(v) and v > 0 else None
    # Next available session within 5 calendar days
    after = work.loc[(work.index.date > day) & (work.index.date <= day + timedelta(days=5))]
    if after.empty:
        return None
    col = prefer if prefer in after.columns else "close"
    v = float(after.iloc[0][col])
    return v if math.isfinite(v) and v > 0 else None


def _dte_window(target_dte: int, dte_tol: int, min_dte: int) -> Tuple[int, int]:
    lo = max(int(min_dte), int(target_dte) - int(dte_tol))
    hi = int(target_dte) + int(dte_tol)
    return lo, hi


def pick_atm_contract(
    chain: pd.DataFrame,
    *,
    day: date,
    spot: float,
    target_dte: int,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte: int = DEFAULT_MIN_DTE,
    min_credit: float = DEFAULT_MIN_CREDIT,
    right: str = "P",
) -> Optional[Dict[str, Any]]:
    """Nearest expiry to target_dte, then strike nearest to spot (no delta)."""
    if chain is None or chain.empty or spot <= 0:
        return None
    df = chain.copy()
    df["dte"] = df["expiration"].map(lambda e: (e - day).days)
    df = df[(df["dte"] >= int(min_dte)) & (df["dte"] <= int(target_dte) + int(dte_tol))]
    if df.empty:
        return None
    best_dte_err = int(df["dte"].sub(int(target_dte)).abs().min())
    df = df[df["dte"].sub(int(target_dte)).abs() == best_dte_err]
    exp = df["expiration"].iloc[0]
    sub = df[df["expiration"] == exp].copy()
    sub = sub[sub["bid"].fillna(0) >= float(min_credit)]
    if sub.empty:
        return None
    right_u = str(right).upper()[0]
    if right_u == "C":
        # Prefer OTM calls (strike >= spot); else nearest
        otm = sub[sub["strike"] >= float(spot)]
        pool = otm if not otm.empty else sub
    else:
        # Prefer OTM puts (strike <= spot); else nearest
        otm = sub[sub["strike"] <= float(spot)]
        pool = otm if not otm.empty else sub
    pool = pool.copy()
    pool["strike_err"] = (pool["strike"] - float(spot)).abs()
    row = pool.sort_values(["strike_err", "strike"]).iloc[0]
    bid = float(row["bid"]) if pd.notna(row.get("bid")) else float("nan")
    ask = float(row["ask"]) if pd.notna(row.get("ask")) else float("nan")
    mid = 0.5 * (bid + ask) if math.isfinite(bid) and math.isfinite(ask) else float("nan")
    delta = float(row["delta"]) if "delta" in row and pd.notna(row.get("delta")) else float("nan")
    return {
        "expiration": exp,
        "strike": float(row["strike"]),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "delta": delta,
        "dte": int(row["dte"]),
    }


def select_contract(
    chain: pd.DataFrame,
    *,
    day: date,
    spot: float,
    target_dte: int,
    target_delta: float,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte: int = DEFAULT_MIN_DTE,
    min_credit: float = DEFAULT_MIN_CREDIT,
    right: str = "P",
    strike_mode: str = "delta",
) -> Optional[Dict[str, Any]]:
    """Delta-target when chain has delta; else ATM/OTM nearest-strike."""
    mode = str(strike_mode or "delta").lower().strip()
    has_delta = (
        chain is not None
        and not chain.empty
        and "delta" in chain.columns
        and chain["delta"].notna().any()
    )
    if mode == "delta" and has_delta:
        got = pick_contract(
            chain,
            day=day,
            target_dte=int(target_dte),
            target_delta=float(target_delta),
            dte_tol=int(dte_tol),
            min_dte=int(min_dte),
            min_credit=float(min_credit),
        )
        if got is not None:
            return got
    return pick_atm_contract(
        chain,
        day=day,
        spot=float(spot),
        target_dte=int(target_dte),
        dte_tol=int(dte_tol),
        min_dte=int(min_dte),
        min_credit=float(min_credit),
        right=right,
    )


def cycle_pnl_per_share(
    *,
    put_credit: float,
    put_strike: float,
    assigned: bool,
    call_credit: float = 0.0,
    call_strike: Optional[float] = None,
    called_away: bool = False,
    share_exit_px: Optional[float] = None,
) -> float:
    """Net $ per share for one CSP->(assign->CC) cycle."""
    pnl = float(put_credit)
    if not assigned:
        return pnl
    # Assigned: long shares at put_strike
    pnl -= float(put_strike)
    pnl += float(call_credit)
    if called_away:
        if call_strike is None:
            raise ValueError("called_away requires call_strike")
        pnl += float(call_strike)
    else:
        if share_exit_px is None:
            raise ValueError("share liquidation requires share_exit_px")
        pnl += float(share_exit_px)
    return pnl


def cash_secured_pct(pnl_per_share: float, put_strike: float) -> Optional[float]:
    k = float(put_strike)
    if k <= 0:
        return None
    return float(pnl_per_share) / k * 100.0


def simulate_one_cycle_wheel(
    row: pd.Series,
    panel_1d: Optional[pd.DataFrame],
    source: QuoteSource,
    *,
    mark_mode: str = "mid",
    target_dte: int = DEFAULT_DTE,
    target_delta: float = 0.30,
    dte_tol: int = DEFAULT_DTE_TOL,
    min_dte: int = DEFAULT_MIN_DTE,
    min_credit: float = DEFAULT_MIN_CREDIT,
    strike_mode: str = "delta",
    entry_slack_days: int = 0,
) -> WheelResult:
    """Walk one wheel cycle for a last-15m fill row.

    ``entry_slack_days=0`` requires a put quote on the signal day (honest).
    """
    stock_gain = _num(row, "gain_pct")
    entry_px = _num(row, "buy_price")
    entry_day = _as_date(row.get("buy_date"))
    sym = str(row.get("stock") or "").upper()
    if not sym or entry_px is None or entry_px <= 0 or entry_day is None:
        return WheelResult(filled=False, reason=REASON_NO_ENTRY, stock_gain_pct=stock_gain)
    if panel_1d is None or panel_1d.empty:
        return WheelResult(filled=False, reason=REASON_NO_DAILY, stock_gain_pct=stock_gain)

    spot0 = _spot_on(panel_1d, entry_day) or float(entry_px)
    dte_lo, dte_hi = _dte_window(target_dte, dte_tol, min_dte)

    # --- open CSP ---
    put_open_day = entry_day
    put_quote = None
    put_pick = None
    put_src = "unknown"
    for slack in range(int(entry_slack_days) + 1):
        day = entry_day + timedelta(days=slack)
        # Skip weekends for quote attempts
        if day.weekday() >= 5:
            continue
        chain = source.chain_on_day(
            sym, day, "P", dte_lo, dte_hi, spot_hint=float(spot0)
        )
        pick = select_contract(
            chain,
            day=day,
            spot=float(spot0),
            target_dte=int(target_dte),
            target_delta=float(target_delta),
            dte_tol=int(dte_tol),
            min_dte=int(min_dte),
            min_credit=float(min_credit),
            right="P",
            strike_mode=strike_mode,
        )
        if pick is None:
            continue
        q = source.quote_on_day(
            sym, pick["expiration"], float(pick["strike"]), "P", day
        )
        if q is None:
            # Chain row already has bid/ask; use it
            if math.isfinite(float(pick.get("bid") or float("nan"))) and math.isfinite(
                float(pick.get("ask") or float("nan"))
            ):
                q = {
                    "bid": float(pick["bid"]),
                    "ask": float(pick["ask"]),
                    "mid": float(pick.get("mid") or 0.5 * (float(pick["bid"]) + float(pick["ask"]))),
                    "source": "chain",
                }
            else:
                continue
        put_open_day = day
        put_pick = pick
        put_quote = q
        put_src = str(q.get("source") or pick.get("source") or "source")
        break

    if put_pick is None or put_quote is None:
        return WheelResult(
            filled=False,
            reason=REASON_NO_PUT_CHAIN if put_pick is None else REASON_NO_PUT_QUOTE,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
        )

    put_cred = entry_credit(put_quote, mark_mode)
    if put_cred < float(min_credit):
        return WheelResult(
            filled=False,
            reason=REASON_NO_PUT_QUOTE,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
        )

    put_leg = WheelLeg(
        right="P",
        expiration=put_pick["expiration"],
        strike=float(put_pick["strike"]),
        credit=float(put_cred),
        entry_date=put_open_day,
        entry_delta=(
            float(put_pick["delta"])
            if put_pick.get("delta") is not None and math.isfinite(float(put_pick["delta"]))
            else None
        ),
        entry_dte=int(put_pick["dte"]),
        source=put_src,
        bid=float(put_quote.get("bid") or 0.0),
        ask=float(put_quote.get("ask") or 0.0),
    )
    events: List[Dict[str, Any]] = [
        {
            "event": "open_put",
            "day": put_open_day.isoformat(),
            "strike": put_leg.strike,
            "exp": put_leg.expiration.isoformat(),
            "credit": put_leg.credit,
            "source": put_src,
        }
    ]

    # --- put expiry ---
    put_exp = put_leg.expiration
    spot_put = _spot_on(panel_1d, put_exp)
    if spot_put is None:
        return WheelResult(
            filled=False,
            reason=REASON_NO_SPOT_EXPIRY,
            put=put_leg,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
            events=events,
        )

    if float(spot_put) >= float(put_leg.strike):
        # OTM expire
        pnl = cycle_pnl_per_share(
            put_credit=put_leg.credit,
            put_strike=put_leg.strike,
            assigned=False,
        )
        events.append(
            {
                "event": "put_expire_otm",
                "day": put_exp.isoformat(),
                "spot": spot_put,
            }
        )
        return WheelResult(
            filled=True,
            reason=REASON_OK,
            exit_reason=EXIT_PUT_OTM,
            put=put_leg,
            put_expiry_spot=float(spot_put),
            assigned=False,
            pnl_per_share=pnl,
            gain_cash_secured_pct=cash_secured_pct(pnl, put_leg.strike),
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
            hold_days=(put_exp - put_open_day).days,
            exit_date=put_exp,
            events=events,
        )

    # Assigned
    events.append(
        {
            "event": "put_assign",
            "day": put_exp.isoformat(),
            "spot": spot_put,
            "strike": put_leg.strike,
        }
    )

    # --- open CC on assignment day (or next session) ---
    call_open_day = put_exp
    call_pick = None
    call_quote = None
    call_src = "unknown"
    for slack in range(0, 4):
        day = put_exp + timedelta(days=slack)
        if day.weekday() >= 5:
            continue
        spot_cc = _spot_on(panel_1d, day) or float(spot_put)
        chain = source.chain_on_day(
            sym, day, "C", dte_lo, dte_hi, spot_hint=float(spot_cc)
        )
        pick = select_contract(
            chain,
            day=day,
            spot=float(spot_cc),
            target_dte=int(target_dte),
            target_delta=float(target_delta),
            dte_tol=int(dte_tol),
            min_dte=int(min_dte),
            min_credit=float(min_credit),
            right="C",
            strike_mode=strike_mode,
        )
        if pick is None:
            continue
        q = source.quote_on_day(
            sym, pick["expiration"], float(pick["strike"]), "C", day
        )
        if q is None:
            if math.isfinite(float(pick.get("bid") or float("nan"))) and math.isfinite(
                float(pick.get("ask") or float("nan"))
            ):
                q = {
                    "bid": float(pick["bid"]),
                    "ask": float(pick["ask"]),
                    "mid": float(pick.get("mid") or 0.5 * (float(pick["bid"]) + float(pick["ask"]))),
                    "source": "chain",
                }
            else:
                continue
        call_open_day = day
        call_pick = pick
        call_quote = q
        call_src = str(q.get("source") or "source")
        break

    if call_pick is None or call_quote is None:
        return WheelResult(
            filled=False,
            reason=REASON_NO_CALL_CHAIN if call_pick is None else REASON_NO_CALL_QUOTE,
            put=put_leg,
            put_expiry_spot=float(spot_put),
            assigned=True,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
            events=events,
        )

    call_cred = entry_credit(call_quote, mark_mode)
    call_leg = WheelLeg(
        right="C",
        expiration=call_pick["expiration"],
        strike=float(call_pick["strike"]),
        credit=float(call_cred),
        entry_date=call_open_day,
        entry_delta=(
            float(call_pick["delta"])
            if call_pick.get("delta") is not None
            and math.isfinite(float(call_pick["delta"]))
            else None
        ),
        entry_dte=int(call_pick["dte"]),
        source=call_src,
        bid=float(call_quote.get("bid") or 0.0),
        ask=float(call_quote.get("ask") or 0.0),
    )
    events.append(
        {
            "event": "open_call",
            "day": call_open_day.isoformat(),
            "strike": call_leg.strike,
            "exp": call_leg.expiration.isoformat(),
            "credit": call_leg.credit,
            "source": call_src,
        }
    )

    call_exp = call_leg.expiration
    spot_call = _spot_on(panel_1d, call_exp)
    if spot_call is None:
        return WheelResult(
            filled=False,
            reason=REASON_NO_SPOT_CC,
            put=put_leg,
            call=call_leg,
            put_expiry_spot=float(spot_put),
            assigned=True,
            stock_gain_pct=stock_gain,
            mark_mode=mark_mode,
            events=events,
        )

    if float(spot_call) >= float(call_leg.strike):
        called_away = True
        share_exit = float(call_leg.strike)
        exit_reason = EXIT_CALL_ASSIGN
        events.append(
            {
                "event": "call_assign",
                "day": call_exp.isoformat(),
                "spot": spot_call,
                "strike": call_leg.strike,
            }
        )
    else:
        called_away = False
        share_exit = float(spot_call)
        exit_reason = EXIT_SHARE_LIQ
        events.append(
            {
                "event": "share_liquidate",
                "day": call_exp.isoformat(),
                "spot": spot_call,
            }
        )

    pnl = cycle_pnl_per_share(
        put_credit=put_leg.credit,
        put_strike=put_leg.strike,
        assigned=True,
        call_credit=call_leg.credit,
        call_strike=call_leg.strike,
        called_away=called_away,
        share_exit_px=None if called_away else share_exit,
    )
    return WheelResult(
        filled=True,
        reason=REASON_OK,
        exit_reason=exit_reason,
        put=put_leg,
        call=call_leg,
        put_expiry_spot=float(spot_put),
        call_expiry_spot=float(spot_call),
        assigned=True,
        called_away=called_away,
        share_exit_px=share_exit,
        pnl_per_share=pnl,
        gain_cash_secured_pct=cash_secured_pct(pnl, put_leg.strike),
        stock_gain_pct=stock_gain,
        mark_mode=mark_mode,
        hold_days=(call_exp - put_open_day).days,
        exit_date=call_exp,
        events=events,
    )


def result_to_record(row: pd.Series, got: WheelResult) -> Dict[str, Any]:
    rec = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    rec["wheel_ok"] = bool(got.filled)
    rec["wheel_reason"] = got.reason
    rec["wheel_exit_reason"] = got.exit_reason
    rec["wheel_mark_mode"] = got.mark_mode
    rec["wheel_assigned"] = bool(got.assigned)
    rec["wheel_called_away"] = bool(got.called_away)
    rec["wheel_pnl_ps"] = got.pnl_per_share
    rec["gain_pct_stock"] = got.stock_gain_pct
    rec["gain_pct"] = got.gain_cash_secured_pct if got.filled else None
    rec["wheel_hold_days"] = got.hold_days
    rec["wheel_exit_date"] = got.exit_date.isoformat() if got.exit_date else None
    if got.put is not None:
        rec["put_strike"] = got.put.strike
        rec["put_credit"] = got.put.credit
        rec["put_expiry"] = got.put.expiration.isoformat()
        rec["put_entry_date"] = got.put.entry_date.isoformat()
        rec["put_delta"] = got.put.entry_delta
        rec["put_dte"] = got.put.entry_dte
        rec["put_source"] = got.put.source
    if got.call is not None:
        rec["call_strike"] = got.call.strike
        rec["call_credit"] = got.call.credit
        rec["call_expiry"] = got.call.expiration.isoformat()
        rec["call_entry_date"] = got.call.entry_date.isoformat()
        rec["call_delta"] = got.call.entry_delta
        rec["call_dte"] = got.call.entry_dte
        rec["call_source"] = got.call.source
    rec["put_expiry_spot"] = got.put_expiry_spot
    rec["call_expiry_spot"] = got.call_expiry_spot
    rec["share_exit_px"] = got.share_exit_px
    return rec


# ---------------------------------------------------------------------------
# Theta equity source (any underlier)
# ---------------------------------------------------------------------------


class ThetaEquitySource:
    """Theta Terminal greeks/eod chain + eod bid/ask for equity options."""

    def __init__(
        self,
        client: Optional[ThetaDataClient] = None,
        *,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.client = client or get_thetadata_client()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._exps: Dict[str, List[date]] = {}
        self._chain_cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}
        self._quote_cache: Dict[Tuple[Any, ...], Optional[Dict[str, Any]]] = {}

    def expirations(self, symbol: str) -> List[date]:
        sym = str(symbol).upper()
        if sym in self._exps:
            return self._exps[sym]
        try:
            exps = list_expirations(self.client, sym)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("theta list_expirations %s failed: %s", sym, exc)
            exps = []
        self._exps[sym] = exps
        return exps

    def _fetch_chain_exp(
        self, symbol: str, day: date, right: str, exp: date
    ) -> pd.DataFrame:
        sym = str(symbol).upper()
        right_s = "call" if str(right).upper().startswith("C") else "put"
        key = (sym, day, right_s[0].upper(), exp)
        if key in self._chain_cache:
            return self._chain_cache[key].copy()
        path = None
        if self.cache_dir is not None:
            path = (
                self.cache_dir
                / f"chain_{sym}_{exp.isoformat()}_{day.isoformat()}_{right_s[0]}.csv"
            )
            if path.exists():
                try:
                    cached = pd.read_csv(path)
                    if not cached.empty and "expiration" in cached.columns:
                        cached["expiration"] = pd.to_datetime(
                            cached["expiration"]
                        ).dt.date
                    self._chain_cache[key] = cached
                    return cached.copy()
                except Exception:  # noqa: BLE001
                    path.unlink(missing_ok=True)
        r = self.client.session.get(
            f"{self.client.base_url}/v3/option/history/greeks/eod",
            params={
                "symbol": sym,
                "expiration": exp.isoformat(),
                "right": right_s,
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
            timeout=90,
        )
        empty = pd.DataFrame(columns=["expiration", "strike", "bid", "ask", "delta"])
        if r.status_code in (403, 472) or not r.content:
            if path is not None:
                empty.to_csv(path, index=False)
            self._chain_cache[key] = empty
            return empty.copy()
        if r.status_code != 200:
            LOG.debug(
                "theta chain %s %s %s %s -> %s",
                sym,
                day,
                exp,
                right_s,
                r.status_code,
            )
            self._chain_cache[key] = empty
            return empty.copy()
        raw = pd.read_csv(StringIO(r.text))
        if raw.empty:
            if path is not None:
                empty.to_csv(path, index=False)
            self._chain_cache[key] = empty
            return empty.copy()
        strike_col = "strike" if "strike" in raw.columns else raw.columns[0]
        rows = []
        for _, row in raw.iterrows():
            try:
                strike = float(row[strike_col])
                bid = float(row["bid"]) if "bid" in raw.columns else float("nan")
                ask = float(row["ask"]) if "ask" in raw.columns else float("nan")
                delta = float(row["delta"]) if "delta" in raw.columns else float("nan")
            except (TypeError, ValueError):
                continue
            if not (bid > 0 and ask > 0):
                continue
            rows.append(
                {
                    "expiration": exp,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "delta": delta,
                }
            )
        df = pd.DataFrame(rows) if rows else empty
        if path is not None:
            df.to_csv(path, index=False)
        self._chain_cache[key] = df
        return df.copy()

    def chain_on_day(
        self,
        symbol: str,
        day: date,
        right: str,
        dte_lo: int,
        dte_hi: int,
        *,
        spot_hint: Optional[float] = None,
    ) -> pd.DataFrame:
        del spot_hint  # Theta returns full chain; spot used at pick time
        cands = [
            e
            for e in self.expirations(symbol)
            if int(dte_lo) <= (e - day).days <= int(dte_hi)
        ]
        if not cands:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        mid = 0.5 * (float(dte_lo) + float(dte_hi))
        cands = sorted(cands, key=lambda e: abs((e - day).days - mid))[:4]
        frames = [self._fetch_chain_exp(symbol, day, right, e) for e in cands]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        return pd.concat(frames, ignore_index=True)

    def quote_on_day(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        day: date,
    ) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()
        right_s = "call" if str(right).upper().startswith("C") else "put"
        key = (sym, expiration, float(strike), right_s[0].upper(), day)
        if key in self._quote_cache:
            return dict(self._quote_cache[key]) if self._quote_cache[key] else None
        r = self.client.session.get(
            f"{self.client.base_url}/v3/option/history/eod",
            params={
                "symbol": sym,
                "expiration": expiration.isoformat(),
                "strike": f"{float(strike):.3f}",
                "right": right_s,
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
            timeout=60,
        )
        if r.status_code != 200 or not r.content or b"No data" in r.content[:80]:
            self._quote_cache[key] = None
            return None
        raw = pd.read_csv(StringIO(r.text))
        if raw.empty:
            self._quote_cache[key] = None
            return None
        bid = float(pd.to_numeric(raw.get("bid"), errors="coerce").iloc[-1])
        ask = float(pd.to_numeric(raw.get("ask"), errors="coerce").iloc[-1])
        if not (bid > 0 and ask > 0):
            self._quote_cache[key] = None
            return None
        out = {
            "bid": bid,
            "ask": ask,
            "mid": 0.5 * (bid + ask),
            "source": "theta",
        }
        self._quote_cache[key] = out
        return dict(out)


# ---------------------------------------------------------------------------
# Polygon day-bar source
# ---------------------------------------------------------------------------


class PolygonDaySource:
    """Polygon contract list + day-agg mid (half-spread for ask/bid marks)."""

    def __init__(
        self,
        client: Optional[PolygonClient] = None,
        *,
        spread_floor: float = 0.05,
        spread_frac: float = 0.08,
        use_bs_delta_hint: bool = True,
    ) -> None:
        self.client = client or get_polygon_client()
        self.spread_floor = float(spread_floor)
        self.spread_frac = float(spread_frac)
        self.use_bs_delta_hint = bool(use_bs_delta_hint)
        self._exp_cache: Dict[str, List[str]] = {}
        self._ctr_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._bar_cache: Dict[str, Optional[Dict[str, float]]] = {}

    def _friday_ladder(self, day: date, dte_lo: int, dte_hi: int) -> List[date]:
        target = day + timedelta(days=int(0.5 * (dte_lo + dte_hi)))
        out: List[date] = []
        for fri in nearby_fridays(target, window_days=max(21, int(dte_hi))):
            dte = (fri - day).days
            if int(dte_lo) <= dte <= int(dte_hi):
                out.append(fri)
        return out

    def _contracts(self, symbol: str, expiry: str, as_of: str) -> List[Dict[str, Any]]:
        ck = "%s|%s|%s" % (symbol, expiry, as_of)
        if ck in self._ctr_cache:
            return self._ctr_cache[ck]
        results: List[Dict[str, Any]] = []
        # Keep attempts tight -- free-tier RPM is ~5 and hybrid can fall here often.
        for kwargs in (
            {"as_of": as_of, "contract_type": None},
            {"as_of": None, "contract_type": None},
        ):
            try:
                resp = self.client.get_contracts_for_expiry(
                    symbol,
                    expiry,
                    as_of=kwargs["as_of"],
                    contract_type=kwargs["contract_type"],
                    limit=1000,
                )
                results = list(resp.get("results") or [])
            except Exception as exc:  # noqa: BLE001
                LOG.debug("polygon contracts %s %s: %s", symbol, expiry, exc)
                results = []
            if results:
                break
        self._ctr_cache[ck] = results
        return results

    def _day_mid(
        self, ticker: str, day: date
    ) -> Optional[Dict[str, float]]:
        ck = "%s|%s" % (ticker, day.isoformat())
        if ck in self._bar_cache:
            return self._bar_cache[ck]
        try:
            resp = self.client.get_option_eod(ticker, day.isoformat())
            bars = bars_from_aggs_response(resp)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("polygon eod %s %s: %s", ticker, day, exc)
            bars = pd.DataFrame()
        if bars is None or bars.empty:
            self._bar_cache[ck] = None
            return None
        row = bars.iloc[0]
        close = float(row.get("close") or 0)
        if close <= 0:
            self._bar_cache[ck] = None
            return None
        # Day aggregates have no NBBO; treat close as mid.
        mid = close
        hs = half_spread(mid, floor=self.spread_floor, frac=self.spread_frac)
        out = {"bid": max(0.01, mid - hs), "ask": mid + hs, "mid": mid}
        self._bar_cache[ck] = out
        return dict(out)

    def chain_on_day(
        self,
        symbol: str,
        day: date,
        right: str,
        dte_lo: int,
        dte_hi: int,
        *,
        spot_hint: Optional[float] = None,
        max_strikes: int = 5,
    ) -> pd.DataFrame:
        """Build a thin near-spot chain (day-bar mids).

        Prefer constructed OCC tickers from ``spot_hint`` (few aggs calls).
        Fall back to contract-list only when spot is unknown.
        """
        sym = str(symbol).upper()
        right_u = "C" if str(right).upper().startswith("C") else "P"
        rows: List[Dict[str, Any]] = []
        fridays = self._friday_ladder(day, dte_lo, dte_hi)[:2]
        if spot_hint is not None and float(spot_hint) > 0:
            strikes = strike_candidates(float(spot_hint), n_each_side=2)[
                : max(1, int(max_strikes))
            ]
            for exp in fridays:
                for strike in strikes:
                    tick = polygon_option_ticker(
                        sym, exp.isoformat(), float(strike), right_u
                    )
                    q = self._day_mid(tick, day)
                    if q is None:
                        continue
                    rows.append(
                        {
                            "expiration": exp,
                            "strike": float(strike),
                            "bid": q["bid"],
                            "ask": q["ask"],
                            "delta": float("nan"),
                            "ticker": tick,
                        }
                    )
            if rows:
                return pd.DataFrame(rows)

        # Slow path: list contracts then nearest strikes
        want_type = "call" if right_u == "C" else "put"
        for exp in fridays:
            ctrs = self._contracts(sym, exp.isoformat(), day.isoformat())
            parsed: List[Tuple[float, str]] = []
            for c in ctrs:
                ctype = str(c.get("contract_type") or "").lower()
                tick = str(c.get("ticker") or "")
                if ctype and ctype not in (want_type, want_type[0], right_u.lower()):
                    continue
                if not ctype:
                    if right_u == "P" and "P" not in tick[-15:]:
                        continue
                    if right_u == "C" and "C" not in tick[-15:]:
                        continue
                try:
                    strike = float(c.get("strike_price") or c.get("strike"))
                except (TypeError, ValueError):
                    continue
                if not tick:
                    tick = polygon_option_ticker(
                        sym, exp.isoformat(), strike, right_u
                    )
                parsed.append((strike, tick))
            if not parsed:
                continue
            if spot_hint is not None and float(spot_hint) > 0:
                parsed.sort(key=lambda x: abs(x[0] - float(spot_hint)))
            parsed = parsed[: max(1, int(max_strikes))]
            for strike, tick in parsed:
                q = self._day_mid(tick, day)
                if q is None:
                    continue
                rows.append(
                    {
                        "expiration": exp,
                        "strike": strike,
                        "bid": q["bid"],
                        "ask": q["ask"],
                        "delta": float("nan"),
                        "ticker": tick,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        return pd.DataFrame(rows)

    def quote_on_day(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        day: date,
    ) -> Optional[Dict[str, Any]]:
        right_u = "C" if str(right).upper().startswith("C") else "P"
        tick = polygon_option_ticker(
            str(symbol).upper(), expiration.isoformat(), float(strike), right_u
        )
        q = self._day_mid(tick, day)
        if q is None:
            return None
        out = dict(q)
        out["source"] = "polygon"
        return out


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


class HybridWheelSource:
    """Theta first; Polygon fallback when chain/quote is empty."""

    def __init__(
        self,
        theta: Optional[ThetaEquitySource] = None,
        polygon: Optional[PolygonDaySource] = None,
    ) -> None:
        self.theta = theta or ThetaEquitySource()
        self.polygon = polygon or PolygonDaySource()

    def chain_on_day(
        self,
        symbol: str,
        day: date,
        right: str,
        dte_lo: int,
        dte_hi: int,
        *,
        spot_hint: Optional[float] = None,
    ) -> pd.DataFrame:
        # Try Theta first on all dates (Standard may cover pre-2024; FREE often 403).
        t = self.theta.chain_on_day(
            symbol, day, right, dte_lo, dte_hi, spot_hint=spot_hint
        )
        if t is not None and not t.empty:
            t = t.copy()
            t["source"] = "theta"
            return t
        # Polygon Starter options aggs are ~2y lookback — older days often 403.
        p = self.polygon.chain_on_day(
            symbol, day, right, dte_lo, dte_hi, spot_hint=spot_hint
        )
        if p is not None and not p.empty:
            p = p.copy()
            p["source"] = "polygon"
        return p

    def quote_on_day(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        day: date,
    ) -> Optional[Dict[str, Any]]:
        q = self.theta.quote_on_day(symbol, expiration, strike, right, day)
        if q is not None:
            return q
        return self.polygon.quote_on_day(symbol, expiration, strike, right, day)


def apply_wheel_book(
    trades: pd.DataFrame,
    panels_1d: Dict[str, pd.DataFrame],
    source: QuoteSource,
    *,
    mark_mode: str = "mid",
    target_dte: int = DEFAULT_DTE,
    target_delta: float = 0.30,
    dte_tol: int = DEFAULT_DTE_TOL,
    strike_mode: str = "delta",
    entry_slack_days: int = 0,
) -> pd.DataFrame:
    rows = []
    for idx, row in trades.iterrows():
        sym = str(row.get("stock", "")).upper()
        got = simulate_one_cycle_wheel(
            row,
            panels_1d.get(sym),
            source,
            mark_mode=mark_mode,
            target_dte=target_dte,
            target_delta=target_delta,
            dte_tol=dte_tol,
            strike_mode=strike_mode,
            entry_slack_days=entry_slack_days,
        )
        rec = result_to_record(row, got)
        rec["src_idx"] = idx
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Occupancy + equal-slot equity (cash-secured collateral model)
# ---------------------------------------------------------------------------


def filled_trades_to_occ_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Normalize claimcheck CSV rows for FCFS occupancy."""
    if df is None or df.empty:
        return []
    work = df.copy()
    if "wheel_ok" in work.columns:
        work = work[work["wheel_ok"] == True]  # noqa: E712
    rows: List[Dict[str, Any]] = []
    for _, r in work.iterrows():
        entry = r.get("put_entry_date") or r.get("buy_date")
        exit_ = r.get("wheel_exit_date") or r.get("put_expiry")
        gain = r.get("gain_pct")
        if entry is None or exit_ is None or gain is None or pd.isna(gain):
            continue
        rows.append(
            {
                "symbol": str(r.get("stock") or r.get("symbol") or "").upper(),
                "entry_date": pd.Timestamp(entry).normalize(),
                "exit_date": pd.Timestamp(exit_).normalize(),
                "gain_pct": float(gain),
                "put_strike": (
                    float(r["put_strike"])
                    if pd.notna(r.get("put_strike"))
                    else None
                ),
                "mark_mode": r.get("wheel_mark_mode") or r.get("mark_mode"),
                "exit_reason": r.get("wheel_exit_reason"),
            }
        )
    return rows


def occupancy_fcfs_wheel(
    trades: Sequence[Dict[str, Any]],
    *,
    max_open: int = 8,
    unique_symbol: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """FCFS occupancy on wheel cycles (entry_date / exit_date / symbol)."""
    if not trades:
        return [], {"n_kept": 0, "n_skipped": 0, "max_open": int(max_open)}
    rows = []
    for i, tr in enumerate(trades):
        buy = pd.Timestamp(tr["entry_date"]).normalize()
        sell = pd.Timestamp(tr["exit_date"]).normalize()
        if buy.tzinfo is not None:
            buy = buy.tz_convert("UTC").tz_localize(None).normalize()
        if sell.tzinfo is not None:
            sell = sell.tz_convert("UTC").tz_localize(None).normalize()
        rows.append(
            {
                **tr,
                "_i": i,
                "_buy": buy,
                "_sell": sell,
                "_sym": str(tr.get("symbol") or "").upper(),
            }
        )
    rows.sort(key=lambda r: (r["_buy"], r["_sym"], r["_i"]))

    open_pos: List[Tuple[pd.Timestamp, str]] = []
    kept: List[Dict[str, Any]] = []
    n_skip = 0
    for row in rows:
        buy = row["_buy"]
        # Slot frees when exit_date < next entry (no same-day flip).
        open_pos = [(s, sym) for s, sym in open_pos if s > buy]
        if unique_symbol and any(sym == row["_sym"] for _, sym in open_pos):
            n_skip += 1
            continue
        if int(max_open) > 0 and len(open_pos) >= int(max_open):
            n_skip += 1
            continue
        kept.append({k: v for k, v in row.items() if not str(k).startswith("_")})
        open_pos.append((row["_sell"], row["_sym"]))
    return kept, {
        "n_kept": len(kept),
        "n_skipped": n_skip,
        "max_open": int(max_open),
    }


def wheel_slot_equity(
    trades: Sequence[Dict[str, Any]],
    calendar: pd.DatetimeIndex,
    *,
    max_open: int = 8,
) -> Tuple[pd.Series, pd.Series]:
    """Equal 1/max_open equity per open cycle; mark collateral at cost until exit.

    On entry: spend ``equity/max_open`` (capped by cash) as cash-secured collateral.
    On exit: return ``collateral * (1 + gain_pct/100)``. Idle cash earns 0.
    Open slots marked at cost (no option MTM) — exit P&L is the only mark change.
    """
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    cal = cal[~cal.duplicated(keep="first")].sort_values()
    if cal.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    slots = max(1, int(max_open))
    alloc = 1.0 / float(slots)

    tdf = pd.DataFrame(list(trades)) if trades else pd.DataFrame()
    by_buy: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    if not tdf.empty:
        tdf = tdf.copy()
        tdf["_buy"] = pd.to_datetime(tdf["entry_date"]).dt.tz_localize(None).dt.normalize()
        tdf["_sell"] = pd.to_datetime(tdf["exit_date"]).dt.tz_localize(None).dt.normalize()
        tdf["gain_pct"] = pd.to_numeric(tdf["gain_pct"], errors="coerce")
        tdf = tdf.dropna(subset=["_buy", "_sell", "gain_pct"])
        tdf = tdf.sort_values(["_buy", "symbol"]).reset_index(drop=True)
        for _, row in tdf.iterrows():
            by_buy.setdefault(pd.Timestamp(row["_buy"]), []).append(row.to_dict())

    cash = 1.0
    # open: list of {sym, sell, capital, r}
    open_pos: List[Dict[str, Any]] = []
    eq_pts: List[float] = []
    inv_pts: List[float] = []

    def _mark() -> float:
        return cash + sum(float(p["capital"]) for p in open_pos)

    for dt in cal:
        dtn = pd.Timestamp(dt).normalize()
        # Exits first (sell < today frees slot for same-day entry; sell == today after entries)
        still: List[Dict[str, Any]] = []
        for pos in open_pos:
            if pd.Timestamp(pos["sell"]) < dtn:
                cash += float(pos["capital"]) * (1.0 + float(pos["r"]))
            else:
                still.append(pos)
        open_pos = still

        for row in by_buy.get(dtn, []):
            if len(open_pos) >= slots:
                break
            sym = str(row.get("symbol") or "").upper()
            if any(p["sym"] == sym for p in open_pos):
                continue
            eq_now = _mark()
            spend = min(eq_now * alloc, cash)
            if spend <= 1e-12:
                continue
            cash -= spend
            open_pos.append(
                {
                    "sym": sym,
                    "sell": pd.Timestamp(row["_sell"]).normalize(),
                    "capital": spend,
                    "r": float(row["gain_pct"]) / 100.0,
                }
            )

        # Same-day exits after entries
        still = []
        for pos in open_pos:
            if pd.Timestamp(pos["sell"]) <= dtn:
                cash += float(pos["capital"]) * (1.0 + float(pos["r"]))
            else:
                still.append(pos)
        open_pos = still

        eq_pts.append(_mark())
        inv_pts.append(len(open_pos) / float(slots))

    eq = pd.Series(eq_pts, index=cal, dtype=float)
    inv = pd.Series(inv_pts, index=cal, dtype=float)
    if float(eq.iloc[0]) > 0:
        eq = eq / float(eq.iloc[0])
    return eq, inv


def spy_buyhold_equity(close: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """SPY price NAV on ``calendar`` (ffill), rebased to 1.0."""
    s = close.astype(float).copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    out = s.reindex(cal).ffill().bfill()
    if out.empty or float(out.iloc[0]) <= 0:
        return pd.Series(dtype=float)
    return out / float(out.iloc[0])
