"""SPY Wheel and Covered Strangle claim-check (OptionsDX EOD + Theta 2024+).

Video: Outlier Trading "Can You Beat The Market With The Wheel?" (wKIxZ9Z9mSE).

Data:
  - OptionsDX in TimescaleDB: SPY EOD bid/ask/delta 2010-01-04 -> 2023-12-29
  - ThetaData Standard: SPY greeks/eod + eod from 2024-01-02 (pre-2012 hist 403)

Books:
  - WHEEL_30_30: CSP -> assign -> CC cycle, 30 DTE / 30 delta, hold to expiry
  - CS_20_30_50_SMA200: 200 shares, 50% call cover, SMA200 put gate, 20 DTE,
    75% max-profit take
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client
from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.forward_factor import list_expirations, load_spy_daily
from utils.research.metrics import max_drawdown, perf_stats

LOG = logging.getLogger(__name__)

SYMBOL = "SPY"
MULT = 100.0
ODX_START = date(2010, 1, 4)
ODX_END = date(2023, 12, 29)
THETA_START = date(2024, 1, 2)

CLAIMED: Dict[str, Any] = {
    "source": "youtu.be/wKIxZ9Z9mSE Outlier Trading Wheel / Covered Strangle",
    "window_claim": "2007-2026-09",
    "window_local": "2010-01-04 through OptionsDX then Theta 2024+",
    "spy_cagr": 0.11,
    "spy_mdd": -0.55,
    "wheel_cagr": 0.08,
    "wheel_mdd": -0.32,
    "cs_cagr": 0.10,
    "cs_mdd": -0.31,
}

EXIT_EXPIRY = "expiry"
EXIT_TP = "tp"
EXIT_ASSIGN = "assign"
EXIT_OTM = "expire_otm"
EXIT_BTC = "btc"


class QuoteSource(Protocol):
    def chain_on_day(
        self, day: date, right: str, dte_lo: int, dte_hi: int
    ) -> pd.DataFrame:
        """Rows: expiration, strike, bid, ask, delta (signed)."""

    def quote_path(
        self,
        expiration: date,
        strike: float,
        right: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Rows: session_date, bid, ask, mid."""


# ---------------------------------------------------------------------------
# OptionsDX
# ---------------------------------------------------------------------------


def _session_date_sql() -> str:
    return "(q.ts AT TIME ZONE 'America/New_York')::date"


class OptionsDXSource:
    """SPY EOD chains from TimescaleDB option_quotes (OptionsDX)."""

    def __init__(self) -> None:
        self._client = get_timescaledb_client()
        self._chain_cache: Dict[Tuple[date, str, int, int], pd.DataFrame] = {}
        self._path_cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}

    def _conn(self):
        self._client.ensure_connection()
        return self._client.connection

    def chain_on_day(
        self, day: date, right: str, dte_lo: int, dte_hi: int
    ) -> pd.DataFrame:
        right_c = "C" if str(right).upper().startswith("C") else "P"
        key = (day, right_c, int(dte_lo), int(dte_hi))
        if key in self._chain_cache:
            return self._chain_cache[key].copy()
        exp_lo = day + timedelta(days=int(dte_lo))
        exp_hi = day + timedelta(days=int(dte_hi))
        sql = f"""
            SELECT oc.expiration,
                   oc.strike_cents / 100.0 AS strike,
                   q.bid::float AS bid,
                   q.ask::float AS ask,
                   q.delta::float AS delta
            FROM option_quotes q
            JOIN option_contracts oc ON oc.option_id = q.option_id
            WHERE oc.underlying = 'SPY'
              AND oc.option_right = %s
              AND {_session_date_sql()} = %s
              AND oc.expiration BETWEEN %s AND %s
              AND q.bid > 0 AND q.ask > 0
              AND q.delta IS NOT NULL
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, (right_c, day, exp_lo, exp_hi))
        rows = cur.fetchall()
        cur.close()
        df = pd.DataFrame(rows, columns=["expiration", "strike", "bid", "ask", "delta"])
        if not df.empty:
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
            df["strike"] = df["strike"].astype(float)
        self._chain_cache[key] = df
        return df.copy()

    def quote_path(
        self,
        expiration: date,
        strike: float,
        right: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        right_c = "C" if str(right).upper().startswith("C") else "P"
        strike_cents = int(round(float(strike) * 100))
        key = (expiration, strike_cents, right_c, start, end)
        if key in self._path_cache:
            return self._path_cache[key].copy()
        sql = f"""
            SELECT {_session_date_sql()} AS session_date,
                   q.bid::float AS bid,
                   q.ask::float AS ask
            FROM option_quotes q
            JOIN option_contracts oc ON oc.option_id = q.option_id
            WHERE oc.underlying = 'SPY'
              AND oc.expiration = %s
              AND oc.strike_cents = %s
              AND oc.option_right = %s
              AND {_session_date_sql()} BETWEEN %s AND %s
              AND q.bid > 0 AND q.ask > 0
            ORDER BY 1
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, (expiration, strike_cents, right_c, start, end))
        rows = cur.fetchall()
        cur.close()
        df = pd.DataFrame(rows, columns=["session_date", "bid", "ask"])
        if df.empty:
            self._path_cache[key] = df
            return df.copy()
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df = df.drop_duplicates("session_date", keep="last")
        self._path_cache[key] = df
        return df.copy()


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------


class ThetaSource:
    """SPY EOD chains via Theta Terminal (post-OptionsDX window)."""

    def __init__(
        self,
        client: Optional[ThetaDataClient] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.client = client or get_thetadata_client()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._exps: Optional[List[date]] = None
        self._chain_cache: Dict[Tuple[date, str, date], pd.DataFrame] = {}
        self._path_cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}

    def expirations(self) -> List[date]:
        if self._exps is None:
            self._exps = list_expirations(self.client, SYMBOL)
        return self._exps

    def _fetch_chain_exp(self, day: date, right: str, exp: date) -> pd.DataFrame:
        right_s = "call" if str(right).upper().startswith("C") else "put"
        key = (day, right_s[0].upper(), exp)
        if key in self._chain_cache:
            return self._chain_cache[key].copy()
        path = None
        if self.cache_dir is not None:
            path = (
                self.cache_dir
                / f"chain_{SYMBOL}_{exp.isoformat()}_{day.isoformat()}_{right_s[0]}.csv"
            )
            if path.exists():
                try:
                    cached = pd.read_csv(path)
                    if cached.empty:
                        self._chain_cache[key] = cached
                        return cached.copy()
                    if "expiration" not in cached.columns:
                        cached["expiration"] = exp.isoformat()
                    cached["expiration"] = pd.to_datetime(cached["expiration"]).dt.date
                    self._chain_cache[key] = cached
                    return cached.copy()
                except Exception:  # noqa: BLE001
                    path.unlink(missing_ok=True)
        r = self.client.session.get(
            f"{self.client.base_url}/v3/option/history/greeks/eod",
            params={
                "symbol": SYMBOL,
                "expiration": exp.isoformat(),
                "right": right_s,
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
            timeout=90,
        )
        if r.status_code in (403, 472) or not r.content:
            empty = pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
            if path is not None:
                empty.to_csv(path, index=False)
            self._chain_cache[key] = empty
            return empty.copy()
        if r.status_code != 200:
            LOG.warning("theta chain %s %s %s -> %s", day, exp, right_s, r.status_code)
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        raw = pd.read_csv(StringIO(r.text))
        if raw.empty:
            empty = pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
            if path is not None:
                empty.to_csv(path, index=False)
            self._chain_cache[key] = empty
            return empty.copy()
        strike_col = "strike" if "strike" in raw.columns else raw.columns[0]
        bid_col = "bid" if "bid" in raw.columns else None
        ask_col = "ask" if "ask" in raw.columns else None
        delta_col = "delta" if "delta" in raw.columns else None
        rows = []
        for _, row in raw.iterrows():
            try:
                strike = float(row[strike_col])
                bid = float(row[bid_col]) if bid_col else float("nan")
                ask = float(row[ask_col]) if ask_col else float("nan")
                delta = float(row[delta_col]) if delta_col else float("nan")
            except (TypeError, ValueError):
                continue
            if not (bid > 0 and ask > 0 and math.isfinite(delta)):
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
        df = pd.DataFrame(rows)
        if path is not None:
            df.to_csv(path, index=False)
        self._chain_cache[key] = df
        return df.copy()

    def chain_on_day(
        self, day: date, right: str, dte_lo: int, dte_hi: int
    ) -> pd.DataFrame:
        cands = [
            e
            for e in self.expirations()
            if int(dte_lo) <= (e - day).days <= int(dte_hi)
        ]
        # Prefer nearest to mid of window; fetch a few closest
        if not cands:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        mid = 0.5 * (float(dte_lo) + float(dte_hi))
        cands = sorted(cands, key=lambda e: abs((e - day).days - mid))[:4]
        frames = [self._fetch_chain_exp(day, right, e) for e in cands]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        return pd.concat(frames, ignore_index=True)

    def quote_path(
        self,
        expiration: date,
        strike: float,
        right: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        if end < start:
            return pd.DataFrame(columns=["session_date", "bid", "ask", "mid"])
        right_s = "call" if str(right).upper().startswith("C") else "put"
        # Theta EOD max ~365d; chunk if needed
        key = (expiration, float(strike), right_s[0].upper(), start, end)
        if key in self._path_cache:
            return self._path_cache[key].copy()
        path = None
        if self.cache_dir is not None:
            path = (
                self.cache_dir
                / (
                    f"eod_{SYMBOL}_{expiration.isoformat()}_"
                    f"{float(strike):.3f}_{right_s[0]}_{start}_{end}.csv"
                )
            )
            if path.exists():
                try:
                    cached = pd.read_csv(path)
                    if not cached.empty and "session_date" in cached.columns:
                        cached["session_date"] = pd.to_datetime(
                            cached["session_date"]
                        ).dt.date
                        self._path_cache[key] = cached
                        return cached.copy()
                except Exception:  # noqa: BLE001
                    path.unlink(missing_ok=True)

        chunks: List[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=364))
            r = self.client.session.get(
                f"{self.client.base_url}/v3/option/history/eod",
                params={
                    "symbol": SYMBOL,
                    "expiration": expiration.isoformat(),
                    "strike": f"{float(strike):.3f}",
                    "right": right_s,
                    "start_date": cursor.isoformat(),
                    "end_date": chunk_end.isoformat(),
                },
                timeout=120,
            )
            if r.status_code == 200 and r.content and b"No data" not in r.content[:80]:
                raw = pd.read_csv(StringIO(r.text))
                if not raw.empty:
                    if "created" in raw.columns:
                        ts = pd.to_datetime(raw["created"], errors="coerce")
                    elif "last_trade" in raw.columns:
                        ts = pd.to_datetime(raw["last_trade"], errors="coerce")
                    else:
                        ts = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
                    part = pd.DataFrame(
                        {
                            "session_date": ts.dt.date,
                            "bid": pd.to_numeric(raw.get("bid"), errors="coerce"),
                            "ask": pd.to_numeric(raw.get("ask"), errors="coerce"),
                        }
                    )
                    part = part.dropna(subset=["session_date", "bid", "ask"])
                    part = part[(part["bid"] > 0) & (part["ask"] > 0)]
                    chunks.append(part)
            cursor = chunk_end + timedelta(days=1)

        if not chunks:
            empty = pd.DataFrame(columns=["session_date", "bid", "ask", "mid"])
            if path is not None:
                empty.to_csv(path, index=False)
            self._path_cache[key] = empty
            return empty.copy()
        df = pd.concat(chunks, ignore_index=True)
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df = df.drop_duplicates("session_date", keep="last").sort_values(
            "session_date"
        )
        if path is not None:
            df.to_csv(path, index=False)
        self._path_cache[key] = df
        return df.copy()


class HybridSource:
    """OptionsDX through 2023-12-29, Theta from 2024-01-02."""

    def __init__(
        self,
        odx: Optional[OptionsDXSource] = None,
        theta: Optional[ThetaSource] = None,
    ) -> None:
        self.odx = odx or OptionsDXSource()
        self.theta = theta or ThetaSource()

    def _src(self, day: date) -> QuoteSource:
        if day <= ODX_END:
            return self.odx
        return self.theta

    def chain_on_day(
        self, day: date, right: str, dte_lo: int, dte_hi: int
    ) -> pd.DataFrame:
        return self._src(day).chain_on_day(day, right, dte_lo, dte_hi)

    def quote_path(
        self,
        expiration: date,
        strike: float,
        right: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        # Prefer ODX for paths wholly inside ODX window; else Theta
        if end <= ODX_END:
            return self.odx.quote_path(expiration, strike, right, start, end)
        if start >= THETA_START:
            return self.theta.quote_path(expiration, strike, right, start, end)
        # Stitch across boundary
        a = self.odx.quote_path(expiration, strike, right, start, ODX_END)
        b = self.theta.quote_path(
            expiration, strike, right, THETA_START, end
        )
        frames = [f for f in (a, b) if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(columns=["session_date", "bid", "ask", "mid"])
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("session_date", keep="last")
            .sort_values("session_date")
        )


# ---------------------------------------------------------------------------
# Contract pick + marks
# ---------------------------------------------------------------------------


def pick_contract(
    chain: pd.DataFrame,
    *,
    day: date,
    target_dte: int,
    target_delta: float,
    dte_tol: int = 10,
    min_dte: int = 7,
    min_credit: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Nearest expiry to target_dte, then strike nearest |delta| to target."""
    if chain is None or chain.empty:
        return None
    df = chain.copy()
    df["dte"] = df["expiration"].map(lambda e: (e - day).days)
    df = df[(df["dte"] >= int(min_dte)) & (df["dte"] <= int(target_dte) + int(dte_tol))]
    if df.empty:
        return None
    # Prefer expiry closest to target_dte
    best_dte_err = int(df["dte"].sub(int(target_dte)).abs().min())
    df = df[df["dte"].sub(int(target_dte)).abs() == best_dte_err]
    exp = df["expiration"].iloc[0]
    sub = df[df["expiration"] == exp].copy()
    sub["abs_delta"] = sub["delta"].abs()
    sub = sub[sub["bid"] >= float(min_credit)]
    if sub.empty:
        return None
    sub["delta_err"] = (sub["abs_delta"] - float(target_delta)).abs()
    row = sub.sort_values(["delta_err", "strike"]).iloc[0]
    mid = 0.5 * (float(row["bid"]) + float(row["ask"]))
    return {
        "expiration": exp,
        "strike": float(row["strike"]),
        "bid": float(row["bid"]),
        "ask": float(row["ask"]),
        "mid": mid,
        "delta": float(row["delta"]),
        "dte": int(row["dte"]),
    }


def entry_credit(quote: Dict[str, Any], mark_mode: str) -> float:
    m = str(mark_mode).lower()
    if m in ("ask_bid", "natural", "askbid"):
        return float(quote["bid"])
    return float(quote["mid"])


def exit_debit(bid: float, ask: float, mark_mode: str) -> float:
    m = str(mark_mode).lower()
    if m in ("ask_bid", "natural", "askbid"):
        return float(ask)
    return 0.5 * (float(bid) + float(ask))


def mtm_mark(bid: float, ask: float, mark_mode: str) -> float:
    """Short-option liability mark (higher = worse for short)."""
    m = str(mark_mode).lower()
    if m in ("ask_bid", "natural", "askbid"):
        return float(ask)
    return 0.5 * (float(bid) + float(ask))


# ---------------------------------------------------------------------------
# Portfolio / state machine
# ---------------------------------------------------------------------------


@dataclass
class ShortLeg:
    right: str
    expiration: date
    strike: float
    contracts: int
    entry_credit: float
    entry_date: date
    entry_delta: float = 0.0
    entry_dte: int = 0


@dataclass
class ClosedLeg:
    right: str
    entry_date: date
    exit_date: date
    expiration: date
    strike: float
    contracts: int
    entry_credit: float
    exit_debit: float
    pnl: float
    reason: str
    shares_after: int = 0


@dataclass
class BookResult:
    label: str
    mode: str
    mark_mode: str
    start: date
    end: date
    equity: pd.Series = field(default_factory=pd.Series)
    spy_equity: pd.Series = field(default_factory=pd.Series)
    closed: List[ClosedLeg] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def stats(self) -> Dict[str, Any]:
        eq = self.equity.dropna()
        spy = self.spy_equity.dropna()
        if eq.empty:
            return {"label": self.label, "n": 0}
        spy_stats = (
            perf_stats("SPY", spy) if len(spy) >= 2 else None
        )
        st = perf_stats(
            self.label,
            eq,
            spy_cagr=spy_stats.cagr if spy_stats else None,
            spy_sharpe=spy_stats.sharpe if spy_stats else None,
            trade_rets=[c.pnl for c in self.closed],
        )
        out = st.as_dict()
        out["claimed"] = CLAIMED
        out["n_closed_legs"] = len(self.closed)
        out["mark_mode"] = self.mark_mode
        out["mode"] = self.mode
        if spy_stats:
            out["spy"] = {
                "cagr": spy_stats.cagr,
                "max_drawdown": spy_stats.max_drawdown,
                "sharpe": spy_stats.sharpe,
            }
        return out


def _sma200(closes: pd.Series) -> pd.Series:
    return closes.rolling(200, min_periods=200).mean()


def target_call_contracts(shares: int, cover_frac: float) -> int:
    """Integer contracts covering cover_frac of shares (50% => shares//200)."""
    if shares < MULT or cover_frac <= 0:
        return 0
    return int(math.floor(shares * float(cover_frac) / MULT))


def run_book(
    *,
    mode: str,
    start: date,
    end: date,
    source: QuoteSource,
    mark_mode: str = "mid",
    target_dte: int = 30,
    target_delta: float = 0.30,
    dte_tol: int = 10,
    tp_frac: Optional[float] = None,
    cover_frac: float = 1.0,
    use_sma_put_gate: bool = False,
    commission_per_contract: float = 0.0,
    spots: Optional[pd.DataFrame] = None,
    min_credit: float = 0.05,
) -> BookResult:
    """Daily wheel / covered-strangle simulation.

    ``tp_frac``: close short when mark <= entry_credit * (1 - tp_frac).
    None => hold to expiry only.
    """
    mode_l = str(mode).lower()
    if mode_l not in ("wheel", "cs", "covered_strangle"):
        raise ValueError(f"unknown mode {mode}")
    is_cs = mode_l in ("cs", "covered_strangle")

    warm_start = start - timedelta(days=400)
    spots = spots if spots is not None else load_spy_daily(warm_start, end)
    if spots is None or spots.empty:
        raise RuntimeError("No SPY daily bars")
    close = spots["close"].astype(float).copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    sma = _sma200(close)

    days = [ts.date() for ts in close.index if start <= ts.date() <= end]
    if not days:
        raise RuntimeError("No sessions in window")

    spot0 = float(close.loc[pd.Timestamp(days[0])])
    if is_cs:
        # 200 shares + cash for 1 CSP
        cash = float(spot0) * 300.0
        shares = 0
        cash -= 200.0 * float(spot0)
        shares = 200
        nav0 = float(spot0) * 300.0
        label_cover = int(round(float(cover_frac) * 100))
        label = (
            f"CS_{target_dte}_{int(round(target_delta*100))}_"
            f"c{label_cover}_{mark_mode}"
        )
        if use_sma_put_gate:
            label += "_SMA200"
        if tp_frac is not None:
            label += f"_tp{int(round(tp_frac*100))}"
    else:
        cash = float(spot0) * 100.0
        shares = 0
        nav0 = cash
        label = f"WHEEL_{target_dte}_{int(round(target_delta*100))}_{mark_mode}"
        if tp_frac is not None:
            label += f"_tp{int(round(tp_frac*100))}"

    short_put: Optional[ShortLeg] = None
    short_call: Optional[ShortLeg] = None
    closed: List[ClosedLeg] = []
    events: List[Dict[str, Any]] = []
    equity_rows: List[Tuple[pd.Timestamp, float]] = []
    spy_shares = nav0 / float(spot0)
    mark_paths: Dict[Tuple[Any, ...], Dict[date, Tuple[float, float]]] = {}

    def _path_map(leg: ShortLeg) -> Dict[date, Tuple[float, float]]:
        key = (leg.expiration, float(leg.strike), leg.right)
        if key not in mark_paths:
            df = source.quote_path(
                leg.expiration,
                leg.strike,
                leg.right,
                leg.entry_date,
                min(end + timedelta(days=5), leg.expiration + timedelta(days=5)),
            )
            mp: Dict[date, Tuple[float, float]] = {}
            if df is not None and not df.empty:
                for _, r in df.iterrows():
                    mp[r["session_date"]] = (float(r["bid"]), float(r["ask"]))
            mark_paths[key] = mp
        return mark_paths[key]

    def _mark_for(leg: ShortLeg, day: date) -> Optional[float]:
        mp = _path_map(leg)
        if day in mp:
            bid, ask = mp[day]
            return mtm_mark(bid, ask, mark_mode)
        # fallback intrinsic
        spot = float(close.loc[pd.Timestamp(day)])
        if leg.right == "P":
            return max(leg.strike - spot, 0.0)
        return max(spot - leg.strike, 0.0)

    def _btc(leg: ShortLeg, day: date, reason: str) -> None:
        nonlocal cash, short_put, short_call
        mp = _path_map(leg)
        if day in mp:
            bid, ask = mp[day]
            debit = exit_debit(bid, ask, mark_mode)
        else:
            debit = _mark_for(leg, day) or 0.0
        fee = float(commission_per_contract) * int(leg.contracts)
        cash -= debit * MULT * int(leg.contracts) + fee
        pnl = (float(leg.entry_credit) - float(debit)) * MULT * int(leg.contracts) - fee
        # entry fee already taken at open; approximate pnl on premium only + exit fee
        closed.append(
            ClosedLeg(
                right=leg.right,
                entry_date=leg.entry_date,
                exit_date=day,
                expiration=leg.expiration,
                strike=leg.strike,
                contracts=leg.contracts,
                entry_credit=leg.entry_credit,
                exit_debit=float(debit),
                pnl=float(pnl),
                reason=reason,
                shares_after=shares,
            )
        )
        events.append(
            {
                "date": str(day),
                "event": "close",
                "right": leg.right,
                "strike": leg.strike,
                "exp": str(leg.expiration),
                "debit": float(debit),
                "reason": reason,
            }
        )
        if leg.right == "P":
            short_put = None
        else:
            short_call = None

    def _settle_expiry(leg: ShortLeg, day: date) -> None:
        nonlocal cash, shares, short_put, short_call
        spot = float(close.loc[pd.Timestamp(day)])
        if leg.right == "P":
            if spot < leg.strike - 1e-9:
                # assigned: buy shares at strike
                cost = float(leg.strike) * MULT * int(leg.contracts)
                cash -= cost
                shares += int(MULT) * int(leg.contracts)
                events.append(
                    {
                        "date": str(day),
                        "event": "assign_put",
                        "strike": leg.strike,
                        "shares": shares,
                        "cash": cash,
                    }
                )
                closed.append(
                    ClosedLeg(
                        right="P",
                        entry_date=leg.entry_date,
                        exit_date=day,
                        expiration=leg.expiration,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        entry_credit=leg.entry_credit,
                        exit_debit=max(leg.strike - spot, 0.0),
                        pnl=(float(leg.entry_credit) - max(leg.strike - spot, 0.0))
                        * MULT
                        * int(leg.contracts),
                        reason=EXIT_ASSIGN,
                        shares_after=shares,
                    )
                )
            else:
                closed.append(
                    ClosedLeg(
                        right="P",
                        entry_date=leg.entry_date,
                        exit_date=day,
                        expiration=leg.expiration,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        entry_credit=leg.entry_credit,
                        exit_debit=0.0,
                        pnl=float(leg.entry_credit) * MULT * int(leg.contracts),
                        reason=EXIT_OTM,
                        shares_after=shares,
                    )
                )
            short_put = None
        else:
            n_share = int(MULT) * int(leg.contracts)
            if spot > leg.strike + 1e-9:
                # called away
                n_share = min(n_share, shares)
                proceeds = float(leg.strike) * float(n_share)
                cash += proceeds
                shares -= n_share
                events.append(
                    {
                        "date": str(day),
                        "event": "assign_call",
                        "strike": leg.strike,
                        "shares": shares,
                        "cash": cash,
                    }
                )
                closed.append(
                    ClosedLeg(
                        right="C",
                        entry_date=leg.entry_date,
                        exit_date=day,
                        expiration=leg.expiration,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        entry_credit=leg.entry_credit,
                        exit_debit=max(spot - leg.strike, 0.0),
                        pnl=(float(leg.entry_credit) - max(spot - leg.strike, 0.0))
                        * MULT
                        * int(leg.contracts),
                        reason=EXIT_ASSIGN,
                        shares_after=shares,
                    )
                )
            else:
                closed.append(
                    ClosedLeg(
                        right="C",
                        entry_date=leg.entry_date,
                        exit_date=day,
                        expiration=leg.expiration,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        entry_credit=leg.entry_credit,
                        exit_debit=0.0,
                        pnl=float(leg.entry_credit) * MULT * int(leg.contracts),
                        reason=EXIT_OTM,
                        shares_after=shares,
                    )
                )
            short_call = None

    def _open_short(day: date, right: str, contracts: int) -> Optional[ShortLeg]:
        nonlocal cash
        if contracts <= 0:
            return None
        dte_lo = max(7, int(target_dte) - int(dte_tol))
        dte_hi = int(target_dte) + int(dte_tol)
        chain = source.chain_on_day(day, right, dte_lo, dte_hi)
        picked = pick_contract(
            chain,
            day=day,
            target_dte=int(target_dte),
            target_delta=float(target_delta),
            dte_tol=int(dte_tol),
            min_credit=float(min_credit),
        )
        if not picked:
            return None
        credit = entry_credit(picked, mark_mode)
        if credit < float(min_credit):
            return None
        # Unit-lot claim-check: keep 1 (or sized) contract even when cash < strike
        # collateral. Cash may go negative after assignment; NAV = cash + shares
        # - short marks is the equity curve. Strict cash-secured gating stalls the
        # wheel once SPY outruns premium growth (see 2014+ flat cash bug).
        fee = float(commission_per_contract) * int(contracts)
        cash += credit * MULT * int(contracts) - fee
        leg = ShortLeg(
            right="C" if right.upper().startswith("C") else "P",
            expiration=picked["expiration"],
            strike=float(picked["strike"]),
            contracts=int(contracts),
            entry_credit=float(credit),
            entry_date=day,
            entry_delta=float(picked["delta"]),
            entry_dte=int(picked["dte"]),
        )
        # warm path cache
        _path_map(leg)
        events.append(
            {
                "date": str(day),
                "event": "open",
                "right": leg.right,
                "strike": leg.strike,
                "exp": str(leg.expiration),
                "credit": credit,
                "delta": leg.entry_delta,
                "dte": leg.entry_dte,
                "contracts": contracts,
            }
        )
        return leg

    for i, day in enumerate(days):
        if i and i % 250 == 0:
            LOG.info("%s %s/%s cash=%.0f shares=%s", label, i, len(days), cash, shares)
        ts = pd.Timestamp(day)
        spot = float(close.loc[ts])

        # 1) Expiry settlement (options typically settle at close on expiry day)
        for leg in (short_put, short_call):
            if leg is not None and leg.expiration <= day:
                _settle_expiry(leg, day)

        # 2) Take-profit management
        if tp_frac is not None and 0 < float(tp_frac) < 1:
            thresh_mult = 1.0 - float(tp_frac)
            for leg in (short_put, short_call):
                if leg is None:
                    continue
                if leg.expiration <= day:
                    continue
                mk = _mark_for(leg, day)
                if mk is not None and mk <= float(leg.entry_credit) * thresh_mult + 1e-12:
                    _btc(leg, day, EXIT_TP)

        # 3) CS: rebuild share target to 200 if under
        if is_cs and shares < 200:
            need = 200 - shares
            cost = need * spot
            if cash >= cost:
                cash -= cost
                shares += need
                events.append(
                    {
                        "date": str(day),
                        "event": "rebuild_shares",
                        "shares": shares,
                        "cash": cash,
                    }
                )

        # 4) Open missing legs
        if is_cs:
            want_calls = target_call_contracts(shares, cover_frac)
            if short_call is None and want_calls > 0:
                short_call = _open_short(day, "C", want_calls)
            put_ok = True
            if use_sma_put_gate:
                sma_v = sma.loc[ts] if ts in sma.index else np.nan
                # prior close vs SMA: use today's close vs SMA (SMA includes today);
                # causal: compare prior session close to prior SMA
                if i == 0:
                    put_ok = False
                else:
                    prev_ts = pd.Timestamp(days[i - 1])
                    prev_close = float(close.loc[prev_ts])
                    prev_sma = float(sma.loc[prev_ts]) if prev_ts in sma.index else float("nan")
                    put_ok = math.isfinite(prev_sma) and prev_close > prev_sma
            if short_put is None and put_ok:
                short_put = _open_short(day, "P", 1)
        else:
            # Wheel: CSP when flat stock; CC when long shares
            if shares <= 0 and short_put is None:
                short_put = _open_short(day, "P", 1)
            want = int(shares // int(MULT))
            if shares >= int(MULT) and short_call is None and want > 0:
                short_call = _open_short(day, "C", want)

        # 5) MTM
        liab = 0.0
        for leg in (short_put, short_call):
            if leg is None:
                continue
            mk = _mark_for(leg, day)
            if mk is not None:
                liab += float(mk) * MULT * int(leg.contracts)
        nav = cash + shares * spot - liab
        equity_rows.append((ts, float(nav)))

    eq = pd.Series(
        {t: v for t, v in equity_rows}, dtype=float
    ).sort_index()
    spy_eq = pd.Series(
        {
            pd.Timestamp(d): spy_shares * float(close.loc[pd.Timestamp(d)])
            for d in days
        },
        dtype=float,
    ).sort_index()

    notes = (
        f"mode={mode_l} marks={mark_mode} dte={target_dte} delta={target_delta} "
        f"tp={tp_frac} cover={cover_frac} sma_put={use_sma_put_gate}; "
        f"unit-lot (cash may go negative for CSP collateral); "
        f"no early assignment; dividends not credited"
    )
    return BookResult(
        label=label,
        mode="cs" if is_cs else "wheel",
        mark_mode=str(mark_mode),
        start=days[0],
        end=days[-1],
        equity=eq,
        spy_equity=spy_eq,
        closed=closed,
        events=events,
        notes=notes,
    )


def closed_to_frame(closed: Sequence[ClosedLeg]) -> pd.DataFrame:
    if not closed:
        return pd.DataFrame()
    return pd.DataFrame([asdict(c) for c in closed])


def default_wheel_params() -> Dict[str, Any]:
    return {
        "mode": "wheel",
        "target_dte": 30,
        "target_delta": 0.30,
        "tp_frac": None,
        "cover_frac": 1.0,
        "use_sma_put_gate": False,
    }


def default_cs_params() -> Dict[str, Any]:
    return {
        "mode": "cs",
        "target_dte": 20,
        "target_delta": 0.30,
        "tp_frac": 0.75,
        "cover_frac": 0.50,
        "use_sma_put_gate": True,
    }
