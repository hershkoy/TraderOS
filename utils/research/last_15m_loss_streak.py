"""Causal consecutive-loss circuit breaker for last-15m H2 fills.

Account-level chronological sit-out by buy_time. Outcomes update state only
for *taken* fills; skipped signals do not invent paper P&L.

States: Active -> Sit (skip S) -> Probe (take P) -> Active on any probe win,
else Sit again. Occupancy is not re-walked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Mode = Literal["count", "dollar"]

DEFAULT_L = 5
DEFAULT_S = 3
DEFAULT_P = 2
DEFAULT_DOLLAR_SUM = -15.0

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _is_win(gain: float) -> bool:
    return gain == gain and float(gain) > 0.0


def _bucket_for_date(ts: pd.Timestamp) -> str:
    if ts is pd.NaT or ts != ts:
        return "UNKNOWN"
    d = ts.normalize()
    for name, start, end in YEAR_BUCKETS:
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return name
    return "OTHER"


def baseline_loss_streak_diag(trades: pd.DataFrame) -> pd.DataFrame:
    """Max / distribution of consecutive losses on the full (unfiltered) book.

    Chronological by buy_time; every row is a taken trade (baseline).
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "n_trades",
                "n_losses",
                "max_consec_loss",
                "mean_run_len",
                "n_runs_ge5",
                "n_runs_ge4",
                "pct_trades_in_run_ge5",
            ]
        )
    df = trades.copy()
    df["_buy"] = pd.to_datetime(df.get("buy_time", df.get("buy_date")), errors="coerce")
    df["_gain"] = pd.to_numeric(df["gain_pct"], errors="coerce")
    df = df.dropna(subset=["_buy"]).sort_values("_buy", kind="mergesort").reset_index(drop=True)
    df["_bucket"] = df["_buy"].map(_bucket_for_date)

    rows: List[dict] = []
    for name, _, _ in YEAR_BUCKETS:
        part = df.loc[df["_bucket"] == name]
        rows.append(_streak_stats_for_gains(name, part["_gain"].tolist()))
    rows.append(_streak_stats_for_gains("FULL", df["_gain"].tolist()))
    return pd.DataFrame(rows)


def _streak_stats_for_gains(bucket: str, gains: List[Any]) -> dict:
    n = len(gains)
    losses = 0
    consec = 0
    max_consec = 0
    run_lens: List[int] = []
    in_run_ge5 = 0
    for g in gains:
        if g == g and float(g) <= 0:
            losses += 1
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            if consec > 0:
                run_lens.append(consec)
                if consec >= 5:
                    in_run_ge5 += consec
            consec = 0
    if consec > 0:
        run_lens.append(consec)
        if consec >= 5:
            in_run_ge5 += consec
    n_ge5 = sum(1 for r in run_lens if r >= 5)
    n_ge4 = sum(1 for r in run_lens if r >= 4)
    mean_run = float(np.mean(run_lens)) if run_lens else 0.0
    return {
        "bucket": bucket,
        "n_trades": n,
        "n_losses": losses,
        "max_consec_loss": int(max_consec),
        "mean_run_len": round(mean_run, 3),
        "n_runs_ge5": int(n_ge5),
        "n_runs_ge4": int(n_ge4),
        "pct_trades_in_run_ge5": round(100.0 * in_run_ge5 / n, 2) if n else 0.0,
    }


def apply_loss_streak_mask(
    trades: pd.DataFrame,
    *,
    L: int = DEFAULT_L,
    S: int = DEFAULT_S,
    P: int = DEFAULT_P,
    mode: Mode = "count",
    dollar_sum: float = DEFAULT_DOLLAR_SUM,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Return keep mask aligned to ``trades.index`` plus diagnostics.

    Parameters
    ----------
    L :
        Consecutive-loss count trigger (mode=count), or ignored for dollar
        except as documentation of the analog.
    S :
        Signals to skip after a trip.
    P :
        Probe trades to take after a sit; any win -> Active; all lose -> Sit.
    mode :
        ``count`` trips when consec losses >= L among taken trades.
        ``dollar`` trips when sum of current consecutive losers' gain_pct
        <= ``dollar_sum`` (e.g. -15).
    """
    if trades.empty:
        empty = pd.Series(dtype=bool)
        return empty, {
            "n_signals": 0,
            "n_kept": 0,
            "n_skipped": 0,
            "n_sit_episodes": 0,
            "skips_by_bucket": {},
            "taken_by_bucket": {},
        }

    L = max(1, int(L))
    S = max(0, int(S))
    P = max(1, int(P))
    dollar_thr = float(dollar_sum)

    work = trades.copy()
    work["_buy"] = pd.to_datetime(work.get("buy_time", work.get("buy_date")), errors="coerce")
    work["_gain"] = pd.to_numeric(work["gain_pct"], errors="coerce")
    work["_orig_idx"] = work.index
    work = work.dropna(subset=["_buy"]).sort_values("_buy", kind="mergesort")

    # state: "active" | "sit" | "probe"
    state = "active"
    consec = 0
    consec_sum = 0.0
    sit_left = 0
    probe_left = 0
    probe_losses = 0
    n_sit_episodes = 0
    keep_by_orig: Dict[Any, bool] = {}
    skips_by_bucket: Dict[str, int] = {}
    taken_by_bucket: Dict[str, int] = {}

    def _trip() -> bool:
        if mode == "dollar":
            return consec > 0 and consec_sum <= dollar_thr
        return consec >= L

    def _reset_streak() -> None:
        nonlocal consec, consec_sum
        consec = 0
        consec_sum = 0.0

    def _on_taken_outcome(gain: float) -> None:
        nonlocal consec, consec_sum, state, sit_left, probe_left, probe_losses, n_sit_episodes
        win = _is_win(gain)
        g = float(gain) if gain == gain else 0.0

        if state == "probe":
            probe_left -= 1
            if win:
                _reset_streak()
                state = "active"
                probe_left = 0
                probe_losses = 0
                return
            # loss during probe
            consec += 1
            consec_sum += g if g <= 0 else 0.0
            probe_losses += 1
            if probe_left <= 0:
                # all P probe trades completed without a win
                state = "sit"
                sit_left = S
                n_sit_episodes += 1
                probe_losses = 0
            return

        # active
        if win:
            _reset_streak()
            return
        consec += 1
        consec_sum += g if g <= 0 else 0.0
        if _trip():
            state = "sit"
            sit_left = S
            n_sit_episodes += 1

    for _, rec in work.iterrows():
        orig = rec["_orig_idx"]
        bucket = _bucket_for_date(rec["_buy"])
        gain = rec["_gain"]

        if state == "sit":
            if sit_left > 0:
                keep_by_orig[orig] = False
                skips_by_bucket[bucket] = skips_by_bucket.get(bucket, 0) + 1
                sit_left -= 1
                if sit_left <= 0:
                    state = "probe"
                    probe_left = P
                    probe_losses = 0
                continue
            # sit_left already 0 (S=0 edge): fall into probe
            state = "probe"
            probe_left = P
            probe_losses = 0

        # active or probe: take
        keep_by_orig[orig] = True
        taken_by_bucket[bucket] = taken_by_bucket.get(bucket, 0) + 1
        _on_taken_outcome(gain)

    keep = pd.Series(True, index=trades.index, dtype=bool)
    for idx in trades.index:
        if idx in keep_by_orig:
            keep.loc[idx] = bool(keep_by_orig[idx])
        # rows without valid buy_time stay True (cannot sequence them)

    n_kept = int(keep.sum())
    n_skip = int((~keep).sum())
    diag: Dict[str, Any] = {
        "n_signals": int(len(trades)),
        "n_kept": n_kept,
        "n_skipped": n_skip,
        "n_sit_episodes": int(n_sit_episodes),
        "skips_by_bucket": dict(skips_by_bucket),
        "taken_by_bucket": dict(taken_by_bucket),
        "L": L,
        "S": S,
        "P": P,
        "mode": mode,
        "dollar_sum": dollar_thr if mode == "dollar" else None,
    }
    return keep, diag


def streak_longer_in_hostile(
    diag_df: pd.DataFrame,
    *,
    hostile: str = "2022-2023",
    compare: str = "2020-2021",
) -> Dict[str, Any]:
    """Whether hostile bucket shows longer streak tail than compare bucket."""
    if diag_df is None or diag_df.empty:
        return {"ok": False, "reason": "empty_diag"}
    by = {str(r["bucket"]): r for _, r in diag_df.iterrows()}
    h = by.get(hostile)
    c = by.get(compare)
    if h is None or c is None:
        return {"ok": False, "reason": "missing_bucket"}
    h_max = int(h["max_consec_loss"])
    c_max = int(c["max_consec_loss"])
    h_ge5 = int(h["n_runs_ge5"])
    c_ge5 = int(c["n_runs_ge5"])
    h_pct = float(h["pct_trades_in_run_ge5"])
    c_pct = float(c["pct_trades_in_run_ge5"])
    # Hostile is "longer-tailed" if max is higher OR more ge5 runs OR higher pct in ge5
    longer = (h_max > c_max) or (h_ge5 > c_ge5 and h_pct >= c_pct)
    return {
        "ok": True,
        "longer_tailed": bool(longer),
        "hostile_max": h_max,
        "compare_max": c_max,
        "hostile_n_runs_ge5": h_ge5,
        "compare_n_runs_ge5": c_ge5,
        "hostile_pct_in_ge5": h_pct,
        "compare_pct_in_ge5": c_pct,
    }
