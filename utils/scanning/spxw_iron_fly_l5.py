"""Live L5/S3/P2 sit-out state for the SPXW 0DTE iron fly.

Reuses the same Active -> Sit S -> Probe P machine as
``utils.research.last_15m_loss_streak.apply_loss_streak_mask``.
Persists under ``logs/scanners/spxw_iron_fly_l5_state.json``.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.research.last_15m_loss_streak import (
    DEFAULT_L,
    DEFAULT_P,
    DEFAULT_S,
    apply_loss_streak_mask,
)

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = ROOT / "logs" / "scanners" / "spxw_iron_fly_l5_state.json"
DEFAULT_SEED_TRADES = (
    ROOT
    / "reports"
    / "research"
    / "spxw_0dte_iron_fly"
    / "2026-10-27_from_20220511"
    / "trades.csv"
)


def _empty_state() -> Dict[str, Any]:
    return {
        "L": DEFAULT_L,
        "S": DEFAULT_S,
        "P": DEFAULT_P,
        "mode": "count",
        "seeded_through": None,
        "outcomes": [],  # [{date, taken, pnl}] chronological candidate days
    }


def load_state(path: Path = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("L5 state read failed (%s); starting empty", exc)
        return _empty_state()
    base = _empty_state()
    base.update(raw or {})
    if not isinstance(base.get("outcomes"), list):
        base["outcomes"] = []
    return base


def save_state(state: Dict[str, Any], path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def seed_from_natural_trades(
    trades_csv: Path = DEFAULT_SEED_TRADES,
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    force: bool = False,
) -> Dict[str, Any]:
    """Seed outcomes from the research natural book (baseline = all taken).

    Then re-apply L5/S3/P2 so live continues from the research overlay's
    terminal sit/probe/active position.
    """
    state = load_state(state_path)
    if state.get("outcomes") and not force:
        return state
    if not trades_csv.exists():
        LOG.warning("No seed trades at %s", trades_csv)
        save_state(state, state_path)
        return state
    df = pd.read_csv(trades_csv)
    df = df.loc[df["fill"].astype(str) == "natural"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=["date", "pnl"]).sort_values("date")
    # Build candidate list = all baseline fills; apply mask to learn which were taken.
    work = df.copy()
    work["buy_date"] = work["date"]
    work["buy_time"] = work["date"]
    work["gain_pct"] = work["pnl"]
    keep, _diag = apply_loss_streak_mask(
        work,
        L=int(state.get("L") or DEFAULT_L),
        S=int(state.get("S") or DEFAULT_S),
        P=int(state.get("P") or DEFAULT_P),
        mode="count",
    )
    outcomes: List[Dict[str, Any]] = []
    for idx, row in work.iterrows():
        taken = bool(keep.loc[idx])
        outcomes.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "taken": taken,
                "pnl": float(row["pnl"]) if taken else None,
            }
        )
    state["outcomes"] = outcomes
    state["seeded_through"] = outcomes[-1]["date"] if outcomes else None
    save_state(state, state_path)
    LOG.info(
        "Seeded L5 state n=%d taken=%d through %s",
        len(outcomes),
        sum(1 for o in outcomes if o.get("taken")),
        state["seeded_through"],
    )
    return state


def _outcomes_frame(outcomes: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        rows.append(
            {
                "buy_date": o.get("date"),
                "buy_time": o.get("date"),
                "gain_pct": float(o["pnl"]) if o.get("taken") and o.get("pnl") is not None else 0.0,
                "taken_flag": bool(o.get("taken")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["buy_date", "buy_time", "gain_pct", "taken_flag"])
    return pd.DataFrame(rows)


def decide_today(
    state: Dict[str, Any],
    today: date,
) -> Dict[str, Any]:
    """Return whether L5/S3/P2 would take ``today`` given prior outcomes.

    Does not mutate state. Call ``record_decision`` / ``record_pnl`` after.
    """
    today_s = today.isoformat()
    outcomes = [o for o in (state.get("outcomes") or []) if str(o.get("date")) < today_s]
    # Replay prior outcomes through the mask by only including taken rows'
    # PnL; skipped rows must still appear as signals with keep=False history.
    # Easiest: rebuild chronological signal list and re-run mask with actual
    # taken PnLs, then append today with placeholder PnL=0 and read keep[-1].
    hist_rows: List[Dict[str, Any]] = []
    for o in outcomes:
        hist_rows.append(
            {
                "buy_date": o["date"],
                "buy_time": o["date"],
                # Mask updates state only when kept; use real pnl for taken.
                "gain_pct": float(o["pnl"]) if o.get("taken") and o.get("pnl") is not None else 0.0,
                "_was_taken": bool(o.get("taken")),
            }
        )
    # Problem: apply_loss_streak_mask decides keep itself; we can't force prior
    # skips. So store only the *decision path* by feeding signals in order with
    # PnL only when historically taken — but the mask will re-decide.
    # Correct approach: walk a reduced machine that trusts stored taken flags.
    return _decide_from_stored(outcomes, today_s, state)


def _decide_from_stored(
    outcomes: List[Dict[str, Any]],
    today_s: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Walk stored taken/skip flags to recover terminal state, then decide today."""
    L = int(state.get("L") or DEFAULT_L)
    S = int(state.get("S") or DEFAULT_S)
    P = int(state.get("P") or DEFAULT_P)

    machine = "active"
    consec = 0
    sit_left = 0
    probe_left = 0

    def on_taken(pnl: float) -> None:
        nonlocal machine, consec, sit_left, probe_left
        win = pnl == pnl and float(pnl) > 0
        if machine == "probe":
            probe_left -= 1
            if win:
                consec = 0
                machine = "active"
                probe_left = 0
                return
            consec += 1
            if probe_left <= 0:
                machine = "sit"
                sit_left = S
            return
        if win:
            consec = 0
            return
        consec += 1
        if consec >= L:
            machine = "sit"
            sit_left = S

    for o in outcomes:
        if str(o.get("date")) >= today_s:
            break
        if not o.get("taken"):
            # Was a sit skip — consume sit_left if in sit
            if machine == "sit" and sit_left > 0:
                sit_left -= 1
                if sit_left <= 0:
                    machine = "probe"
                    probe_left = P
            continue
        if machine == "sit" and sit_left > 0:
            # Should not happen if stored consistently; treat as skip
            sit_left -= 1
            if sit_left <= 0:
                machine = "probe"
                probe_left = P
            continue
        on_taken(float(o.get("pnl") or 0.0))

    take = True
    reason = "active"
    if machine == "sit" and sit_left > 0:
        take = False
        reason = f"sit({sit_left}_left)"
    elif machine == "sit" and sit_left <= 0:
        machine = "probe"
        probe_left = P
        take = True
        reason = f"probe({probe_left})"
    elif machine == "probe":
        take = True
        reason = f"probe({probe_left})"
    else:
        reason = f"active(consec={consec})"

    return {
        "take": take,
        "reason": reason,
        "machine": machine,
        "consec": consec,
        "sit_left": sit_left,
        "probe_left": probe_left,
        "L": L,
        "S": S,
        "P": P,
        "today": today_s,
    }


def record_signal_day(
    state: Dict[str, Any],
    day: date,
    *,
    taken: bool,
    pnl: Optional[float] = None,
    state_path: Path = DEFAULT_STATE_PATH,
) -> Dict[str, Any]:
    """Append or update today's outcome; pnl may be filled later at settle."""
    day_s = day.isoformat()
    outcomes = list(state.get("outcomes") or [])
    found = False
    for o in outcomes:
        if str(o.get("date")) == day_s:
            o["taken"] = bool(taken)
            if pnl is not None:
                o["pnl"] = float(pnl)
            found = True
            break
    if not found:
        outcomes.append(
            {
                "date": day_s,
                "taken": bool(taken),
                "pnl": None if pnl is None else float(pnl),
            }
        )
    outcomes.sort(key=lambda x: str(x.get("date")))
    state["outcomes"] = outcomes
    save_state(state, state_path)
    return state
