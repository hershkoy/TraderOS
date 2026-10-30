#!/usr/bin/env python3
"""Causal consecutive-loss sit-out on SPXW 0DTE iron-fly natural book.

Reuses ``apply_loss_streak_mask`` from the last-15m harness. Locked params only
(no full-sample L/S/P grid). Skipped signals do not invent paper P&L.

Locked:
  count L=5 S=3 P=2 (primary)
  count L=5 S=3 P=3
  dollar sum<=-$1500 S=3 P=2  (~3 avg losses; fly P&L is dollars)
Exploratory report-only: L in {3,4,5} S=3 P=2 (do not promote from this grid)

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_spxw_iron_fly_loss_streak.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.last_15m_loss_streak import (  # noqa: E402
    apply_loss_streak_mask,
    baseline_loss_streak_diag,
    streak_longer_in_hostile,
)
from utils.research.spxw_0dte_iron_fly import FILL_NATURAL, summarize_pnls  # noqa: E402
from utils.research.spxw_0dte_iron_fly_path import phase5_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("overlay_spxw_iron_fly_loss_streak")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "research"
    / "spxw_0dte_iron_fly"
    / "2026-10-27_from_20220511"
    / "trades.csv"
)
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "research"
    / "spxw_0dte_iron_fly"
    / "2026-10-27_from_20220511"
    / "loss_streak"
)
START_EQUITY = 25_000.0

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)

# Locked (pre-registered) — do not fish.
LOCKED: List[Dict[str, Any]] = [
    {"label": "count_L5_S3_P2", "mode": "count", "L": 5, "S": 3, "P": 2},
    {"label": "count_L5_S3_P3", "mode": "count", "L": 5, "S": 3, "P": 3},
    {
        "label": "dollar_m1500_S3_P2",
        "mode": "dollar",
        "L": 5,
        "S": 3,
        "P": 2,
        "dollar_sum": -1500.0,
    },
]
# Report-only exploratory L sweep (same family as last-15m; no promote).
EXPLORATORY_L = (3, 4, 5)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _load_natural(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["fill"].astype(str) == FILL_NATURAL].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=["date", "pnl"]).sort_values("date").reset_index(drop=True)
    # Mask harness expects buy_date / gain_pct (pct or $ both fine as signed outcome).
    df["buy_date"] = df["date"]
    df["buy_time"] = df["date"]
    df["gain_pct"] = df["pnl"]
    return df


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["date"], errors="coerce")
    return (d >= start) & (d <= end)


def _book_slice(df: pd.DataFrame) -> Dict[str, Any]:
    pnls = [float(x) for x in df["pnl"].tolist()] if not df.empty else []
    out = summarize_pnls(pnls)
    out["total"] = float(sum(pnls)) if pnls else 0.0
    return out


def _year_blocks(df: pd.DataFrame) -> Dict[str, Any]:
    blocks: Dict[str, Any] = {}
    for name, start, end in YEAR_BUCKETS:
        part = df.loc[_in_range(df, start, end)]
        blocks[name] = _book_slice(part)
    return blocks


def _load_spy(start: date, end: date) -> pd.Series:
    import yfinance as yf

    raw = yf.download(
        "SPY",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = pd.to_numeric(raw["Close"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).normalize())
    return s.dropna()


def _phase5(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    if df.empty:
        return {"gate_pass": False, "gate": "empty", "n_trades": 0}
    d0 = pd.Timestamp(df["date"].min()).date()
    d1 = pd.Timestamp(df["date"].max()).date()
    spy = _load_spy(d0, d1)
    if spy.empty:
        return {"gate_pass": False, "gate": "no_spy", "n_trades": len(df)}
    sessions = [ts.date() for ts in spy.index]
    trades = [
        {"date": pd.Timestamp(r["date"]).date(), "pnl": float(r["pnl"])}
        for _, r in df.iterrows()
    ]
    return phase5_report(
        name, trades, sessions, spy, start_equity=START_EQUITY
    )


def _score_rule(
    base: pd.DataFrame,
    *,
    label: str,
    mode: str,
    L: int,
    S: int,
    P: int,
    dollar_sum: float = -1500.0,
) -> Dict[str, Any]:
    keep, diag = apply_loss_streak_mask(
        base,
        L=L,
        S=S,
        P=P,
        mode=mode,  # type: ignore[arg-type]
        dollar_sum=dollar_sum,
    )
    kept = base.loc[keep].copy()
    leftover_hostile = kept.loc[_in_range(kept, "2022-01-01", "2023-12-31")]
    baseline_hostile = base.loc[_in_range(base, "2022-01-01", "2023-12-31")]
    book = _book_slice(kept)
    years = _year_blocks(kept)
    p5 = _phase5(kept, f"iron_fly_natural_{label}")
    LOG.info(
        "%s kept=%d skip=%d sit_ep=%d | E=%.1f PF=%s | 2022-23 E=%s PF=%s | Phase5 %s Sharpe=%s MDD=%s",
        label,
        diag.get("n_kept"),
        diag.get("n_skipped"),
        diag.get("n_sit_episodes"),
        book.get("expectancy") or 0.0,
        None if book.get("pf") is None else round(float(book["pf"]), 2),
        None
        if years["2022-2023"].get("expectancy") is None
        else round(float(years["2022-2023"]["expectancy"]), 1),
        None
        if years["2022-2023"].get("pf") is None
        else round(float(years["2022-2023"]["pf"]), 2),
        p5.get("gate"),
        None
        if not (p5.get("strat") or {}).get("sharpe")
        else round(float(p5["strat"]["sharpe"]), 2),
        None
        if not (p5.get("strat") or {}).get("max_drawdown")
        else f"{100.0 * abs(float(p5['strat']['max_drawdown'])):.1f}%",
    )
    return {
        "label": label,
        "mode": mode,
        "L": L,
        "S": S,
        "P": P,
        "dollar_sum": dollar_sum if mode == "dollar" else None,
        "diag": diag,
        "book": book,
        "years": years,
        "leftover_2022_23": _book_slice(leftover_hostile),
        "baseline_2022_23": _book_slice(baseline_hostile),
        "phase5": {
            "gate": p5.get("gate"),
            "gate_pass": p5.get("gate_pass"),
            "strat": p5.get("strat"),
            "spy": p5.get("spy"),
        },
        "kept_dates": [
            pd.Timestamp(d).date().isoformat() for d in kept["date"].tolist()
        ],
    }


def _verdict(baseline_p5: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for r in rules:
        p5 = r.get("phase5") or {}
        if not p5.get("gate_pass"):
            continue
        left = r.get("leftover_2022_23") or {}
        base_left = r.get("baseline_2022_23") or {}
        # Soft reject if leftover expectancy worse than baseline hostile window.
        left_e = left.get("expectancy")
        base_e = base_left.get("expectancy")
        worse = (
            left_e is not None
            and base_e is not None
            and float(left_e) < float(base_e)
        )
        cand = {
            "label": r["label"],
            "gate": p5.get("gate"),
            "leftover_worse": bool(worse),
            "leftover_E": left_e,
            "baseline_leftover_E": base_e,
        }
        if best is None:
            best = cand
        # Prefer not-worse leftover among PASS rules.
        if best.get("leftover_worse") and not worse:
            best = cand
    if best is None:
        return {
            "promote": False,
            "soft_promote": False,
            "reason": "No locked rule clears Phase-5 vs SPY (MDD/Sharpe)",
        }
    if best["leftover_worse"]:
        return {
            "promote": False,
            "soft_promote": False,
            "best_rule": best["label"],
            "reason": (
                f"{best['label']} Phase-5 PASS but 2022-23 leftover E worse "
                f"({best['leftover_E']} vs baseline {best['baseline_leftover_E']}) — no promote"
            ),
        }
    return {
        "promote": False,
        "soft_promote": True,
        "best_rule": best["label"],
        "reason": (
            f"{best['label']} Phase-5 PASS and 2022-23 leftover not worse — "
            "soft-promote overlay only; do not wire; do not L/S/P-fish further"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fill", default=FILL_NATURAL)
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    base = _load_natural(args.trades)
    if args.fill != FILL_NATURAL:
        LOG.warning("Only natural ranking book is scored; got fill=%s", args.fill)
    LOG.info("Baseline natural n=%d %s -> %s", len(base), base["date"].min().date(), base["date"].max().date())

    # Remap YEAR_BUCKETS inside baseline_loss_streak_diag uses fixed 2018.. buckets;
    # still useful for FULL + 2022-2023 vs 2024-2026 via our own year blocks.
    diag = baseline_loss_streak_diag(base)
    # Compare hostile 2022-2023 vs later calm 2024-2026 (no 2020-21 in this book).
    longer = streak_longer_in_hostile(
        diag, hostile="2022-2023", compare="2024-2026"
    )
    LOG.info("Streak diag:\n%s", diag.to_string(index=False))
    LOG.info("Hostile longer-tailed vs 2024-2026: %s", longer)

    baseline_book = _book_slice(base)
    baseline_years = _year_blocks(base)
    baseline_p5 = _phase5(base, "iron_fly_natural_baseline")
    LOG.info(
        "Baseline Phase5 %s Sharpe=%s MDD=%s",
        baseline_p5.get("gate"),
        None
        if not (baseline_p5.get("strat") or {}).get("sharpe")
        else round(float(baseline_p5["strat"]["sharpe"]), 2),
        None
        if not (baseline_p5.get("strat") or {}).get("max_drawdown")
        else f"{100.0 * abs(float(baseline_p5['strat']['max_drawdown'])):.1f}%",
    )

    locked_out: List[Dict[str, Any]] = []
    for rule in LOCKED:
        locked_out.append(
            _score_rule(
                base,
                label=str(rule["label"]),
                mode=str(rule["mode"]),
                L=int(rule["L"]),
                S=int(rule["S"]),
                P=int(rule["P"]),
                dollar_sum=float(rule.get("dollar_sum", -1500.0)),
            )
        )

    explor_out: List[Dict[str, Any]] = []
    for L in EXPLORATORY_L:
        explor_out.append(
            _score_rule(
                base,
                label=f"explor_L{L}_S3_P2",
                mode="count",
                L=int(L),
                S=3,
                P=2,
            )
        )

    # Locked secondary: tighter VIX open (already <20 in claim book).
    vix_out: List[Dict[str, Any]] = []
    if "vix" in base.columns:
        for vmax in (18.0, 16.0):
            sub = base.loc[pd.to_numeric(base["vix"], errors="coerce") < vmax].copy()
            label = f"vix_lt_{int(vmax)}"
            book = _book_slice(sub)
            years = _year_blocks(sub)
            p5 = _phase5(sub, f"iron_fly_natural_{label}")
            LOG.info(
                "%s n=%d E=%.1f PF=%s | 2022-23 E=%s | Phase5 %s MDD=%s",
                label,
                book.get("n"),
                book.get("expectancy") or 0.0,
                None if book.get("pf") is None else round(float(book["pf"]), 2),
                None
                if years["2022-2023"].get("expectancy") is None
                else round(float(years["2022-2023"]["expectancy"]), 1),
                p5.get("gate"),
                None
                if not (p5.get("strat") or {}).get("max_drawdown")
                else f"{100.0 * abs(float(p5['strat']['max_drawdown'])):.1f}%",
            )
            vix_out.append(
                {
                    "label": label,
                    "vix_max": vmax,
                    "book": book,
                    "years": years,
                    "leftover_2022_23": _book_slice(
                        sub.loc[_in_range(sub, "2022-01-01", "2023-12-31")]
                    ),
                    "baseline_2022_23": _book_slice(
                        base.loc[_in_range(base, "2022-01-01", "2023-12-31")]
                    ),
                    "phase5": {
                        "gate": p5.get("gate"),
                        "gate_pass": p5.get("gate_pass"),
                        "strat": p5.get("strat"),
                        "spy": p5.get("spy"),
                    },
                }
            )

    verdict = _verdict(baseline_p5, locked_out + vix_out)
    LOG.info("Verdict: %s", verdict)

    # Drop bulky date lists from JSON summary (keep in per-rule CSVs).
    slim_locked = []
    for r in locked_out:
        kept_dates = r.pop("kept_dates", [])
        slim_locked.append(r)
        pd.DataFrame({"date": kept_dates}).to_csv(
            outdir / f"kept_{r['label']}.csv", index=False
        )
    slim_explor = []
    for r in explor_out:
        kept_dates = r.pop("kept_dates", [])
        slim_explor.append(r)
        pd.DataFrame({"date": kept_dates}).to_csv(
            outdir / f"kept_{r['label']}.csv", index=False
        )

    payload = {
        "params": {
            "trades": str(args.trades),
            "fill": FILL_NATURAL,
            "start_equity": START_EQUITY,
            "locked": LOCKED,
            "exploratory_L": list(EXPLORATORY_L),
        },
        "baseline": {
            "book": baseline_book,
            "years": baseline_years,
            "phase5": {
                "gate": baseline_p5.get("gate"),
                "gate_pass": baseline_p5.get("gate_pass"),
                "strat": baseline_p5.get("strat"),
                "spy": baseline_p5.get("spy"),
            },
        },
        "streak_diag": diag.to_dict(orient="records"),
        "hostile_longer_vs_2024_2026": longer,
        "locked": slim_locked,
        "vix_tighten": vix_out,
        "exploratory": slim_explor,
        "verdict": verdict,
    }
    out_path = outdir / "summary_loss_streak.json"
    out_path.write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    LOG.info("Wrote %s", out_path)
    print(
        json.dumps(
            _jsonable(
                {
                    "verdict": verdict,
                    "hostile_longer": longer,
                    "baseline_phase5": payload["baseline"]["phase5"]["gate"],
                    "locked": [
                        {
                            "label": r["label"],
                            "n": (r.get("book") or {}).get("n"),
                            "E": (r.get("book") or {}).get("expectancy"),
                            "PF": (r.get("book") or {}).get("pf"),
                            "y22_23_E": (r.get("years") or {})
                            .get("2022-2023", {})
                            .get("expectancy"),
                            "phase5": (r.get("phase5") or {}).get("gate"),
                        }
                        for r in slim_locked
                    ],
                    "vix_tighten": [
                        {
                            "label": r["label"],
                            "n": (r.get("book") or {}).get("n"),
                            "E": (r.get("book") or {}).get("expectancy"),
                            "phase5": (r.get("phase5") or {}).get("gate"),
                        }
                        for r in vix_out
                    ],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
