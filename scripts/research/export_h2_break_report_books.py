#!/usr/bin/env python3
"""Slice H2 resist-break books for TV HTML reports (no rescan).

Writes:
  - span<=365 breakout-only CSV (n~2253)
  - frozen L3 keeper + that sleeve (pre RS-cap) for HTML max/day=1
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_h2_break import _daily_base, _flat_scan_params  # noqa: E402
from backtest_channel_touch_trades import (  # noqa: E402
    _export_trade_columns,
    _summarize,
    apply_friction,
    filter_trades,
    select_same_day_rs,
)
from scripts.research.generate_channel_touch_tv_report import (  # noqa: E402
    summary_sidecar_path,
    write_summary_sidecar,
)
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

BRK = resolve_artifact("channel_touch_h2_resist_break_20260829_204905.csv")
KEEPER = resolve_artifact("channel_touch_trades_20260828_194314.csv")
FRICTION = 0.25


def _net(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "gain_pct_net" in df.columns:
        return df
    return apply_friction(df, FRICTION)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    cols = _export_trade_columns(df)
    df.loc[:, cols].to_csv(path, index=False)
    print("wrote %s n=%d" % (path.name, len(df)))


def _write_book_summary(path: Path, df: pd.DataFrame, *, title: str, notes: list[str]) -> None:
    params = _flat_scan_params(_daily_base())
    params.update(
        {
            "provider": "ALPACA",
            "timeframe": "1d",
            "fallback_provider": "IB",
            "merge_mode": "prefix",
            "friction_pct": FRICTION,
            "max_channel_span_days": 365.0,
            "h2_resist_break": True,
            "h2_resist_break_only": True,
            "causal_h2": True,
        }
    )
    gain_col = "gain_pct"
    if "gain_pct_net" in df.columns and df["gain_pct_net"].notna().all():
        gain_col = "gain_pct_net"
    results = _summarize(_net(df), gain_col=gain_col) if not df.empty else {}
    write_summary_sidecar(
        summary_sidecar_path(path),
        title=title,
        params=params,
        results=results,
        notes=notes,
    )
    print("wrote %s" % summary_sidecar_path(path).name)


def main() -> int:
    brk = pd.read_csv(BRK)
    keeper = pd.read_csv(KEEPER)
    brk_span = filter_trades(brk, max_channel_span_days=365.0)
    if "resist_break" not in keeper.columns:
        keeper = keeper.copy()
        keeper["resist_break"] = False
    else:
        keeper = keeper.copy()
        keeper["resist_break"] = keeper["resist_break"].fillna(False)

    brk_n = _net(brk)
    brk_span_n = _net(brk_span)
    keeper_n = _net(keeper)
    combo = pd.concat([keeper, brk_span], ignore_index=True, sort=False)
    combo_rs = select_same_day_rs(combo, rs_col="rs_spy_126d", max_per_day=1)
    combo_rs_n = apply_friction(combo_rs, FRICTION)
    n_brk_kept = int(combo_rs["resist_break"].fillna(False).astype(bool).sum())

    print("keeper has gain_pct_net", "gain_pct_net" in keeper.columns)
    print("brk has gain_pct_net", "gain_pct_net" in brk.columns)

    print("=== resist-break all ===", _summarize(brk_n, gain_col="gain_pct_net"))
    print("=== resist-break span<=365 ===", _summarize(brk_span_n, gain_col="gain_pct_net"))
    print("=== keeper ===", _summarize(keeper_n, gain_col="gain_pct_net"))
    print("=== keeper + span365 re-RS ===", _summarize(combo_rs_n, gain_col="gain_pct_net"))
    print("resist-break kept after RS=%d / span365=%d / union=%d" % (n_brk_kept, len(brk_span), len(combo)))

    outdir = dated_outdir()
    span_path = outdir / "channel_touch_h2_break_span365.csv"
    union_path = outdir / "channel_touch_h2_break_keeper_plus_span365.csv"
    _write_csv(span_path, brk_span)
    _write_csv(union_path, combo)
    _write_book_summary(
        span_path,
        brk_span,
        title="H2 resistance-break span<=365",
        notes=[
            "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
            "H2 resist-break fills a close above resistance after H2 (not L3 support tag)",
        ],
    )
    _write_book_summary(
        union_path,
        combo,
        title="Retired L3 keeper + H2 resist-break span<=365 (pre RS-cap union)",
        notes=[
            "Combined book for HTML max/day=1 re-RS. L3 sleeve: in-channel + span365 + beyond 0.25 + RSI 50.",
            "H2 sleeve: resist-break span<=365. Unique-symbol/day is the live nightly occupancy, not this union.",
            "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
