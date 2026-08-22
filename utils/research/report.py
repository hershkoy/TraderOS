"""Phase 5 scorecard, JSON fragments, and equity/drawdown plots."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from utils.research.metrics import (
    PerfStats,
    annual_returns,
    format_annual_table,
    format_elapsed,
    format_table,
    passes_phase5_gates,
)


def _json_ready(stats: PerfStats) -> dict:
    d = stats.as_dict()
    return d


def save_fragment(
    reports_dir: Path,
    name: str,
    stats_list: Sequence[PerfStats],
    timings: Dict[str, float],
    notes: Sequence[str],
    curves: Dict[str, pd.Series],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    frag_dir = reports_dir / "fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stats": [_json_ready(s) for s in stats_list],
        "timings": timings,
        "notes": list(notes),
    }
    path = frag_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if curves:
        df = pd.DataFrame(curves)
        df.to_csv(reports_dir / f"equity_{name}.csv")
    return path


def plot_equity_and_dd(curves: Dict[str, pd.Series], out_path: Path) -> Optional[Path]:
    if not curves:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
    for name, eq in curves.items():
        eq = eq.dropna()
        if eq.empty:
            continue
        axes[0].plot(eq.index, eq.values, label=name, linewidth=1.2)
        peak = eq.cummax()
        dd = eq / peak - 1.0
        axes[1].plot(dd.index, dd.values, label=name, linewidth=1.0)
    axes[0].set_title("Equity (start=1.0)")
    axes[0].set_ylabel("Equity")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_phase5_scorecard(
    out_dir: Path,
    reports_dir: Path,
    spy_stats: PerfStats,
    all_stats: List[PerfStats],
    timings: Dict[str, float],
    window_note: str,
    curves: Dict[str, pd.Series],
    extra_sections: Sequence[str] = (),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{ts}_edge_hunt_phase5_scorecard.md"

    winners = []
    for s in all_stats:
        if s.name == spy_stats.name or s.name.startswith("SPY_buy_hold"):
            continue
        ok, gate = passes_phase5_gates(s, spy_stats)
        if ok:
            winners.append((s, gate))

    annual_map = {s.name: annual_returns(curves[s.name]) for s in all_stats if s.name in curves}

    body: List[str] = []
    body.append("# Edge hunt Phase 5 - XS mom delta + swing mean reversion")
    body.append("")
    body.append(f"Date: {ts}")
    body.append("")
    body.append("## Objective")
    body.append("")
    body.append(
        "- Gate: Sharpe **> 1.0**, Sharpe **>= SPY**, and **MDD better than SPY** "
        "(stricter than Phase 1-4). CAGR is reported, not the ranking objective."
    )
    body.append(
        "- Hold: swing / multi-day primary; 12-1 monthly is a one-shot delta vs Phase 1 "
        "(PIT top-500 ADV + SPY>SMA200), not a lookback/top-N grid."
    )
    body.append("- Incumbent (not the gate): Phase 4 KEEP `Blend_SPY70_BV30`.")
    body.append("")
    body.append("## Window / data")
    body.append("")
    body.append(f"- {window_note}")
    body.append(f"- Full eval: `{spy_stats.start}` -> `{spy_stats.end}`")
    body.append("- IS: `2018-01-01` -> `2022-12-31` (clipped to available bars)")
    body.append("- OOS: `2023-01-01` -> `2025-11-26`")
    body.append("- Costs: 10 bps round-trip. Universe: ALPACA daily panel, PIT 30d dollar volume.")
    body.append(
        "- Coverage caveat: ALPACA `1d` bars for most names begin ~2020-2022 (SPY itself from 2018-11). "
        "The wide panel index is the union of those dates, so 2018-2019 is mostly empty. "
        "SPY SMA200 is computed on IB SPY (2017+) then aligned."
    )
    body.append("- Do not re-run: vanilla SPX 12-1, SPY-only SMA200, dual mom, BigVol standalone, blend weights.")
    body.append("")
    body.append("## Scorecard")
    body.append("")
    body.append(format_table(all_stats, spy_stats))
    body.append("")
    body.append("## Annual returns")
    body.append("")
    body.append(format_annual_table(annual_map) if annual_map else "_No overlapping equity curves._")
    body.append("")
    body.append("## Notes per candidate")
    body.append("")
    for s in all_stats:
        body.append(f"- **{s.name}**: {s.notes or 'n/a'}")
    body.append("")
    body.append("## Timings")
    body.append("")
    for k, v in timings.items():
        body.append(f"- {k}: {format_elapsed(v)}")
    body.append("")
    if extra_sections:
        for block in extra_sections:
            body.append(block)
            body.append("")
    body.append("## Promotion")
    body.append("")
    if winners:
        winners_sorted = sorted(winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = winners_sorted[0]
        body.append(f"**Promoted (Phase 5):** `{best.name}` - {gate}")
        body.append("")
        body.append(
            f"Best passer: CAGR={best.cagr:.2%}, Sharpe={best.sharpe:.2f}, Sortino={best.sortino:.2f}, "
            f"MDD={best.max_drawdown:.2%} vs SPY CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} "
            f"MDD={spy_stats.max_drawdown:.2%}."
        )
        if len(winners) > 1:
            body.append("")
            body.append("Other passers: " + ", ".join(f"`{s.name}` ({g})" for s, g in winners_sorted[1:]))
    else:
        body.append("**No Phase-5 candidate cleared the risk-adjusted gate.**")
        body.append("")
        body.append(
            "If 12-1 PIT+SMA200 failed, that family is closed (no lookback/top-N fishing). "
            "Swing MR follow-on (Phase 5c) is 15m entry timing on IB 15m intersect ALPACA daily only if a daily variant is close."
        )
    body.append("")

    path.write_text("\n".join(body), encoding="utf-8")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "spy": _json_ready(spy_stats),
        "stats": [_json_ready(s) for s in all_stats],
        "timings": timings,
        "winners": [s.name for s, _ in winners],
    }
    (reports_dir / "scorecard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if curves:
        pd.DataFrame(curves).to_csv(reports_dir / "equity_curves.csv")
        plot_equity_and_dd(curves, reports_dir / "equity_drawdown.png")
    return path


def write_phase6_scorecard(
    out_dir: Path,
    reports_dir: Path,
    spy_stats: PerfStats,
    all_stats: List[PerfStats],
    timings: Dict[str, float],
    window_note: str,
    curves: Dict[str, pd.Series],
    extra_sections: Sequence[str] = (),
) -> Path:
    """Phase 6 portfolio overlays scorecard (same gate as Phase 5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{ts}_edge_hunt_phase6_scorecard.md"

    winners = []
    for s in all_stats:
        if s.name == spy_stats.name or s.name.startswith("SPY_buy_hold"):
            continue
        # Full-sample promotion only; IS/OOS rows scored vs window SPY below
        if s.name.endswith("_IS") or s.name.endswith("_OOS"):
            continue
        ok, gate = passes_phase5_gates(s, spy_stats)
        if ok:
            winners.append((s, gate))

    spy_oos = next((s for s in all_stats if s.name == "SPY_buy_hold_OOS"), None)
    oos_winners = []
    if spy_oos is not None:
        for s in all_stats:
            if not s.name.endswith("_OOS") or s.name.startswith("SPY_buy_hold"):
                continue
            ok, gate = passes_phase5_gates(s, spy_oos)
            if ok:
                oos_winners.append((s, gate))

    annual_map = {s.name: annual_returns(curves[s.name]) for s in all_stats if s.name in curves}

    body: List[str] = []
    body.append("# Edge hunt Phase 6 - Portfolio overlays (vol-target / CTA / low-vol)")
    body.append("")
    body.append(f"Date: {ts}")
    body.append("")
    body.append("## Objective")
    body.append("")
    body.append(
        "- Gate (unchanged): Sharpe **> 1.0**, Sharpe **>= SPY**, and **MDD better than SPY**."
    )
    body.append(
        "- Shift from single-stock signal fishing (Phase 5 FAIL) to portfolio construction: "
        "vol targeting, multi-asset trend, low-vol basket."
    )
    body.append("- Incumbent: Phase 4 KEEP `Blend_SPY70_BV30`.")
    body.append("")
    body.append("## Window / data")
    body.append("")
    body.append(f"- {window_note}")
    body.append(f"- Full eval: `{spy_stats.start}` -> `{spy_stats.end}`")
    body.append("- IS: `2018-01-01` -> `2022-12-31`; OOS: `2023-01-01` -> `2025-11-26`")
    body.append(
        "- VIX term-structure overlay: **skipped** (no VIX/VXV / VIX futures in TimescaleDB)."
    )
    body.append(
        "- Quality/low-vol: **realized-vol only** (no ROE/FCF fundamentals in `market_data`)."
    )
    body.append(
        "- Gate column in the table compares every row to **full-sample** SPY; "
        "formal promotion uses full-sample rows only. OOS is judged vs `SPY_buy_hold_OOS` in Promotion."
    )
    body.append("")
    body.append("## Scorecard")
    body.append("")
    body.append(format_table(all_stats, spy_stats))
    body.append("")
    body.append("## Annual returns")
    body.append("")
    body.append(format_annual_table(annual_map) if annual_map else "_No overlapping equity curves._")
    body.append("")
    body.append("## Notes per candidate")
    body.append("")
    for s in all_stats:
        body.append(f"- **{s.name}**: {s.notes or 'n/a'}")
    body.append("")
    body.append("## Timings")
    body.append("")
    for k, v in timings.items():
        body.append(f"- {k}: {format_elapsed(v)}")
    body.append("")
    if extra_sections:
        for block in extra_sections:
            body.append(block)
            body.append("")
    body.append("## Promotion")
    body.append("")
    if winners:
        winners_sorted = sorted(winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = winners_sorted[0]
        body.append(f"**Promoted (Phase 6 full sample):** `{best.name}` - {gate}")
        body.append("")
        body.append(
            f"Best passer: CAGR={best.cagr:.2%}, Sharpe={best.sharpe:.2f}, Sortino={best.sortino:.2f}, "
            f"MDD={best.max_drawdown:.2%} vs SPY CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} "
            f"MDD={spy_stats.max_drawdown:.2%}."
        )
        if len(winners) > 1:
            body.append("")
            body.append("Other passers: " + ", ".join(f"`{s.name}` ({g})" for s, g in winners_sorted[1:]))
    else:
        body.append("**No Phase-6 full-sample candidate cleared the risk-adjusted gate.**")
        body.append("")
        near = [
            s
            for s in all_stats
            if not s.name.startswith("SPY_buy_hold")
            and not s.name.endswith("_IS")
            and not s.name.endswith("_OOS")
            and abs(s.max_drawdown) + 1e-12 < abs(spy_stats.max_drawdown)
            and s.sharpe + 1e-12 >= spy_stats.sharpe
        ]
        if near:
            body.append("Near-miss (Sharpe>=SPY and better MDD, but Sharpe<=1.0):")
            for s in sorted(near, key=lambda x: x.sharpe, reverse=True):
                body.append(
                    f"- `{s.name}`: CAGR={s.cagr:.2%} Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}"
                )
            body.append("")
        body.append(
            "Next: tighten vol-target (or blend with Phase 4 BigVol) to push full-sample Sharpe above 1.0; "
            "do not re-open Phase 5 stock-signal grids. VIX curve still needs futures data."
        )
    body.append("")
    if spy_oos is not None:
        body.append("### OOS robustness (vs SPY_buy_hold_OOS)")
        body.append("")
        if oos_winners:
            ow = sorted(oos_winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
            body.append(
                "OOS passers: "
                + ", ".join(
                    f"`{s.name}` Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}" for s, _ in ow
                )
            )
        else:
            body.append("No OOS candidate beat OOS SPY on the Phase 5/6 gate.")
        body.append("")
    body.append("")

    path.write_text("\n".join(body), encoding="utf-8")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "spy": _json_ready(spy_stats),
        "stats": [_json_ready(s) for s in all_stats],
        "timings": timings,
        "winners": [s.name for s, _ in winners],
        "oos_winners": [s.name for s, _ in oos_winners],
    }
    (reports_dir / "scorecard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if curves:
        pd.DataFrame(curves).to_csv(reports_dir / "equity_curves.csv")
        plot_equity_and_dd(curves, reports_dir / "equity_drawdown.png")
    return path
