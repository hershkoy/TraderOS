#!/usr/bin/env python3
"""
Audit channel-touch trade quality vs classical geometry, then A/B post-filters.

Does NOT rewrite find_channels. Frozen trade CSVs often omit geometry columns
(REPORT_COLS historically dropped them) — this script re-attaches channel_pos /
width / slope / bars_span by re-running find_channels on cached OHLCV.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\audit_channel_touch_quality.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import find_channels  # noqa: E402
from backtest_channel_touch_trades import (  # noqa: E402
    apply_friction,
    filter_trades,
    select_same_day_rs,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("audit_channel_touch_quality")

DEFAULT_TRADES = [
    ROOT / "reports/ascending_channels/channel_touch_trades_20260823_225856.csv",
    ROOT / "reports/ascending_channels/channel_touch_trades_20260824_015252.csv",
]


def _line_at(y0: float, x0: int, slope: float, x: int) -> float:
    return y0 + slope * float(x - x0)


def _summarize(trades: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    if trades is None or trades.empty or gain_col not in trades.columns:
        return {
            "n_trades": 0,
            "n_symbols": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "avg_hold_days": None,
        }
    g = trades[gain_col].astype(float)
    wins = g[g > 0]
    losses = g[g <= 0]
    sum_win = float(wins.sum()) if len(wins) else 0.0
    sum_loss = float((-losses).sum()) if len(losses) else 0.0
    pf = (sum_win / sum_loss) if sum_loss > 0 else (None if sum_win <= 0 else float("inf"))
    hold = trades["hold_days"].astype(float) if "hold_days" in trades.columns else pd.Series(dtype=float)
    return {
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["stock"].nunique()) if "stock" in trades.columns else 0,
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "expectancy_pct": round(float(g.mean()), 3),
        "profit_factor": None if pf is None or not np.isfinite(pf) else round(float(pf), 3),
        "avg_win_pct": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss_pct": round(float(losses.mean()), 3) if len(losses) else None,
        "avg_hold_days": round(float(hold.mean()), 2) if len(hold) else None,
    }


def enrich_dates(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    cs = pd.to_datetime(out["channel_start"], errors="coerce")
    ce = pd.to_datetime(out["channel_end"], errors="coerce")
    bd = pd.to_datetime(out["buy_date"], errors="coerce")
    out["channel_span_days"] = (ce - cs).dt.days
    out["channel_age_at_buy_days"] = (bd - cs).dt.days
    out["days_since_channel_end"] = (bd - ce).dt.days
    return out


def enrich_flags(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if "channel_pos" in out.columns:
        pos = out["channel_pos"].astype(float)
        out["entry_above_resist"] = pos > 1.0
        out["entry_upper_half"] = pos > 0.5
        out["entry_lower_40"] = pos <= 0.40
    else:
        out["entry_above_resist"] = False
        out["entry_upper_half"] = False
        out["entry_lower_40"] = False
    if "room_to_resist_pct" in out.columns:
        out["entry_above_resist_room"] = out["room_to_resist_pct"].astype(float) < 0
    return out


def attach_geometry_from_ohlcv(
    trades: pd.DataFrame,
    *,
    provider: str = "ALPACA",
    timeframe: str = "1d",
    start: datetime = datetime(2018, 11, 1),
    end: datetime = datetime(2026, 8, 23),
    pivot_len: int = 15,
    load_workers: int = 8,
) -> pd.DataFrame:
    """Re-detect channels and attach channel_pos / width / slope at buy bar."""
    if trades.empty:
        return trades
    need = not (
        "channel_pos" in trades.columns
        and trades["channel_pos"].notna().any()
        and "channel_width_pct" in trades.columns
        and trades["channel_width_pct"].notna().any()
    )
    if not need:
        logger.info("Geometry columns already present; skipping OHLCV reattach")
        return trades

    symbols = sorted(trades["stock"].astype(str).str.upper().unique().tolist())
    logger.info("Loading OHLCV for %d symbols to reattach geometry...", len(symbols))
    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe=timeframe,
        provider=provider,
        start=start,
        end=end,
        workers=load_workers,
        use_cache=True,
    )
    logger.info("OHLCV load done in %.1fs (%d panels)", time.perf_counter() - t0, len(panels))

    pos_list: List[Optional[float]] = [None] * len(trades)
    room_list: List[Optional[float]] = [None] * len(trades)
    width_list: List[Optional[float]] = [None] * len(trades)
    slope_list: List[Optional[float]] = [None] * len(trades)
    bars_list: List[Optional[int]] = [None] * len(trades)
    matched = 0
    missing_panel = 0
    missing_channel = 0

    grouped = trades.groupby(trades["stock"].astype(str).str.upper(), sort=False)
    for stock, g in grouped:
        df = panels.get(stock)
        if df is None or df.empty:
            missing_panel += len(g)
            continue
        channels = find_channels(df, pivot_len=pivot_len)
        by_key = {(c["start_date"], c["end_date"]): c for c in channels}
        dates = df.index
        close = df["close"].to_numpy(dtype=float)
        date_to_i = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}

        for idx, row in g.iterrows():
            ch = by_key.get((str(row["channel_start"]), str(row["channel_end"])))
            if ch is None:
                # Fuzzy: match start only + nearest end
                cands = [c for c in channels if c["start_date"] == str(row["channel_start"])]
                ch = cands[0] if len(cands) == 1 else None
            if ch is None:
                missing_channel += 1
                continue
            buy_s = str(row["buy_date"])[:10]
            entry_i = date_to_i.get(buy_s)
            if entry_i is None:
                # nearest date
                bd = pd.Timestamp(buy_s)
                diffs = [(abs((d - bd).days), i) for i, d in enumerate(dates)]
                diffs.sort()
                entry_i = diffs[0][1] if diffs else None
            if entry_i is None:
                missing_channel += 1
                continue
            sx0 = int(ch["support_x0"])
            sy0 = float(ch["support_y0"])
            sslope = float(ch["support_slope"])
            width = float(ch["channel_width"])
            entry_px = float(close[entry_i])
            support_at = _line_at(sy0, sx0, sslope, entry_i)
            resist_at = support_at + width
            channel_pos = (entry_px - support_at) / width if width > 0 else float("nan")
            room = (resist_at - entry_px) / entry_px * 100.0 if entry_px > 0 else float("nan")
            loc = trades.index.get_loc(idx)
            if isinstance(loc, slice):
                continue
            if isinstance(loc, np.ndarray):
                loc = int(np.flatnonzero(loc)[0])
            pos_list[loc] = round(float(channel_pos), 3) if np.isfinite(channel_pos) else None
            room_list[loc] = round(float(room), 3) if np.isfinite(room) else None
            width_list[loc] = ch.get("channel_width_pct")
            slope_list[loc] = ch.get("slope_pct_per_bar")
            bars_list[loc] = ch.get("bars_span")
            matched += 1

    out = trades.copy()
    out["channel_pos"] = pos_list
    out["room_to_resist_pct"] = room_list
    out["channel_width_pct"] = width_list
    out["slope_pct_per_bar"] = slope_list
    out["bars_span"] = bars_list
    logger.info(
        "Geometry attach: matched=%d missing_panel=%d missing_channel=%d",
        matched,
        missing_panel,
        missing_channel,
    )
    return out


def quality_report(trades: pd.DataFrame, label: str, gain_col: str) -> List[str]:
    lines: List[str] = [f"## Quality audit: {label}", f"n={len(trades)} gain_col={gain_col}"]
    if trades.empty:
        return lines + ["(empty)"]
    if "channel_pos" not in trades.columns or trades["channel_pos"].isna().all():
        return lines + ["(no channel_pos — geometry attach failed)"]

    def pct(m: pd.Series) -> str:
        return f"{100.0 * float(m.mean()):.1f}%"

    lines.append(
        f"- entry above resist (channel_pos>1): {pct(trades['entry_above_resist'])} "
        f"({int(trades['entry_above_resist'].sum())})"
    )
    if "entry_above_resist_room" in trades.columns:
        lines.append(
            f"- entry above resist (room_to_resist<0): {pct(trades['entry_above_resist_room'])} "
            f"({int(trades['entry_above_resist_room'].sum())})"
        )
    lines.append(f"- entry upper half (pos>0.5): {pct(trades['entry_upper_half'])}")
    lines.append(f"- entry lower 40% (pos<=0.40): {pct(trades['entry_lower_40'])}")

    span = trades["channel_span_days"].dropna()
    age = trades["channel_age_at_buy_days"].dropna()
    lines.append(
        f"- channel_span_days: median={span.median():.0f} p90={span.quantile(0.9):.0f} "
        f"max={span.max():.0f}"
    )
    lines.append(
        f"- channel_age_at_buy_days: median={age.median():.0f} p90={age.quantile(0.9):.0f} "
        f"max={age.max():.0f}"
    )
    for thr in (180, 365, 730, 1000):
        m = trades["channel_span_days"].fillna(0) > thr
        lines.append(f"- span > {thr}d: {pct(m)} ({int(m.sum())})")

    buckets = [
        ("in_channel pos<=1", trades["channel_pos"].astype(float) <= 1.0),
        ("above_resist pos>1", trades["channel_pos"].astype(float) > 1.0),
        ("lower40", trades["channel_pos"].astype(float) <= 0.40),
        (
            "mid 0.4-1.0",
            (trades["channel_pos"].astype(float) > 0.40)
            & (trades["channel_pos"].astype(float) <= 1.0),
        ),
        ("span<=365d", trades["channel_span_days"].fillna(1e9) <= 365),
        ("span>365d", trades["channel_span_days"].fillna(0) > 365),
        ("span>730d", trades["channel_span_days"].fillna(0) > 730),
        ("age_at_buy<=365d", trades["channel_age_at_buy_days"].fillna(1e9) <= 365),
        ("age_at_buy>730d", trades["channel_age_at_buy_days"].fillna(0) > 730),
    ]
    lines.append("")
    lines.append("| Bucket | n | E% | PF | win% |")
    lines.append("|--------|---|----|----|------|")
    for name, mask in buckets:
        sub = trades.loc[mask]
        s = _summarize(sub, gain_col=gain_col)
        lines.append(
            f"| {name} | {s['n_trades']} | {s['expectancy_pct']} | {s['profit_factor']} | {s['win_rate_pct']} |"
        )

    bad = trades.loc[trades["entry_above_resist"]].copy()
    if not bad.empty:
        bad = bad.sort_values(["channel_span_days", "channel_pos"], ascending=[False, False])
        cols = [
            c
            for c in [
                "stock",
                "buy_date",
                "buy_price",
                "channel_start",
                "channel_end",
                "channel_pos",
                "room_to_resist_pct",
                "channel_span_days",
                "gain_pct",
            ]
            if c in bad.columns
        ]
        lines.append("")
        lines.append("### Top above-resist examples (by span)")
        lines.append(bad[cols].head(15).to_string(index=False))
    return lines


def apply_quality_filter(trades: pd.DataFrame, name: str) -> pd.DataFrame:
    t = trades
    if name == "baseline":
        return t
    if name == "in_channel":
        return filter_trades(t, require_in_channel=True)
    if name == "in_channel_soft":
        return filter_trades(t, max_channel_pos=1.15)
    if name == "lower40":
        return filter_trades(t, max_channel_pos=0.40)
    if name == "geometry_h3":
        return filter_trades(
            t,
            max_channel_pos=0.40,
            min_width_pct=3.0,
            max_width_pct=35.0,
            min_slope_pct=0.02,
            max_slope_pct=0.50,
        )
    if name == "span_le_365":
        return filter_trades(t, max_channel_span_days=365)
    if name == "span_le_730":
        return filter_trades(t, max_channel_span_days=730)
    if name == "age_le_730":
        return filter_trades(t, max_channel_age_days=730)
    if name == "in_channel+span365":
        return filter_trades(t, require_in_channel=True, max_channel_span_days=365)
    if name == "in_channel+span730":
        return filter_trades(t, require_in_channel=True, max_channel_span_days=730)
    if name == "lower40+span730":
        return filter_trades(t, max_channel_pos=0.40, max_channel_span_days=730)
    if name == "geometry_h3+span730":
        return filter_trades(
            t,
            max_channel_pos=0.40,
            min_width_pct=3.0,
            max_width_pct=35.0,
            min_slope_pct=0.02,
            max_slope_pct=0.50,
            max_channel_span_days=730,
        )
    raise ValueError(f"unknown filter {name}")


SCENARIOS: Sequence[str] = (
    "baseline",
    "in_channel",
    "in_channel_soft",
    "lower40",
    "geometry_h3",
    "span_le_365",
    "span_le_730",
    "age_le_730",
    "in_channel+span365",
    "in_channel+span730",
    "lower40+span730",
    "geometry_h3+span730",
)


def finalize(
    trades: pd.DataFrame,
    *,
    rs_top1: bool,
    friction_pct: float,
) -> Tuple[pd.DataFrame, str]:
    out = trades
    if rs_top1 and "rs_spy_126d" in out.columns:
        out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    gain_col = "gain_pct"
    if friction_pct and friction_pct > 0:
        # Avoid double-friction if CSV already netted
        if "gain_pct_net" in out.columns and "gain_pct" in out.columns:
            # Recompute clean net from gross
            out = out.copy()
            out["gain_pct_net"] = out["gain_pct"].astype(float) - float(friction_pct)
        else:
            out = apply_friction(out, friction_pct)
        gain_col = "gain_pct_net"
    return out, gain_col


def run_ab(
    raw: pd.DataFrame,
    *,
    rs_top1: bool,
    friction_pct: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    for name in SCENARIOS:
        filtered = apply_quality_filter(raw, name)
        final, gain_col = finalize(filtered, rs_top1=rs_top1, friction_pct=friction_pct)
        s = _summarize(final, gain_col=gain_col)
        rows.append({"scenario": name, "gain_col": gain_col, **s})
    return pd.DataFrame(rows)


def load_trades(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    t["stock"] = t["stock"].astype(str).str.strip().str.upper()
    return enrich_dates(t)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit + A/B channel-touch geometry quality filters")
    ap.add_argument("--trades", type=Path, nargs="*", default=None)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports/ascending_channels")
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--no-rs-top1", action="store_true")
    ap.add_argument("--skip-geometry-attach", action="store_true")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-23")
    ap.add_argument(
        "--status-md",
        type=Path,
        default=ROOT
        / "docs/status_log/edge_hunt/channel_touch/2026-08-25_channel_touch_quality_audit.md",
    )
    args = ap.parse_args()

    paths = list(args.trades) if args.trades else [p for p in DEFAULT_TRADES if p.exists()]
    if not paths:
        logger.error("No trade CSVs found")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rs_top1 = not args.no_rs_top1
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    all_ab: List[pd.DataFrame] = []
    report_lines: List[str] = [
        f"# Channel-touch geometry quality audit ({stamp[:8]})",
        "",
        "Detector unchanged (`find_ascending_channels.find_channels`). "
        "Post-hoc filters only. Frozen CSVs lacked `channel_pos` in REPORT_COLS — "
        "geometry re-attached via OHLCV + `find_channels`.",
        "",
        f"RS top1={rs_top1}, friction={args.friction_pct}%.",
        "",
        "## TV draw note (separate from detector)",
        "",
        "TradingView left-endpoint time-snap can move long multi-year rail anchors "
        "forward while keeping old prices — rails then float under recent candles. "
        "Fix in draw playbook / verify timestamps; not a `find_channels` bug.",
        "",
    ]

    enriched_paths: List[Path] = []
    for path in paths:
        logger.info("Loading %s", path)
        raw = load_trades(path)
        if not args.skip_geometry_attach:
            raw = attach_geometry_from_ohlcv(
                raw,
                start=start,
                end=end,
                load_workers=args.load_workers,
            )
        raw = enrich_dates(raw)
        raw = enrich_flags(raw)

        enriched = args.outdir / f"channel_touch_trades_geom_{path.stem}_{stamp}.csv"
        raw.to_csv(enriched, index=False)
        enriched_paths.append(enriched)
        logger.info("Wrote enriched trades -> %s", enriched)

        audit_base, gain_col = finalize(raw, rs_top1=rs_top1, friction_pct=args.friction_pct)
        audit_base = enrich_flags(enrich_dates(audit_base))
        report_lines.extend(quality_report(audit_base, path.name, gain_col))
        report_lines.append("")

        ab = run_ab(raw, rs_top1=rs_top1, friction_pct=args.friction_pct)
        ab.insert(0, "source", path.name)
        all_ab.append(ab)
        report_lines.append(f"### A/B filters on {path.name}")
        report_lines.append("")
        report_lines.append("| scenario | n | E% | PF | win% | avg_win | avg_loss |")
        report_lines.append("|----------|---|----|----|------|---------|----------|")
        base_e = None
        for _, r in ab.iterrows():
            if r["scenario"] == "baseline":
                base_e = r["expectancy_pct"]
            delta = ""
            if base_e is not None and r["expectancy_pct"] is not None and r["scenario"] != "baseline":
                delta = f" ({r['expectancy_pct'] - base_e:+.2f})"
            report_lines.append(
                f"| {r['scenario']} | {r['n_trades']} | {r['expectancy_pct']}{delta} | "
                f"{r['profit_factor']} | {r['win_rate_pct']} | {r['avg_win_pct']} | {r['avg_loss_pct']} |"
            )
        report_lines.append("")

        idcc = audit_base.loc[audit_base["stock"] == "IDCC"]
        if not idcc.empty:
            report_lines.append("### IDCC spotlight")
            cols = [
                c
                for c in [
                    "buy_date",
                    "buy_price",
                    "touch_price",
                    "channel_start",
                    "channel_end",
                    "channel_pos",
                    "room_to_resist_pct",
                    "channel_span_days",
                    "bars_span",
                    "gain_pct",
                    "exit_reason",
                ]
                if c in idcc.columns
            ]
            report_lines.append(idcc[cols].to_string(index=False))
            report_lines.append("")

    ab_all = pd.concat(all_ab, ignore_index=True)
    ab_csv = args.outdir / f"channel_touch_quality_ab_{stamp}.csv"
    ab_all.to_csv(ab_csv, index=False)

    # Auto verdict from first source baseline vs filters
    verdict_lines = ["## Verdict (auto)", ""]
    first = all_ab[0]
    base_row = first.loc[first["scenario"] == "baseline"].iloc[0]
    keepers = []
    rejects = []
    for _, r in first.iterrows():
        if r["scenario"] == "baseline":
            continue
        if r["n_trades"] is None or int(r["n_trades"]) < 50:
            rejects.append(f"- `{r['scenario']}`: n too small ({r['n_trades']})")
            continue
        e = r["expectancy_pct"]
        pf = r["profit_factor"]
        be = base_row["expectancy_pct"]
        bpf = base_row["profit_factor"]
        if e is None or pf is None or be is None or bpf is None:
            continue
        # keep if E and PF within ~10% relative of baseline or better, and n>=50%
        n_ok = int(r["n_trades"]) >= max(50, int(0.5 * int(base_row["n_trades"])))
        e_ok = e >= be * 0.9
        pf_ok = pf >= bpf * 0.9
        if n_ok and e_ok and pf_ok and (e >= be or pf >= bpf):
            keepers.append(
                f"- **soft keep** `{r['scenario']}`: n={r['n_trades']} E={e} PF={pf} "
                f"(base E={be} PF={bpf})"
            )
        elif e < be * 0.75 or pf < bpf * 0.75:
            rejects.append(
                f"- reject `{r['scenario']}`: n={r['n_trades']} E={e} PF={pf} (hurts edge)"
            )
        else:
            rejects.append(
                f"- no promote `{r['scenario']}`: n={r['n_trades']} E={e} PF={pf}"
            )
    if keepers:
        verdict_lines.append("Candidates that hold edge vs baseline:")
        verdict_lines.extend(keepers)
    else:
        verdict_lines.append(
            "No filter clearly holds edge with adequate sample — "
            "do **not** rewrite detector; keep live baseline frozen."
        )
    if rejects:
        verdict_lines.append("")
        verdict_lines.append("Others:")
        verdict_lines.extend(rejects[:12])
    verdict_lines.extend(
        [
            "",
            "### Recommended next step",
            "",
            "- If above-resist / long-span buckets have **higher** E than in-channel, "
            "edge is partly post-touch breakout momentum — classical tightening will likely cut winners.",
            "- Prefer optional CLI gates "
            "(`--require-in-channel`, `--max-channel-span-days`) over rewriting swings.",
            "- Do not promote until retested on ATR k=2.0 keeper stack if that is live.",
            "",
            f"Artifacts: `{ab_csv.as_posix()}`",
            *[f"- enriched: `{p.as_posix()}`" for p in enriched_paths],
        ]
    )
    report_lines.extend(verdict_lines)

    args.status_md.parent.mkdir(parents=True, exist_ok=True)
    args.status_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.status_md)
    logger.info("Wrote %s", ab_csv)
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
