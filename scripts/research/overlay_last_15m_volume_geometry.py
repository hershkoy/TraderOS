"""Same-list volume-delta / geometry overlays on the last-15m 15m-next-mid book.

Smoke: --symbols AMPL,VST,TARS
Full book: omit --symbols. Occupancy is not re-walked.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_volume_geometry.py --symbols AMPL,VST,TARS
  python scripts\\research\\overlay_last_15m_volume_geometry.py --workers 8
  python scripts\\research\\overlay_last_15m_volume_geometry.py --followups --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    EXIT_DOJI_STAR,
    EXIT_FAILED_BREAKOUT,
    EXIT_SELLER_SESSIONS,
    FORM_CAP,
    FRICTION_PCT,
    POS_CAPS,
    apply_early_exit,
    book_stats,
    delayed_second_close,
    fill_day_et,
    geom_mean_year_pf,
    skip_mask,
    winner_cut_early_exit,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.session_volume_delta import (  # noqa: E402
    VOLUME_MODE_15M_SUM,
    VOLUME_MODE_SESSION_OHLC,
    consecutive_seller_sessions,
    sessions_from_by_day,
)
from utils.research.evening_doji_star import find_evening_doji_star  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_vol")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)


def _log_stats(label: str, gains: pd.Series, years: pd.DataFrame) -> None:
    s = book_stats(gains)
    gmean = geom_mean_year_pf(years)
    gtxt = "n/a" if gmean is None else "%.3f" % gmean
    pf = s["profit_factor"]
    ptxt = "n/a" if pf is None else "%.3f" % pf
    LOG.info(
        "%s n=%d E=%.3f PF=%s geo_year_PF=%s WR=%.1f",
        label,
        s["n"],
        s["expectancy_pct"] if s["expectancy_pct"] is not None else float("nan"),
        ptxt,
        gtxt,
        s["win_rate_pct"] if s["win_rate_pct"] is not None else float("nan"),
    )


def _print_cut(title: str, cut: dict) -> None:
    print("---- %s ----" % title)
    for k, v in cut.items():
        print("  %s=%s" % (k, v))


def _print_years(years: pd.DataFrame) -> None:
    if years is None or years.empty:
        return
    print(years.to_string(index=False))


def _smoke_print(row: pd.Series, by_day: dict, *, require_gap: bool) -> None:
    sym = str(row["stock"]).upper()
    fill_day = fill_day_et(row)
    print("")
    print("==== SMOKE %s fill=%s px=%.4f pos=%s form=%s gain=%.3f ====" % (
        sym,
        row.get("buy_time") or row.get("buy_date"),
        float(row["buy_price"]),
        row.get("channel_pos"),
        row.get("formation_beyond_width"),
        float(row["gain_pct"]),
    ))
    if not by_day or fill_day is None:
        print("  no 15m / fill_day")
        return
    sessions = sessions_from_by_day(by_day)
    until_day = None
    sell_d = row.get("sell_date")
    if sell_d:
        try:
            until_day = pd.Timestamp(str(sell_d)[:10]).date()
        except (TypeError, ValueError):
            until_day = None
    doji = find_evening_doji_star(
        sessions, fill_day=fill_day, require_gap=True, until_day=until_day
    )
    doji_ng = find_evening_doji_star(
        sessions, fill_day=fill_day, require_gap=False, until_day=until_day
    )
    seller = consecutive_seller_sessions(
        sessions, fill_day=fill_day, until_day=until_day
    )
    seller_ohlc = consecutive_seller_sessions(
        sessions_from_by_day(by_day, volume_mode=VOLUME_MODE_SESSION_OHLC),
        fill_day=fill_day,
        until_day=until_day,
    )
    print("  doji_gap=%s doji_nongap=%s seller2_15m=%s seller2_ohlc=%s" % (
        doji, doji_ng, seller, seller_ohlc
    ))
    print("  session  date        o      h      l      c   buy%  sell%  ohlc_buy%  ohlc_sell%")
    sessions_ohlc = sessions_from_by_day(by_day, volume_mode=VOLUME_MODE_SESSION_OHLC)
    ohlc_by_day = {s.session_date: s for s in sessions_ohlc}
    for sess in sessions:
        if fill_day is not None and sess.session_date < fill_day:
            continue
        sell_d = row.get("sell_date")
        if sell_d and str(sess.session_date) > str(sell_d)[:10]:
            continue
        ohlc = ohlc_by_day.get(sess.session_date)
        print(
            "  sess %s %6.2f %6.2f %6.2f %6.2f %5.1f %5.1f %9.1f %10.1f"
            % (
                sess.session_date.isoformat(),
                sess.open,
                sess.high,
                sess.low,
                sess.close,
                sess.buy_pct,
                sess.sell_pct,
                ohlc.buy_pct if ohlc is not None else float("nan"),
                ohlc.sell_pct if ohlc is not None else float("nan"),
            )
        )


def _run_early_overlay(
    trades: pd.DataFrame,
    indexed: dict,
    *,
    label: str,
    require_gap: bool,
    enable_doji: bool,
    enable_seller: bool,
    enable_failed_breakout: bool,
    volume_mode: str,
    baseline_gain: pd.Series,
) -> dict:
    rows: List[dict] = []
    overlay_gain = []
    for _, row in trades.iterrows():
        by_day = indexed.get(str(row["stock"]).upper()) or {}
        got = apply_early_exit(
            row,
            by_day,
            require_gap=require_gap,
            enable_doji=enable_doji,
            enable_seller=enable_seller,
            enable_failed_breakout=enable_failed_breakout,
            volume_mode=volume_mode,
        )
        overlay_gain.append(got.gain_pct)
        rows.append(
            {
                "stock": str(row["stock"]).upper(),
                "buy_date": str(row.get("buy_date") or ""),
                "gain_before": float(row["gain_pct"]),
                "gain_after": got.gain_pct,
                "used_overlay": got.used_overlay,
                "exit_reason_after": got.exit_reason,
                "doji_day": got.doji_day.isoformat() if got.doji_day else "",
                "seller_day": got.seller_day.isoformat() if got.seller_day else "",
                "failed_breakout_day": (
                    got.failed_breakout_day.isoformat() if got.failed_breakout_day else ""
                ),
                "sell_px_after": round(got.sell_px, 4),
            }
        )
    a_df = pd.DataFrame(rows)
    g = pd.Series(overlay_gain, index=trades.index)
    years = summarize_by_year(
        trades.assign(gain_pct_ov=g), gain_col="gain_pct_ov", buckets=YEAR_BUCKETS
    )
    cut = winner_cut_early_exit(baseline_gain, g)
    cut["n_overlay_used"] = int(a_df["used_overlay"].sum())
    cut["n_cut_earlier"] = int(a_df["used_overlay"].sum())
    n_doji = int((a_df["exit_reason_after"] == EXIT_DOJI_STAR).sum())
    n_seller = int((a_df["exit_reason_after"] == EXIT_SELLER_SESSIONS).sum())
    n_fail = int((a_df["exit_reason_after"] == EXIT_FAILED_BREAKOUT).sum())
    _log_stats(label, g, years)
    _print_cut(label + " winner-cut", cut)
    _print_years(years)
    LOG.info(
        "%s fills doji_star=%d seller_sessions=%d failed_breakout=%d used=%d",
        label,
        n_doji,
        n_seller,
        n_fail,
        int(a_df["used_overlay"].sum()),
    )
    return {
        "label": label,
        "df": a_df,
        "gains": g,
        "years": years,
        "cut": cut,
        "n_doji": n_doji,
        "n_seller": n_seller,
        "n_fail": n_fail,
        "stats": book_stats(g),
        "geo_year_pf": geom_mean_year_pf(years),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pad-before", type=int, default=5)
    ap.add_argument("--pad-after", type=int, default=21)
    ap.add_argument("--symbols", type=str, default="", help="Optional comma list (smoke)")
    ap.add_argument("--no-require-gap", action="store_true", help="Evening star without a true gap-up")
    ap.add_argument("--delayed-second-close", action="store_true", help="Force overlay C even if B is not blunt")
    ap.add_argument(
        "--followups",
        action="store_true",
        help="Skip A/B/C; run isolated doji-only, failed-breakout, and session-OHLC seller overlays",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require_gap = not bool(args.no_require_gap)

    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    want = str(args.symbols or "").strip().upper()
    smoke_syms = set()
    if want:
        smoke_syms = {s.strip() for s in want.split(",") if s.strip()}
        trades = trades[trades["stock"].isin(smoke_syms)].copy()
        LOG.info("Restricted to %s n=%d", sorted(smoke_syms), len(trades))
    if trades.empty:
        LOG.error("No trades in %s", args.trades)
        return 1

    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    years0 = summarize_by_year(trades, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    _log_stats("baseline_gross", baseline_gain, years0)
    net0 = baseline_gain - FRICTION_PCT
    years0n = summarize_by_year(
        trades.assign(gain_pct_net=net0), gain_col="gain_pct_net", buckets=YEAR_BUCKETS
    )
    _log_stats("baseline_fric0.25", net0, years0n)

    followups = bool(args.followups)
    b_rows: List[dict] = []
    b_blunt: Dict[str, bool] = {}
    if not followups:
        # Overlay B does not need 15m.
        b_combos: List[tuple] = [
            (None, 1.25),
            (None, 1.50),
            (FORM_CAP, None),
            (FORM_CAP, 1.25),
            (FORM_CAP, 1.50),
        ]
        for form_cap, pos_cap in b_combos:
            keep = skip_mask(trades, max_formation_beyond=form_cap, max_channel_pos=pos_cap)
            part = trades.loc[keep].copy()
            g = pd.to_numeric(part["gain_pct"], errors="coerce")
            years = summarize_by_year(part, gain_col="gain_pct", buckets=YEAR_BUCKETS)
            cut = winner_cut_skip(baseline_gain, keep)
            label = "B"
            if form_cap is not None:
                label += "_form%.2f" % form_cap
            if pos_cap is not None:
                label += "_pos%.2f" % pos_cap
            _log_stats(label, g, years)
            _print_cut(label + " winner-cut", cut)
            _print_years(years)
            b_blunt[label] = bool(cut["blunt"])
            b_rows.append(
                {
                    "overlay": label,
                    "form_cap": form_cap,
                    "pos_cap": pos_cap,
                    **book_stats(g),
                    "geo_year_pf": geom_mean_year_pf(years),
                    **{("cut_" + k): v for k, v in cut.items()},
                }
            )

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
        days=int(args.pad_before)
    )
    sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
        days=int(args.pad_after)
    )
    LOG.info(
        "Loading IB 15m symbols=%d window %s -> %s",
        len(symbols),
        buy_min.strftime("%Y-%m-%d"),
        sell_max.strftime("%Y-%m-%d"),
    )
    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min,
        end=sell_max,
        workers=int(args.workers),
        use_cache=True,
    )
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    LOG.info("Loaded IB 15m %d/%d in %.1fs", n_have, len(symbols), time.perf_counter() - t0)
    indexed = build_hold_session_index(
        panels,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )
    LOG.info("Indexed 15m sessions for %d symbols", len(indexed))

    if smoke_syms:
        for _, row in trades.iterrows():
            _smoke_print(row, indexed.get(str(row["stock"]).upper()) or {}, require_gap=require_gap)

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    if followups:
        specs = [
            {
                "label": "F1_doji_only",
                "enable_doji": True,
                "enable_seller": False,
                "enable_failed_breakout": False,
                "volume_mode": VOLUME_MODE_15M_SUM,
                "csv": "last_15m_overlay_F1_doji_only.csv",
            },
            {
                "label": "F2_failed_breakout",
                "enable_doji": False,
                "enable_seller": False,
                "enable_failed_breakout": True,
                "volume_mode": VOLUME_MODE_15M_SUM,
                "csv": "last_15m_overlay_F2_failed_breakout.csv",
            },
            {
                "label": "F3_seller_session_ohlc",
                "enable_doji": False,
                "enable_seller": True,
                "enable_failed_breakout": False,
                "volume_mode": VOLUME_MODE_SESSION_OHLC,
                "csv": "last_15m_overlay_F3_seller_session_ohlc.csv",
            },
        ]
        summary_lines = [
            "last-15m follow-up overlays (same-list, occupancy not re-walked)",
            "trades=%s" % args.trades,
            "n=%d" % len(trades),
            "require_gap=%s" % require_gap,
            "baseline geo_year_PF=%s" % geom_mean_year_pf(years0),
        ]
        for spec in specs:
            got = _run_early_overlay(
                trades,
                indexed,
                label=spec["label"],
                require_gap=require_gap,
                enable_doji=spec["enable_doji"],
                enable_seller=spec["enable_seller"],
                enable_failed_breakout=spec["enable_failed_breakout"],
                volume_mode=spec["volume_mode"],
                baseline_gain=baseline_gain,
            )
            path = outdir / spec["csv"]
            got["df"].to_csv(path, index=False)
            LOG.info("Wrote %s", path)
            st = got["stats"]
            summary_lines.append(
                "%s n=%s E=%s PF=%s geo=%s used=%s doji=%s seller=%s fail=%s cut=%s"
                % (
                    spec["label"],
                    st.get("n"),
                    st.get("expectancy_pct"),
                    st.get("profit_factor"),
                    got["geo_year_pf"],
                    got["cut"].get("n_overlay_used"),
                    got["n_doji"],
                    got["n_seller"],
                    got["n_fail"],
                    got["cut"],
                )
            )
        sum_path = outdir / "last_15m_overlay_followups_summary.txt"
        sum_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print("out:", outdir)
        LOG.info("Wrote %s", sum_path)
        return 0

    got_a = _run_early_overlay(
        trades,
        indexed,
        label="A_early_exit",
        require_gap=require_gap,
        enable_doji=True,
        enable_seller=True,
        enable_failed_breakout=False,
        volume_mode=VOLUME_MODE_15M_SUM,
        baseline_gain=baseline_gain,
    )
    a_df = got_a["df"]
    n_doji = got_a["n_doji"]
    n_seller = got_a["n_seller"]
    cut_a = got_a["cut"]
    years_a = got_a["years"]

    # Overlay C if B pos caps are blunt (or forced).
    pos_blunt = any(b_blunt.get("B_pos%.2f" % c, False) for c in POS_CAPS)
    form_pos_blunt = any(
        b_blunt.get("B_form0.25_pos%.2f" % c, False) for c in POS_CAPS
    )
    run_c = bool(args.delayed_second_close) or pos_blunt or form_pos_blunt
    c_rows: List[dict] = []
    if run_c:
        LOG.info("Running overlay C delayed 2nd close (pos skips only)")
        keep_pos = skip_mask(trades, max_formation_beyond=None, max_channel_pos=1.25)
        keep_form_only = skip_mask(trades, max_formation_beyond=FORM_CAP, max_channel_pos=None)
        delayed_idx = trades.index[~keep_pos & keep_form_only]
        c_gains = pd.Series(index=trades.index, dtype=float)
        n_delayed = 0
        n_miss = 0
        for idx in delayed_idx:
            row = trades.loc[idx]
            by_day = indexed.get(str(row["stock"]).upper()) or {}
            got = delayed_second_close(row, by_day)
            if got is None:
                n_miss += 1
                continue
            n_delayed += 1
            c_gains.loc[idx] = got["gain_pct"]
            c_rows.append({"stock": str(row["stock"]).upper(), "buy_date": str(row["buy_date"]), **got})
        g_c = pd.Series(index=trades.index, dtype=float)
        g_c.loc[keep_form_only & keep_pos] = baseline_gain.loc[keep_form_only & keep_pos]
        for idx in delayed_idx:
            if pd.notna(c_gains.loc[idx]):
                g_c.loc[idx] = c_gains.loc[idx]
        g_c = g_c.dropna()
        part_c = trades.loc[g_c.index].copy()
        part_c["gain_pct_c"] = g_c
        years_c = summarize_by_year(part_c, gain_col="gain_pct_c", buckets=YEAR_BUCKETS)
        _log_stats("C_delay_pos1.25", g_c, years_c)
        LOG.info("C delayed fills=%d no_fill_skip=%d", n_delayed, n_miss)
        _print_years(years_c)
    else:
        LOG.info("Skipping overlay C (B pos caps not blunt)")

    a_path = outdir / "last_15m_overlay_A_early_exit.csv"
    a_df.to_csv(a_path, index=False)
    b_path = outdir / "last_15m_overlay_B_skip_summary.csv"
    pd.DataFrame(b_rows).to_csv(b_path, index=False)
    if c_rows:
        c_path = outdir / "last_15m_overlay_C_delayed.csv"
        pd.DataFrame(c_rows).to_csv(c_path, index=False)
        LOG.info("Wrote %s", c_path)
    summary_lines = [
        "last-15m volume-delta / geometry overlays (same-list, occupancy not re-walked)",
        "trades=%s" % args.trades,
        "n=%d" % len(trades),
        "require_gap=%s" % require_gap,
        "A used=%d doji=%d seller=%s" % (int(a_df["used_overlay"].sum()), n_doji, n_seller),
        "A cut=%s" % cut_a,
        "baseline geo_year_PF=%s" % geom_mean_year_pf(years0),
        "A geo_year_PF=%s" % geom_mean_year_pf(years_a),
    ]
    for row in b_rows:
        summary_lines.append(
            "%s n=%s E=%s PF=%s geo=%s blunt=%s"
            % (
                row["overlay"],
                row.get("n"),
                row.get("expectancy_pct"),
                row.get("profit_factor"),
                row.get("geo_year_pf"),
                row.get("cut_blunt"),
            )
        )
    sum_path = outdir / "last_15m_overlay_volume_geometry_summary.txt"
    sum_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("out:", outdir)
    LOG.info("Wrote %s %s %s", a_path, b_path, sum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
