"""Causal VIX backwardation + expanding HMM skip on the last-15m 1d book.

Pre-registered (same-list, occupancy not re-walked). Fill-day daily closes
are illegal: joins use prior-session completed_asof.

  VIX_back          skip if prior VIX/VIX3M >= 1
  VIX_back_min20    skip if backwardation AND VIX >= 20
  HMM2_p50/p65      2-state SPY+VIX+VIX3M, skip if P(bull) < thresh
  HMM3_p50/p65      3-state, same
  HMM2_macro_p50    2-state + TLT/GLD/UUP when those series exist; missing p_bull keeps

Grade: 2022 itself non-red, 2022-23 not a failed-breakout slaughter, without
clipping 2020-21 / 2024-26 winner dollars (winner-$ gate).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_hmm_vix_skip.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.research.causal_regime_hmm import (  # noqa: E402
    CORE_FEATURE_COLS,
    MACRO_FEATURE_COLS,
    attach_prior_session_regime,
    build_regime_features,
    causal_bull_proba,
    keep_if_bull,
    keep_unless_backwardation,
)
from utils.research.cta_trend import load_macro_closes  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.panel import load_spy_close  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_hmm_vix")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)

WINNER_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame) -> dict:
    g = pd.to_numeric(df["gain_pct"], errors="coerce")
    years = summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS)
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")]
    y23 = df.loc[_in_range(df, "2023-01-01", "2023-12-31")]
    s22 = book_stats(pd.to_numeric(y22["gain_pct"], errors="coerce"))
    s23 = book_stats(pd.to_numeric(y23["gain_pct"], errors="coerce"))
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")]
    s223 = book_stats(pd.to_numeric(both["gain_pct"], errors="coerce"))
    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s | 2022 n=%d E=%s PF=%s | 2023 n=%d E=%s PF=%s | 2022-23 E=%s PF=%s",
        label,
        s["n"],
        "n/a" if s["expectancy_pct"] is None else "%.3f" % s["expectancy_pct"],
        "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
        "n/a" if geo is None else "%.3f" % geo,
        s22["n"],
        "n/a" if s22["expectancy_pct"] is None else "%.3f" % s22["expectancy_pct"],
        "n/a" if s22["profit_factor"] is None else "%.3f" % s22["profit_factor"],
        s23["n"],
        "n/a" if s23["expectancy_pct"] is None else "%.3f" % s23["expectancy_pct"],
        "n/a" if s23["profit_factor"] is None else "%.3f" % s23["profit_factor"],
        "n/a" if s223["expectancy_pct"] is None else "%.3f" % s223["expectancy_pct"],
        "n/a" if s223["profit_factor"] is None else "%.3f" % s223["profit_factor"],
    )
    print("---- %s ----" % label)
    print(years.to_string(index=False))
    return {
        "label": label,
        "n": s["n"],
        "expectancy_pct": s["expectancy_pct"],
        "profit_factor": s["profit_factor"],
        "geo_year_pf": geo,
        "n_2022": s22["n"],
        "e_2022": s22["expectancy_pct"],
        "pf_2022": s22["profit_factor"],
        "n_2023": s23["n"],
        "e_2023": s23["expectancy_pct"],
        "pf_2023": s23["profit_factor"],
        "n_2022_23": s223["n"],
        "e_2022_23": s223["expectancy_pct"],
        "pf_2022_23": s223["profit_factor"],
    }


def _winner_windows(trades: pd.DataFrame, keep: pd.Series, rec: dict) -> None:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    for name, start, end in WINNER_WINDOWS:
        m = _in_range(trades, start, end)
        cut = winner_cut_skip(g.loc[m], keep.loc[m])
        rec["cut_%s_winners" % name] = cut["winners_dropped"]
        rec["cut_%s_losers" % name] = cut["losers_dropped"]
        rec["cut_%s_win_sum" % name] = cut["winner_drop_gain_sum"]
        rec["cut_%s_lose_sum" % name] = cut["loser_drop_gain_sum"]
        rec["cut_%s_blunt" % name] = cut["blunt"]
        print(
            "  winner-cut %s dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
            % (
                name,
                cut["n_dropped"],
                cut["winners_dropped"],
                cut["losers_dropped"],
                cut["winner_drop_gain_sum"],
                cut["loser_drop_gain_sum"],
                cut["blunt"],
            )
        )


def _score_keep(
    label: str,
    trades: pd.DataFrame,
    keep: pd.Series,
    baseline_gain: pd.Series,
    rows: List[dict],
) -> None:
    keep = keep.reindex(trades.index).fillna(True).astype(bool)
    part = trades.loc[keep].copy()
    rec = _print_block(label, part)
    cut = winner_cut_skip(baseline_gain, keep)
    rec.update({("cut_" + k): v for k, v in cut.items()})
    print(
        "  winner-cut ALL dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
        % (
            cut["n_dropped"],
            cut["winners_dropped"],
            cut["losers_dropped"],
            cut["winner_drop_gain_sum"],
            cut["loser_drop_gain_sum"],
            cut["blunt"],
        )
    )
    _winner_windows(trades, keep, rec)
    rec["n_skipped"] = int((~keep).sum())
    rows.append(rec)


def _load_panels(start: datetime, end: datetime):
    spy = load_spy_close(start, end, preferred_provider="IB", fallback_provider="ALPACA")
    macros = load_macro_closes(
        ["VIX", "VIX3M", "TLT", "GLD", "UUP"],
        start=start,
        end=end,
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    vix = macros.get("VIX")
    vix3m = macros.get("VIX3M")
    if vix is None or vix.empty or vix3m is None or vix3m.empty:
        raise RuntimeError("Need IB (or Alpaca) daily VIX and VIX3M in TimescaleDB")
    LOG.info(
        "SPY n=%d %s->%s | VIX n=%d | VIX3M n=%d | TLT=%s GLD=%s UUP=%s",
        len(spy),
        spy.index.min().date(),
        spy.index.max().date(),
        len(vix),
        len(vix3m),
        "yes" if macros.get("TLT") is not None and not macros["TLT"].empty else "no",
        "yes" if macros.get("GLD") is not None and not macros["GLD"].empty else "no",
        "yes" if macros.get("UUP") is not None and not macros["UUP"].empty else "no",
    )
    return spy, vix, vix3m, macros


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--min-train", type=int, default=252)
    ap.add_argument("--refit-every", type=int, default=21)
    ap.add_argument("--hmm-iter", type=int, default=12)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    rows: List[dict] = []
    rows.append(_print_block("baseline_last15m", trades))

    start = datetime(2017, 1, 1)
    end = datetime(2026, 9, 12)
    spy, vix, vix3m, macros = _load_panels(start, end)
    feat = build_regime_features(
        spy,
        vix,
        vix3m,
        macros={k: macros[k] for k in ("TLT", "GLD", "UUP") if k in macros},
    )
    LOG.info("regime features %s -> %s n=%d", feat.index.min().date(), feat.index.max().date(), len(feat))

    tagged = attach_prior_session_regime(trades, feat)
    _score_keep(
        "VIX_back",
        tagged,
        keep_unless_backwardation(tagged, back_thresh=1.0),
        baseline_gain,
        rows,
    )
    _score_keep(
        "VIX_back_min20",
        tagged,
        keep_unless_backwardation(tagged, back_thresh=1.0, min_vix=20.0),
        baseline_gain,
        rows,
    )

    LOG.info("Fitting causal HMM K=2 core...")
    hmm2 = causal_bull_proba(
        feat,
        n_states=2,
        feature_cols=CORE_FEATURE_COLS,
        min_train=args.min_train,
        refit_every=args.refit_every,
        n_iter=args.hmm_iter,
        seed=0,
    )
    tagged2 = attach_prior_session_regime(trades, feat, p_bull=hmm2["p_bull"])
    _score_keep("HMM2_p50", tagged2, keep_if_bull(tagged2, min_p_bull=0.50), baseline_gain, rows)
    _score_keep("HMM2_p65", tagged2, keep_if_bull(tagged2, min_p_bull=0.65), baseline_gain, rows)

    LOG.info("Fitting causal HMM K=3 core...")
    hmm3 = causal_bull_proba(
        feat,
        n_states=3,
        feature_cols=CORE_FEATURE_COLS,
        min_train=args.min_train,
        refit_every=args.refit_every,
        n_iter=args.hmm_iter,
        seed=0,
    )
    tagged3 = attach_prior_session_regime(trades, feat, p_bull=hmm3["p_bull"])
    _score_keep("HMM3_p50", tagged3, keep_if_bull(tagged3, min_p_bull=0.50), baseline_gain, rows)
    _score_keep("HMM3_p65", tagged3, keep_if_bull(tagged3, min_p_bull=0.65), baseline_gain, rows)

    macro_cols = list(CORE_FEATURE_COLS) + list(MACRO_FEATURE_COLS)
    if feat[list(MACRO_FEATURE_COLS)].notna().any().any():
        LOG.info("Fitting causal HMM K=2 core+macro...")
        hmm_m = causal_bull_proba(
            feat,
            n_states=2,
            feature_cols=macro_cols,
            min_train=args.min_train,
            refit_every=args.refit_every,
            n_iter=args.hmm_iter,
            seed=0,
        )
        tagged_m = attach_prior_session_regime(trades, feat, p_bull=hmm_m["p_bull"])
        _score_keep(
            "HMM2_macro_p50",
            tagged_m,
            keep_if_bull(tagged_m, min_p_bull=0.50),
            baseline_gain,
            rows,
        )
        tagged2 = tagged2.copy()
        tagged2["p_bull_macro_asof"] = tagged_m["p_bull_asof"]
    else:
        LOG.warning("No TLT/GLD/UUP coverage; skip HMM2_macro_p50")

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    path = outdir / "last_15m_hmm_vix_skip_summary.csv"
    summary.to_csv(path, index=False)
    tagged2.to_csv(outdir / "last_15m_hmm_vix_skip_trades.csv", index=False)
    hmm2.to_csv(outdir / "last_15m_hmm2_p_bull_daily.csv")
    hmm3.to_csv(outdir / "last_15m_hmm3_p_bull_daily.csv")
    lines = [
        "last-15m causal VIX/HMM skip (same-list, occupancy not re-walked, prior-session asof)"
    ]
    for rec in rows:
        lines.append(
            "%s n=%s E=%s PF=%s geo=%s skipped=%s 2022 n=%s E=%s PF=%s 2023 n=%s E=%s PF=%s 2022-23 E=%s PF=%s blunt=%s"
            % (
                rec.get("label"),
                rec.get("n"),
                rec.get("expectancy_pct"),
                rec.get("profit_factor"),
                rec.get("geo_year_pf"),
                rec.get("n_skipped"),
                rec.get("n_2022"),
                rec.get("e_2022"),
                rec.get("pf_2022"),
                rec.get("n_2023"),
                rec.get("e_2023"),
                rec.get("pf_2023"),
                rec.get("e_2022_23"),
                rec.get("pf_2022_23"),
                rec.get("cut_blunt"),
            )
        )
    txt = outdir / "last_15m_hmm_vix_skip_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", outdir)
    LOG.info("Wrote %s %s", path, txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
