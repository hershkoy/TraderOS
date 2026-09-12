"""2022-23 diagnosis and causal regime skips on the last-15m book.

Pre-registered (same-list, occupancy not re-walked):
  H1 prior-session SPY close > SMA200
  H2 prior-session SPY 20d return > 0
  H3 stock SMA50 > SMA200 (stored fill-day snapshot; last-15m is 15:45 ET)
  H4 name RS vs SPY 63d > 0 (same snapshot)

Also scores 15m L3 wait-12 as an alternate playbook in the same years.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_2022_23_regime.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_entry_features import enrich_spy_entry_features  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_2022_23")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
L3_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_l3_wait12.csv"


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
    LOG.info(
        "%s n=%d E=%.3f PF=%s geo=%.3f | 2022 n=%d E=%s PF=%s | 2023 n=%d E=%s PF=%s",
        label,
        s["n"],
        s["expectancy_pct"] if s["expectancy_pct"] is not None else float("nan"),
        "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
        geo if geo is not None else float("nan"),
        s22["n"],
        "n/a" if s22["expectancy_pct"] is None else "%.3f" % s22["expectancy_pct"],
        "n/a" if s22["profit_factor"] is None else "%.3f" % s22["profit_factor"],
        s23["n"],
        "n/a" if s23["expectancy_pct"] is None else "%.3f" % s23["expectancy_pct"],
        "n/a" if s23["profit_factor"] is None else "%.3f" % s23["profit_factor"],
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
        "n_2022_23": int(_in_range(df, "2022-01-01", "2023-12-31").sum()),
        "e_2022_23": book_stats(
            pd.to_numeric(
                df.loc[_in_range(df, "2022-01-01", "2023-12-31"), "gain_pct"],
                errors="coerce",
            )
        )["expectancy_pct"],
        "pf_2022_23": book_stats(
            pd.to_numeric(
                df.loc[_in_range(df, "2022-01-01", "2023-12-31"), "gain_pct"],
                errors="coerce",
            )
        )["profit_factor"],
    }


def _load_spy() -> pd.DataFrame:
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2017, 1, 1),
        end=datetime(2026, 9, 12),
        fallback_provider="IB",
        merge_mode="prefix",
        workers=1,
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        raise RuntimeError("SPY panel missing")
    LOG.info("SPY %s -> %s n=%d", spy.index.min(), spy.index.max(), len(spy))
    return spy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--l3-trades", type=Path, default=L3_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    rows: List[dict] = []
    rows.append(_print_block("baseline_last15m", trades))

    spy = _load_spy()
    trades = enrich_spy_entry_features(trades, spy)

    both = _in_range(trades, "2022-01-01", "2023-12-31")
    print("\n==== prior-session SPY mix in 2022-23 ====")
    sub = trades.loc[both]
    for col in ("spy_above_sma50", "spy_above_sma200"):
        vc = pd.to_numeric(sub[col], errors="coerce").value_counts(dropna=False)
        print(col, vc.to_dict())
    ret = pd.to_numeric(sub["spy_ret_20d"], errors="coerce")
    print(
        "spy_ret_20d 2022-23 mean=%.3f med=%.3f pct>0=%.1f"
        % (ret.mean(), ret.median(), 100.0 * (ret > 0).mean())
    )

    filters: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("H1_spy_above_sma200", lambda d: pd.to_numeric(d["spy_above_sma200"], errors="coerce") == 1),
        ("H2_spy_ret20_gt0", lambda d: pd.to_numeric(d["spy_ret_20d"], errors="coerce") > 0),
        (
            "H3_stock_sma50_gt_sma200",
            lambda d: pd.to_numeric(d["sma50_gt_sma200"], errors="coerce") == 1,
        ),
        (
            "H4_rs_spy_63d_gt0",
            lambda d: pd.to_numeric(d["rs_spy_63d"], errors="coerce") > 0,
        ),
        (
            "H1_and_H2",
            lambda d: (pd.to_numeric(d["spy_above_sma200"], errors="coerce") == 1)
            & (pd.to_numeric(d["spy_ret_20d"], errors="coerce") > 0),
        ),
    ]
    for label, fn in filters:
        keep = fn(trades).fillna(False)
        part = trades.loc[keep].copy()
        rec = _print_block(label, part)
        cut = winner_cut_skip(baseline_gain, keep)
        rec.update({("cut_" + k): v for k, v in cut.items()})
        print(
            "  winner-cut dropped=%s winners=%s losers=%s win$=%s lose$=%s blunt=%s"
            % (
                cut["n_dropped"],
                cut["winners_dropped"],
                cut["losers_dropped"],
                cut["winner_drop_gain_sum"],
                cut["loser_drop_gain_sum"],
                cut["blunt"],
            )
        )
        # 2022-23-only skip stats (did we make the sleeve itself green?)
        keep_b = keep & both
        sleeve = trades.loc[keep_b]
        ss = book_stats(pd.to_numeric(sleeve["gain_pct"], errors="coerce"))
        rec["sleeve_2022_23_n"] = ss["n"]
        rec["sleeve_2022_23_e"] = ss["expectancy_pct"]
        rec["sleeve_2022_23_pf"] = ss["profit_factor"]
        rows.append(rec)

    if args.l3_trades.is_file():
        l3 = pd.read_csv(args.l3_trades)
        rec = _print_block("ALT_15m_L3_wait12", l3)
        rec["cut_n_dropped"] = None
        rows.append(rec)
    else:
        LOG.warning("No 15m L3 CSV at %s", args.l3_trades)

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    path = outdir / "last_15m_2022_23_regime_summary.csv"
    out.to_csv(path, index=False)
    lines = ["last-15m 2022-23 regime overlays (same-list, occupancy not re-walked)"]
    for rec in rows:
        lines.append(
            "%s n=%s E=%s PF=%s geo=%s 2022 n=%s E=%s PF=%s 2023 n=%s E=%s PF=%s 2022-23 E=%s PF=%s"
            % (
                rec.get("label"),
                rec.get("n"),
                rec.get("expectancy_pct"),
                rec.get("profit_factor"),
                rec.get("geo_year_pf"),
                rec.get("n_2022"),
                rec.get("e_2022"),
                rec.get("pf_2022"),
                rec.get("n_2023"),
                rec.get("e_2023"),
                rec.get("pf_2023"),
                rec.get("e_2022_23"),
                rec.get("pf_2022_23"),
            )
        )
    txt = outdir / "last_15m_2022_23_regime_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", outdir)
    LOG.info("Wrote %s %s", path, txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
