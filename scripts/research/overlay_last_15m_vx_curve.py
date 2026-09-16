"""Causal VIX-curve slope skip on the last-15m 1d book.

Pine: ((VX2 / VX1) - 1) * 100 computed in Python from separate TV daily dumps
(CBOE_DLY:VX1! and VX2!), not a Pine plot. Skip if prior-session slope < 4.

  VX12_lt4    skip if prior (VX2/VX1 - 1)*100 < 4  (the Pine rule)
  VX60_lt4    cash analog (optional; --with-analog)
  VIX3M_lt4   90d cash analog (optional; --with-analog)

Same-list, occupancy not re-walked. Fill-day CFE close is illegal at 15:45 ET.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_vx_curve.py --vx1 reports\\ascending_channels\\2026-09-14\\tv_cboe_vx1_daily.json --vx2 reports\\ascending_channels\\2026-09-14\\tv_cboe_vx2_daily.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from overlay_last_15m_hmm_vix_skip import (  # noqa: E402
    DEFAULT_TRADES,
    _print_block,
    _score_keep,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.vx_futures_curve import (  # noqa: E402
    USER_SLOPE_MIN,
    attach_prior_session_slope,
    curve_slope_pct,
    keep_unless_slope_below,
    load_cboe_index_close,
    load_tv_dump_close,
    pine_band_label,
    vx60_slope_pct,
)

LOG = logging.getLogger("overlay_last_15m_vx_curve")


def _print_slope_diag(label: str, tagged: pd.DataFrame, col: str) -> None:
    s = pd.to_numeric(tagged[col], errors="coerce")
    print("---- %s slope asof ----" % label)
    print(s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print("  n_missing=%d  pct_lt_%g=%.1f" % (int(s.isna().sum()), USER_SLOPE_MIN, float((s < USER_SLOPE_MIN).mean() * 100.0)))
    bands = pine_band_label(s)
    print("  pine bands:\n%s" % bands.value_counts(dropna=False).to_string())
    d = pd.to_datetime(tagged["buy_date"], errors="coerce")
    for year in (2022, 2023):
        m = d.dt.year == year
        part = s.loc[m]
        if part.empty:
            continue
        print(
            "  %d n=%d median=%.2f pct_lt_%g=%.1f"
            % (
                year,
                int(part.notna().sum()),
                float(part.median()) if part.notna().any() else float("nan"),
                USER_SLOPE_MIN,
                float((part < USER_SLOPE_MIN).mean() * 100.0),
            )
        )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--vx1", type=Path, default=None, help="TV dump JSON for CBOE:VX1!")
    ap.add_argument("--vx2", type=Path, default=None, help="TV dump JSON for CBOE:VX2!")
    ap.add_argument("--with-analog", action="store_true", help="Also score cash VIX/VIX3M analogs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    baseline_gain = pd.to_numeric(trades["gain_pct"], errors="coerce")
    rows: List[dict] = []
    rows.append(_print_block("baseline_last15m", trades))

    tagged12 = None
    if args.vx1 is None or args.vx2 is None:
        raise SystemExit("Need --vx1 and --vx2 TV daily dumps (CBOE:VX1! and CBOE:VX2!), not a Pine ratio")
    vx1 = load_tv_dump_close(args.vx1, name="VX1")
    vx2 = load_tv_dump_close(args.vx2, name="VX2")
    LOG.info(
        "TV VX1 n=%d %s->%s | VX2 n=%d %s->%s",
        len(vx1),
        vx1.index.min().date(),
        vx1.index.max().date(),
        len(vx2),
        vx2.index.min().date(),
        vx2.index.max().date(),
    )
    slope12 = curve_slope_pct(vx1, vx2)
    tagged12 = attach_prior_session_slope(trades, slope12, col="vx_slope_asof")
    tagged12["vx1_asof"] = attach_prior_session_slope(trades, vx1, col="vx_slope_asof")["vx_slope_asof"]
    tagged12["vx2_asof"] = attach_prior_session_slope(trades, vx2, col="vx_slope_asof")["vx_slope_asof"]
    _print_slope_diag("VX12 (TV VX2/VX1)", tagged12, "vx_slope_asof")
    _score_keep(
        "VX12_lt4",
        tagged12,
        keep_unless_slope_below(tagged12, thresh=USER_SLOPE_MIN),
        baseline_gain,
        rows,
    )

    if args.with_analog:
        LOG.info("Downloading CBOE VIX + VIX3M daily closes for analog...")
        vix = load_cboe_index_close("VIX")
        vix3m = load_cboe_index_close("VIX3M")
        slope60 = vx60_slope_pct(vix, vix3m)
        slope3m = curve_slope_pct(vix, vix3m)
        tagged60 = attach_prior_session_slope(trades, slope60, col="vx_slope_asof")
        tagged3m = attach_prior_session_slope(trades, slope3m, col="vx_slope_asof")
        _print_slope_diag("VX60 (30d vs interp 60d)", tagged60, "vx_slope_asof")
        _print_slope_diag("VIX3M (spot vs 90d)", tagged3m, "vx_slope_asof")
        _score_keep(
            "VX60_lt4",
            tagged60,
            keep_unless_slope_below(tagged60, thresh=USER_SLOPE_MIN),
            baseline_gain,
            rows,
        )
        _score_keep(
            "VIX3M_lt4",
            tagged3m,
            keep_unless_slope_below(tagged3m, thresh=USER_SLOPE_MIN),
            baseline_gain,
            rows,
        )

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    path = outdir / "last_15m_vx_curve_skip_summary.csv"
    summary.to_csv(path, index=False)
    tagged12.to_csv(outdir / "last_15m_vx_curve_skip_trades.csv", index=False)
    slope12.to_csv(outdir / "tv_vx12_slope_daily.csv", header=True)
    lines = [
        "last-15m causal VX-curve slope skip (same-list, occupancy not re-walked, prior-session asof)",
        "Pine ((VX2/VX1)-1)*100 < 4 from separate TV daily dumps, ratio in Python",
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
    txt = outdir / "last_15m_vx_curve_skip_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", outdir)
    LOG.info("Wrote %s %s", path, txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
