"""Time-split logistic + tiny MLP on channel-touch entry features.

Train rows are strictly earlier buy_dates than test. Features are the entry
bar snapshot only.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\train_channel_touch_entry_model.py --trades reports\\ascending_channels\\channel_touch_15m_trades_raw_STAMP.csv --friction-pct 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import apply_friction  # noqa: E402
from utils.research.channel_touch_entry_model import (  # noqa: E402
    DEFAULT_CUTOFF,
    fit_and_eval,
    scored_to_row,
    walk_forward,
)
from utils.research.report_paths import dated_outdir


def main() -> int:
    ap = argparse.ArgumentParser(description="Time-split entry scorer (no future features)")
    ap.add_argument("--trades", required=True, type=Path)
    ap.add_argument("--friction-pct", type=float, default=0.10)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)

    df = pd.read_csv(args.trades)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    if "gain_pct_net" not in df.columns and "gain_pct" in df.columns and args.friction_pct:
        df = apply_friction(df, args.friction_pct)

    rows = []
    for kind in ("logistic", "mlp"):
        split = fit_and_eval(df, cutoff=args.cutoff, model_kind=kind)
        split.name = f"{kind} cutoff={args.cutoff}"
        rows.append(scored_to_row(split))
        for wf in walk_forward(df, model_kind=kind):
            rows.append(scored_to_row(wf))

    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / f"{args.trades.stem}_entry_model.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
