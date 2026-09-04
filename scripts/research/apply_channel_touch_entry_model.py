"""Apply a time-split entry scorer then RS-top1 (no future rows in the fit)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    apply_friction,
    filter_trades,
    select_same_day_rs,
    summarize_by_year,
    _summarize,
)
from utils.research.channel_touch_entry_model import (  # noqa: E402
    DEFAULT_CUTOFF,
    attach_scores,
    fit_and_eval,
    scored_to_row,
    time_split,
    trade_metrics,
    walk_forward,
)
from utils.research.report_paths import dated_outdir


def pack(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    out = filter_trades(
        df,
        require_in_channel=True,
        max_channel_span_days=10.0,
        max_beyond_width=0.25,
    )
    out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=1)
    return apply_friction(out, friction)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--friction-pct", type=float, default=0.10)
    ap.add_argument("--kind", choices=("logistic", "mlp"), default="mlp")
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)

    raw = pd.read_csv(args.raw)
    raw["buy_date"] = pd.to_datetime(raw["buy_date"])
    if "gain_pct_net" not in raw.columns and args.friction_pct:
        raw = apply_friction(raw, args.friction_pct)

    scored, thr = attach_scores(raw, cutoff=args.cutoff, model_kind=args.kind)
    print("threshold", round(thr, 4), "pass_frac", float(scored["entry_pass"].mean()))

    rows = []
    base = pack(raw, args.friction_pct)
    rows.append({"name": "BASE keeper", **_summarize(base, gain_col="gain_pct_net")})
    kept_raw = scored.loc[scored["entry_pass"]].copy()
    model_then_rs = pack(kept_raw, args.friction_pct)
    rows.append({"name": f"{args.kind} then RS", **_summarize(model_then_rs, gain_col="gain_pct_net")})

    tr, te = time_split(scored, args.cutoff)
    for label, part in (("train", tr), ("test", te)):
        gcol = "gain_pct_net" if "gain_pct_net" in part.columns else "gain_pct"
        rows.append({"name": f"{label} all", **trade_metrics(part[gcol].to_numpy())})
        passed = part.loc[part["entry_pass"]]
        rows.append({"name": f"{label} model-pass", **trade_metrics(passed[gcol].to_numpy())})
        packed = pack(passed, args.friction_pct)
        rows.append({"name": f"{label} model+keeper", **_summarize(packed, gain_col="gain_pct_net")})
        print(f"\n==== {label} model+keeper years ====")
        print(summarize_by_year(packed, gain_col="gain_pct_net").to_string(index=False))

    print("\n==== walk-forward ====")
    wf_rows = [scored_to_row(s) for s in walk_forward(raw, model_kind=args.kind)]
    wf = pd.DataFrame(wf_rows)
    print(wf.to_string(index=False) if not wf.empty else "(none)")

    split = fit_and_eval(raw, cutoff=args.cutoff, model_kind=args.kind)
    print("\n==== holdout", args.cutoff, "====")
    print(pd.DataFrame([scored_to_row(split)]).to_string(index=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / f"{args.raw.stem}_{args.kind}_scored.csv"
    scored.to_csv(dest, index=False)
    summary = args.outdir / f"{args.raw.stem}_{args.kind}_model_ab.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    wf_path = args.outdir / f"{args.raw.stem}_{args.kind}_walkforward.csv"
    wf.to_csv(wf_path, index=False)
    print("wrote", dest)
    print("wrote", summary)
    print("wrote", wf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
