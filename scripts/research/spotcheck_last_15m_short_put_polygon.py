"""Spot-check last-15m short-put BS hard-stops vs Polygon Options Basic (free).

Pulls daily option aggs for a small recent hard-stop sample and compares
live exit marks to constant-IV BS (and implied stop-IV multiple).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=5
  python scripts\\research\\spotcheck_last_15m_short_put_polygon.py --n 12
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "5")

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.short_put_polygon_spotcheck import (  # noqa: E402
    bars_from_aggs_response,
    build_spot_plan,
    closest_listed_expiry,
    pick_entry_exit_bars,
    polygon_option_ticker,
    select_hard_stop_sample,
)

LOG = logging.getLogger("spotcheck_short_put_polygon")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-26"
    / "last_15m_short_put_bs_dte45_atm_trades_20260926_213902.csv"
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-buy", type=str, default="2024-10-01")
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trades = pd.read_csv(args.trades)
    sample = select_hard_stop_sample(
        trades, n=int(args.n), min_buy=str(args.min_buy), min_adv=float(args.min_adv)
    )
    if sample.empty:
        LOG.error("No hard-stop rows in window min_buy=%s min_adv=%s", args.min_buy, args.min_adv)
        return 1
    LOG.info("Sample n=%d symbols=%s", len(sample), list(sample["stock"].astype(str).str.upper()))

    client = get_polygon_client()
    rows = []
    t0 = time.perf_counter()
    try:
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            plan = build_spot_plan(row)
            start = (plan.buy_date - timedelta(days=5)).isoformat()
            end = (plan.exit_date + timedelta(days=5)).isoformat()
            LOG.info(
                "[%d/%d] %s %s -> %s strike=%s ticker=%s",
                i,
                len(sample),
                plan.stock,
                plan.buy_date,
                plan.exit_date,
                plan.strike,
                plan.ticker,
            )
            try:
                resp = client.get_options_aggregates(plan.ticker, start, end, timespan="day")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("API error %s: %s", plan.ticker, exc)
                rec = {
                    "stock": plan.stock,
                    "buy_date": plan.buy_date.isoformat(),
                    "exit_date": plan.exit_date.isoformat(),
                    "strike": plan.strike,
                    "expiry": plan.expiry,
                    "ticker": plan.ticker,
                    "bs_credit": plan.bs_credit,
                    "bs_exit": plan.bs_exit,
                    "bs_gain_cs": plan.bs_gain_cs,
                    "status": "api_error",
                    "error": str(exc)[:200],
                }
                rows.append(rec)
                continue

            bars = bars_from_aggs_response(resp)
            # If synthetic DTE expiry has no prints, snap to nearest listed expiry.
            if bars.empty:
                target = plan.buy_date + timedelta(days=45)
                try:
                    exps = client.list_expirations(
                        plan.stock,
                        start=(plan.buy_date + timedelta(days=25)).isoformat(),
                        end=(plan.buy_date + timedelta(days=70)).isoformat(),
                        include_expired=True,
                        max_contracts=1200,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("list_expirations failed %s: %s", plan.stock, exc)
                    exps = []
                listed = closest_listed_expiry(exps, target)
                if listed and listed != plan.expiry:
                    plan.expiry = listed
                    plan.ticker = polygon_option_ticker(
                        plan.stock, listed, plan.strike, "P"
                    )
                    LOG.info("retry listed expiry %s ticker=%s", listed, plan.ticker)
                    try:
                        resp = client.get_options_aggregates(
                            plan.ticker, start, end, timespan="day"
                        )
                        bars = bars_from_aggs_response(resp)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("retry API error %s: %s", plan.ticker, exc)

            entry, exit_bar, status = pick_entry_exit_bars(
                bars, plan.buy_date, plan.exit_date
            )
            rec = {
                "stock": plan.stock,
                "buy_date": plan.buy_date.isoformat(),
                "exit_date": plan.exit_date.isoformat(),
                "strike": plan.strike,
                "expiry": plan.expiry,
                "ticker": plan.ticker,
                "bs_credit": plan.bs_credit,
                "bs_exit": plan.bs_exit,
                "bs_gain_cs": plan.bs_gain_cs,
                "n_bars": int(len(bars)),
                "status": status,
            }
            if entry is not None:
                rec["live_entry"] = float(entry["close"]) if entry.get("close") is not None else None
                rec["live_entry_date"] = entry["date"].isoformat()
                rec["live_entry_vol"] = entry.get("volume")
            if exit_bar is not None:
                rec["live_exit"] = float(exit_bar["close"]) if exit_bar.get("close") is not None else None
                rec["live_exit_date"] = exit_bar["date"].isoformat()
                rec["live_exit_vol"] = exit_bar.get("volume")
            if (
                rec.get("live_entry") is not None
                and rec.get("live_exit") is not None
                and plan.strike > 0
            ):
                live_cs = 100.0 * (rec["live_entry"] - rec["live_exit"]) / plan.strike
                rec["live_gain_cs"] = live_cs
                if plan.bs_exit > 1e-9:
                    rec["exit_vs_bs"] = rec["live_exit"] / plan.bs_exit
                if plan.bs_credit > 1e-9:
                    rec["entry_vs_bs"] = rec["live_entry"] / plan.bs_credit
                rec["delta_cs_live_minus_bs"] = live_cs - plan.bs_gain_cs
            rows.append(rec)
    finally:
        client.close()

    out = pd.DataFrame(rows)
    ok = out[out["status"] == "ok"].copy() if not out.empty else out
    print("---- sample statuses ----")
    if not out.empty:
        print(out["status"].value_counts().to_string())
    print("---- ok rows ----")
    cols = [
        "stock",
        "buy_date",
        "exit_date",
        "strike",
        "live_entry",
        "live_exit",
        "bs_credit",
        "bs_exit",
        "exit_vs_bs",
        "entry_vs_bs",
        "live_gain_cs",
        "bs_gain_cs",
        "delta_cs_live_minus_bs",
    ]
    show = [c for c in cols if c in (ok.columns if not ok.empty else [])]
    if not ok.empty:
        print(ok[show].to_string(index=False, float_format=lambda x: "%.3f" % x))
        print(
            "n_ok=%d median exit_vs_bs=%.3f mean exit_vs_bs=%.3f median live_cs=%.3f median bs_cs=%.3f"
            % (
                len(ok),
                float(ok["exit_vs_bs"].median()) if "exit_vs_bs" in ok else float("nan"),
                float(ok["exit_vs_bs"].mean()) if "exit_vs_bs" in ok else float("nan"),
                float(ok["live_gain_cs"].median()) if "live_gain_cs" in ok else float("nan"),
                float(ok["bs_gain_cs"].median()) if "bs_gain_cs" in ok else float("nan"),
            )
        )
        # Classify vs IV stress bands using exit mark ratio
        if "exit_vs_bs" in ok.columns:
            r = pd.to_numeric(ok["exit_vs_bs"], errors="coerce")
            print(
                "exit_vs_bs bands: <0.9=%d | 0.9-1.1=%d | 1.1-1.35=%d | >1.35=%d"
                % (
                    int((r < 0.9).sum()),
                    int(((r >= 0.9) & (r < 1.1)).sum()),
                    int(((r >= 1.1) & (r <= 1.35)).sum()),
                    int((r > 1.35).sum()),
                )
            )
    else:
        print("(none)")

    outdir = args.outdir or dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = outdir / ("last_15m_short_put_polygon_spot_%s.csv" % stamp)
    out.to_csv(path, index=False)
    LOG.info("Wrote %s wall=%.1fs", path, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
