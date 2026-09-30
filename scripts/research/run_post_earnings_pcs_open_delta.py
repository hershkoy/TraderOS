#!/usr/bin/env python3
"""Open-vs-close + mid-implied 20-delta vertical spot-check (Polygon day aggs).

Answers: was >=$0.80 credit available at the open (vs close) on liquid earns,
and how does that compare to the BS claim-check credit?

Usage:
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=4
  python scripts\\research\\run_post_earnings_pcs_open_delta.py --n 12
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "4")

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.research.post_earnings_pcs_live import select_recent_pcs_sample  # noqa: E402
from utils.research.post_earnings_pcs_open_delta import evaluate_trade_open_close  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.short_put_polygon_spotcheck import bars_from_aggs_response  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("pcs_open_delta")
logging.getLogger("utils.api.polygon.polygon_client").setLevel(logging.WARNING)

DEFAULT_TRADES = (
    ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-29" / "trades.csv"
)

# Prefer mega-liquid names for this probe
LIQUID = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "AVGO", "COST", "JPM", "V", "MA", "HD", "UNH", "XOM", "WMT",
}


def _key(sym: str, entry: Any) -> str:
    return "%s|%s" % (str(sym).upper(), pd.Timestamp(entry).strftime("%Y-%m-%d"))


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[obj.get("key") or _key(obj.get("symbol"), obj.get("entry_date"))] = obj
    return out


def _append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-entry", default="2024-10-01")
    ap.add_argument("--max-entry", default="2025-12-31", help="Avoid thin 2026 free-tier gaps")
    ap.add_argument("--ladder-n", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    outdir = args.outdir or dated_outdir(
        ROOT / "reports" / "research" / "post_earnings_pcs"
    )
    # force a clear folder name if dated
    if args.outdir is None:
        outdir = ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-29_open_delta"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "open_delta_cache.jsonl"

    trades = pd.read_csv(args.trades)
    sample = select_recent_pcs_sample(
        trades, n=int(args.n) * 3, min_entry=str(args.min_entry)
    )
    sample = sample[sample["symbol"].astype(str).str.upper().isin(LIQUID)].copy()
    sample["entry_date"] = pd.to_datetime(sample["entry_date"])
    sample = sample[sample["entry_date"] <= pd.Timestamp(args.max_entry)]
    sample = sample.sort_values("entry_date").head(int(args.n)).reset_index(drop=True)
    LOG.info("Sample n=%d liquids", len(sample))
    if sample.empty:
        LOG.error("Empty sample")
        return 1

    client = get_polygon_client()
    bar_cache: Dict[str, pd.DataFrame] = {}

    def fetch_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
        ck = "%s|%s|%s" % (ticker, start, end)
        if ck in bar_cache:
            return bar_cache[ck]
        try:
            resp = client.get_options_aggregates(ticker, start, end, timespan="day")
            df = bars_from_aggs_response(resp if isinstance(resp, dict) else {})
        except Exception as exc:  # noqa: BLE001
            LOG.warning("aggs fail %s: %s", ticker, exc)
            df = pd.DataFrame()
        bar_cache[ck] = df
        return df

    # Patch ladder size via env of evaluate - pass by wrapping strike_ladder
    import utils.research.post_earnings_pcs_open_delta as mod

    _orig = mod.strike_ladder

    def _ladder(px: float, *, n: int = 10):
        return _orig(px, n=int(args.ladder_n))

    mod.strike_ladder = _ladder  # type: ignore

    cache = {} if args.refresh else _load_cache(cache_path)
    rows: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(sample.iterrows()):
        key = _key(row["symbol"], row["entry_date"])
        if key in cache and not args.refresh:
            rows.append(cache[key])
            LOG.info("[%d/%d] cache %s %s", i + 1, len(sample), key, cache[key].get("status"))
            continue
        LOG.info("[%d/%d] eval %s ...", i + 1, len(sample), key)
        obj = evaluate_trade_open_close(
            symbol=str(row["symbol"]).upper(),
            entry_date=pd.Timestamp(row["entry_date"]).date(),
            entry_px=float(row["entry_px"]),
            bs_short_k=float(row["short_k"]),
            bs_long_k=float(row["long_k"]),
            bs_credit=float(row["credit"]),
            width=float(row["width"]),
            dte=int(row["dte"]),
            fetch_bars=fetch_bars,
        )
        obj["key"] = key
        _append_cache(cache_path, obj)
        cache[key] = obj
        rows.append(obj)
        LOG.info(
            "  -> %s open_bs=%.3f close_bs=%.3f d20_open=%.3f d20_close=%.3f meet_open=%s",
            obj.get("status"),
            float(obj.get("bs_strike_credit_open") or 0),
            float(obj.get("bs_strike_credit_close") or 0),
            float(obj.get("d20_credit_open") or 0),
            float(obj.get("d20_credit_close") or 0),
            obj.get("d20_open_meets_min"),
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "open_delta_spotcheck.csv", index=False)

    ok = df[df["status"] == "ok"].copy() if not df.empty else df

    def _summ(col: str) -> Dict[str, Any]:
        s = pd.to_numeric(ok[col], errors="coerce").dropna() if col in ok.columns else pd.Series(dtype=float)
        if s.empty:
            return {"n": 0}
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    summary = {
        "attempted": len(rows),
        "status_counts": df["status"].value_counts().to_dict() if not df.empty else {},
        "ok_n": int(len(ok)),
        "bs_strike_credit_open": _summ("bs_strike_credit_open"),
        "bs_strike_credit_close": _summ("bs_strike_credit_close"),
        "d20_credit_open": _summ("d20_credit_open"),
        "d20_credit_close": _summ("d20_credit_close"),
        "bs_credit": _summ("bs_credit"),
        "open_vs_bs_d20": _summ("d20_open_vs_bs"),
        "close_vs_bs_d20": _summ("d20_close_vs_bs"),
        "open_vs_bs_strike": _summ("bs_strike_open_vs_bs"),
        "close_vs_bs_strike": _summ("bs_strike_close_vs_bs"),
        "d20_open_meets_min_pct": (
            100.0 * float(ok["d20_open_meets_min"].astype(bool).mean())
            if len(ok) and "d20_open_meets_min" in ok.columns
            else None
        ),
        "d20_close_meets_min_pct": (
            100.0 * float(ok["d20_close_meets_min"].astype(bool).mean())
            if len(ok) and "d20_close_meets_min" in ok.columns
            else None
        ),
        "open_gt_close_pct": (
            100.0
            * float(
                (
                    pd.to_numeric(ok["d20_credit_open"], errors="coerce")
                    > pd.to_numeric(ok["d20_credit_close"], errors="coerce")
                ).mean()
            )
            if len(ok)
            else None
        ),
        "wall_s": time.perf_counter() - t0,
        "outdir": str(outdir),
        "note": "Historical option snapshots 403 on free tier; delta from mid-implied BS IV on day open.",
    }
    (outdir / "open_delta_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "Post-earnings PCS open-vs-close + mid-implied 20d spot-check",
        "attempted=%d ok=%d statuses=%s"
        % (len(rows), len(ok), summary["status_counts"]),
        "BS model credit: med=%s mean=%s" % (
            summary["bs_credit"].get("median"),
            summary["bs_credit"].get("mean"),
        ),
        "BS-strike listed open/close credit: med %s / %s (vs BS med ratios %s / %s)"
        % (
            summary["bs_strike_credit_open"].get("median"),
            summary["bs_strike_credit_close"].get("median"),
            summary["open_vs_bs_strike"].get("median"),
            summary["close_vs_bs_strike"].get("median"),
        ),
        "Implied-20d listed open/close credit: med %s / %s (vs BS med ratios %s / %s)"
        % (
            summary["d20_credit_open"].get("median"),
            summary["d20_credit_close"].get("median"),
            summary["open_vs_bs_d20"].get("median"),
            summary["close_vs_bs_d20"].get("median"),
        ),
        "Meet min credit (0.16*width) open%%=%s close%%=%s"
        % (summary["d20_open_meets_min_pct"], summary["d20_close_meets_min_pct"]),
        "Open>close credit %%=%s" % summary["open_gt_close_pct"],
        "wall=%.1fs out=%s" % (summary["wall_s"], outdir),
        summary["note"],
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "open_delta_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("\n%s", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
