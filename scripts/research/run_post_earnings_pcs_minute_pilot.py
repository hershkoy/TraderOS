#!/usr/bin/env python3
"""Starter minute-bar open/first-hour PCS pilot.

Reuses day-open ~20d strike picks from the open_delta cache when present,
pulls 1-minute option aggs for short+long on entry day, marks credit at
RTH open / 10:00 / first-hour VWAP / day close, then walks TP50/SL2x on
day closes.

Usage:
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=30
  python scripts\\research\\run_post_earnings_pcs_minute_pilot.py --n 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "30")

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.research.post_earnings_pcs_live import select_recent_pcs_sample  # noqa: E402
from utils.research.post_earnings_pcs_minute import (  # noqa: E402
    evaluate_minute_vertical,
    summarize_minute_pilot,
)
from utils.research.post_earnings_pcs_open_delta import evaluate_trade_open_close  # noqa: E402
from utils.research.short_put_polygon_spotcheck import bars_from_aggs_response  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("pcs_minute")
logging.getLogger("utils.api.polygon.polygon_client").setLevel(logging.WARNING)

DEFAULT_TRADES = (
    ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-29" / "trades.csv"
)
OPEN_DELTA_CACHE = (
    ROOT
    / "reports"
    / "research"
    / "post_earnings_pcs"
    / "2026-09-29_open_delta"
    / "open_delta_cache.jsonl"
)

LIQUID = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "NFLX",
    "AVGO",
    "COST",
    "JPM",
    "V",
    "MA",
    "HD",
    "UNH",
    "XOM",
    "WMT",
    "ADBE",
    "CRM",
    "ORCL",
    "INTC",
    "QCOM",
    "TXN",
    "IBM",
    "BA",
    "CAT",
    "GS",
}


def _key(sym: str, entry: Any) -> str:
    return "%s|%s" % (str(sym).upper(), pd.Timestamp(entry).strftime("%Y-%m-%d"))


def _load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
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


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _pick_vertical(
    row: pd.Series,
    *,
    od: Optional[Dict[str, Any]],
    fetch_day,
) -> Optional[Dict[str, Any]]:
    """Resolve short/long tickers + width from open_delta cache or day ladder."""
    if od and od.get("status") == "ok" and od.get("d20_short_ticker") and od.get(
        "d20_long_ticker"
    ):
        width = float(od.get("d20_short_k") or 0) - float(od.get("d20_long_k") or 0)
        if width <= 0:
            width = float(row.get("width") or 5.0)
        return {
            "short_ticker": od["d20_short_ticker"],
            "long_ticker": od["d20_long_ticker"],
            "width": width,
            "expiry": date.fromisoformat(str(od["d20_expiry"])[:10]),
            "source": "open_delta_cache",
            "d20_delta_open": od.get("d20_delta_open"),
            "d20_credit_open_day": od.get("d20_credit_open"),
        }

    obj = evaluate_trade_open_close(
        symbol=str(row["symbol"]).upper(),
        entry_date=pd.Timestamp(row["entry_date"]).date(),
        entry_px=float(row["entry_px"]),
        bs_short_k=float(row["short_k"]),
        bs_long_k=float(row["long_k"]),
        bs_credit=float(row["credit"]),
        width=float(row["width"]),
        dte=int(row["dte"]),
        fetch_bars=fetch_day,
    )
    if obj.get("status") != "ok":
        return None
    width = float(obj["d20_short_k"]) - float(obj["d20_long_k"])
    return {
        "short_ticker": obj["d20_short_ticker"],
        "long_ticker": obj["d20_long_ticker"],
        "width": width,
        "expiry": date.fromisoformat(str(obj["d20_expiry"])[:10]),
        "source": "day_ladder",
        "d20_delta_open": obj.get("d20_delta_open"),
        "d20_credit_open_day": obj.get("d20_credit_open"),
        "open_delta": obj,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-entry", default="2024-10-01")
    ap.add_argument("--max-entry", default="2026-09-28")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument(
        "--open-delta-cache",
        type=Path,
        default=OPEN_DELTA_CACHE,
        help="Reuse day-open 20d picks to save API calls",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    outdir = args.outdir or (
        ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-30_minute_pilot"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "minute_pilot_cache.jsonl"

    trades = pd.read_csv(args.trades)
    sample = select_recent_pcs_sample(
        trades, n=int(args.n) * 4, min_entry=str(args.min_entry)
    )
    sample = sample[sample["symbol"].astype(str).str.upper().isin(LIQUID)].copy()
    sample["entry_date"] = pd.to_datetime(sample["entry_date"])
    sample = sample[sample["entry_date"] <= pd.Timestamp(args.max_entry)]
    sample = sample.sort_values("entry_date", ascending=False).head(int(args.n))
    sample = sample.sort_values("entry_date").reset_index(drop=True)
    LOG.info("Sample n=%d liquids", len(sample))
    if sample.empty:
        LOG.error("Empty sample")
        return 1

    client = get_polygon_client()
    day_cache: Dict[str, pd.DataFrame] = {}
    min_cache: Dict[str, Dict[str, Any]] = {}

    def fetch_day(ticker: str, start: str, end: str) -> pd.DataFrame:
        ck = "%s|%s|%s|day" % (ticker, start, end)
        if ck in day_cache:
            return day_cache[ck]
        try:
            resp = client.get_options_aggregates(ticker, start, end, timespan="day")
            df = bars_from_aggs_response(resp if isinstance(resp, dict) else {})
        except Exception as exc:  # noqa: BLE001
            LOG.warning("day aggs fail %s: %s", ticker, exc)
            df = pd.DataFrame()
        day_cache[ck] = df
        return df

    def fetch_minute(ticker: str, day: str) -> Dict[str, Any]:
        ck = "%s|%s|min" % (ticker, day)
        if ck in min_cache:
            return min_cache[ck]
        try:
            resp = client.get_options_aggregates(ticker, day, day, timespan="minute")
            if not isinstance(resp, dict):
                resp = {}
        except Exception as exc:  # noqa: BLE001
            LOG.warning("minute aggs fail %s %s: %s", ticker, day, exc)
            resp = {}
        min_cache[ck] = resp
        return resp

    od_cache = _load_jsonl(args.open_delta_cache)
    cache = {} if args.refresh else _load_jsonl(cache_path)
    rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(sample.iterrows()):
        key = _key(row["symbol"], row["entry_date"])
        if key in cache and not args.refresh:
            rows.append(cache[key])
            LOG.info(
                "[%d/%d] cache %s %s",
                i + 1,
                len(sample),
                key,
                cache[key].get("status"),
            )
            continue

        LOG.info("[%d/%d] pick+minute %s ...", i + 1, len(sample), key)
        pick = _pick_vertical(row, od=od_cache.get(key), fetch_day=fetch_day)
        if pick is None:
            obj = {
                "key": key,
                "symbol": str(row["symbol"]).upper(),
                "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
                "status": "no_vertical_pick",
            }
            _append_jsonl(cache_path, obj)
            cache[key] = obj
            rows.append(obj)
            continue

        obj = evaluate_minute_vertical(
            symbol=str(row["symbol"]).upper(),
            entry_date=pd.Timestamp(row["entry_date"]).date(),
            short_ticker=pick["short_ticker"],
            long_ticker=pick["long_ticker"],
            width=float(pick["width"]),
            expiry=pick["expiry"],
            fetch_minute=fetch_minute,
            fetch_day=fetch_day,
        )
        obj["key"] = key
        obj["pick_source"] = pick["source"]
        obj["d20_delta_open"] = pick.get("d20_delta_open")
        obj["d20_credit_open_day"] = pick.get("d20_credit_open_day")
        obj["bs_credit"] = float(row["credit"])
        _append_jsonl(cache_path, obj)
        cache[key] = obj
        rows.append(obj)
        LOG.info(
            "  -> %s rth=%.3f t10=%.3f fh=%.3f close=%.3f meet_rth=%s walk_rth=%s",
            obj.get("status"),
            float(obj.get("credit_rth_open") or 0),
            float(obj.get("credit_t1000") or 0),
            float(obj.get("credit_first_hour_vwap") or 0),
            float(obj.get("credit_day_close") or 0),
            obj.get("meets_min_rth_open"),
            obj.get("walk_rth_open_exit"),
        )

    df = pd.DataFrame(rows)
    # Drop nested walks for flat CSV (kept in jsonl)
    flat = df.drop(columns=["walks"], errors="ignore")
    flat.to_csv(outdir / "minute_pilot.csv", index=False)

    summary = summarize_minute_pilot(rows)
    summary["wall_s"] = time.perf_counter() - t0
    summary["outdir"] = str(outdir)
    summary["note"] = (
        "Polygon Options Starter minute aggs; ~20d strikes from day-open ladder; "
        "exits on day closes TP50/SL2x; mid/trade prints (no 8pct half-spread -- "
        "that haircut zeroed open credits on expensive legs)."
    )
    (outdir / "minute_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    def _book_line(label: str, b: Dict[str, Any]) -> str:
        return (
            "%s: n=%s WR=%.0f%% hold=%.1fd RoR=%.2f%% PF=%s tp=%.0f%% sl=%.0f%%"
            % (
                label,
                b.get("n"),
                b.get("win_rate_pct") or 0,
                b.get("avg_hold_days") or 0,
                b.get("avg_ror_pct") or 0,
                (
                    ("%.2f" % b["profit_factor"])
                    if isinstance(b.get("profit_factor"), (int, float))
                    else b.get("profit_factor")
                ),
                b.get("tp_pct") or 0,
                b.get("sl_pct") or 0,
            )
        )

    lines = [
        "Post-earnings PCS minute open/first-hour pilot (Starter)",
        "attempted=%d ok=%d statuses=%s"
        % (summary["attempted"], summary["ok_n"], summary["status_counts"]),
        "Credits med (meet min%%): rth=%s (%.0f%%) t1000=%s (%.0f%%) fh=%s (%.0f%%) close=%s (%.0f%%)"
        % (
            (summary.get("credit_rth_open") or {}).get("median"),
            (summary.get("credit_rth_open") or {}).get("meets_min_pct") or 0,
            (summary.get("credit_t1000") or {}).get("median"),
            (summary.get("credit_t1000") or {}).get("meets_min_pct") or 0,
            (summary.get("credit_first_hour_vwap") or {}).get("median"),
            (summary.get("credit_first_hour_vwap") or {}).get("meets_min_pct") or 0,
            (summary.get("credit_day_close") or {}).get("median"),
            (summary.get("credit_day_close") or {}).get("meets_min_pct") or 0,
        ),
        "RTH open > day close credit %%=%s; med open/close ratio=%s"
        % (
            summary.get("rth_open_gt_day_close_pct"),
            summary.get("rth_open_vs_day_close_med"),
        ),
        _book_line("Book RTH open", summary.get("book_rth_open") or {}),
        _book_line("Book 10:00 ET", summary.get("book_t1000") or {}),
        _book_line("Book first-hour VWAP", summary.get("book_first_hour_vwap") or {}),
        _book_line("Book day close", summary.get("book_day_close") or {}),
        "wall=%.1fs out=%s" % (summary["wall_s"], outdir),
        summary["note"],
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "minute_pilot_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("\n%s", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
