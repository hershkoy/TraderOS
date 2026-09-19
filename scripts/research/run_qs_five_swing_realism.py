#!/usr/bin/env python3
"""QS five-swing fill realism suite (SPY daily + 5m/15m).

Compares:
  - oracle_moc: signal+fill settled close (claim-check)
  - proxy_HHMM: arm on developing RTH OHLC at clock; fill settled close
  - proxy_HHMM_buf: same with IBS buffer (default 0.05)
  - next_open: settled signal; fill next open
  - oracle_fric_*: oracle with round-turn friction

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_qs_five_swing_realism.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_qs_five_swing_realism.py --intraday-tf 15m --start 2018-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.generate_deep_pullback_report import (  # noqa: E402
    YAHOO_CACHE,
    load_yahoo,
)
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.qs_five_swing_realism import (  # noqa: E402
    PROXY_CLOCKS_1M_MOC,
    PROXY_CLOCKS_5M,
    run_moc_1m_suite,
    run_realism_suite,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_qs_five_swing_realism")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "qs_five_swing_realism"


def load_intraday(
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    client = get_timescaledb_client()
    sql = (
        "SELECT ts, open, high, low, close, volume FROM market_data "
        "WHERE symbol=%s AND timeframe=%s AND ts >= %s"
    )
    params: List[Any] = [symbol, timeframe, start]
    if end:
        sql += " AND ts <= %s"
        params.append(end)
    sql += " ORDER BY ts"
    rows = client.execute_query(sql, tuple(params))
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _slim(book: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"trades", "equity_monthly", "bh_monthly", "by_year", "year_buckets"}
    return {k: v for k, v in book.items() if k not in skip}


def _row(book: Dict[str, Any]) -> Dict[str, Any]:
    agr = book.get("agreement") or {}
    return {
        "book": book.get("book"),
        "n": book.get("n"),
        "E": book.get("avg_gain_pct"),
        "PF": book.get("profit_factor"),
        "WR": book.get("win_rate_pct"),
        "CAGR": book.get("cagr_full_window_pct"),
        "exp": book.get("exposure_pct"),
        "MDD": book.get("mdd_mtm_pct"),
        "final_100k": book.get("final_equity_100k"),
        "precision": agr.get("precision"),
        "recall": agr.get("recall"),
        "proxy_only": agr.get("proxy_only_false_pos"),
        "oracle_only": agr.get("oracle_only_miss"),
        "no_intraday": agr.get("no_intraday"),
        "clock": book.get("decision_clock"),
        "ibs_buffer": book.get("ibs_buffer"),
        "friction": book.get("friction_pct"),
    }


def render_html(
    *,
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    title: str,
) -> str:
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>"
            + "".join(
                "<td>%s</td>"
                % (
                    "-"
                    if r.get(k) is None
                    else (
                        "%.3f" % r[k]
                        if isinstance(r.get(k), float)
                        else str(r.get(k))
                    )
                )
                for k in (
                    "book",
                    "n",
                    "E",
                    "PF",
                    "CAGR",
                    "exp",
                    "MDD",
                    "final_100k",
                    "precision",
                    "recall",
                    "proxy_only",
                    "oracle_only",
                )
            )
            + "</tr>"
        )
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>%s</title>
<style>
body{font-family:Segoe UI,sans-serif;background:#0f1419;color:#e7ecf3;margin:24px}
table{border-collapse:collapse;width:100%%;font-size:0.9rem}
th,td{border-bottom:1px solid #2a3548;padding:6px 8px;text-align:left}
th{color:#9aa7b8} .muted{color:#9aa7b8}
</style></head><body>
<h1>%s</h1>
<p class="muted">%s | daily %s -> %s | intra %s %s bars %s -> %s | wall %.1fs</p>
<table><thead><tr>
<th>book</th><th>n</th><th>E</th><th>PF</th><th>CAGR</th><th>exp</th><th>MDD</th>
<th>$100k</th><th>prec</th><th>recall</th><th>false+</th><th>miss</th>
</tr></thead><tbody>%s</tbody></table>
<p class="muted">Proxy: arm on developing OHLC at clock, fill settled close (MOC).
Next-open: settled signal, fill next open. Exit always settled close &gt; prior high.</p>
</body></html>""" % (
        title,
        title,
        meta.get("generated"),
        meta.get("daily_first"),
        meta.get("daily_last"),
        meta.get("intraday_tf"),
        meta.get("intraday_n"),
        meta.get("intra_first"),
        meta.get("intra_last"),
        meta.get("wall_s", 0),
        "\n".join(body_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="QS five-swing fill realism")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--intraday-tf", choices=("1m", "5m", "15m"), default="5m")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--ibs-buffer", type=float, default=0.05)
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument(
        "--clocks",
        default="",
        help="ET clocks; default 15:45,15:50,15:55 for 5m; 15:57,15:58 for 1m",
    )
    ap.add_argument(
        "--with-5m-baseline",
        action="store_true",
        help="With --intraday-tf 1m also include proxy_1555_5m baseline",
    )
    args = ap.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    yahoo = load_yahoo(YAHOO_CACHE, refresh=args.refresh_yahoo)
    daily = yahoo[yahoo["ts"] >= args.start].copy()
    if args.end:
        daily = daily[daily["ts"] <= args.end]
    daily = daily.reset_index(drop=True)

    intra = load_intraday(args.symbol, args.intraday_tf, args.start, args.end)
    if intra.empty:
        logger.error(
            "No %s %s bars from %s — backfill first",
            args.symbol,
            args.intraday_tf,
            args.start,
        )
        sys.exit(1)

    intra_first = str(intra["ts"].iloc[0].date())
    intra_last = str(intra["ts"].iloc[-1].date())
    daily = daily[
        (daily["ts"] >= intra["ts"].iloc[0].normalize())
        & (daily["ts"] <= pd.Timestamp(intra_last, tz="UTC") + pd.Timedelta(days=1))
    ].reset_index(drop=True)

    clocks: List[Tuple[int, int]] = []
    clock_str = str(args.clocks).strip()
    if not clock_str:
        if args.intraday_tf == "1m":
            clocks = list(PROXY_CLOCKS_1M_MOC)
        elif args.intraday_tf == "15m":
            clocks = [(15, 45)]
        else:
            clocks = list(PROXY_CLOCKS_5M)
    else:
        for part in clock_str.split(","):
            part = part.strip()
            if not part:
                continue
            hh, mm = part.split(":")
            clocks.append((int(hh), int(mm)))

    logger.info(
        "daily=%d %s->%s | %s=%d %s->%s | clocks=%s",
        len(daily),
        daily["ts"].iloc[0].date() if len(daily) else None,
        daily["ts"].iloc[-1].date() if len(daily) else None,
        args.intraday_tf,
        len(intra),
        intra_first,
        intra_last,
        clocks,
    )

    if args.intraday_tf == "1m":
        intra_15 = load_intraday(args.symbol, "15m", args.start, args.end)
        if intra_15.empty:
            logger.error("No 15m bars for proximity gate")
            sys.exit(1)
        intra_5 = None
        if args.with_5m_baseline:
            intra_5 = load_intraday(args.symbol, "5m", args.start, args.end)
        suite = run_moc_1m_suite(
            daily,
            intra,
            intra_15,
            clocks=clocks,
            symbol=args.symbol,
            include_5m_1555=intra_5,
        )
        stem = "1m_moc"
    else:
        suite = run_realism_suite(
            daily,
            intra,
            clocks=clocks,
            ibs_buffer=float(args.ibs_buffer),
            symbol=args.symbol,
        )
        stem = args.intraday_tf

    wall = time.perf_counter() - t0
    order = sorted(suite["books"].keys())
    rows = [_row(suite["books"][k]) for k in order]

    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily_first": str(daily["ts"].iloc[0].date()),
        "daily_last": str(daily["ts"].iloc[-1].date()),
        "intraday_tf": args.intraday_tf,
        "intraday_n": len(intra),
        "intra_first": intra_first,
        "intra_last": intra_last,
        "wall_s": round(wall, 2),
        "ibs_buffer": args.ibs_buffer,
        "clocks": ["%02d:%02d" % c for c in clocks],
    }

    title = "QS Five Swing fill realism — %s %s" % (args.symbol, stem)
    html = render_html(rows=rows, meta=meta, title=title)
    html_path = outdir / ("qs_five_swing_realism_%s.html" % stem)
    html_path.write_text(html, encoding="utf-8")

    summary = {
        "meta": meta,
        "table": rows,
        "books": {k: _slim(v) for k, v in suite["books"].items()},
    }
    json_path = outdir / ("qs_five_swing_realism_%s_summary.json" % stem)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for r in rows:
        logger.info(
            "%s n=%s E=%s PF=%s CAGR=%s MDD=%s prec=%s recall=%s",
            r["book"],
            r["n"],
            r["E"],
            r["PF"],
            r["CAGR"],
            r["MDD"],
            r.get("precision"),
            r.get("recall"),
        )
    logger.info("Wrote %s (wall %.1fs)", html_path, wall)


if __name__ == "__main__":
    main()