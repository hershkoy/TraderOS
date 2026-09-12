"""Pull IB historical top-of-book (not L2) for last-15m smoke losers.

IB has no historical DOM. This requests BID/ASK/TRADES bars and, when the
farm still has them, BidAsk/Last ticks (~6 months). Client 8828, port 4001.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\pull_ib_historical_l1.py --smoke
  python scripts\\research\\pull_ib_historical_l1.py --symbols AMPL --sessions 2024-02-09,2024-02-13,2024-02-21
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ib_insync import IB  # noqa: E402
from utils.research.ib_historical_l1 import (  # noqa: E402
    canonical_symbol,
    metrics_row,
    parse_session_date,
    pull_session_top_of_book,
    save_frames,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("pull_ib_historical_l1")

DEFAULT_CLIENT_ID = 8828
DEFAULT_PORT = 4001
DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)

# Follow-up overlay smoke (docs/status_log/.../2026-09-12_last_15m_followup_overlays.md).
# Aliases APML/TATS are accepted via canonical_symbol.
SMOKE_JOBS: List[Dict[str, Any]] = [
    {
        "stock": "AMPL",
        "session": "2024-02-09",
        "fill_px": 14.17,
        "gain_pct": -21.3,
        "note": "fill",
    },
    {
        "stock": "AMPL",
        "session": "2024-02-13",
        "fill_px": 14.17,
        "gain_pct": -21.3,
        "note": "F2 fail-close",
    },
    {
        "stock": "AMPL",
        "session": "2024-02-21",
        "fill_px": 14.17,
        "gain_pct": -21.3,
        "note": "gap 14.07 to 9.22",
    },
    {
        "stock": "VST",
        "session": "2021-02-18",
        "fill_px": 23.20,
        "gain_pct": -19.6,
        "note": "fill form 1.82",
    },
    {
        "stock": "VST",
        "session": "2021-02-19",
        "fill_px": 23.20,
        "gain_pct": -19.6,
        "note": "F2 fail-close",
    },
    {
        "stock": "TARS",
        "session": "2023-07-20",
        "fill_px": 23.24,
        "gain_pct": -19.4,
        "note": "fill pos 1.49",
    },
]


def _load_trade_row(path: Path, stock: str, session: str) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        LOG.exception("failed to read trades %s", path)
        return None
    if df.empty or "stock" not in df.columns:
        return None
    want = canonical_symbol(stock)
    day = str(session)[:10]
    work = df.copy()
    work["_sym"] = work["stock"].astype(str).str.upper()
    buy_col = "buy_time" if "buy_time" in work.columns else "buy_date"
    work["_day"] = work[buy_col].astype(str).str[:10]
    hit = work[(work["_sym"] == want) & (work["_day"] == day)]
    if hit.empty:
        # AMPL 02-13 is not a fill day; use the fill row for rail / pos.
        hit = work[work["_sym"] == want]
    if hit.empty:
        return None
    return hit.iloc[0]


def _rail_from_row(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    for col in ("resist", "resistance", "resist_px", "h2_resist"):
        if col in row.index:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            if val == val and val > 0:
                return val
    return None


def _jobs_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.smoke:
        return [dict(j) for j in SMOKE_JOBS]
    symbols = [canonical_symbol(s) for s in str(args.symbols or "").split(",") if s.strip()]
    sessions = [s.strip() for s in str(args.sessions or "").split(",") if s.strip()]
    if not symbols or not sessions:
        raise SystemExit("need --smoke or both --symbols and --sessions")
    jobs = []
    for stock in symbols:
        for session in sessions:
            jobs.append({"stock": stock, "session": session, "note": "", "fill_px": None})
    return jobs


def _enrich(job: Dict[str, Any], trades_path: Path) -> Dict[str, Any]:
    out = dict(job)
    row = _load_trade_row(trades_path, out["stock"], out["session"])
    if row is None:
        return out
    rail = _rail_from_row(row)
    if rail is not None:
        out["rail"] = rail
    if out.get("fill_px") is None:
        for col in ("buy_price", "fill_px", "entry_px"):
            if col in row.index:
                try:
                    out["fill_px"] = float(row[col])
                    break
                except (TypeError, ValueError):
                    continue
    for col in ("channel_pos", "formation_beyond_width", "gain_pct", "support"):
        if col in row.index and out.get(col) is None:
            try:
                out[col] = float(row[col])
            except (TypeError, ValueError):
                out[col] = row[col]
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IB historical BID/ASK/TRADES (not L2)")
    p.add_argument("--smoke", action="store_true", help="AMPL/VST/TARS follow-up days")
    p.add_argument("--symbols", default="", help="Comma symbols (APML->AMPL, TATS->TARS)")
    p.add_argument("--sessions", default="", help="Comma YYYY-MM-DD session dates")
    p.add_argument("--ib-client-id", type=int, default=DEFAULT_CLIENT_ID)
    p.add_argument("--ib-port", type=int, default=DEFAULT_PORT)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--no-ticks", action="store_true", help="Skip reqHistoricalTicks")
    p.add_argument("--no-5s", action="store_true", help="Skip 5-second bars")
    p.add_argument("--trades", default=str(DEFAULT_TRADES))
    p.add_argument("--outdir", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


async def _connect_light_async(ib: IB, host: str, port: int, client_id: int, timeout: float):
    import asyncio

    await ib.client.connectAsync(host, port, clientId=int(client_id), timeout=float(timeout))
    await asyncio.sleep(1.0)
    if not ib.client.isReady():
        raise ConnectionError("IB client not ready (client %s port %s)" % (client_id, port))
    return ib


def connect_ib(*, client_id: int, port: int, host: str = "127.0.0.1", timeout: float = 20.0):
    """Hist-only handshake. Skip open-orders/executions (wedges next to 5m backfill)."""
    ib = IB()
    ib._run(_connect_light_async(ib, host, int(port), int(client_id), timeout))
    if not ib.client.isReady():
        try:
            ib.disconnect()
        except Exception:
            pass
        raise SystemExit("IB API handshake failed (client %s port %s)" % (client_id, port))
    LOG.info("IB light-connect ready client=%s port=%s accounts=%s", client_id, port, ib.client.getAccounts())
    return ib


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    jobs = _jobs_from_args(args)
    trades_path = Path(args.trades)
    jobs = [_enrich(j, trades_path) for j in jobs]
    outdir = Path(args.outdir) if args.outdir else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "ib_l1_smoke_summary.csv"
    LOG.info(
        "jobs=%d client=%s port=%s outdir=%s",
        len(jobs),
        args.ib_client_id,
        args.ib_port,
        outdir,
    )
    if args.dry_run:
        for job in jobs:
            LOG.info("dry-run %s %s note=%s", job["stock"], job["session"], job.get("note"))
        pd.DataFrame(jobs).to_csv(outdir / "ib_l1_smoke_jobs_dry_run.csv", index=False)
        return 0

    ib = None
    rows: List[dict] = []
    t0 = time.monotonic()
    try:
        ib = connect_ib(client_id=int(args.ib_client_id), port=int(args.ib_port))
        for i, job in enumerate(jobs, start=1):
            stock = canonical_symbol(job["stock"])
            session = parse_session_date(job["session"]).isoformat()
            LOG.info(
                "[%d/%d] %s %s note=%s",
                i,
                len(jobs),
                stock,
                session,
                job.get("note") or "",
            )
            try:
                payload = pull_session_top_of_book(
                    ib,
                    stock,
                    session,
                    rail=job.get("rail"),
                    fill_px=job.get("fill_px"),
                    sleep_s=float(args.sleep),
                    want_ticks=not args.no_ticks,
                    want_5s=not args.no_5s,
                )
            except Exception as exc:
                LOG.exception("%s %s pull failed", stock, session)
                rows.append(
                    {
                        "stock": stock,
                        "session": session,
                        "note": job.get("note") or "",
                        "error": str(exc),
                    }
                )
                continue
            stem = "%s_%s" % (stock, session.replace("-", ""))
            written = save_frames(payload, outdir, stem=stem)
            row = metrics_row(payload, note=str(job.get("note") or ""))
            row["channel_pos"] = job.get("channel_pos")
            row["formation_beyond_width"] = job.get("formation_beyond_width")
            row["gain_pct"] = job.get("gain_pct")
            row["n_files"] = len(written)
            rows.append(row)
            LOG.info(
                "%s %s last15m quotes=%s trades=%s spread_bps=%s tags=%s files=%d",
                stock,
                session,
                row.get("n_quote_last15m"),
                row.get("n_trade_last15m"),
                row.get("median_spread_bps"),
                row.get("tags"),
                len(written),
            )
    finally:
        if ib is not None:
            try:
                ib.disconnect()
            except Exception:
                LOG.exception("IB disconnect failed")
    elapsed = time.monotonic() - t0
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    meta = {
        "client_id": int(args.ib_client_id),
        "port": int(args.ib_port),
        "elapsed_s": round(elapsed, 1),
        "n_jobs": len(jobs),
        "n_ok": int(summary["n_quote_session"].fillna(0).gt(0).sum()) if not summary.empty and "n_quote_session" in summary.columns else 0,
        "summary": str(summary_path),
        "note": "IB has no historical L2; this is BID/ASK/TRADES top of book.",
    }
    (outdir / "ib_l1_smoke_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("done elapsed=%.1fs summary=%s", elapsed, summary_path)
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
