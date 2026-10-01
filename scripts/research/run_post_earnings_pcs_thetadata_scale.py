#!/usr/bin/env python3
"""Scale post-earnings PCS with honest ThetaData NBBO entry/exit.

Entry: first meet-min marketable credit (short_bid - long_ask) 09:30-11:00 ET.
Exit: daily 15:45 marketable debit (short_ask - long_bid) TP50 / SL2x / expiry.

Requires Theta Terminal:
  python scripts\\pipeline\\thetadata_terminal.py

Usage:
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_post_earnings_pcs_thetadata_scale.py
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.notify.telegram_pinger import send_message  # noqa: E402
from utils.research.pcs_thetadata_scale_status import (  # noqa: E402
    JOB_NAME,
    public_status,
    read_status,
    write_status,
)
from utils.research.post_earnings_pcs_live import (  # noqa: E402
    book_stats_from_live,
    resolve_vertical_tickers,
)
from utils.research.post_earnings_pcs_thetadata import (  # noqa: E402
    fetch_exit_daily_debits,
    first_marketable_vertical,
    walk_vertical_nbbo_debit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("pcs_theta_scale")

DEFAULT_TRADES = (
    ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-29" / "trades.csv"
)

_STATUS: Dict[str, Any] = {}
_FINALIZED = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tg(msg: str) -> None:
    try:
        send_message(msg)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Telegram failed: %s", exc)


def _key(sym: str, entry: Any) -> str:
    return "%s|%s" % (str(sym).upper(), pd.Timestamp(entry).strftime("%Y-%m-%d"))


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
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


def _heartbeat(**patch: Any) -> None:
    global _STATUS
    _STATUS.update(patch)
    _STATUS.setdefault("job", JOB_NAME)
    _STATUS.setdefault("pid", os.getpid())
    write_status(_STATUS)


def _finalize(state: str, *, error: Optional[str] = None, summary: Optional[str] = None) -> None:
    global _FINALIZED
    if _FINALIZED:
        return
    _FINALIZED = True
    _heartbeat(state=state, last_error=error)
    if state == "failed":
        _tg(
            "PCS ThetaData scale FAILED\n%s\nlast=%s"
            % (error or "unknown", _STATUS.get("last_key"))
        )
    elif state == "stopped":
        _tg(
            "PCS ThetaData scale STOPPED\n%d/%d done\nlast=%s"
            % (
                int(_STATUS.get("done") or 0),
                int(_STATUS.get("total") or 0),
                _STATUS.get("last_key"),
            )
        )
    elif state == "completed" and summary:
        _tg(summary)


def _on_signal(signum, _frame) -> None:  # noqa: ANN001
    LOG.warning("Signal %s — marking stopped", signum)
    _finalize("stopped")
    raise SystemExit(130)


def _atexit_guard() -> None:
    if _FINALIZED:
        return
    if str(_STATUS.get("state")) == "running":
        _finalize("failed", error="process exited without completion")


def _eval_one(row: pd.Series, client) -> Dict[str, Any]:
    sym = str(row["symbol"]).upper()
    ed = pd.Timestamp(row["entry_date"]).date()
    key = _key(sym, ed)
    st, lt, expiry_s, sk, lk = resolve_vertical_tickers(
        sym,
        ed,
        float(row["short_k"]),
        float(row["long_k"]),
        int(row["dte"]),
    )
    width = float(sk) - float(lk)
    if width <= 0:
        width = float(row.get("width") or 5.0)
    expiry = date.fromisoformat(str(expiry_s)[:10])

    entry = first_marketable_vertical(
        client,
        short_ticker=st,
        long_ticker=lt,
        entry_date=ed,
        width=width,
    )
    obj: Dict[str, Any] = {
        "key": key,
        "symbol": sym,
        "entry_date": ed.isoformat(),
        "bs_credit": float(row.get("credit") or 0),
        "bs_short_k": float(row["short_k"]),
        "bs_long_k": float(row["long_k"]),
        "short_ticker": st,
        "long_ticker": lt,
        "short_k": sk,
        "long_k": lk,
        "width": width,
        "expiry": expiry.isoformat(),
        "entry_status": entry.get("status"),
        "fill_clock": entry.get("fill_clock"),
        "wait_minutes": entry.get("wait_minutes"),
        "credit_bidask": entry.get("credit_bidask"),
        "credit_mid": entry.get("credit_mid"),
        "min_credit": entry.get("min_credit"),
        "any_positive_credit": entry.get("any_positive_credit"),
        "any_positive_clock": entry.get("any_positive_clock"),
        "any_positive_wait": entry.get("any_positive_wait"),
        "n_aligned_min": entry.get("n_aligned_min"),
        "n_short_min": entry.get("n_short_min"),
        "n_long_min": entry.get("n_long_min"),
        "error": entry.get("error"),
    }

    if entry.get("status") != "ok":
        obj["status"] = entry.get("status") or "no_credit"
        # Diagnostic: still walk any-positive first fill when meet-min failed
        ap_cred = entry.get("any_positive_credit")
        if (
            ap_cred is not None
            and float(ap_cred) > 0
            and entry.get("status") == "no_credit"
        ):
            try:
                daily = fetch_exit_daily_debits(
                    client,
                    short_ticker=st,
                    long_ticker=lt,
                    entry_date=ed,
                    expiry=expiry,
                )
                wr_ap = walk_vertical_nbbo_debit(
                    daily,
                    entry_date=ed,
                    credit=float(ap_cred),
                    width=width,
                    expiry=expiry,
                    include_entry_day=True,
                )
                obj["any_walk_status"] = wr_ap.status
                obj["any_exit_reason"] = wr_ap.exit_reason
                obj["any_hold_days"] = wr_ap.hold_days
                obj["any_ror_pct"] = wr_ap.ror_pct
                obj["any_exit_date"] = (
                    wr_ap.exit_date.isoformat() if wr_ap.exit_date else None
                )
                obj["any_pnl"] = wr_ap.pnl
            except Exception as exc:  # noqa: BLE001
                obj["any_walk_status"] = "error"
                obj["error"] = (obj.get("error") or "") + (" | any_walk: %s" % exc)
        return obj

    daily = fetch_exit_daily_debits(
        client,
        short_ticker=st,
        long_ticker=lt,
        entry_date=ed,
        expiry=expiry,
    )
    wr = walk_vertical_nbbo_debit(
        daily,
        entry_date=ed,
        credit=float(entry["credit_bidask"]),
        width=width,
        expiry=expiry,
        include_entry_day=True,
    )
    obj.update(
        {
            "status": wr.status if wr.status != "ok" else "ok",
            "walk_status": wr.status,
            "exit_reason": wr.exit_reason,
            "exit_date": wr.exit_date.isoformat() if wr.exit_date else None,
            "hold_days": wr.hold_days,
            "exit_debit": wr.exit_debit,
            "pnl": wr.pnl,
            "ror_pct": wr.ror_pct,
            "n_exit_days": wr.n_days,
        }
    )

    # Diagnostic any-positive walk (same exits, different entry credit/clock)
    ap_cred = entry.get("any_positive_credit")
    if ap_cred is not None and float(ap_cred) > 0:
        wr_ap = walk_vertical_nbbo_debit(
            daily,
            entry_date=ed,
            credit=float(ap_cred),
            width=width,
            expiry=expiry,
            include_entry_day=True,
        )
        obj["any_walk_status"] = wr_ap.status
        obj["any_exit_reason"] = wr_ap.exit_reason
        obj["any_hold_days"] = wr_ap.hold_days
        obj["any_ror_pct"] = wr_ap.ror_pct
        obj["any_exit_date"] = (
            wr_ap.exit_date.isoformat() if wr_ap.exit_date else None
        )
        obj["any_pnl"] = wr_ap.pnl

    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--min-entry", default="2020-01-01")
    ap.add_argument("--max-entry", default=None)
    ap.add_argument("--n", type=int, default=0, help="Cap sample size (0=all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    global _STATUS, _FINALIZED
    _FINALIZED = False

    t0 = time.perf_counter()
    outdir = args.outdir or (
        ROOT
        / "reports"
        / "research"
        / "post_earnings_pcs"
        / ("%s_thetadata_scale" % date.today().isoformat())
    )
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "scale_cache.jsonl"

    # Detect abandoned prior run
    prior = read_status()
    if prior and str(prior.get("state")) == "running":
        pub = public_status()
        if pub and pub.get("state") == "stale":
            LOG.warning("Found stale prior run; notifying")
            if not args.no_telegram:
                _tg(
                    "PCS ThetaData scale: prior run STALE (abandoned)\nlast=%s done=%s/%s"
                    % (prior.get("last_key"), prior.get("done"), prior.get("total"))
                )

    trades = pd.read_csv(args.trades)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades = trades.dropna(subset=["symbol", "entry_date", "short_k", "long_k", "width", "dte"])
    trades = trades[trades["entry_date"] >= pd.Timestamp(args.min_entry)]
    if args.max_entry:
        trades = trades[trades["entry_date"] <= pd.Timestamp(args.max_entry)]
    trades = trades.sort_values("entry_date").reset_index(drop=True)
    if int(args.n) > 0:
        trades = trades.head(int(args.n))

    total = int(len(trades))
    LOG.info("Sample n=%d from %s", total, args.trades)
    if total == 0:
        LOG.error("Empty sample")
        return 1

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not reachable — start thetadata_terminal.py")
        return 1

    cache = {} if args.refresh else _load_cache(cache_path)
    _STATUS = {
        "job": JOB_NAME,
        "state": "running",
        "started_at": _utc_now(),
        "done": 0,
        "total": total,
        "ok": 0,
        "no_credit": 0,
        "last_key": None,
        "last_error": None,
        "outdir": str(outdir),
        "pid": os.getpid(),
    }
    write_status(_STATUS)

    try:
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _on_signal)
    except Exception:  # noqa: BLE001
        pass
    atexit.register(_atexit_guard)

    if not args.no_telegram:
        _tg(
            "PCS ThetaData scale STARTED\nn=%d workers=%d\nout=%s"
            % (total, int(args.workers), outdir)
        )

    rows: List[Dict[str, Any]] = []
    ok_n = 0
    no_credit_n = 0
    workers = max(1, min(4, int(args.workers)))

    pending_idx = []
    for i, (_, row) in enumerate(trades.iterrows()):
        key = _key(row["symbol"], row["entry_date"])
        if key in cache and not args.refresh:
            obj = cache[key]
            rows.append(obj)
            if obj.get("status") == "ok":
                ok_n += 1
            if obj.get("status") == "no_credit":
                no_credit_n += 1
            _heartbeat(
                done=len(rows),
                ok=ok_n,
                no_credit=no_credit_n,
                last_key=key,
            )
            LOG.info("[%d/%d] cache %s %s", len(rows), total, key, obj.get("status"))
        else:
            pending_idx.append((i, row))

    def _work(item):
        _i, _row = item
        return _eval_one(_row, get_thetadata_client())

    # Sequential is safer for Terminal queue; still allow workers<=4
    if workers <= 1 or len(pending_idx) <= 1:
        for i, row in pending_idx:
            key = _key(row["symbol"], row["entry_date"])
            LOG.info("[%d/%d] eval %s ...", len(rows) + 1, total, key)
            try:
                obj = _eval_one(row, client)
            except Exception as exc:  # noqa: BLE001
                obj = {
                    "key": key,
                    "symbol": str(row["symbol"]).upper(),
                    "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
                    "status": "error",
                    "error": str(exc),
                }
                LOG.exception("eval failed %s", key)
            _append_cache(cache_path, obj)
            cache[key] = obj
            rows.append(obj)
            if obj.get("status") == "ok":
                ok_n += 1
            if obj.get("status") == "no_credit":
                no_credit_n += 1
            _heartbeat(
                done=len(rows),
                ok=ok_n,
                no_credit=no_credit_n,
                last_key=key,
                last_error=obj.get("error"),
            )
            LOG.info(
                "  -> %s credit=%s clock=%s exit=%s ror=%s",
                obj.get("status"),
                obj.get("credit_bidask"),
                obj.get("fill_clock"),
                obj.get("exit_reason"),
                obj.get("ror_pct"),
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, item): item for item in pending_idx}
            for fut in as_completed(futs):
                i, row = futs[fut]
                key = _key(row["symbol"], row["entry_date"])
                try:
                    obj = fut.result()
                except Exception as exc:  # noqa: BLE001
                    obj = {
                        "key": key,
                        "symbol": str(row["symbol"]).upper(),
                        "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
                        "status": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc()[-500:],
                    }
                _append_cache(cache_path, obj)
                cache[key] = obj
                rows.append(obj)
                if obj.get("status") == "ok":
                    ok_n += 1
                if obj.get("status") == "no_credit":
                    no_credit_n += 1
                _heartbeat(
                    done=len(rows),
                    ok=ok_n,
                    no_credit=no_credit_n,
                    last_key=key,
                    last_error=obj.get("error"),
                )
                LOG.info(
                    "[%d/%d] %s %s credit=%s exit=%s",
                    len(rows),
                    total,
                    key,
                    obj.get("status"),
                    obj.get("credit_bidask"),
                    obj.get("exit_reason"),
                )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "scale_trades.csv", index=False)

    meet_book = [
        {
            "ror_hs": r.get("ror_pct"),
            "hold_days": r.get("hold_days"),
            "exit_reason": r.get("exit_reason"),
        }
        for r in rows
        if r.get("status") == "ok" and r.get("walk_status") == "ok"
    ]
    any_book = [
        {
            "ror_hs": r.get("any_ror_pct"),
            "hold_days": r.get("any_hold_days"),
            "exit_reason": r.get("any_exit_reason"),
        }
        for r in rows
        if r.get("any_walk_status") == "ok" and r.get("any_ror_pct") is not None
    ]

    summary = {
        "attempted": len(rows),
        "status_counts": df["status"].value_counts().to_dict() if not df.empty else {},
        "meet_min_book": book_stats_from_live(meet_book, ror_key="ror_hs"),
        "any_positive_book": book_stats_from_live(any_book, ror_key="ror_hs"),
        "wall_s": time.perf_counter() - t0,
        "outdir": str(outdir),
        "note": (
            "Entry=first meet-min marketable NBBO 09:30-11:00 ET; "
            "exit=daily 15:45 marketable debit TP50/SL2x."
        ),
    }
    (outdir / "scale_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    mb = summary["meet_min_book"]
    ab = summary["any_positive_book"]

    def _bl(label: str, b: Dict[str, Any]) -> str:
        pf = b.get("profit_factor")
        pf_s = ("%.2f" % pf) if isinstance(pf, (int, float)) else str(pf)
        return (
            "%s: n=%s WR=%.0f%% hold=%.1fd RoR=%.2f%% PF=%s"
            % (
                label,
                b.get("n"),
                b.get("win_rate_pct") or 0,
                b.get("avg_hold_days") or 0,
                b.get("avg_ror_pct") or 0,
                pf_s,
            )
        )

    lines = [
        "Post-earnings PCS ThetaData scale (first meet-min entry + NBBO debit exits)",
        "attempted=%d statuses=%s" % (len(rows), summary["status_counts"]),
        _bl("Meet-min book", mb),
        _bl("Any-positive diagnostic", ab),
        "wall=%.1fs out=%s" % (summary["wall_s"], outdir),
        summary["note"],
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "scale_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("\n%s", text)

    done_msg = (
        "PCS ThetaData scale COMPLETED\n"
        "attempted=%d meet_min_n=%s WR=%.0f%% RoR=%.2f%% PF=%s\n"
        "wall=%.0fs"
        % (
            len(rows),
            mb.get("n"),
            mb.get("win_rate_pct") or 0,
            mb.get("avg_ror_pct") or 0,
            (
                ("%.2f" % mb["profit_factor"])
                if isinstance(mb.get("profit_factor"), (int, float))
                else mb.get("profit_factor")
            ),
            summary["wall_s"],
        )
    )
    if not args.no_telegram:
        _finalize("completed", summary=done_msg)
    else:
        _finalize("completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("fatal")
        _finalize("failed", error="%s: %s" % (type(exc).__name__, exc))
        raise
