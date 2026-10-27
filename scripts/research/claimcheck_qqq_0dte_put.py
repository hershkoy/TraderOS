#!/usr/bin/env python3
"""Claim-check the 09:45 QQQ 0-DTE short put (YouTube 7S-mGV4D-fI).

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_qqq_0dte_put.py
  python scripts\\research\\claimcheck_qqq_0dte_put.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import ThetaDataClient, get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.qqq_0dte_put import (  # noqa: E402
    CLAIMED,
    DEFAULT_FEE_SIDE,
    FILL_MID,
    FILL_NATURAL,
    FILL_TARGET,
    buyhold_stats,
    claimed_one_lot_pnl,
    compound_book,
    daily_returns_on_sessions,
    fetch_chain,
    fetch_path,
    max_losing_streak,
    select_put,
    sharpe_ratio,
    simulate_short_put,
    summarize_pnls,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_qqq_0dte_put")

DEFAULT_OUT = ROOT / "reports" / "research" / "qqq_0dte_put" / "2026-10-24"
MODES = (FILL_TARGET, FILL_MID, FILL_NATURAL)


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    return value


def _yahoo_closes(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily closes. Theta stock history 403s before mid-2023 on the free plan."""
    import yfinance as yf
    from datetime import timedelta

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        LOG.warning("Yahoo empty for %s", ticker)
        return pd.DataFrame(columns=["date", "close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    out = raw[["close"]].copy()
    out["date"] = pd.to_datetime(out.index).date
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out[["date", "close"]]


def _expiry_days(client: Any, start: date, end: date) -> List[date]:
    exps: List[date] = []
    for _ in range(3):
        exps = list_expirations(client, "QQQ")
        if exps:
            break
        time.sleep(2.0)
    return [d for d in exps if start <= d <= end]


def _simulate_modes(path: pd.DataFrame, strike: float) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        out[mode] = simulate_short_put(path, strike=strike, mode=mode)
    natural = out[FILL_NATURAL]
    fee = dict(natural)
    if fee.get("pnl_1lot") is not None:
        fee["pnl_1lot"] = float(fee["pnl_1lot"]) - 2.0 * DEFAULT_FEE_SIDE
    out["natural_fee"] = fee
    return out


def _trade_row(day: date, rule: str, sel: Dict[str, Any], mode: str, sim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": day.isoformat(),
        "rule": rule,
        "mode": mode,
        "strike": sel["strike"],
        "spot": sel["spot"],
        "delta": sel["delta"],
        "points_otm": sel["points_otm"],
        "spread": sel["spread"],
        "iv": sel.get("iv"),
        "credit": sim.get("credit"),
        "exit_px": sim.get("exit_px"),
        "exit_reason": sim.get("exit_reason"),
        "exit_time": sim.get("exit_time"),
        "pnl_1lot": sim.get("pnl_1lot"),
    }


_THREAD_CLIENT = threading.local()


def _thread_client() -> ThetaDataClient:
    client = getattr(_THREAD_CLIENT, "client", None)
    if client is None:
        client = ThetaDataClient()
        _THREAD_CLIENT.client = client
    return client


def _one_day(day: date, cache_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    client = _thread_client()
    rows: List[Dict[str, Any]] = []
    skips: Counter = Counter()
    try:
        chain = fetch_chain(client, day, cache_dir)
    except Exception as exc:
        LOG.warning("chain %s failed: %s", day.isoformat(), exc)
        return [], {"fetch_fail": 1}
    if chain is None:
        return [], {"fetch_fail": 1}
    if chain.empty:
        return [], {"no_chain": 1}
    for rule, kwargs in (
        ("delta30", {"target_delta": -0.30}),
        ("points2_5", {"points_below": 2.5}),
    ):
        sel, reason = select_put(chain, **kwargs)
        if sel is None:
            skips[f"{rule}:{reason}"] += 1
            continue
        try:
            path = fetch_path(client, day, float(sel["strike"]), cache_dir)
        except Exception as exc:
            LOG.warning("path %s %s failed: %s", day.isoformat(), sel["strike"], exc)
            skips[f"{rule}:fetch_fail"] += 1
            continue
        if path is None:
            skips[f"{rule}:fetch_fail"] += 1
            continue
        sims = _simulate_modes(path, float(sel["strike"]))
        if sims[FILL_TARGET].get("pnl_1lot") is None:
            skips[f"{rule}:{sims[FILL_TARGET].get('exit_reason') or 'no_path'}"] += 1
            continue
        for mode, sim in sims.items():
            rows.append(_trade_row(day, rule, sel, mode, sim))
    return rows, dict(skips)


def run(client: Any, days: List[date], cache_dir: Path, limit: int = 0, workers: int = 4) -> pd.DataFrame:
    del client  # workers own their sessions; the caller client is for the expiration list
    rows: List[Dict[str, Any]] = []
    skips: Counter = Counter()
    todo = days[:limit] if limit else days
    t0 = time.time()
    workers = max(1, int(workers))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_day, day, cache_dir) for day in todo]
        for fut in as_completed(futures):
            day_rows, day_skips = fut.result()
            rows.extend(day_rows)
            skips.update(day_skips)
            done += 1
            if done == 1 or done % 50 == 0 or done == len(todo):
                LOG.info(
                    "progress %d/%d elapsed %.0fs trades %d skips %s",
                    done,
                    len(todo),
                    time.time() - t0,
                    len(rows),
                    dict(skips),
                )
    frame = pd.DataFrame(rows)
    frame.attrs["skips"] = dict(skips)
    frame.attrs["elapsed"] = time.time() - t0
    return frame


def _book(frame: pd.DataFrame, rule: str, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.loc[(frame["rule"] == rule) & (frame["mode"] == mode)].copy()
    return out.sort_values("date")


def _year_rows(book: pd.DataFrame) -> List[Dict[str, Any]]:
    if book.empty:
        return []
    tmp = book.copy()
    tmp["year"] = tmp["date"].str.slice(0, 4)
    rows = []
    for year, grp in tmp.groupby("year"):
        stats = summarize_pnls(grp["pnl_1lot"].tolist())
        stats["year"] = str(year)
        reasons = grp["exit_reason"].value_counts().to_dict()
        stats["exits"] = {str(k): int(v) for k, v in reasons.items()}
        rows.append(stats)
    return rows


def _sized(book: pd.DataFrame, mode: str, bp: float = 10_000.0) -> Dict[str, Any]:
    trades = book.to_dict("records")
    sized = compound_book(trades, pnl_key="pnl_1lot", start=100_000.0, bp=bp, mode=mode)
    curve = sized.pop("curve")
    sized["avg_contracts"] = float(curve.loc[curve["contracts"] > 0, "contracts"].mean()) if not curve.empty else None
    sized["max_contracts"] = int(curve["contracts"].max()) if not curve.empty else 0
    return sized


def _sharpe_block(book: pd.DataFrame, sessions: pd.Series, mode: str) -> Optional[float]:
    sized = compound_book(book.to_dict("records"), pnl_key="pnl_1lot", start=100_000.0, mode=mode)
    rets = daily_returns_on_sessions(sized["curve"], sessions.index, start=100_000.0)
    if rets.empty:
        return None
    # Align to the benchmark sessions and keep flats.
    return sharpe_ratio(rets.tolist())


def summarize(frame: pd.DataFrame, qqq: pd.DataFrame, spy: pd.DataFrame) -> Dict[str, Any]:
    books: Dict[str, Any] = {}
    for rule in ("delta30", "points2_5"):
        for mode in ("target", "mid", "natural", "natural_fee"):
            book = _book(frame, rule, mode)
            stats = summarize_pnls(book["pnl_1lot"].tolist() if not book.empty else [])
            if not book.empty:
                stats["median_credit"] = float(book["credit"].median())
                stats["mean_delta"] = float(book["delta"].mean())
                stats["median_points_otm"] = float(book["points_otm"].median())
                stats["median_spread"] = float(book["spread"].median())
                stats["max_losing_streak"] = max_losing_streak(book["pnl_1lot"].tolist())
                reasons = book["exit_reason"].value_counts().to_dict()
                stats["exits"] = {str(k): int(v) for k, v in reasons.items()}
            else:
                stats["exits"] = {}
            stats["years"] = _year_rows(book)
            stats["flat_10k"] = _sized(book, "flat_bp")
            stats["reg_t"] = _sized(book, "reg_t")
            books[f"{rule}_{mode}"] = stats
    window_closes = {}
    for name, eod in (("qqq", qqq), ("spy", spy)):
        if eod is None or eod.empty:
            window_closes[name] = {}
            continue
        s = eod.copy()
        if not frame.empty:
            d0 = frame["date"].min()
            d1 = frame["date"].max()
            s = s.loc[(s["date"].astype(str) >= d0) & (s["date"].astype(str) <= d1)]
        series = s.set_index("date")["close"]
        window_closes[name] = buyhold_stats(series)
    primary = _book(frame, "delta30", "target")
    qqq_idx = qqq.set_index("date")["close"] if qqq is not None and not qqq.empty else pd.Series(dtype=float)
    return {
        "claimed": CLAIMED,
        "claimed_one_lot_pnl": claimed_one_lot_pnl(),
        "books": books,
        "buyhold": window_closes,
        "sharpe_flat_10k_delta_target": _sharpe_block(primary, qqq_idx, "flat_bp") if not primary.empty else None,
        "sharpe_reg_t_delta_target": _sharpe_block(primary, qqq_idx, "reg_t") if not primary.empty else None,
        "sharpe_flat_10k_delta_natural": _sharpe_block(
            _book(frame, "delta30", "natural"), qqq_idx, "flat_bp"
        )
        if not frame.empty
        else None,
        "skips": frame.attrs.get("skips", {}),
        "elapsed_sec": frame.attrs.get("elapsed"),
        "n_rows": int(len(frame)),
    }


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt_num(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _print_report(summary: Dict[str, Any]) -> None:
    claim = summary["claimed"]
    print("")
    print("Claim (video, before fees and slippage)")
    print(
        f"  n={claim['n']} winners={claim['winners']} WR={_fmt_pct(claim['win_rate'])} "
        f"avg win ${_fmt_num(claim['avg_win'])} avg loss ${_fmt_num(claim['avg_loss'])}"
    )
    print(
        f"  those averages on 1 contract imply ${_fmt_num(summary['claimed_one_lot_pnl'])} "
        f"not the stated ${_fmt_num(claim['profit'])} account profit"
    )
    print(
        f"  account: start ${_fmt_num(claim['start_equity'])} profit ${_fmt_num(claim['profit'])} "
        f"CAGR {_fmt_pct(claim['cagr'])} max DD {_fmt_pct(claim['max_dd'])}"
    )
    print("")
    print(f"{'book':<28} {'n':>5} {'WR':>7} {'avgW':>8} {'avgL':>8} {'PF':>6} {'1lot PnL':>10} {'10k profit':>12} {'10k DD':>8} {'10k CAGR':>9}")
    for key, stats in summary["books"].items():
        sized = stats.get("flat_10k") or {}
        print(
            f"{key:<28} {stats.get('n') or 0:5d} {_fmt_pct(stats.get('win_rate')):>7} "
            f"{_fmt_num(stats.get('avg_win')):>8} {_fmt_num(stats.get('avg_loss')):>8} "
            f"{_fmt_num(stats.get('pf'), 2):>6} {_fmt_num(stats.get('pnl')):>10} "
            f"{_fmt_num(sized.get('profit')):>12} {_fmt_pct(sized.get('max_dd')):>8} "
            f"{_fmt_pct(sized.get('cagr')):>9}"
        )
    print("")
    bh = summary.get("buyhold") or {}
    for name, stats in bh.items():
        print(
            f"buyhold {name:<6} CAGR {_fmt_pct(stats.get('cagr'))} "
            f"max DD {_fmt_pct(stats.get('max_dd'))} Sharpe {_fmt_num(stats.get('sharpe'), 2)}"
        )
    print(
        f"delta30 target sized Sharpe flat-10k {_fmt_num(summary.get('sharpe_flat_10k_delta_target'), 2)} "
        f"reg-T {_fmt_num(summary.get('sharpe_reg_t_delta_target'), 2)} "
        f"natural flat-10k {_fmt_num(summary.get('sharpe_flat_10k_delta_natural'), 2)}"
    )
    for key in ("delta30_target", "delta30_natural", "points2_5_target"):
        stats = summary["books"].get(key) or {}
        reg = stats.get("reg_t") or {}
        print(
            f"{key} exits {stats.get('exits')} streak {stats.get('max_losing_streak')} "
            f"med credit {_fmt_num(stats.get('median_credit'), 2)} "
            f"mean delta {_fmt_num(stats.get('mean_delta'), 3)} "
            f"med pts {_fmt_num(stats.get('median_points_otm'), 2)} "
            f"regT profit {_fmt_num(reg.get('profit'))} DD {_fmt_pct(reg.get('max_dd'))} "
            f"CAGR {_fmt_pct(reg.get('cagr'))}"
        )
        for year in stats.get("years") or []:
            print(
                f"  {year.get('year')} n={year.get('n')} WR={_fmt_pct(year.get('win_rate'))} "
                f"PF={_fmt_num(year.get('pf'), 2)} pnl={_fmt_num(year.get('pnl'))} "
                f"exits={year.get('exits')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-check the 09:45 QQQ 0-DTE short put")
    parser.add_argument("--start", default="2022-05-02")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N expiration days (smoke)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "cache"
    client = get_thetadata_client()
    if not client.is_ready():
        raise SystemExit("Theta Terminal is not responding on :25503")
    days = _expiry_days(client, start, end)
    LOG.info("QQQ expirations %s -> %s: %d", start, end, len(days))
    t0 = time.time()
    frame = run(client, days, cache, limit=int(args.limit or 0), workers=int(args.workers))
    qqq = _yahoo_closes("QQQ", start, end)
    spy = _yahoo_closes("SPY", start, end)
    summary = summarize(frame, qqq, spy)
    summary["expirations"] = len(days)
    summary["start"] = start.isoformat()
    summary["end"] = end.isoformat()
    summary["limit"] = int(args.limit or 0)
    summary["wall_sec"] = round(time.time() - t0, 1)
    if not frame.empty:
        frame.to_csv(outdir / "trades.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _print_report(summary)
    LOG.info("wrote %s (%.0fs)", outdir, summary["wall_sec"])


if __name__ == "__main__":
    main()
