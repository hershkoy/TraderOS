"""Matched listed short-put + half-spread fills + occupancy (Polygon free).

Expands the BS hard-stop spot-check toward live-ish fills:
  1) snap synthetic expiry/strike to listed contracts
  2) day-bar mid +/- half-spread (floor $0.05 or 8% of mid)
  3) FCFS occupancy at max_open 4/8/15

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=5
  python scripts\\research\\run_last_15m_short_put_live_realism.py --n 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "5")

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.short_put_live_realism import (  # noqa: E402
    cash_secured_gain,
    credit_after_spread,
    debit_after_spread,
    occupancy_fcfs_put,
    strike_candidates,
    summarize_gains,
)
from utils.research.short_put_polygon_spotcheck import (  # noqa: E402
    bars_from_aggs_response,
    closest_listed_expiry,
    pick_entry_exit_bars,
    polygon_option_ticker,
    select_hard_stop_sample,
)

LOG = logging.getLogger("short_put_live_realism")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-26"
    / "last_15m_short_put_bs_dte45_atm_trades_20260926_213902.csv"
)


def _row_key(stock: str, buy_date: Any) -> str:
    d = pd.Timestamp(buy_date).strftime("%Y-%m-%d")
    return "%s|%s" % (str(stock).upper(), d)


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
            k = obj.get("key") or _row_key(obj.get("stock"), obj.get("buy_date"))
            out[k] = obj
    return out


def _append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _list_expiries_cached(
    client: Any,
    cache: Dict[str, List[str]],
    sym: str,
    buy_date,
    *,
    dte_lo: int = 25,
    dte_hi: int = 70,
) -> List[str]:
    start = (pd.Timestamp(buy_date) + timedelta(days=dte_lo)).date().isoformat()
    end = (pd.Timestamp(buy_date) + timedelta(days=dte_hi)).date().isoformat()
    ck = "%s|%s|%s" % (sym, start, end)
    if ck in cache:
        return cache[ck]
    try:
        exps = client.list_expirations(sym, start, end, max_contracts=1500)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("%s list_expirations failed: %s", sym, exc)
        exps = []
    cache[ck] = list(exps or [])
    return cache[ck]


def _contracts_cached(
    client: Any,
    cache: Dict[str, List[Dict[str, Any]]],
    sym: str,
    expiry: str,
    as_of: str,
) -> List[Dict[str, Any]]:
    ck = "%s|%s|%s" % (sym, expiry, as_of)
    if ck in cache:
        return cache[ck]
    results: List[Dict[str, Any]] = []
    # Free tier often returns empty with contract_type+as_of; try bare expiry first.
    for kwargs in (
        {"as_of": as_of, "contract_type": None},
        {"as_of": as_of, "contract_type": "put"},
        {"as_of": None, "contract_type": None},
    ):
        try:
            resp = client.get_contracts_for_expiry(
                sym,
                expiry,
                as_of=kwargs["as_of"],
                contract_type=kwargs["contract_type"],
                limit=1000,
            )
            results = list(resp.get("results") or [])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s contracts %s failed: %s", sym, expiry, exc)
            results = []
        if results:
            break
    cache[ck] = results
    return results


def _fetch_one(
    client: Any,
    row: pd.Series,
    *,
    exp_cache: Dict[str, List[str]],
    ctr_cache: Dict[str, List[Dict[str, Any]]],
    spread_floor: float,
    spread_frac: float,
    skip_contracts: bool = True,
    use_list_expiries: bool = False,
    max_aggs_attempts: int = 4,
) -> Dict[str, Any]:
    from datetime import date

    from utils.research.short_put_live_realism import nearby_fridays

    stock = str(row["stock"]).upper()
    buy_d = pd.Timestamp(row["buy_date"]).date()
    exit_d = pd.Timestamp(row["put_exit_date"]).date()
    target_strike = float(row["put_strike"])
    dte = int(row["put_dte"]) if pd.notna(row.get("put_dte")) else 45
    if pd.notna(row.get("buy_price")) and float(row["buy_price"]) > 0:
        target_strike = float(row["buy_price"])

    out: Dict[str, Any] = {
        "key": _row_key(stock, buy_d),
        "stock": stock,
        "buy_date": buy_d.isoformat(),
        "bs_exit_date": exit_d.isoformat(),
        "bs_strike": float(row["put_strike"]),
        "bs_credit": float(row["put_credit"]),
        "bs_exit": float(row["put_exit_premium"]),
        "bs_gain_cs": float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else None,
        "adv_20": float(row["adv_20"]) if pd.notna(row.get("adv_20")) else None,
        "status": "pending",
    }

    target_exp = buy_d + timedelta(days=dte)
    expiry_candidates: List[str] = []
    if use_list_expiries:
        expiries = _list_expiries_cached(client, exp_cache, stock, buy_d)
        listed_exp = closest_listed_expiry(expiries, target_exp) if expiries else None
        if listed_exp:
            expiry_candidates.append(listed_exp)
    for fri in nearby_fridays(target_exp, window_days=21):
        s = fri.isoformat()
        if s not in expiry_candidates:
            expiry_candidates.append(s)
    if not expiry_candidates:
        out["status"] = "no_expiry_match"
        return out

    strikes = strike_candidates(target_strike, n_each_side=2)
    start = (buy_d - timedelta(days=2)).isoformat()
    end = (exit_d + timedelta(days=5)).isoformat()
    last_st = "no_bars"

    # Try nearest Friday+ATM first, then ATM on next Fridays, then strike neighbors.
    pairs: List[Tuple[str, float]] = []
    for exp in expiry_candidates[:3]:
        pairs.append((exp, strikes[0]))
    for k in strikes[1:3]:
        pairs.append((expiry_candidates[0], k))

    attempts = 0
    for expiry, strike in pairs:
        if attempts >= int(max_aggs_attempts):
            break
        attempts += 1
        ticker = polygon_option_ticker(stock, expiry, strike, "P")
        try:
            resp = client.get_options_aggregates(
                ticker, start, end, timespan="day"
            )
            bars = bars_from_aggs_response(resp if isinstance(resp, dict) else {})
        except Exception as exc:  # noqa: BLE001
            last_st = "aggs_error:%s" % exc
            continue
        entry, exit_bar, st = pick_entry_exit_bars(bars, buy_d, exit_d)
        last_st = st
        if entry is None or exit_bar is None or st != "ok":
            continue

        mid_e = float(entry["close"])
        mid_x = float(exit_bar["close"])
        cred = credit_after_spread(mid_e, floor=spread_floor, frac=spread_frac)
        deb = debit_after_spread(mid_x, floor=spread_floor, frac=spread_frac)
        gain = cash_secured_gain(cred, deb, strike)
        gain_mid = cash_secured_gain(mid_e, mid_x, strike)
        exp_d = date.fromisoformat(expiry[:10])
        out.update(
            {
                "status": "ok",
                "live_ticker": ticker,
                "live_strike": strike,
                "live_expiry": expiry,
                "strike_diff": strike - float(target_strike),
                "expiry_diff_days": (exp_d - target_exp).days,
                "match_source": "friday_ladder",
                "aggs_attempts": attempts,
                "live_entry_date": entry["date"].isoformat()
                if hasattr(entry["date"], "isoformat")
                else str(entry["date"]),
                "live_exit_date": exit_bar["date"].isoformat()
                if hasattr(exit_bar["date"], "isoformat")
                else str(exit_bar["date"]),
                "live_mid_entry": mid_e,
                "live_mid_exit": mid_x,
                "live_credit_spread": cred,
                "live_debit_spread": deb,
                "live_gain_cs_mid": gain_mid,
                "live_gain_cs_spread": gain,
                "exit_vs_bs": (mid_x / float(row["put_exit_premium"]))
                if float(row["put_exit_premium"]) > 0
                else None,
                "n_bars": int(len(bars)),
                "bar_status": st,
            }
        )
        return out

    out["status"] = last_st
    out["live_expiry"] = expiry_candidates[0]
    out["live_strike"] = strikes[0]
    out["live_ticker"] = polygon_option_ticker(
        stock, expiry_candidates[0], strikes[0], "P"
    )
    out["aggs_attempts"] = attempts
    return out


def _write_summary(
    df: pd.DataFrame,
    outdir: Path,
    *,
    max_opens: List[int],
    spread_floor: float,
    spread_frac: float,
) -> Path:
    ok_all = df[df["status"] == "ok"].copy()
    # Honest: option entry print on signal calendar day (no slid entry).
    ok = ok_all.copy()
    if not ok.empty and "live_entry_date" in ok.columns and "buy_date" in ok.columns:
        be = pd.to_datetime(ok["buy_date"], errors="coerce").dt.normalize()
        le = pd.to_datetime(ok["live_entry_date"], errors="coerce").dt.normalize()
        n_slid = int((le != be).sum())
        ok = ok.loc[le == be].copy()
    else:
        n_slid = 0

    lines = []
    lines.append("last-15m short-put live realism (listed + half-spread + occupancy)")
    lines.append("spread = max(%.2f, %.0f%% * mid)" % (spread_floor, spread_frac * 100))
    lines.append(
        "attempted=%d status_ok=%d slid_entry=%d honest_ok=%d"
        % (len(df), len(ok_all), n_slid, len(ok))
    )
    if n_slid:
        lines.append(
            "note: honest stats require live_entry_date==buy_date; slid rows dropped"
        )
    if not ok.empty and "exit_vs_bs" in ok.columns:
        r = pd.to_numeric(ok["exit_vs_bs"], errors="coerce").dropna()
        lines.append(
            "exit_vs_bs mid/bs: med=%.3f mean=%.3f"
            % (float(r.median()), float(r.mean()))
        )
    for col, label in (
        ("live_gain_cs_mid", "mid fills"),
        ("live_gain_cs_spread", "half-spread fills"),
        ("bs_gain_cs", "BS cash-secured (same rows)"),
    ):
        if col not in ok.columns or ok.empty:
            continue
        g = pd.to_numeric(ok[col], errors="coerce").dropna()
        s = summarize_gains(g.tolist())
        lines.append(
            "unconstrained %s: n=%d E=%+.2f PF=%.2f WR=%.1f%% med=%+.2f"
            % (label, s["n"], s["E"], s["PF"], s["WR"], s.get("med", float("nan")))
        )

    occ_rows = []
    if not ok.empty:
        trades = []
        for _, r in ok.iterrows():
            trades.append(
                {
                    "symbol": r["stock"],
                    "entry_date": r["live_entry_date"],
                    "exit_date": r["live_exit_date"],
                    "gain_pct": float(r["live_gain_cs_spread"]),
                }
            )
        for mo in max_opens:
            kept, meta = occupancy_fcfs_put(trades, max_open=int(mo), unique_symbol=True)
            gains = [float(t["gain_pct"]) for t in kept]
            s = summarize_gains(gains)
            lines.append(
                "occupancy max_open=%d: kept=%d skip=%d E=%+.2f PF=%.2f WR=%.1f%%"
                % (
                    mo,
                    meta["n_kept"],
                    meta["n_skipped"],
                    s["E"],
                    s["PF"],
                    s["WR"],
                )
            )
            occ_rows.append(
                {
                    "max_open": mo,
                    "n_kept": meta["n_kept"],
                    "n_skipped": meta["n_skipped"],
                    **s,
                }
            )

    text = "\n".join(lines) + "\n"
    path = outdir / "last_15m_short_put_live_realism_summary.txt"
    path.write_text(text, encoding="utf-8")
    if occ_rows:
        pd.DataFrame(occ_rows).to_csv(
            outdir / "last_15m_short_put_live_realism_occupancy.csv", index=False
        )
    LOG.info("Summary:\n%s", text)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-buy", type=str, default="2024-10-01")
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--spread-floor", type=float, default=0.05)
    ap.add_argument("--spread-frac", type=float, default=0.08)
    ap.add_argument("--max-open", type=str, default="4,8,15")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument(
        "--use-contracts",
        action="store_true",
        default=False,
        help="Query Polygon contract metadata (slower; often empty on free tier)",
    )
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
        LOG.error("No hard-stop rows")
        return 1

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else outdir / "live_realism_cache.jsonl"

    done = _load_cache(cache_path) if args.resume else {}
    # Re-fetch failed rows from older cache versions
    done = {k: v for k, v in done.items() if v.get("status") == "ok"}
    LOG.info(
        "Sample n=%d cache_ok=%d path=%s",
        len(sample),
        len(done),
        cache_path,
    )

    client = get_polygon_client()
    exp_cache: Dict[str, List[str]] = {}
    ctr_cache: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    skip_contracts = not bool(args.use_contracts)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        key = _row_key(row["stock"], row["buy_date"])
        if key in done:
            rows.append(done[key])
            LOG.info("[%d/%d] cache hit %s", i, len(sample), key)
            continue
        LOG.info("[%d/%d] fetch %s", i, len(sample), key)
        try:
            obj = _fetch_one(
                client,
                row,
                exp_cache=exp_cache,
                ctr_cache=ctr_cache,
                spread_floor=float(args.spread_floor),
                spread_frac=float(args.spread_frac),
                skip_contracts=skip_contracts,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("fetch failed %s", key)
            obj = {
                "key": key,
                "stock": str(row["stock"]).upper(),
                "buy_date": pd.Timestamp(row["buy_date"]).date().isoformat(),
                "status": "error:%s" % exc,
            }
        rows.append(obj)
        done[key] = obj
        _append_cache(cache_path, obj)
        LOG.info(
            "  -> %s live_gain_spread=%s exit_vs_bs=%s attempts=%s",
            obj.get("status"),
            obj.get("live_gain_cs_spread"),
            obj.get("exit_vs_bs"),
            obj.get("aggs_attempts"),
        )
        # Pace free-tier: each trade may use several aggs calls
        time.sleep(12.0)

    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / ("last_15m_short_put_live_realism_%s.csv" % stamp)
    df.to_csv(csv_path, index=False)

    max_opens = [int(x.strip()) for x in str(args.max_open).split(",") if x.strip()]
    _write_summary(
        df,
        outdir,
        max_opens=max_opens,
        spread_floor=float(args.spread_floor),
        spread_frac=float(args.spread_frac),
    )
    LOG.info(
        "Wrote %s (%.1fs wall)",
        csv_path,
        time.perf_counter() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
