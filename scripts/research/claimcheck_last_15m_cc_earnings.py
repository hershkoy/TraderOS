"""Compare buy-write vs wheel, each with earnings flatten<=10.

Uses the full-pool wheel trade list (ADV>=$2M, buy>=2022). Wheel marks are
loaded from prior CSVs; buy-write is simulated (Theta cache) then overlaid.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_last_15m_cc_earnings.py --smoke
  python scripts\\research\\claimcheck_last_15m_cc_earnings.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

load_dotenv(ROOT / ".env")

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from overlay_last_15m_earnings_event_filters import load_earnings_map  # noqa: E402
from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_buy_write import (  # noqa: E402
    close_call_debit,
    result_to_record as bw_result_to_record,
    simulate_buy_write,
    simulate_buy_write_listed_or_bs,
)
from utils.research.last_15m_cc_earnings import (  # noqa: E402
    apply_buy_write_earnings,
    apply_wheel_earnings,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
)
from utils.research.last_15m_wheel import (  # noqa: E402
    DEFAULT_DTE,
    HybridWheelSource,
    ThetaEquitySource,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("claimcheck_last_15m_cc_earn")

WHEEL_DIR = ROOT / "reports" / "ascending_channels" / "2026-10-09"
DEFAULT_WHEEL_MID = WHEEL_DIR / "last_15m_wheel_mid_repaired_20261010_112646.csv"
DEFAULT_WHEEL_ASK = WHEEL_DIR / "last_15m_wheel_ask_bid_20261010_112516.csv"
DEFAULT_EARN_CACHE = (
    ROOT / "reports" / "ascending_channels" / "earnings_event_filters" / "earnings_cache.json"
)


def _row_key(stock: str, buy_date: Any, mark_mode: str) -> str:
    d = pd.Timestamp(buy_date).strftime("%Y-%m-%d")
    return "%s|%s|%s" % (str(stock).upper(), d, str(mark_mode).lower())


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
            k = obj.get("key")
            if not k:
                continue
            prev = out.get(k)
            if prev is not None and prev.get("bw_ok") and not obj.get("bw_ok"):
                continue
            out[k] = obj
    return out


def _append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _score(
    label: str,
    trades: pd.DataFrame,
    gain: pd.Series,
) -> Dict[str, Any]:
    part = trades.loc[gain.notna()].copy()
    g = gain.loc[gain.notna()]
    if part.empty:
        LOG.info("%s n=0", label)
        return {"label": label, "n": 0}
    years = summarize_by_year(
        part.assign(gain_pct_ov=g), gain_col="gain_pct_ov", buckets=YEAR_BUCKETS
    )
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    s22 = book_stats(g.loc[_in_range(part, "2022-01-01", "2022-12-31")])
    s23 = book_stats(g.loc[_in_range(part, "2023-01-01", "2023-12-31")])
    s2223 = book_stats(g.loc[_in_range(part, "2022-01-01", "2023-12-31")])
    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s | 2022 E=%s PF=%s | 2023 E=%s PF=%s",
        label,
        s["n"],
        "n/a" if s["expectancy_pct"] is None else "%.3f" % s["expectancy_pct"],
        "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
        "n/a" if geo is None else "%.3f" % geo,
        "n/a" if s22["expectancy_pct"] is None else "%.3f" % s22["expectancy_pct"],
        "n/a" if s22["profit_factor"] is None else "%.3f" % s22["profit_factor"],
        "n/a" if s23["expectancy_pct"] is None else "%.3f" % s23["expectancy_pct"],
        "n/a" if s23["profit_factor"] is None else "%.3f" % s23["profit_factor"],
    )
    print("---- %s ----" % label)
    print(years.to_string(index=False))
    return {
        "label": label,
        "n": s["n"],
        "expectancy_pct": s["expectancy_pct"],
        "profit_factor": s["profit_factor"],
        "geo_year_pf": geo,
        "win_rate_pct": s.get("win_rate_pct"),
        "e_2022": s22["expectancy_pct"],
        "pf_2022": s22["profit_factor"],
        "n_2022": s22["n"],
        "e_2023": s23["expectancy_pct"],
        "pf_2023": s23["profit_factor"],
        "n_2023": s23["n"],
        "e_2022_23": s2223["expectancy_pct"],
        "pf_2022_23": s2223["profit_factor"],
        "n_2022_23": s2223["n"],
    }


def _overlay_wheel(
    df: pd.DataFrame,
    earn_map: Mapping[str, Sequence],
    indexed: Mapping[str, Mapping],
    *,
    max_sessions: int,
) -> Tuple[pd.Series, pd.Series, List[dict]]:
    """Return (keep_mask, overlay_gain, detail)."""
    keep = []
    gains = []
    detail = []
    for _, row in df.iterrows():
        sym = str(row["stock"]).upper()
        if not bool(row.get("wheel_ok")):
            keep.append(False)
            gains.append(float("nan"))
            detail.append({"stock": sym, "reason": "not_filled", "keep": False})
            continue
        got = apply_wheel_earnings(
            row,
            earn_map.get(sym) or [],
            indexed.get(sym) or {},
            max_sessions=int(max_sessions),
        )
        keep.append(bool(got.keep))
        gains.append(got.gain_pct if got.keep else float("nan"))
        detail.append(
            {
                "stock": sym,
                "buy_date": str(row.get("buy_date") or ""),
                "keep": bool(got.keep),
                "used": bool(got.used),
                "skipped": bool(got.skipped),
                "reason": got.reason,
                "sessions_ahead": got.sessions_ahead,
                "gain_before": float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else None,
                "gain_after": got.gain_pct if got.keep else None,
            }
        )
    return (
        pd.Series(keep, index=df.index, dtype=bool),
        pd.Series(gains, index=df.index, dtype=float),
        detail,
    )


def _overlay_buy_write(
    df: pd.DataFrame,
    earn_map: Mapping[str, Sequence],
    indexed: Mapping[str, Mapping],
    *,
    max_sessions: int,
    source: Optional[Any],
    mark_mode: str,
) -> Tuple[pd.Series, pd.Series, List[dict]]:
    keep = []
    gains = []
    detail = []
    for _, row in df.iterrows():
        sym = str(row["stock"]).upper()
        if not bool(row.get("bw_ok")):
            keep.append(False)
            gains.append(float("nan"))
            detail.append({"stock": sym, "reason": "not_filled", "keep": False})
            continue
        debit = None
        win_flat = None
        from utils.research.last_15m_cc_earnings import earnings_window_for_row

        win = earnings_window_for_row(
            row,
            earn_map.get(sym) or [],
            indexed.get(sym) or {},
            max_sessions=int(max_sessions),
        )
        if win is not None and source is not None:
            _ev, flat_day, ahead = win
            win_flat = flat_day
            if ahead > 0 and row.get("call_expiry") and row.get("call_strike"):
                try:
                    exp = date.fromisoformat(str(row["call_expiry"])[:10])
                    debit = close_call_debit(
                        source,
                        symbol=sym,
                        expiration=exp,
                        strike=float(row["call_strike"]),
                        day=flat_day,
                        mark_mode=mark_mode,
                        spot_hint=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.debug("call BTC quote fail %s %s: %s", sym, flat_day, exc)
                    debit = None
        got = apply_buy_write_earnings(
            row,
            earn_map.get(sym) or [],
            indexed.get(sym) or {},
            max_sessions=int(max_sessions),
            call_close_debit=debit,
        )
        keep.append(bool(got.keep))
        gains.append(got.gain_pct if got.keep else float("nan"))
        detail.append(
            {
                "stock": sym,
                "buy_date": str(row.get("buy_date") or ""),
                "keep": bool(got.keep),
                "used": bool(got.used),
                "skipped": bool(got.skipped),
                "reason": got.reason,
                "sessions_ahead": got.sessions_ahead,
                "flatten_session": win_flat.isoformat() if win_flat else None,
                "call_close_debit": got.call_close_debit,
                "gain_before": float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else None,
                "gain_after": got.gain_pct if got.keep else None,
            }
        )
    return (
        pd.Series(keep, index=df.index, dtype=bool),
        pd.Series(gains, index=df.index, dtype=float),
        detail,
    )


def _run_buy_write(
    base_rows: pd.DataFrame,
    *,
    mark_mode: str,
    panels: Mapping[str, pd.DataFrame],
    source: Any,
    cache_path: Path,
    smoke_n: Optional[int],
    listed_only: bool = False,
    fresh_cache: bool = False,
    strike_mode: str = "delta",
) -> pd.DataFrame:
    cache = {} if fresh_cache else _load_cache(cache_path)
    LOG.info(
        "buy-write cache %s hits=%d mark=%s listed_only=%s fresh=%s",
        cache_path,
        len(cache),
        mark_mode,
        listed_only,
        fresh_cache,
    )
    work = base_rows.copy()
    if smoke_n is not None and len(work) > int(smoke_n):
        work = work.head(int(smoke_n))
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (_, row) in enumerate(work.iterrows(), start=1):
        key = _row_key(row["stock"], row["buy_date"], mark_mode)
        cached_rec = cache.get(key)
        src_tag = str(cached_rec.get("bw_source") or "") if cached_rec else ""
        # Re-sim when prior run only stored listed misses (before BS fallback),
        # or when listed_only and cache is a BS fill.
        reuse = (
            cached_rec is not None
            and cached_rec.get("bw_reason") is not None
            and (
                bool(cached_rec.get("bw_ok"))
                or src_tag in ("bs", "listed")
            )
        )
        if listed_only and src_tag == "bs":
            reuse = False
        if reuse:
            rows.append(dict(cached_rec))
        else:
            sym = str(row["stock"]).upper()
            if listed_only:
                got = simulate_buy_write(
                    row,
                    panels.get(sym),
                    source,
                    mark_mode=mark_mode,
                    strike_mode=strike_mode,
                )
                bw_src = "listed" if got.filled else "listed_miss"
            else:
                got = simulate_buy_write_listed_or_bs(
                    row,
                    panels.get(sym),
                    source,
                    mark_mode=mark_mode,
                    strike_mode=strike_mode,
                )
                bw_src = "bs" if (got.notes or "").startswith("bs") else "listed"
            rec = bw_result_to_record(row, got)
            rec["key"] = key
            rec["bw_source"] = bw_src
            _append_cache(cache_path, rec)
            cache[key] = rec
            rows.append(rec)
        if i % 50 == 0 or i == len(work):
            LOG.info(
                "buy-write %s %d/%d elapsed=%.0fs",
                mark_mode,
                i,
                len(work),
                time.time() - t0,
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel-mid", type=Path, default=DEFAULT_WHEEL_MID)
    ap.add_argument("--wheel-askbid", type=Path, default=DEFAULT_WHEEL_ASK)
    ap.add_argument("--earn-cache", type=Path, default=DEFAULT_EARN_CACHE)
    ap.add_argument("--max-sessions", type=int, default=10)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=25)
    ap.add_argument("--skip-buy-write", action="store_true")
    ap.add_argument(
        "--buy-write-only",
        action="store_true",
        help="Skip wheel scoring (still loads earnings/15m for BW overlay)",
    )
    ap.add_argument(
        "--option-feat",
        type=Path,
        default=ROOT
        / "reports"
        / "ascending_channels"
        / "2026-10-18"
        / "last15m_option_features.csv",
    )
    ap.add_argument("--mark-modes", type=str, default="mid,ask_bid")
    ap.add_argument(
        "--listed-only",
        action="store_true",
        help="Require Theta/Polygon listed calls (no BS fallback); ignore bs cache hits",
    )
    ap.add_argument(
        "--fresh-bw-cache",
        action="store_true",
        help="Ignore existing buy-write jsonl cache for this run",
    )
    ap.add_argument(
        "--strike-mode",
        choices=("delta", "atm"),
        default="delta",
        help="delta needs greeks; atm is the Value-plan path (no greeks/eod)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    t_wall0 = time.perf_counter()

    if not args.wheel_mid.exists():
        LOG.error("missing wheel mid CSV: %s", args.wheel_mid)
        return 2
    if not args.wheel_askbid.exists():
        LOG.error("missing wheel ask_bid CSV: %s", args.wheel_askbid)
        return 2

    wheel_mid = pd.read_csv(args.wheel_mid)
    wheel_ask = pd.read_csv(args.wheel_askbid)
    for df in (wheel_mid, wheel_ask):
        df["stock"] = df["stock"].astype(str).str.upper()
        df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")

    if args.smoke:
        n = int(args.smoke_n)
        wheel_mid = wheel_mid.head(n).copy()
        # Align ask_bid on same keys when possible
        keys = set(
            wheel_mid.apply(
                lambda r: _row_key(r["stock"], r["buy_date"], "ask_bid"), axis=1
            )
        )
        wheel_ask = wheel_ask[
            wheel_ask.apply(
                lambda r: _row_key(r["stock"], r["buy_date"], "ask_bid") in keys, axis=1
            )
        ].copy()
        if wheel_ask.empty:
            wheel_ask = pd.read_csv(args.wheel_askbid).head(n)
            wheel_ask["stock"] = wheel_ask["stock"].astype(str).str.upper()
            wheel_ask["buy_date"] = pd.to_datetime(wheel_ask["buy_date"], errors="coerce")

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(
        set(wheel_mid["stock"].tolist()) | set(wheel_ask["stock"].tolist())
    )
    earn_map = load_earnings_map(
        symbols,
        Path(args.earn_cache),
        workers=4,
        refresh_empty=False,
    )
    n_earn = sum(1 for s in symbols if earn_map.get(s))
    LOG.info("Earnings coverage %d/%d", n_earn, len(symbols))

    # 15m for flatten mids — pad through ~2x DTE past last buy
    buy_min = min(wheel_mid["buy_date"].min(), wheel_ask["buy_date"].min()) - timedelta(
        days=5
    )
    buy_max = max(wheel_mid["buy_date"].max(), wheel_ask["buy_date"].max()) + timedelta(
        days=int(DEFAULT_DTE) * 2 + 40
    )
    LOG.info(
        "Loading IB 15m symbols=%d %s..%s",
        len(symbols),
        buy_min.date(),
        buy_max.date(),
    )
    t0 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min,
        end=buy_max,
        workers=int(args.workers),
        use_cache=True,
    )
    LOG.info(
        "15m loaded %d/%d in %.1fs",
        sum(1 for s in symbols if panels_15m.get(s) is not None and not panels_15m[s].empty),
        len(symbols),
        time.perf_counter() - t0,
    )
    # build_hold_session_index expects trades with buy/sell — use wheel_mid as spine
    indexed = build_hold_session_index(
        panels_15m, wheel_mid, pad_before=5, pad_after=int(DEFAULT_DTE) * 2 + 40
    )
    LOG.info("Indexed 15m sessions for %d symbols", len(indexed))

    summary_rows: List[dict] = []
    mark_modes = [m.strip() for m in str(args.mark_modes).split(",") if m.strip()]

    # ---- Wheel baselines + earnings overlays ----
    if not args.buy_write_only:
        for label, df in (("wheel_mid", wheel_mid), ("wheel_ask_bid", wheel_ask)):
            filled = df[df["wheel_ok"] == True].copy()  # noqa: E712
            g0 = pd.to_numeric(filled["gain_pct"], errors="coerce")
            summary_rows.append(_score("%s_baseline" % label, filled, g0))

            keep, g1, detail = _overlay_wheel(
                filled, earn_map, indexed, max_sessions=int(args.max_sessions)
            )
            part = filled.loc[keep].copy()
            summary_rows.append(
                _score(
                    "%s_earn_flat%d" % (label, int(args.max_sessions)),
                    part,
                    g1.loc[keep],
                )
            )
            det_path = outdir / ("%s_earn_flat_detail.csv" % label)
            pd.DataFrame(detail).to_csv(det_path, index=False)
            n_skip = sum(1 for d in detail if d.get("skipped"))
            n_flat = sum(1 for d in detail if d.get("used") and not d.get("skipped"))
            LOG.info(
                "%s earn overlay skip=%d flatten=%d -> %s",
                label,
                n_skip,
                n_flat,
                det_path,
            )
    else:
        # Keep prior wheel rows in the combined summary if present.
        prior = outdir / "last_15m_cc_earnings_summary.csv"
        if prior.exists():
            old = pd.read_csv(prior)
            for _, r in old.iterrows():
                if str(r.get("label", "")).startswith("wheel_"):
                    summary_rows.append(r.to_dict())

    # ---- Buy-write ----
    if not args.skip_buy_write:
        theta_ok = False
        try:
            theta_ok = bool(get_thetadata_client().is_ready())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("theta not ready: %s", exc)
        if not theta_ok:
            LOG.warning(
                "Theta Terminal not ready — buy-write will miss most chains. "
                "Start scripts/pipeline/thetadata_terminal.py"
            )

        if args.listed_only:
            theta_cache = outdir / "theta_value_chain_cache"
        else:
            theta_cache = WHEEL_DIR / "theta_wheel_cache"
        theta_cache.mkdir(parents=True, exist_ok=True)
        source = ThetaEquitySource(cache_dir=theta_cache)
        LOG.info("Theta chain cache %s", theta_cache)

        # Daily panels for expiry spots
        start_1d = (buy_min - timedelta(days=5)).date()
        end_1d = buy_max.date()
        LOG.info("Loading 1d panels n=%d", len(symbols))
        panels_1d = load_ohlcv_many(
            symbols,
            timeframe="1d",
            start=start_1d,
            end=end_1d,
            provider="ALPACA",
            fallback_provider="IB",
            merge_mode="prefix",
            use_cache=True,
        )

        bw_cache = outdir / "last_15m_buy_write_listed_cache.jsonl"
        if args.listed_only:
            # Keep listed results separate from the prior BS cache.
            bw_cache = outdir / "last_15m_buy_write_listed_cache.jsonl"
        else:
            bw_cache = outdir / "last_15m_buy_write_cache.jsonl"
        smoke_n = int(args.smoke_n) if args.smoke else None
        # Use mid CSV as the attempted universe spine (same 1118 rows).
        spine = wheel_mid.copy()
        # Attach name ATM IV when available (better BS fallback than ATR-only).
        feat_path = Path(args.option_feat)
        if feat_path.exists():
            feat = pd.read_csv(feat_path)
            feat["stock"] = feat["stock"].astype(str).str.upper()
            feat["buy_date"] = pd.to_datetime(feat["buy_date"], errors="coerce")
            spine["_bd"] = pd.to_datetime(spine["buy_date"], errors="coerce").dt.normalize()
            feat["_bd"] = feat["buy_date"].dt.normalize()
            cols = ["stock", "_bd", "name_atm_iv"]
            spine = spine.merge(feat[cols], on=["stock", "_bd"], how="left")
            LOG.info(
                "name_atm_iv marked %d/%d",
                int(spine["name_atm_iv"].notna().sum()),
                len(spine),
            )
            spine = spine.drop(columns=["_bd"], errors="ignore")

        for mark_mode in mark_modes:
            strike_mode = str(args.strike_mode)
            if args.listed_only and strike_mode == "delta":
                # Value plan has no greeks/eod; ATM is the honest listed path.
                strike_mode = "atm"
                LOG.info("listed-only: forcing strike_mode=atm (no greeks on Value)")
            bw = _run_buy_write(
                spine,
                mark_mode=mark_mode,
                panels=panels_1d,
                source=source,
                cache_path=bw_cache,
                smoke_n=smoke_n,
                listed_only=bool(args.listed_only),
                fresh_cache=bool(args.fresh_bw_cache),
                strike_mode=strike_mode,
            )
            stem = "last_15m_buy_write_listed" if args.listed_only else "last_15m_buy_write"
            bw_path = outdir / ("%s_%s.csv" % (stem, mark_mode))
            bw.to_csv(bw_path, index=False)
            LOG.info("Wrote %s n=%d ok=%s", bw_path, len(bw), int(bw["bw_ok"].sum()))

            filled = bw[bw["bw_ok"] == True].copy()  # noqa: E712
            g0 = pd.to_numeric(filled["gain_pct"], errors="coerce")
            summary_rows.append(_score("buy_write_%s_baseline" % mark_mode, filled, g0))

            keep, g1, detail = _overlay_buy_write(
                filled,
                earn_map,
                indexed,
                max_sessions=int(args.max_sessions),
                source=source,
                mark_mode=mark_mode,
            )
            part = filled.loc[keep].copy()
            summary_rows.append(
                _score(
                    "buy_write_%s_earn_flat%d" % (mark_mode, int(args.max_sessions)),
                    part,
                    g1.loc[keep],
                )
            )
            det_path = outdir / ("buy_write_%s_earn_flat_detail.csv" % mark_mode)
            pd.DataFrame(detail).to_csv(det_path, index=False)

    summary = pd.DataFrame(summary_rows)
    sum_path = outdir / "last_15m_cc_earnings_summary.csv"
    summary.to_csv(sum_path, index=False)
    meta = {
        "wall_s": round(time.perf_counter() - t_wall0, 1),
        "max_sessions": int(args.max_sessions),
        "wheel_mid": str(args.wheel_mid),
        "wheel_askbid": str(args.wheel_askbid),
        "smoke": bool(args.smoke),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (outdir / "last_15m_cc_earnings_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    LOG.info("Summary -> %s wall=%.1fs", sum_path, meta["wall_s"])
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    # Winner pick among earn-flat books (prefer ask_bid when present).
    earn = summary[summary["label"].str.contains("earn_flat", na=False)].copy()
    if not earn.empty and earn["expectancy_pct"].notna().any():
        best = earn.sort_values(
            ["profit_factor", "expectancy_pct", "geo_year_pf"],
            ascending=False,
        ).iloc[0]
        print(
            "\nBEST earn-flat by PF/E/geo: %s E=%s PF=%s geo=%s n=%s"
            % (
                best["label"],
                best["expectancy_pct"],
                best["profit_factor"],
                best["geo_year_pf"],
                best["n"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
