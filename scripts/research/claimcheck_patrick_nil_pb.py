#!/usr/bin/env python3
"""Claim-check Patrick Nil P/B ranges on daily bars (100-name cap panel).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python -u scripts\\research\\claimcheck_patrick_nil_pb.py --n-per-bucket 33 --start 2006-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_volume_geometry import book_stats, geom_mean_year_pf  # noqa: E402
from utils.research.metrics import passes_phase5_gates, perf_stats  # noqa: E402
from utils.research.panel import frames_to_wide, to_naive_index  # noqa: E402
from utils.research.patrick_nil_pb import (  # noqa: E402
    BOOK_CONFIRMED,
    BOOK_FAKE,
    BOOK_RANGE,
    CLAIMED,
    NilParams,
    YEAR_BUCKETS,
    apply_friction,
    build_cap_universe,
    collect_symbol_trades,
    drop_top_n_expectancy,
    max_concurrent,
    occupancy_equity_from_trades,
    occupancy_take,
    params_dict,
    sitout_after_loss_streak,
    summarize_by_year,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_patrick_nil_pb")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "patrick_nil_pb"
BOOKS = (BOOK_RANGE, BOOK_CONFIRMED, BOOK_FAKE)


def _stats_block(label: str, trades: pd.DataFrame, gain_col: str = "gain_pct") -> Dict[str, Any]:
    g = (
        pd.to_numeric(trades[gain_col], errors="coerce")
        if not trades.empty
        else pd.Series(dtype=float)
    )
    s = book_stats(g)
    years = summarize_by_year(trades, gain_col=gain_col, buckets=YEAR_BUCKETS)
    geo = geom_mean_year_pf(years)
    drop3 = drop_top_n_expectancy(g.tolist(), 3)
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s drop-top-3 E=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s.get("win_rate_pct"),
        s.get("median_pct"),
        None if drop3 is None else round(drop3, 4),
    )
    if not years.empty:
        print(years.to_string(index=False))
    if not trades.empty and "buy_date" in trades.columns:
        bd = pd.to_datetime(trades["buy_date"], errors="coerce")
        y22 = trades.loc[bd.dt.year == 2022]
        if not y22.empty:
            s22 = book_stats(pd.to_numeric(y22[gain_col], errors="coerce"))
            LOG.info(
                "%s calendar 2022 n=%s E=%s PF=%s WR=%s",
                label,
                s22.get("n"),
                s22.get("expectancy_pct"),
                s22.get("profit_factor"),
                s22.get("win_rate_pct"),
            )
        if "exit_reason" in trades.columns:
            print(trades["exit_reason"].value_counts().to_string())
        if "cap_bucket" in trades.columns:
            print(trades["cap_bucket"].value_counts().to_string())
    return {
        "label": label,
        "stats": s,
        "geo_pf": None if geo is None else round(float(geo), 3),
        "drop_top3_e": None if drop3 is None else round(float(drop3), 4),
        "years": years.to_dict(orient="records") if years is not None and not years.empty else [],
    }


def _spy_buy_hold(spy_close: pd.Series, start: str, end: str):
    s = to_naive_index(spy_close.astype(float)).dropna().sort_index()
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if s.empty or len(s) < 2:
        return None, None, None
    eq = s / float(s.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    stats = perf_stats("SPY_buy_hold", eq, invested=inv)
    return eq, inv, stats


def _phase5_for_book(
    label: str,
    trades: pd.DataFrame,
    close_panel: pd.DataFrame,
    spy_stats,
    *,
    max_open: int,
    eval_start: str,
    eval_end: str,
) -> Dict[str, Any]:
    if trades.empty or spy_stats is None:
        return {"label": label, "max_open": max_open, "gate": "SKIP", "reason": "empty"}
    raw = trades.to_dict(orient="records")
    # occupancy_take needs buy_ts/sell_ts comparable
    for r in raw:
        r["buy_ts"] = pd.Timestamp(r["buy_ts"])
        r["sell_ts"] = pd.Timestamp(r["sell_ts"])
    kept = occupancy_take(raw, max_open=max(1, int(max_open)))
    occ = pd.DataFrame(kept)
    if occ.empty:
        return {"label": label, "max_open": max_open, "gate": "SKIP", "reason": "occ empty"}
    eq, inv = occupancy_equity_from_trades(occ, close_panel, max_open=int(max_open))
    if eq.empty:
        return {"label": label, "max_open": max_open, "gate": "SKIP", "reason": "no equity"}
    eq = eq.loc[(eq.index >= pd.Timestamp(eval_start)) & (eq.index <= pd.Timestamp(eval_end))]
    inv = inv.reindex(eq.index).fillna(0.0)
    if eq.empty:
        return {"label": label, "max_open": max_open, "gate": "SKIP", "reason": "clipped empty"}
    gains = pd.to_numeric(occ["gain_pct"], errors="coerce").dropna() / 100.0
    st = perf_stats(
        label,
        eq / float(eq.iloc[0]),
        invested=inv,
        spy_cagr=spy_stats.cagr,
        spy_sharpe=spy_stats.sharpe,
        trade_rets=gains.tolist(),
    )
    ok, msg = passes_phase5_gates(st, spy_stats)
    # 2022 window
    eq22 = eq.loc[(eq.index >= "2022-01-01") & (eq.index <= "2022-12-31")]
    gate22 = None
    if len(eq22) >= 5:
        if "SPY" in close_panel.columns:
            spy_c = to_naive_index(close_panel["SPY"].astype(float)).dropna()
            spy_c = spy_c.loc[(spy_c.index >= "2022-01-01") & (spy_c.index <= "2022-12-31")]
            if len(spy_c) >= 5:
                spy22 = perf_stats("SPY_2022", spy_c / float(spy_c.iloc[0]))
                st22 = perf_stats("book_2022", eq22 / float(eq22.iloc[0]))
                ok22, msg22 = passes_phase5_gates(st22, spy22, window="2022")
                gate22 = {"ok": ok22, "msg": msg22, "sharpe": st22.sharpe, "mdd": st22.max_drawdown}
    LOG.info(
        "%s occ%d Phase5 %s Sharpe=%.2f MDD=%.2f vs SPY Sharpe=%.2f MDD=%.2f",
        label,
        max_open,
        msg,
        st.sharpe,
        st.max_drawdown,
        spy_stats.sharpe,
        spy_stats.max_drawdown,
    )
    return {
        "label": label,
        "max_open": max_open,
        "gate": msg,
        "ok": bool(ok),
        "sharpe": st.sharpe,
        "cagr": st.cagr,
        "mdd": st.max_drawdown,
        "n_trades": int(len(occ)),
        "gate_2022": gate22,
    }


def _run_book(
    book: str,
    frames: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    params: NilParams,
    *,
    close_panel: pd.DataFrame,
    spy_stats,
    outdir: Path,
    friction_pcts: Sequence[float],
    occ_caps: Sequence[int],
    eval_start: str,
    eval_end: str,
) -> Dict[str, Any]:
    sym_to_cap = {
        str(r["symbol"]).upper(): str(r["cap_bucket"])
        for _, r in meta.iterrows()
    }
    raw: List[dict] = []
    for i, (sym, df) in enumerate(frames.items(), 1):
        if sym.upper() == "SPY":
            continue
        rows = collect_symbol_trades(
            df,
            symbol=sym,
            cap_bucket=sym_to_cap.get(str(sym).upper(), ""),
            params=params,
            books=(book,),
        )
        raw.extend(rows)
        if i % 25 == 0:
            LOG.info("%s scan %d/%d trades=%d", book, i, len(frames), len(raw))

    trades = pd.DataFrame(raw)
    if not trades.empty:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y-%m-%d")
        trades["sell_date"] = pd.to_datetime(trades["sell_date"]).dt.strftime("%Y-%m-%d")

    stem = book
    csv_path = outdir / ("nil_pb_%s_trades.csv" % stem)
    trades.to_csv(csv_path, index=False)
    peak, med = max_concurrent(raw) if raw else (0, 0.0)
    LOG.info(
        "%s n=%d max_conc=%s median_conc=%s -> %s",
        book,
        len(trades),
        peak,
        med,
        csv_path,
    )

    payload: Dict[str, Any] = {
        "book": book,
        "claimed": CLAIMED,
        "params": params_dict(params),
        "n_raw": len(raw),
        "max_concurrent": peak,
        "median_concurrent": med,
        "pooled": _stats_block("%s pooled" % book, trades),
        "by_cap": {},
        "friction": {},
        "occupancy_gates": [],
        "sitout3": None,
    }

    for bucket in ("small", "mid", "large"):
        if trades.empty or "cap_bucket" not in trades.columns:
            continue
        sub = trades.loc[trades["cap_bucket"] == bucket]
        payload["by_cap"][bucket] = _stats_block("%s %s" % (book, bucket), sub)

    for fr in friction_pcts:
        if trades.empty:
            continue
        net = apply_friction(trades, float(fr))
        payload["friction"][str(fr)] = _stats_block(
            "%s fric%.2f" % (book, float(fr)), net, gain_col="gain_pct_net"
        )

    # Hygiene diagnostic (not a promote overlay)
    if raw:
        sit = sitout_after_loss_streak(raw, max_losses=3)
        sit_df = pd.DataFrame(sit)
        if not sit_df.empty:
            sit_df["buy_date"] = pd.to_datetime(sit_df["buy_date"]).dt.strftime("%Y-%m-%d")
            sit_df["sell_date"] = pd.to_datetime(sit_df["sell_date"]).dt.strftime("%Y-%m-%d")
        payload["sitout3"] = _stats_block("%s sitout3" % book, sit_df)

    for cap in occ_caps:
        if trades.empty:
            continue
        gate = _phase5_for_book(
            "%s_occ%d" % (book, int(cap)),
            trades,
            close_panel,
            spy_stats,
            max_open=int(cap),
            eval_start=eval_start,
            eval_end=eval_end,
        )
        payload["occupancy_gates"].append(gate)

    (outdir / ("nil_pb_%s_summary.json" % stem)).write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Patrick Nil P/B 1d claim-check")
    ap.add_argument("--n-per-bucket", type=int, default=33)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--symbols-csv", type=Path, default=None, help="Reuse a prior symbol list")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--books", default="all", help="all|range_fade|confirmed_breakout|fake_breakout")
    ap.add_argument("--occ", default="1,3,8")
    ap.add_argument("--friction", default="0.10,0.25")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--write-status", action="store_true", help="Write docs/status_log note")
    args = ap.parse_args()

    outdir = args.outdir or dated_outdir(DEFAULT_OUT)
    outdir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime.utcnow()
    )
    eval_start = "2018-01-01"
    eval_end = end.strftime("%Y-%m-%d")

    t0 = time.perf_counter()
    if args.symbols_csv and Path(args.symbols_csv).exists():
        meta = pd.read_csv(args.symbols_csv)
        meta["symbol"] = meta["symbol"].astype(str).str.upper()
    else:
        meta = build_cap_universe(
            n_per_bucket=int(args.n_per_bucket),
            min_adv=float(args.min_adv),
        )
        meta.to_csv(outdir / "nil_pb_symbols.csv", index=False)
    timings["universe"] = time.perf_counter() - t0
    LOG.info(
        "Universe %d names (small=%d mid=%d large=%d) %.1fs",
        len(meta),
        int((meta["cap_bucket"] == "small").sum()) if not meta.empty else 0,
        int((meta["cap_bucket"] == "mid").sum()) if not meta.empty else 0,
        int((meta["cap_bucket"] == "large").sum()) if not meta.empty else 0,
        timings["universe"],
    )
    if meta.empty:
        LOG.error("Empty universe")
        return 1

    symbols = sorted(set(meta["symbol"].astype(str).str.upper().tolist()) | {"SPY"})
    t0 = time.perf_counter()
    frames = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
        fallback_provider="IB",
        merge_mode="prefix",
    )
    timings["load"] = time.perf_counter() - t0
    LOG.info("Loaded %d/%d frames (%.1fs)", len(frames), len(symbols), timings["load"])

    close_panel = frames_to_wide(frames, "close")
    spy_stats = None
    if "SPY" in close_panel.columns:
        _, _, spy_stats = _spy_buy_hold(close_panel["SPY"], eval_start, eval_end)
        if spy_stats is not None:
            LOG.info(
                "SPY buy-hold Sharpe=%.2f CAGR=%.2f%% MDD=%.2f%%",
                spy_stats.sharpe,
                100.0 * spy_stats.cagr,
                100.0 * spy_stats.max_drawdown,
            )

    book_arg = str(args.books).strip().lower()
    if book_arg in ("all", "*"):
        books = list(BOOKS)
    else:
        books = [b.strip() for b in book_arg.split(",") if b.strip()]

    friction_pcts = [float(x) for x in str(args.friction).split(",") if x.strip()]
    occ_caps = [int(x) for x in str(args.occ).split(",") if x.strip()]
    params = NilParams()

    summaries: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for book in books:
        LOG.info("=== Book %s ===", book)
        summaries[book] = _run_book(
            book,
            frames,
            meta,
            params,
            close_panel=close_panel,
            spy_stats=spy_stats,
            outdir=outdir,
            friction_pcts=friction_pcts,
            occ_caps=occ_caps,
            eval_start=eval_start,
            eval_end=eval_end,
        )
    timings["fills"] = time.perf_counter() - t0
    timings["wall"] = time.perf_counter() - t_all
    LOG.info("Wall %.1fs %s", timings["wall"], timings)

    (outdir / "nil_pb_timings.txt").write_text(
        "\n".join("%s=%.3f" % (k, v) for k, v in timings.items()) + "\n",
        encoding="utf-8",
    )
    (outdir / "nil_pb_run_summary.json").write_text(
        json.dumps(
            {
                "claimed": CLAIMED,
                "params": params_dict(params),
                "n_symbols": len(meta),
                "timings": timings,
                "books": {k: {"n_raw": v.get("n_raw"), "pooled": v.get("pooled")} for k, v in summaries.items()},
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.write_status:
        _write_status(outdir, summaries, timings, meta)

    return 0


def _write_status(
    outdir: Path,
    summaries: Dict[str, Any],
    timings: Dict[str, float],
    meta: pd.DataFrame,
) -> None:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    status_path = ROOT / "docs" / "status_log" / "edge_hunt" / (
        "%s_patrick_nil_pb_claimcheck.md" % day
    )
    lines = [
        "# Patrick Nil P/B 1d claim-check (%s)" % day,
        "",
        "User-pasted Nil sketch mapped to daily bars + weekly SMA50 HTF. "
        "Three books on ~100 names (small/mid/large SEC mkt_cap).",
        "",
        "Harness: `utils/research/patrick_nil_pb.py`, "
        "`scripts/research/claimcheck_patrick_nil_pb.py`",
        "Artifacts: `%s`" % outdir.as_posix().replace("\\", "/"),
        "Wall clock: **%.1fs** (universe %.1fs, load %.1fs, fills %.1fs)"
        % (
            timings.get("wall", 0),
            timings.get("universe", 0),
            timings.get("load", 0),
            timings.get("fills", 0),
        ),
        "",
        "## Universe",
        "",
        "| bucket | n | sec | adv-proxy |",
        "|--------|--:|----:|----------:|",
    ]
    for b in ("small", "mid", "large"):
        sub = meta.loc[meta["cap_bucket"] == b] if not meta.empty else meta
        n_sec = int((sub["cap_proxy"] == "sec").sum()) if not sub.empty else 0
        n_adv = int((sub["cap_proxy"] == "adv").sum()) if not sub.empty else 0
        lines.append("| %s | %d | %d | %d |" % (b, len(sub), n_sec, n_adv))
    lines.extend(["", "## Results", ""])

    any_promote = False
    for book, payload in summaries.items():
        pooled = payload.get("pooled") or {}
        stats = pooled.get("stats") or {}
        lines.append("### %s" % book)
        lines.append("")
        lines.append(
            "Pooled **n=%s E=%s PF=%s geo=%s** (drop-top-3 E=%s)"
            % (
                stats.get("n"),
                stats.get("expectancy_pct"),
                stats.get("profit_factor"),
                pooled.get("geo_pf"),
                pooled.get("drop_top3_e"),
            )
        )
        fr = payload.get("friction") or {}
        for k, block in fr.items():
            st = (block or {}).get("stats") or {}
            lines.append(
                "- fric %s: E=%s PF=%s" % (k, st.get("expectancy_pct"), st.get("profit_factor"))
            )
        for bucket, block in (payload.get("by_cap") or {}).items():
            st = (block or {}).get("stats") or {}
            lines.append(
                "- %s: n=%s E=%s PF=%s"
                % (bucket, st.get("n"), st.get("expectancy_pct"), st.get("profit_factor"))
            )
        for g in payload.get("occupancy_gates") or []:
            lines.append(
                "- occ%s: %s (Sharpe=%.2f MDD=%.2f n=%s)"
                % (
                    g.get("max_open"),
                    g.get("gate"),
                    float(g.get("sharpe") or 0),
                    float(g.get("mdd") or 0),
                    g.get("n_trades"),
                )
            )
            if g.get("ok"):
                any_promote = True
        lines.append("")

    verdict = (
        "**Soft-interesting only if occupancy cleared Phase-5 — review gates above.**"
        if any_promote
        else "**Verdict: no promote.**"
    )
    lines.insert(3, verdict)
    lines.insert(4, "")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Wrote %s", status_path)

    # Roll-up one-liner
    roll = ROOT / "docs" / "status_log" / "current_status.md"
    if roll.exists():
        marker = "### Patrick Nil P/B 1d ranges"
        block = (
            "\n%s — %s\n\n"
            "Status: [%s](edge_hunt/%s). Daily P/B fair-value ranges "
            "(range fade / confirmed BO / fake BO) on ~100 SEC cap-bucket names. "
            "%s Code: `claimcheck_patrick_nil_pb.py`.\n"
            % (
                marker,
                day,
                day + "_patrick_nil_pb_claimcheck.md",
                day + "_patrick_nil_pb_claimcheck.md",
                "No promote." if not any_promote else "See status for occupancy gates.",
            )
        )
        text = roll.read_text(encoding="utf-8")
        if marker not in text:
            # Insert after first "## Edge hunt"ish section or append near Silent Flip
            needle = "### Silent Flip 15m open-range"
            if needle in text:
                text = text.replace(needle, block + "\n" + needle, 1)
            else:
                text = text.rstrip() + "\n" + block
            roll.write_text(text, encoding="utf-8")
            LOG.info("Updated current_status.md")


if __name__ == "__main__":
    raise SystemExit(main())
