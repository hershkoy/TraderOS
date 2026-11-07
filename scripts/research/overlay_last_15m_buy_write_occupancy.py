"""Buy-write mid +/- slip occupancy (+ optional call MTM) vs SPY.

Reads listed buy-write mid CSV, builds slip books ($0 / $0.05 / $0.10 credit
haircut), FCFS equal-slot mark-at-cost occupancy, and optional EOD call MTM
on max_open=8 (Theta Value history/eod).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_buy_write_occupancy.py
  python scripts\\research\\overlay_last_15m_buy_write_occupancy.py --with-mtm
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_buy_write import (  # noqa: E402
    buy_write_unrealized_frac,
    filled_buy_write_to_occ_rows,
)
from utils.research.last_15m_short_put_bs import _as_date, _panel_dates  # noqa: E402
from utils.research.last_15m_volume_geometry import book_stats  # noqa: E402
from utils.research.last_15m_wheel import (  # noqa: E402
    ThetaEquitySource,
    _mark_map_from_path,
    occupancy_fcfs_wheel,
    spy_buyhold_equity,
    wheel_slot_equity,
)
from utils.research.metrics import (  # noqa: E402
    passes_phase5_gates,
    perf_stats,
    rebase_equity,
    window_equity,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.spy_wheel import mtm_mark  # noqa: E402

LOG = logging.getLogger("overlay_bw_occ")

DEFAULT_MID = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-11-03_cc_earnings_listed"
    / "last_15m_buy_write_listed_mid.csv"
)
DEFAULT_ASKBID = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-11-03_cc_earnings_listed"
    / "last_15m_buy_write_listed_ask_bid.csv"
)
SLIPS = (0.0, 0.05, 0.10)
MAX_OPENS = (1, 4, 8, 15)
WINDOWS = (
    ("FULL", None, None),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2022-23", "2022-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", "2026-12-31"),
)


def _stats_dict(st) -> Dict[str, Any]:
    return {
        "cagr": round(float(st.cagr), 4),
        "sharpe": round(float(st.sharpe), 3),
        "max_drawdown": round(float(st.max_drawdown), 4),
        "pct_time_invested": round(float(st.pct_time_invested), 3),
        "n_trades": int(st.n_trades),
        "win_rate": round(float(st.win_rate), 4),
        "profit_factor": round(float(st.profit_factor), 3),
        "expectancy": round(float(st.expectancy), 4),
    }


def _score_window(
    label: str,
    eq: pd.Series,
    inv: pd.Series,
    spy_eq: pd.Series,
    trades: Sequence[Dict[str, Any]],
    *,
    start: Optional[str],
    end: Optional[str],
    window: Optional[str],
) -> Dict[str, Any]:
    if start and end:
        eq_w, inv_w = window_equity(eq, inv, start, end)
        spy_w, _ = window_equity(spy_eq, None, start, end)
    else:
        eq_w = rebase_equity(eq)
        inv_w = inv.reindex(eq_w.index).fillna(0.0)
        spy_w = rebase_equity(spy_eq.reindex(eq_w.index).ffill().bfill())
    common = eq_w.index.intersection(spy_w.index)
    if len(common) < 2:
        return {"label": label, "gate_pass": False, "gate_reason": "insufficient equity"}
    eq_w = eq_w.reindex(common).ffill()
    inv_w = inv_w.reindex(common).fillna(0.0)
    spy_w = spy_w.reindex(common).ffill()
    eq_w = eq_w / float(eq_w.iloc[0])
    spy_w = spy_w / float(spy_w.iloc[0])
    if start and end:
        t_win = [
            t
            for t in trades
            if start <= pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d") <= end
        ]
    else:
        t_win = list(trades)
    rets = [float(t["gain_pct"]) / 100.0 for t in t_win]
    spy_st = perf_stats("SPY", spy_w, notes="price NAV", trade_rets=[])
    sleeve = perf_stats(
        label,
        eq_w,
        inv_w,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes="buy-write equal-slot stock notional",
        trade_rets=rets,
    )
    ok, reason = passes_phase5_gates(sleeve, spy_st, window=window)
    LOG.info(
        "%s n=%s E=%.2f PF=%.2f Sharpe=%.2f CAGR=%.1f%% MDD=%.1f%% | "
        "SPY Sharpe=%.2f MDD=%.1f%% | %s",
        label,
        sleeve.n_trades,
        100.0 * sleeve.expectancy,
        sleeve.profit_factor,
        sleeve.sharpe,
        100.0 * sleeve.cagr,
        100.0 * sleeve.max_drawdown,
        spy_st.sharpe,
        100.0 * spy_st.max_drawdown,
        reason,
    )
    return {
        "label": label,
        "gate_pass": bool(ok),
        "gate_reason": reason,
        "sleeve": _stats_dict(sleeve),
        "spy": _stats_dict(spy_st),
        "occ_n": len(t_win),
    }


def _pooled_slip_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for slip in SLIPS:
        occ = filled_buy_write_to_occ_rows(df, slip=float(slip))
        g = pd.Series([r["gain_pct"] for r in occ], dtype=float)
        s = book_stats(g)
        # year slices
        by_year = {}
        for lab, a, b in (
            ("2022", "2022-01-01", "2022-12-31"),
            ("2023", "2023-01-01", "2023-12-31"),
            ("2022-23", "2022-01-01", "2023-12-31"),
        ):
            sub = [
                r
                for r in occ
                if a <= pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d") <= b
            ]
            by_year[lab] = book_stats(pd.Series([x["gain_pct"] for x in sub], dtype=float))
        rows.append(
            {
                "slip": slip,
                "n": s["n"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "e_2022": by_year["2022"]["expectancy_pct"],
                "pf_2022": by_year["2022"]["profit_factor"],
                "e_2023": by_year["2023"]["expectancy_pct"],
                "pf_2023": by_year["2023"]["profit_factor"],
                "e_2022_23": by_year["2022-23"]["expectancy_pct"],
                "pf_2022_23": by_year["2022-23"]["profit_factor"],
            }
        )
        LOG.info(
            "pooled slip=%.2f n=%s E=%s PF=%s | 2022 E=%s PF=%s",
            slip,
            s["n"],
            s["expectancy_pct"],
            s["profit_factor"],
            by_year["2022"]["expectancy_pct"],
            by_year["2022"]["profit_factor"],
        )
    return rows


def _liq_filter_from_askbid(
    mid_df: pd.DataFrame,
    ask_df: pd.DataFrame,
    *,
    min_adv: Optional[float],
    max_half_pct: Optional[float],
) -> pd.DataFrame:
    """Keep mid fills whose (mid_credit - bid_credit)/mid <= max_half_pct and ADV gate."""
    m = mid_df[mid_df["bw_ok"] == True].copy()  # noqa: E712
    a = ask_df[ask_df["bw_ok"] == True].copy()  # noqa: E712
    m["_k"] = (
        m["stock"].astype(str).str.upper()
        + "|"
        + pd.to_datetime(m["buy_date"]).dt.strftime("%Y-%m-%d")
    )
    a["_k"] = (
        a["stock"].astype(str).str.upper()
        + "|"
        + pd.to_datetime(a["buy_date"]).dt.strftime("%Y-%m-%d")
    )
    j = m.merge(
        a[["_k", "call_credit"]].rename(columns={"call_credit": "call_credit_bid"}),
        on="_k",
        how="inner",
    )
    mid_c = pd.to_numeric(j["call_credit"], errors="coerce")
    bid_c = pd.to_numeric(j["call_credit_bid"], errors="coerce")
    half_pct = 100.0 * (mid_c - bid_c).clip(lower=0) / mid_c.replace(0, pd.NA)
    j["half_pct"] = half_pct
    keep = pd.Series(True, index=j.index)
    if min_adv is not None:
        keep &= pd.to_numeric(j["adv_20"], errors="coerce").fillna(0) >= float(min_adv)
    if max_half_pct is not None:
        keep &= j["half_pct"].fillna(999) <= float(max_half_pct)
    out = j.loc[keep].drop(columns=["_k", "call_credit_bid", "half_pct"], errors="ignore")
    LOG.info(
        "liq filter adv>=%s half<=%s%%: matched=%d kept=%d med_half%%=%s",
        min_adv,
        max_half_pct,
        len(j),
        len(out),
        None if out.empty else round(float(half_pct.loc[keep].median()), 2),
    )
    return out


def bw_slot_equity_mtm(
    trades: Sequence[Dict[str, Any]],
    calendar: pd.DatetimeIndex,
    *,
    max_open: int,
    spot_panels: Dict[str, pd.DataFrame],
    call_marks: Dict[int, Dict[date, float]],
) -> Tuple[pd.Series, pd.Series]:
    """Equal-slot MTM: long stock + short call vs entry notional."""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize()
    cal = cal[~cal.duplicated(keep="first")].sort_values()
    slots = max(1, int(max_open))
    alloc = 1.0 / float(slots)

    indexed = []
    for i, tr in enumerate(trades):
        indexed.append({**tr, "_i": i})

    by_buy: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for tr in indexed:
        buy = pd.Timestamp(tr["entry_date"]).normalize()
        by_buy.setdefault(buy, []).append(tr)

    cash = 1.0
    open_pos: List[Dict[str, Any]] = []
    eq_pts: List[float] = []
    inv_pts: List[float] = []

    def _spot(sym: str, day: date) -> Optional[float]:
        panel = spot_panels.get(sym)
        if panel is None or panel.empty:
            return None
        work = panel.copy()
        work.index = _panel_dates(work)
        exact = work.loc[work.index.date == day]
        if not exact.empty:
            return float(exact.iloc[0]["close"])
        before = work.loc[work.index.date <= day]
        if before.empty:
            return None
        return float(before.iloc[-1]["close"])

    def _mark_open() -> float:
        total = cash
        for pos in open_pos:
            total += float(pos["capital"]) * (1.0 + float(pos["u"]))
        return total

    for dt in cal:
        dtn = pd.Timestamp(dt).normalize()
        dday = dtn.date()
        # Exits before today
        still = []
        for pos in open_pos:
            if pd.Timestamp(pos["sell"]) < dtn:
                cash += float(pos["capital"]) * (1.0 + float(pos["r"]))
            else:
                still.append(pos)
        open_pos = still

        for row in by_buy.get(dtn, []):
            if len(open_pos) >= slots:
                break
            sym = str(row.get("symbol") or "").upper()
            if any(p["sym"] == sym for p in open_pos):
                continue
            eq_now = _mark_open()
            spend = min(eq_now * alloc, cash)
            if spend <= 1e-12:
                continue
            cash -= spend
            open_pos.append(
                {
                    "sym": sym,
                    "sell": pd.Timestamp(row["exit_date"]).normalize(),
                    "capital": spend,
                    "r": float(row["gain_pct"]) / 100.0,
                    "u": 0.0,
                    "tr": row,
                    "idx": int(row["_i"]),
                }
            )

        # MTM open
        still = []
        for pos in open_pos:
            sell = pd.Timestamp(pos["sell"]).normalize()
            if sell <= dtn:
                cash += float(pos["capital"]) * (1.0 + float(pos["r"]))
                continue
            tr = pos["tr"]
            entry_px = float(tr["buy_price"] or 0.0)
            credit = float(tr.get("call_credit") or 0.0)
            # Apply slip already baked into gain; for MTM use slipped credit ≈
            # credit - slip stored on row.
            slip = float(tr.get("slip") or 0.0)
            credit_u = max(0.0, credit - slip)
            spot = _spot(pos["sym"], dday)
            cm = call_marks.get(int(pos["idx"]), {}).get(dday)
            u = buy_write_unrealized_frac(
                dday,
                entry_px=entry_px,
                call_credit=credit_u,
                call_strike=tr.get("call_strike"),
                call_expiry=_as_date(tr.get("call_expiry") or tr["exit_date"]),
                spot=spot,
                call_mark=cm,
            )
            pos["u"] = float(u)
            still.append(pos)
        open_pos = still

        eq_pts.append(_mark_open())
        inv_pts.append(len(open_pos) / float(slots))

    eq = pd.Series(eq_pts, index=cal, dtype=float)
    inv = pd.Series(inv_pts, index=cal, dtype=float)
    if float(eq.iloc[0]) > 0:
        eq = eq / float(eq.iloc[0])
    return eq, inv


def _load_call_marks(
    trades: Sequence[Dict[str, Any]],
    source: ThetaEquitySource,
    *,
    mark_mode: str = "mid",
) -> Dict[int, Dict[date, float]]:
    out: Dict[int, Dict[date, float]] = {}
    for i, tr in enumerate(trades):
        sym = str(tr.get("symbol") or "").upper()
        exp = _as_date(tr.get("call_expiry") or tr.get("exit_date"))
        strike = tr.get("call_strike")
        entry = _as_date(tr.get("call_entry_date") or tr.get("entry_date"))
        if not sym or exp is None or strike is None or entry is None:
            out[i] = {}
            continue
        try:
            path = source.quote_path(sym, exp, float(strike), "C", entry, exp)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("call path %s: %s", sym, exc)
            path = None
        out[i] = _mark_map_from_path(path, mark_mode)
        if (i + 1) % 25 == 0 or i + 1 == len(trades):
            LOG.info("call MTM paths %d/%d days=%d", i + 1, len(trades), len(out[i]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-mid", type=Path, default=DEFAULT_MID)
    ap.add_argument("--trades-askbid", type=Path, default=DEFAULT_ASKBID)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--with-mtm", action="store_true")
    ap.add_argument("--mtm-max-open", type=int, default=8)
    ap.add_argument("--liq-adv25-half15", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    t0 = time.perf_counter()
    if not args.trades_mid.exists():
        LOG.error("missing %s", args.trades_mid)
        return 2

    mid_df = pd.read_csv(args.trades_mid)
    ask_df = (
        pd.read_csv(args.trades_askbid) if args.trades_askbid.exists() else mid_df
    )
    if args.liq_adv25_half15:
        mid_df = _liq_filter_from_askbid(
            mid_df, ask_df, min_adv=25e6, max_half_pct=15.0
        )

    outdir = Path(args.outdir) if args.outdir else dated_outdir(
        ROOT / "reports" / "ascending_channels"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    pooled = _pooled_slip_table(mid_df)
    pd.DataFrame(pooled).to_csv(outdir / "bw_listed_slip_pooled.csv", index=False)

    # Calendar from trade span
    all_occ = filled_buy_write_to_occ_rows(mid_df, slip=0.0)
    if not all_occ:
        LOG.error("no filled buy-write rows")
        return 2
    start = min(pd.Timestamp(r["entry_date"]) for r in all_occ) - pd.Timedelta(days=5)
    end = max(pd.Timestamp(r["exit_date"]) for r in all_occ) + pd.Timedelta(days=5)

    LOG.info("Loading SPY 1d %s..%s", start.date(), end.date())
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        start=start.date(),
        end=end.date(),
        provider="ALPACA",
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        LOG.error("SPY panel missing")
        return 2
    spy = spy.copy()
    spy.index = _panel_dates(spy)
    calendar = spy.index
    spy_close = spy["close"].astype(float)

    results: Dict[str, Any] = {
        "pooled_slip": pooled,
        "books": {},
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    for slip in SLIPS:
        name = "mid" if slip == 0 else ("mid_slip_%.2f" % slip).replace(".", "p")
        rows = filled_buy_write_to_occ_rows(mid_df, slip=float(slip))
        book_out: Dict[str, Any] = {"n_raw": len(rows), "max_open": {}}
        for mo in MAX_OPENS:
            kept, meta = occupancy_fcfs_wheel(
                rows, max_open=int(mo), unique_symbol=True
            )
            eq, inv = wheel_slot_equity(kept, calendar, max_open=int(mo))
            spy_eq = spy_buyhold_equity(spy_close, calendar)
            windows = {}
            for wlabel, wstart, wend in WINDOWS:
                windows[wlabel] = _score_window(
                    "%s_occ%s_%s" % (name, mo, wlabel),
                    eq,
                    inv,
                    spy_eq,
                    kept,
                    start=wstart,
                    end=wend,
                    window=wlabel,
                )
            book_out["max_open"][str(mo)] = {"occupancy": meta, "windows": windows}
            full = windows["FULL"]
            print(
                "---- %s occ%s kept=%s FULL Sharpe=%s vs SPY %s | %s"
                % (
                    name,
                    mo,
                    meta["n_kept"],
                    (full.get("sleeve") or {}).get("sharpe"),
                    (full.get("spy") or {}).get("sharpe"),
                    full.get("gate_reason"),
                )
            )
        results["books"][name] = book_out

    if args.with_mtm:
        # Prefer mid_slip $0.05 as the intermediate book for MTM.
        slip_mtm = 0.05
        name = "mid_slip_0p05_mtm"
        rows = filled_buy_write_to_occ_rows(mid_df, slip=slip_mtm)
        mo = int(args.mtm_max_open)
        kept, meta = occupancy_fcfs_wheel(rows, max_open=mo, unique_symbol=True)
        # Re-index kept for mark path keys
        kept_i = []
        for i, tr in enumerate(kept):
            kept_i.append({**tr, "_i": i})

        syms = sorted({str(t["symbol"]).upper() for t in kept_i})
        LOG.info("MTM loading 1d panels n=%d", len(syms))
        spot_panels = load_ohlcv_many(
            syms,
            timeframe="1d",
            start=start.date(),
            end=end.date(),
            provider="ALPACA",
            fallback_provider="IB",
            merge_mode="prefix",
            use_cache=True,
        )

        # Ensure Theta up
        theta_cache = (
            ROOT
            / "reports"
            / "ascending_channels"
            / "2026-11-03_cc_earnings_listed"
            / "theta_value_chain_cache"
        )
        source = ThetaEquitySource(cache_dir=theta_cache)
        LOG.info("Fetching call EOD marks for %d kept trades", len(kept_i))
        call_marks = _load_call_marks(kept_i, source, mark_mode="mid")

        eq_m, inv_m = bw_slot_equity_mtm(
            kept_i,
            calendar,
            max_open=mo,
            spot_panels=spot_panels,
            call_marks=call_marks,
        )
        # also mark-at-cost on same kept for compare
        eq_c, inv_c = wheel_slot_equity(kept, calendar, max_open=mo)
        spy_eq = spy_buyhold_equity(spy_close, calendar)
        mtm_windows = {}
        cost_windows = {}
        for wlabel, wstart, wend in WINDOWS:
            mtm_windows[wlabel] = _score_window(
                "%s_occ%s_%s" % (name, mo, wlabel),
                eq_m,
                inv_m,
                spy_eq,
                kept,
                start=wstart,
                end=wend,
                window=wlabel,
            )
            cost_windows[wlabel] = _score_window(
                "mid_slip_0p05_cost_occ%s_%s" % (mo, wlabel),
                eq_c,
                inv_c,
                spy_eq,
                kept,
                start=wstart,
                end=wend,
                window=wlabel,
            )
        results["mtm"] = {
            "slip": slip_mtm,
            "max_open": mo,
            "occupancy": meta,
            "mtm_windows": mtm_windows,
            "mark_at_cost_windows": cost_windows,
        }
        print(
            "==== MTM slip0.05 occ%s FULL: %s | cost: %s"
            % (
                mo,
                mtm_windows["FULL"].get("gate_reason"),
                cost_windows["FULL"].get("gate_reason"),
            )
        )

    out_json = outdir / "bw_listed_occupancy.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    # Flat CSV of gate results
    flat = []
    for bname, book in results["books"].items():
        for mo, blob in book["max_open"].items():
            for wlabel, w in blob["windows"].items():
                flat.append(
                    {
                        "book": bname,
                        "max_open": mo,
                        "window": wlabel,
                        "gate_pass": w.get("gate_pass"),
                        "gate_reason": w.get("gate_reason"),
                        "n": (w.get("sleeve") or {}).get("n_trades"),
                        "sharpe": (w.get("sleeve") or {}).get("sharpe"),
                        "cagr": (w.get("sleeve") or {}).get("cagr"),
                        "mdd": (w.get("sleeve") or {}).get("max_drawdown"),
                        "spy_sharpe": (w.get("spy") or {}).get("sharpe"),
                        "spy_mdd": (w.get("spy") or {}).get("max_drawdown"),
                        "E": (w.get("sleeve") or {}).get("expectancy"),
                        "PF": (w.get("sleeve") or {}).get("profit_factor"),
                    }
                )
    if "mtm" in results:
        for kind, key in (("mtm", "mtm_windows"), ("cost", "mark_at_cost_windows")):
            for wlabel, w in results["mtm"][key].items():
                flat.append(
                    {
                        "book": "mid_slip_0p05_%s" % kind,
                        "max_open": results["mtm"]["max_open"],
                        "window": wlabel,
                        "gate_pass": w.get("gate_pass"),
                        "gate_reason": w.get("gate_reason"),
                        "n": (w.get("sleeve") or {}).get("n_trades"),
                        "sharpe": (w.get("sleeve") or {}).get("sharpe"),
                        "cagr": (w.get("sleeve") or {}).get("cagr"),
                        "mdd": (w.get("sleeve") or {}).get("max_drawdown"),
                        "spy_sharpe": (w.get("spy") or {}).get("sharpe"),
                        "spy_mdd": (w.get("spy") or {}).get("max_drawdown"),
                        "E": (w.get("sleeve") or {}).get("expectancy"),
                        "PF": (w.get("sleeve") or {}).get("profit_factor"),
                    }
                )
    flat_df = pd.DataFrame(flat)
    flat_path = outdir / "bw_listed_occupancy_gates.csv"
    flat_df.to_csv(flat_path, index=False)
    LOG.info("Wrote %s and %s wall=%.1fs", out_json, flat_path, time.perf_counter() - t0)
    print("\n===== GATES (FULL) =====")
    print(
        flat_df[flat_df["window"] == "FULL"][
            ["book", "max_open", "gate_pass", "sharpe", "spy_sharpe", "mdd", "spy_mdd", "gate_reason"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
