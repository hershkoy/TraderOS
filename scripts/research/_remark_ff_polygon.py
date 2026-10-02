"""Cross-check Forward Factor calendar legs on Massive/Polygon day bars.

Polygon Starter: day aggs OK; historical NBBO quotes are 403 (Advanced).
Uses day close as mid proxy and a half-spread haircut for friction estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.api.polygon.polygon_client import get_polygon_client
from utils.research.forward_factor import MULT, FFTrade, book_stats, trades_to_frame
from utils.research.short_put_live_realism import half_spread
from utils.research.short_put_polygon_spotcheck import (
    bars_from_aggs_response,
    polygon_option_ticker,
)

COMM = 0.65 * 4


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _day_close(client, ticker: str, day: date, cache: Dict[str, pd.DataFrame]) -> Optional[float]:
    key = ticker
    if key not in cache:
        # Pull a wide window once per contract when possible
        cache[key] = pd.DataFrame()
    df = cache[key]
    if not df.empty and (df["date"] == day).any():
        row = df[df["date"] == day].iloc[-1]
        return float(row["close"]) if pd.notna(row["close"]) else None
    try:
        resp = client.get_option_eod(ticker, day.isoformat())
    except Exception as exc:
        print(f"agg fail {ticker} {day}: {exc}")
        return None
    bars = bars_from_aggs_response(resp if isinstance(resp, dict) else {})
    if bars.empty:
        return None
    # merge into cache
    if df.empty:
        cache[key] = bars
    else:
        cache[key] = (
            pd.concat([df, bars], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
    rows = cache[key][cache[key]["date"] == day]
    if rows.empty:
        return None
    c = rows.iloc[-1]["close"]
    return float(c) if pd.notna(c) else None


def _pnl_from_closes(
    front_e: float,
    back_e: float,
    front_x: float,
    back_x: float,
    *,
    mark_mode: str,
    hs_floor: float = 0.05,
    hs_frac: float = 0.08,
) -> tuple[float, float, float]:
    """Long back / short front calendar."""
    if mark_mode == "mid":
        entry = back_e - front_e
        exit_ = front_x - back_x  # buy front, sell back to close? wait
        # Entry: buy back @ mid, sell front @ mid => debit = back - front
        # Exit: sell back @ mid, buy front @ mid => credit = back_x - front_x? 
        # Actually calendar exit: sell the long back, buy back the short front
        # credit = back_x - front_x ... no: sell back receive back_x, buy front pay front_x => credit = back_x - front_x
        # But forward_factor uses: exit_credit = front_mid_x - back_mid_x for mid? Let me check _pnl_calendar
        raise RuntimeError("use _pnl_calendar path")
    # haircut mid: sell front at mid-hs, buy back at mid+hs
    fe_sell = max(0.0, front_e - half_spread(front_e, floor=hs_floor, frac=hs_frac))
    be_buy = back_e + half_spread(back_e, floor=hs_floor, frac=hs_frac)
    entry_debit = be_buy - fe_sell
    # exit: buy front (ask~mid+hs), sell back (bid~mid-hs)
    fx_buy = front_x + half_spread(front_x, floor=hs_floor, frac=hs_frac)
    bx_sell = max(0.0, back_x - half_spread(back_x, floor=hs_floor, frac=hs_frac))
    exit_credit = bx_sell - fx_buy  # can be negative (pay to close)
    # Wait - for calendar close you sell long back and buy short front:
    # credit = back_sell - front_buy = bx_sell - fx_buy
    # PnL per share = exit_credit - entry_debit ... if exit_credit negative it's a debit to close
    pnl = (exit_credit - entry_debit) * MULT - COMM
    # Actually standard: entry debit paid; exit receive credit for unwinding
    # If unwind costs more, exit_credit is negative meaning you pay net
    return entry_debit, exit_credit, pnl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trades",
        type=Path,
        default=ROOT
        / "reports"
        / "research"
        / "forward_factor"
        / "2026-10-02_spy"
        / "SPY_60_90_mid_ff0.2_trades.csv",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "research" / "forward_factor" / "2026-10-02_spy",
    )
    ap.add_argument("--mark", default="half_spread", choices=["mid", "half_spread"])
    args = ap.parse_args()

    if not args.trades.exists():
        print("missing trades", args.trades)
        return 1

    client = get_polygon_client()
    df = pd.read_csv(args.trades)
    cache: Dict[str, pd.DataFrame] = {}
    trades = []
    skipped = []
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        ed, xd = _d(row["entry_date"]), _d(row["exit_date"])
        front, back = _d(row["front"]), _d(row["back"])
        strike = float(row["strike"])
        ft = polygon_option_ticker(sym, front.isoformat(), strike, "C")
        bt = polygon_option_ticker(sym, back.isoformat(), strike, "C")
        time.sleep(0.25)
        fe = _day_close(client, ft, ed, cache)
        be = _day_close(client, bt, ed, cache)
        fx = _day_close(client, ft, xd, cache)
        bx = _day_close(client, bt, xd, cache)
        if None in (fe, be, fx, bx):
            skipped.append({"symbol": sym, "entry": str(ed), "reason": "no_agg"})
            print("skip", sym, ed, fe, be, fx, bx)
            continue
        if args.mark == "mid":
            entry_debit = be - fe
            # close: sell back, buy front
            exit_credit = bx - fx
            pnl = (exit_credit - entry_debit) * MULT - COMM
        else:
            entry_debit, exit_credit, pnl = _pnl_from_closes(
                fe, be, fx, bx, mark_mode="half_spread"
            )
        if entry_debit <= 0.05:
            skipped.append({"symbol": sym, "entry": str(ed), "reason": "not_debit", "debit": entry_debit})
            continue
        ret_pct = 100.0 * pnl / (entry_debit * MULT)
        trades.append(
            FFTrade(
                symbol=sym,
                entry_date=ed,
                exit_date=xd,
                front=front,
                back=back,
                strike=strike,
                spot_entry=float(row["spot_entry"]),
                front_iv=float(row["front_iv"]),
                back_iv=float(row["back_iv"]),
                fwd_iv=float(row["fwd_iv"]),
                ff=float(row["ff"]),
                mark_mode=f"poly_{args.mark}",
                entry_debit=float(entry_debit),
                exit_credit=float(exit_credit),
                pnl=float(pnl),
                ret_pct=float(ret_pct),
                hold_days=int(row["hold_days"]),
                commission=COMM,
            )
        )
        print(
            f"{sym} {ed} poly_{args.mark} debit={entry_debit:.2f} "
            f"pnl={pnl:.0f} ret={ret_pct:.1f}% closes={fe:.2f}/{be:.2f}->{fx:.2f}/{bx:.2f}"
        )

    stats = book_stats(trades)
    print("STATS", stats)
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"poly_{args.mark}_ff_trades"
    trades_to_frame(trades).to_csv(args.outdir / f"{stem}.csv", index=False)
    out: Dict[str, Any] = {"stats": stats, "n_skipped": len(skipped), "skipped": skipped}
    with (args.outdir / f"{stem}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
