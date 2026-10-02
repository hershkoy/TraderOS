"""Re-mark existing Forward Factor trades with ask/bid (EOD only, paced)."""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client
from utils.research.forward_factor import (
    MULT,
    FFTrade,
    _pnl_calendar,
    book_stats,
    eod_on,
    fetch_eod,
    trades_to_frame,
)

TRADES_CSV = (
    ROOT
    / "reports"
    / "research"
    / "forward_factor"
    / "2026-10-01_panel"
    / "SPY_QQQ_IWM_panel_60_90_mid_ff0.2_trades.csv"
)
OUTDIR = ROOT / "reports" / "research" / "forward_factor" / "2026-10-01_panel"


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _eod_row(client, cache, symbol, exp, strike, day):
    for attempt in range(6):
        try:
            df = fetch_eod(
                client,
                symbol=symbol,
                expiration=exp,
                strike=strike,
                right="call",
                start=day,
                end=day,
                cache=cache,
            )
            row = eod_on(df, day)
            if row is not None:
                return {
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                    "mid": float(row["mid"]),
                }
            return None
        except Exception as exc:
            if "429" in str(exc) or "timed out" in str(exc).lower():
                wait = 5 * (attempt + 1)
                print(f"429/timeout sleep {wait}s")
                time.sleep(wait)
                continue
            raise
    return None


def main() -> int:
    client = get_thetadata_client()
    for i in range(60):
        if client.is_ready():
            try:
                r = client.session.get(
                    f"{client.base_url}/v3/option/list/expirations",
                    params={"symbol": "SPY"},
                    timeout=30,
                )
                if r.status_code == 429:
                    print("still 429", i)
                    time.sleep(10)
                    continue
                if r.status_code == 200:
                    break
            except Exception as exc:
                print("probe", exc)
                time.sleep(10)
                continue
        else:
            print("not ready", i)
            time.sleep(5)
    else:
        print("Theta not ready")
        return 1

    df = pd.read_csv(TRADES_CSV)
    cache = {}
    trades = []
    skipped = []
    commission = 0.65 * 4
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        ed = _d(row["entry_date"])
        xd = _d(row["exit_date"])
        front = _d(row["front"])
        back = _d(row["back"])
        strike = float(row["strike"])
        time.sleep(1.5)
        fe = _eod_row(client, cache, sym, front, strike, ed)
        time.sleep(0.5)
        be = _eod_row(client, cache, sym, back, strike, ed)
        time.sleep(0.5)
        fx = _eod_row(client, cache, sym, front, strike, xd)
        time.sleep(0.5)
        bx = _eod_row(client, cache, sym, back, strike, xd)
        if None in (fe, be, fx, bx):
            skipped.append({"symbol": sym, "entry": str(ed), "reason": "no_eod"})
            print("skip", sym, ed)
            continue
        entry_debit, exit_credit, pnl = _pnl_calendar(
            front_bid_e=fe["bid"],
            front_ask_e=fe["ask"],
            front_mid_e=fe["mid"],
            back_bid_e=be["bid"],
            back_ask_e=be["ask"],
            back_mid_e=be["mid"],
            front_bid_x=fx["bid"],
            front_ask_x=fx["ask"],
            front_mid_x=fx["mid"],
            back_bid_x=bx["bid"],
            back_ask_x=bx["ask"],
            back_mid_x=bx["mid"],
            mark_mode="ask_bid",
            commission=commission,
        )
        if entry_debit <= 0.05:
            skipped.append(
                {"symbol": sym, "entry": str(ed), "reason": "not_debit", "debit": entry_debit}
            )
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
                mark_mode="ask_bid",
                entry_debit=entry_debit,
                exit_credit=exit_credit,
                pnl=pnl,
                ret_pct=ret_pct,
                hold_days=int(row["hold_days"]),
                commission=commission,
            )
        )
        print(f"{sym} {ed} debit={entry_debit:.2f} pnl={pnl:.0f} ret={ret_pct:.1f}%")

    stats = book_stats(trades)
    print("STATS", stats)
    trades_to_frame(trades).to_csv(OUTDIR / "panel_ff0.2_ask_bid_trades.csv", index=False)
    with (OUTDIR / "panel_ff0.2_ask_bid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"stats": stats, "n_skipped": len(skipped), "skipped": skipped},
            f,
            indent=2,
            default=str,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
