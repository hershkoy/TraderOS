"""Print bid/ask detail for selected mid vs ask_bid wheel trades."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_20261009_133405.csv"
AB = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_ask_bid_20261009_133643.csv"

mid = pd.read_csv(MID)
ab = pd.read_csv(AB)
print("COLUMNS:", list(mid.columns))
print()

# Prefer stored bid/ask if present
bid_cols = [c for c in mid.columns if "bid" in c.lower() or "ask" in c.lower()]
print("bid/ask-ish cols:", bid_cols)

m = mid[mid["wheel_ok"] == True].copy()  # noqa: E712
a = ab[ab["wheel_ok"] == True].copy()  # noqa: E712
for df in (m, a):
    df["_k"] = (
        df["stock"].astype(str).str.upper()
        + "|"
        + df["put_entry_date"].astype(str)
        + "|"
        + df["put_expiry"].astype(str)
        + "|"
        + pd.to_numeric(df["put_strike"], errors="coerce").round(4).astype(str)
    )

# Pick concrete examples for the report
picks = [
    ("ACGL", "2022-01-10"),  # clean OTM, modest haircut
    ("ICLR", "2023-08-11"),  # OTM tight spread
    ("VRRM", "2023-12-11"),  # OTM wide % of mid (cheap put)
    ("PNC", "2024-08-06"),  # OTM liquid-ish
    ("VECO", "2022-01-11"),  # call_assign modest
    ("FULT", "2022-02-25"),  # call_assign extreme
    ("KEY", "2022-02-07"),  # share_liquidate tight
    ("NMIH", "2023-08-25"),  # share_liquidate wide CC
]

mm = m.set_index("_k")
aa = a.set_index("_k")

for stock, buy in picks:
    sub_m = m[(m["stock"].astype(str).str.upper() == stock) & (m["buy_date"].astype(str).str.startswith(buy))]
    if sub_m.empty:
        print(f"\nMISSING {stock} {buy}")
        continue
    mr = sub_m.iloc[0]
    k = mr["_k"]
    if k not in aa.index:
        print(f"\nNO ASK_BID MATCH {stock} {buy}")
        continue
    ar = aa.loc[k]
    if isinstance(ar, pd.DataFrame):
        ar = ar.iloc[0]

    put_mid = float(mr["put_credit"])
    put_ab = float(ar["put_credit"])
    # Reconstruct bid/ask: ask_bid credit = bid; mid = (bid+ask)/2 => ask = 2*mid - bid
    put_bid = put_ab
    put_ask = 2.0 * put_mid - put_bid
    half = put_mid - put_bid

    call_mid = float(mr["call_credit"]) if pd.notna(mr.get("call_credit")) else None
    call_ab = float(ar["call_credit"]) if pd.notna(ar.get("call_credit")) else None
    call_line = ""
    if call_mid is not None and call_ab is not None:
        c_bid = call_ab
        c_ask = 2.0 * call_mid - c_bid
        call_line = (
            f"  CC  strike={mr.get('call_strike')} entry={mr.get('call_entry_date')} "
            f"expiry={mr.get('call_expiry')}\n"
            f"      mid credit={call_mid:.4f}  bid(sell)={c_bid:.4f}  "
            f"implied ask={c_ask:.4f}  half-spread={call_mid - c_bid:.4f}\n"
        )

    k_put = float(mr["put_strike"])
    print(f"\n=== {stock} buy={buy} exit={mr.get('wheel_exit_reason')} ===")
    print(
        f"  CSP strike={k_put} dte={mr.get('put_dte')} delta={mr.get('put_delta')} "
        f"entry={mr.get('put_entry_date')} expiry={mr.get('put_expiry')}"
    )
    print(
        f"  PUT mid credit={put_mid:.4f}  bid(sell)={put_bid:.4f}  "
        f"implied ask={put_ask:.4f}  half-spread=${half:.4f} "
        f"({100*half/put_mid:.1f}% of mid)"
    )
    print(call_line, end="")
    print(
        f"  gain CS% mid={float(mr['gain_pct']):+.4f}  "
        f"ask_bid={float(ar['gain_pct']):+.4f}  "
        f"diff={float(mr['gain_pct']) - float(ar['gain_pct']):+.4f}pp"
    )
    print(
        f"  put haircut alone = ${(put_mid-put_bid):.4f} / ${k_put:.2f} "
        f"= {100*(put_mid-put_bid)/k_put:.4f}pp of CS"
    )

# Why book E collapses: decompose
cmp = pd.read_csv(
    ROOT
    / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_vs_askbid_compare.csv"
)
print("\n=== Decomposition: mean gain_diff by exit ===")
for reason, g in cmp.groupby("exit_reason"):
    print(
        f"  {reason}: n={len(g)} mean_gain_diff={g['gain_diff'].mean():.4f}pp "
        f"mean_put_hair_cs={g['put_haircut_cs_pct'].mean():.4f}pp "
        f"mean_call_diff_$/share={g['call_credit_diff'].mean() if g['call_credit_diff'].notna().any() else float('nan'):.4f}"
    )

# Spread as % of credit buckets
print("\n=== Half-spread as % of mid put credit (distribution) ===")
pct = cmp["put_credit_diff_pct_of_mid"]
for thr in (5, 10, 20, 30, 50):
    print(f"  >={thr}% of mid credit: {(pct >= thr).mean()*100:.1f}% of trades")
