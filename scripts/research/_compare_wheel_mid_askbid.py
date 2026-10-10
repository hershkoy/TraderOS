"""Compare mid vs ask_bid marks on matched last-15m wheel fills."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_20261009_133405.csv"
AB = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_ask_bid_20261009_133643.csv"
OUT = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_vs_askbid_compare.csv"

mid = pd.read_csv(MID)
ab = pd.read_csv(AB)
mid = mid[mid["wheel_ok"] == True].copy()  # noqa: E712
ab = ab[ab["wheel_ok"] == True].copy()  # noqa: E712

# Match on stock + put entry/expiry/strike (same contract)
for df in (mid, ab):
    df["_k"] = (
        df["stock"].astype(str).str.upper()
        + "|"
        + df["put_entry_date"].astype(str)
        + "|"
        + df["put_expiry"].astype(str)
        + "|"
        + pd.to_numeric(df["put_strike"], errors="coerce").round(4).astype(str)
    )

m = mid.set_index("_k")
a = ab.set_index("_k")
common = m.index.intersection(a.index)
print("filled mid", len(mid), "ask_bid", len(ab), "matched keys", len(common))

cols = [
    "stock",
    "buy_date",
    "put_entry_date",
    "put_expiry",
    "put_strike",
    "put_dte",
    "put_delta",
    "put_credit",
    "call_strike",
    "call_credit",
    "call_entry_date",
    "call_expiry",
    "wheel_exit_reason",
    "wheel_assigned",
    "gain_pct",
    "put_expiry_spot",
    "share_exit_px",
]
rows = []
for k in common:
    mr = m.loc[k]
    ar = a.loc[k]
    if isinstance(mr, pd.DataFrame):
        mr = mr.iloc[0]
    if isinstance(ar, pd.DataFrame):
        ar = ar.iloc[0]
    put_mid = float(mr["put_credit"])
    put_nat = float(ar["put_credit"])
    call_mid = float(mr["call_credit"]) if pd.notna(mr.get("call_credit")) else None
    call_nat = float(ar["call_credit"]) if pd.notna(ar.get("call_credit")) else None
    g_mid = float(mr["gain_pct"])
    g_nat = float(ar["gain_pct"])
    k_put = float(mr["put_strike"])
    # Haircut on put credit as % of strike (CS basis)
    put_hair_ps = put_mid - put_nat
    put_hair_cs = put_hair_ps / k_put * 100.0 if k_put else None
    call_hair_ps = None
    if call_mid is not None and call_nat is not None:
        call_hair_ps = call_mid - call_nat
    rows.append(
        {
            "stock": mr["stock"],
            "buy_date": mr["buy_date"],
            "put_entry_date": mr["put_entry_date"],
            "put_expiry": mr["put_expiry"],
            "put_strike": k_put,
            "put_dte": mr.get("put_dte"),
            "put_delta": mr.get("put_delta"),
            "exit_reason": mr.get("wheel_exit_reason"),
            "assigned": bool(mr.get("wheel_assigned")),
            "put_credit_mid": round(put_mid, 4),
            "put_credit_askbid": round(put_nat, 4),
            "put_credit_diff": round(put_hair_ps, 4),
            "put_credit_diff_pct_of_mid": round(
                100.0 * put_hair_ps / put_mid, 2
            )
            if put_mid
            else None,
            "put_haircut_cs_pct": round(put_hair_cs, 4) if put_hair_cs is not None else None,
            "call_credit_mid": None if call_mid is None else round(call_mid, 4),
            "call_credit_askbid": None if call_nat is None else round(call_nat, 4),
            "call_credit_diff": None if call_hair_ps is None else round(call_hair_ps, 4),
            "gain_mid": round(g_mid, 4),
            "gain_askbid": round(g_nat, 4),
            "gain_diff": round(g_mid - g_nat, 4),
            "put_expiry_spot": mr.get("put_expiry_spot"),
        }
    )

cmp_ = pd.DataFrame(rows)
cmp_.to_csv(OUT, index=False)
print("wrote", OUT)

# Summary
print("\n=== Put credit: mid vs sell-at-bid (ask_bid entry) ===")
print(
    cmp_[["put_credit_mid", "put_credit_askbid", "put_credit_diff", "put_credit_diff_pct_of_mid", "put_haircut_cs_pct"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(4)
    .to_string()
)

print("\n=== Gain CS%: mid vs ask_bid ===")
print(
    cmp_[["gain_mid", "gain_askbid", "gain_diff"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(4)
    .to_string()
)

print("\n=== By exit reason: mean put haircut & gain diff ===")
agg = (
    cmp_.groupby("exit_reason")
    .agg(
        n=("stock", "size"),
        put_mid=("put_credit_mid", "mean"),
        put_ab=("put_credit_askbid", "mean"),
        put_diff=("put_credit_diff", "mean"),
        put_diff_pct_mid=("put_credit_diff_pct_of_mid", "mean"),
        put_hair_cs=("put_haircut_cs_pct", "mean"),
        call_diff=("call_credit_diff", "mean"),
        gain_mid=("gain_mid", "mean"),
        gain_ab=("gain_askbid", "mean"),
        gain_diff=("gain_diff", "mean"),
    )
    .round(4)
)
print(agg.to_string())

# Sample trades: largest gain_diff, typical OTM, assigned
print("\n=== Sample: 8 largest gain_mid - gain_askbid ===")
show = [
    "stock",
    "buy_date",
    "put_strike",
    "put_dte",
    "exit_reason",
    "put_credit_mid",
    "put_credit_askbid",
    "put_credit_diff",
    "put_credit_diff_pct_of_mid",
    "call_credit_mid",
    "call_credit_askbid",
    "call_credit_diff",
    "gain_mid",
    "gain_askbid",
    "gain_diff",
]
print(cmp_.nlargest(8, "gain_diff")[show].to_string(index=False))

print("\n=== Sample: 8 random-ish OTM expires (sorted by buy_date) ===")
otm = cmp_[cmp_["exit_reason"] == "put_expire_otm"].sort_values("buy_date")
step = max(1, len(otm) // 8)
print(otm.iloc[::step].head(8)[show].to_string(index=False))

print("\n=== Sample: assigned cycles (first 8) ===")
asg = cmp_[cmp_["assigned"] == True].sort_values("buy_date")  # noqa: E712
print(asg.head(8)[show].to_string(index=False))

# How entry_credit works reminder
print(
    "\nNote: ask_bid ENTRY uses option BID (sell credit);"
    " mid uses (bid+ask)/2."
    " Hold-to-expiry OTM puts never pay an exit debit —"
    " so the ONLY mark difference on OTM cycles is entry credit."
    " Assigned cycles also haircut the CC entry credit (sell call at bid)."
)

summary = {
    "n_matched": int(len(cmp_)),
    "mean_put_credit_mid": float(cmp_["put_credit_mid"].mean()),
    "mean_put_credit_askbid": float(cmp_["put_credit_askbid"].mean()),
    "mean_put_diff_dollars": float(cmp_["put_credit_diff"].mean()),
    "mean_put_diff_pct_of_mid": float(cmp_["put_credit_diff_pct_of_mid"].mean()),
    "median_put_diff_pct_of_mid": float(cmp_["put_credit_diff_pct_of_mid"].median()),
    "mean_put_haircut_cs_pct": float(cmp_["put_haircut_cs_pct"].mean()),
    "mean_gain_mid": float(cmp_["gain_mid"].mean()),
    "mean_gain_askbid": float(cmp_["gain_askbid"].mean()),
    "mean_gain_diff": float(cmp_["gain_diff"].mean()),
    "otm_n": int((cmp_["exit_reason"] == "put_expire_otm").sum()),
    "otm_mean_gain_diff": float(
        cmp_.loc[cmp_["exit_reason"] == "put_expire_otm", "gain_diff"].mean()
    ),
    "assigned_n": int(cmp_["assigned"].sum()),
    "assigned_mean_gain_diff": float(
        cmp_.loc[cmp_["assigned"] == True, "gain_diff"].mean()  # noqa: E712
    ),
}
print("\n=== Compact summary ===")
print(json.dumps(summary, indent=2))
sum_path = OUT.with_name("last_15m_wheel_mid_vs_askbid_summary.json")
sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("wrote", sum_path)
