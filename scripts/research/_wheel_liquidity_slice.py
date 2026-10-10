"""Stratify last-15m wheel mid vs ask_bid by ADV / half-spread liquidity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_20261009_133405.csv"
AB = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_ask_bid_20261009_133643.csv"
CMP = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_vs_askbid_compare.csv"
OUT = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_liquidity_slices.json"


def pf_e_wr(gains: pd.Series) -> dict:
    g = pd.to_numeric(gains, errors="coerce").dropna()
    if g.empty:
        return {"n": 0, "E": None, "PF": None, "WR": None}
    wins = g[g > 0].sum()
    losses = (-g[g < 0]).sum()
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else None)
    return {
        "n": int(len(g)),
        "E": round(float(g.mean()), 4),
        "PF": None if pf is None or not np.isfinite(pf) else round(pf, 4),
        "WR": round(float((g > 0).mean()) * 100.0, 1),
    }


def year_of(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.year


mid = pd.read_csv(MID)
ab = pd.read_csv(AB)
mid = mid[mid["wheel_ok"] == True].copy()  # noqa: E712
ab = ab[ab["wheel_ok"] == True].copy()  # noqa: E712

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

rows = []
for k in common:
    mr = m.loc[k]
    ar = a.loc[k]
    if isinstance(mr, pd.DataFrame):
        mr = mr.iloc[0]
    if isinstance(ar, pd.DataFrame):
        ar = ar.iloc[0]
    put_mid = float(mr["put_credit"])
    put_ab = float(ar["put_credit"])
    half = put_mid - put_ab
    half_pct = 100.0 * half / put_mid if put_mid else np.nan
    k_put = float(mr["put_strike"])
    call_mid = float(mr["call_credit"]) if pd.notna(mr.get("call_credit")) else np.nan
    call_ab = float(ar["call_credit"]) if pd.notna(ar.get("call_credit")) else np.nan
    rows.append(
        {
            "stock": mr["stock"],
            "buy_date": mr["buy_date"],
            "year": int(year_of(pd.Series([mr["buy_date"]])).iloc[0]),
            "adv_20": float(mr["adv_20"]) if pd.notna(mr.get("adv_20")) else np.nan,
            "put_strike": k_put,
            "put_credit_mid": put_mid,
            "put_credit_ab": put_ab,
            "half_spread": half,
            "half_pct_of_mid": half_pct,
            "half_cs_pct": 100.0 * half / k_put if k_put else np.nan,
            "call_half": (call_mid - call_ab)
            if np.isfinite(call_mid) and np.isfinite(call_ab)
            else np.nan,
            "exit_reason": mr.get("wheel_exit_reason"),
            "assigned": bool(mr.get("wheel_assigned")),
            "gain_mid": float(mr["gain_pct"]),
            "gain_ab": float(ar["gain_pct"]),
            "gain_stock": float(mr["gain_pct_stock"])
            if pd.notna(mr.get("gain_pct_stock"))
            else np.nan,
        }
    )

df = pd.DataFrame(rows)
print("matched", len(df), "adv null", df["adv_20"].isna().sum())
print("adv describe:\n", df["adv_20"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))
print("half_pct describe:\n", df["half_pct_of_mid"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

# Liquidity filters
filters = {
    "ALL": df.index,
    "ADV>=5M": df.index[df["adv_20"] >= 5e6],
    "ADV>=10M": df.index[df["adv_20"] >= 10e6],
    "ADV>=25M": df.index[df["adv_20"] >= 25e6],
    "ADV>=50M": df.index[df["adv_20"] >= 50e6],
    "ADV>=100M": df.index[df["adv_20"] >= 100e6],
    "half<=10%_mid": df.index[df["half_pct_of_mid"] <= 10],
    "half<=15%_mid": df.index[df["half_pct_of_mid"] <= 15],
    "half<=20%_mid": df.index[df["half_pct_of_mid"] <= 20],
    "half_cs<=0.25pp": df.index[df["half_cs_pct"] <= 0.25],
    "half_cs<=0.50pp": df.index[df["half_cs_pct"] <= 0.50],
    "ADV>=10M & half<=15%": df.index[
        (df["adv_20"] >= 10e6) & (df["half_pct_of_mid"] <= 15)
    ],
    "ADV>=25M & half<=15%": df.index[
        (df["adv_20"] >= 25e6) & (df["half_pct_of_mid"] <= 15)
    ],
    "ADV>=10M & half_cs<=0.5": df.index[
        (df["adv_20"] >= 10e6) & (df["half_cs_pct"] <= 0.50)
    ],
}

out: dict = {"n_matched": int(len(df)), "slices": {}}

print("\n=== Liquidity slices (wheel mid / ask_bid / stock) ===")
hdr = (
    f"{'slice':28s} {'n':>4} {'mid_E':>7} {'mid_PF':>7} "
    f"{'ab_E':>7} {'ab_PF':>7} {'stk_E':>7} "
    f"{'dE':>6} {'med_half%':>9} {'mean_half$':>10}"
)
print(hdr)
for name, idx in filters.items():
    sub = df.loc[idx]
    mid_s = pf_e_wr(sub["gain_mid"])
    ab_s = pf_e_wr(sub["gain_ab"])
    stk_s = pf_e_wr(sub["gain_stock"])
    dE = None
    if mid_s["E"] is not None and ab_s["E"] is not None:
        dE = round(mid_s["E"] - ab_s["E"], 4)
    row = {
        "n": mid_s["n"],
        "mid": mid_s,
        "ask_bid": ab_s,
        "stock": stk_s,
        "mean_gain_diff": dE,
        "median_half_pct_of_mid": round(float(sub["half_pct_of_mid"].median()), 2)
        if len(sub)
        else None,
        "mean_half_spread": round(float(sub["half_spread"].mean()), 4)
        if len(sub)
        else None,
        "mean_half_cs_pct": round(float(sub["half_cs_pct"].mean()), 4)
        if len(sub)
        else None,
        "assigned_pct": round(float(sub["assigned"].mean()) * 100, 1)
        if len(sub)
        else None,
    }
    # year splits for promising filters
    years = {}
    for y in (2022, 2023):
        ys = sub[sub["year"] == y]
        years[str(y)] = {
            "mid": pf_e_wr(ys["gain_mid"]),
            "ask_bid": pf_e_wr(ys["gain_ab"]),
            "stock": pf_e_wr(ys["gain_stock"]),
        }
    y2223 = sub[sub["year"].isin([2022, 2023])]
    years["2022-23"] = {
        "mid": pf_e_wr(y2223["gain_mid"]),
        "ask_bid": pf_e_wr(y2223["gain_ab"]),
        "stock": pf_e_wr(y2223["gain_stock"]),
    }
    row["years"] = years
    out["slices"][name] = row
    print(
        f"{name:28s} {mid_s['n']:4d} "
        f"{mid_s['E'] if mid_s['E'] is not None else float('nan'):7.2f} "
        f"{(mid_s['PF'] or float('nan')):7.2f} "
        f"{ab_s['E'] if ab_s['E'] is not None else float('nan'):7.2f} "
        f"{(ab_s['PF'] or float('nan')):7.2f} "
        f"{stk_s['E'] if stk_s['E'] is not None else float('nan'):7.2f} "
        f"{(dE if dE is not None else float('nan')):6.2f} "
        f"{(row['median_half_pct_of_mid'] or float('nan')):9.1f} "
        f"{(row['mean_half_spread'] or float('nan')):10.3f}"
    )

# Year detail for key slices
print("\n=== Year detail for key slices ===")
for name in (
    "ALL",
    "ADV>=10M",
    "ADV>=25M",
    "ADV>=50M",
    "half<=15%_mid",
    "ADV>=10M & half<=15%",
    "ADV>=25M & half<=15%",
):
    s = out["slices"][name]
    print(f"\n{name} n={s['n']}")
    for yk, yv in s["years"].items():
        am = yv["ask_bid"]
        mm = yv["mid"]
        print(
            f"  {yk}: mid E/PF {mm['E']}/{mm['PF']} n={mm['n']} | "
            f"ask_bid E/PF {am['E']}/{am['PF']} n={am['n']}"
        )

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote", OUT)

# Top names by ADV in liquid slice
liq = df[df["adv_20"] >= 25e6].sort_values("adv_20", ascending=False)
print("\n=== ADV>=25M names (sample) ===")
print(
    liq[["stock", "buy_date", "adv_20", "half_pct_of_mid", "gain_mid", "gain_ab"]]
    .head(15)
    .to_string(index=False)
)
