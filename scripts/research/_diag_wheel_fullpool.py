"""Diagnose full-pool mid vs ask_bid year coverage and 2022 gap."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/ascending_channels/2026-10-09"
mid = pd.read_csv(OUT / "last_15m_wheel_mid_20261010_112205.csv")
ab = pd.read_csv(OUT / "last_15m_wheel_ask_bid_20261010_112516.csv")
old_mid = pd.read_csv(OUT / "last_15m_wheel_mid_20261009_133405.csv")

for name, df in [("mid_new", mid), ("ab_new", ab), ("mid_old", old_mid)]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["buy_date"], errors="coerce").dt.year
    ok = df[df["wheel_ok"] == True]  # noqa: E712
    print(f"\n=== {name} attempted={len(df)} filled={len(ok)} ===")
    print("attempted by year:\n", df["year"].value_counts().sort_index().to_string())
    print("filled by year:\n", ok["year"].value_counts().sort_index().to_string())
    if "wheel_reason" in df.columns:
        y22 = df[df["year"] == 2022]
        print("2022 reasons:\n", y22["wheel_reason"].value_counts().to_string())

# Compare 2022 keys mid_old vs mid_new
old = old_mid.copy()
old["year"] = pd.to_datetime(old["buy_date"], errors="coerce").dt.year
old22 = old[old["year"] == 2022]
new22 = mid.copy()
new22["year"] = pd.to_datetime(new22["buy_date"], errors="coerce").dt.year
new22 = new22[new22["year"] == 2022]
print("\nold 2022 filled", (old22["wheel_ok"] == True).sum(), "of", len(old22))
print("new 2022 filled", (new22["wheel_ok"] == True).sum(), "of", len(new22))

# key overlap
def k(df):
    return (
        df["stock"].astype(str).str.upper()
        + "|"
        + pd.to_datetime(df["buy_date"]).dt.strftime("%Y-%m-%d")
    )

old_keys = set(k(old22[old22["wheel_ok"] == True]))
new_keys = set(k(new22))
print("old filled 2022 keys in new attempted:", len(old_keys & new_keys), "/", len(old_keys))
missing = old_keys - new_keys
print("old filled 2022 NOT in new sample:", len(missing), list(missing)[:5])

# Of overlapping, what happened in new?
if old_keys & new_keys:
    sub = new22[k(new22).isin(old_keys)]
    print("overlap new reasons:\n", sub["wheel_reason"].value_counts().to_string())
    print("sample overlap rows:")
    print(sub[["stock", "buy_date", "wheel_ok", "wheel_reason"]].head(10).to_string(index=False))
