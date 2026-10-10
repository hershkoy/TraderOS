"""Print pool sizes for last-15m wheel n-increase levers."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
trades = ROOT / "reports/ascending_channels/current_best/channel_touch_1d_last_15m.csv"
# try alternates
cands = [
    trades,
    ROOT / "reports/ascending_channels/current_best/1d_last_15m_trades.csv",
    ROOT / "reports/ascending_channels/current_best/1d_channel_touch.csv",
]
for p in ROOT.glob("reports/ascending_channels/**/*last_15m*.csv"):
    if "wheel" in p.name:
        continue
    cands.append(p)

seen = set()
for p in cands:
    if p in seen or not p.exists():
        continue
    seen.add(p)
    df = pd.read_csv(p, nrows=5)
    full = pd.read_csv(p)
    print(f"\n{p.relative_to(ROOT)} rows={len(full)} cols_has_adv={'adv_20' in full.columns}")
    if "buy_date" in full.columns and "adv_20" in full.columns:
        full["year"] = pd.to_datetime(full["buy_date"], errors="coerce").dt.year
        sub = full[full["year"] >= 2022]
        print(f"  buy>=2022: {len(sub)}")
        for thr in (2e6, 5e6, 10e6, 25e6, 50e6):
            print(f"  ADV>={thr/1e6:.0f}M & 2022+: {(sub['adv_20'].fillna(0)>=thr).sum()}")

mid = ROOT / "reports/ascending_channels/2026-10-09/last_15m_wheel_mid_20261009_133405.csv"
if mid.exists():
    m = pd.read_csv(mid)
    print("\nclaimcheck mid CSV:", len(m), "wheel_ok", (m.get("wheel_ok")==True).sum() if "wheel_ok" in m.columns else "?")
    if "wheel_reason" in m.columns:
        print(m["wheel_reason"].value_counts().head(15).to_string())
    if "wheel_ok" in m.columns:
        bad = m[m["wheel_ok"]!=True]
        if "wheel_reason" in bad.columns:
            print("failures:\n", bad["wheel_reason"].value_counts().to_string())
