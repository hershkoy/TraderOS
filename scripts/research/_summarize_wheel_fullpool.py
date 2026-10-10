"""Pooled E/PF year slices for repaired mid + ask_bid full pool."""
from pathlib import Path
import json
import pandas as pd

OUT = Path("reports/ascending_channels/2026-10-09")
mid = pd.read_csv(OUT / "last_15m_wheel_mid_repaired_20261010_112646.csv")
ab = pd.read_csv(OUT / "last_15m_wheel_ask_bid_20261010_112516.csv")


def stats(df, label):
    d = df[df["wheel_ok"] == True].copy()  # noqa: E712
    d["year"] = pd.to_datetime(d["buy_date"]).dt.year
    d["g"] = pd.to_numeric(d["gain_pct"], errors="coerce")
    rows = {}
    for name, mask in [
        ("FULL", d.index),
        ("2022", d["year"] == 2022),
        ("2023", d["year"] == 2023),
        ("2022-23", d["year"].isin([2022, 2023])),
        ("2024-26", d["year"] >= 2024),
    ]:
        g = d.loc[mask, "g"].dropna()
        if g.empty:
            rows[name] = {"n": 0}
            continue
        wins = g[g > 0].sum()
        losses = (-g[g < 0]).sum()
        pf = float(wins / losses) if losses > 0 else None
        rows[name] = {
            "n": int(len(g)),
            "E": round(float(g.mean()), 4),
            "PF": None if pf is None else round(pf, 4),
            "WR": round(float((g > 0).mean()) * 100, 1),
        }
    print(label, json.dumps(rows, indent=2))
    return rows


s_mid = stats(mid, "mid_repaired")
s_ab = stats(ab, "ask_bid")
# matched liquid sizes already known
(OUT / "last_15m_wheel_fullpool_pooled.json").write_text(
    json.dumps({"mid_repaired": s_mid, "ask_bid": s_ab}, indent=2),
    encoding="utf-8",
)
