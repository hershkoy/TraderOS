"""Re-append repaired mid fills into jsonl so prefer-ok load_cache recovers them."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("reports/ascending_channels/2026-10-09")
latest = (OUT / "last_15m_wheel_mid_repaired_latest.txt").read_text(encoding="utf-8").strip()
csv_path = Path(latest)
cache = OUT / "last_15m_wheel_cache.jsonl"
df = pd.read_csv(csv_path)
df = df[df["wheel_ok"] == True]  # noqa: E712
n = 0
with cache.open("a", encoding="utf-8") as f:
    for _, r in df.iterrows():
        rec = r.to_dict()
        buy = pd.Timestamp(r["buy_date"]).strftime("%Y-%m-%d")
        rec["key"] = "%s|%s|mid" % (str(r["stock"]).upper(), buy)
        # json-friendly
        for k, v in list(rec.items()):
            if pd.isna(v):
                rec[k] = None
        f.write(json.dumps(rec, default=str) + "\n")
        n += 1
print("appended", n, "mid fills to", cache)
