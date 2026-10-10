from pathlib import Path
import json

p = Path("reports/ascending_channels/2026-10-09/last_15m_wheel_cache.jsonl")
hits = []
for line in p.open(encoding="utf-8"):
    o = json.loads(line)
    k = o.get("key", "")
    if k.startswith("ACGL|2022-01-10|"):
        hits.append((k, o.get("wheel_ok"), o.get("wheel_reason"), o.get("put_credit")))
print("n", len(hits))
for h in hits:
    print(h)

# count mid 2022 ok vs fail last-wins
last = {}
n_lines = 0
for line in p.open(encoding="utf-8"):
    n_lines += 1
    o = json.loads(line)
    k = o.get("key")
    if k:
        last[k] = o
print("cache lines", n_lines, "unique keys", len(last))
mid22_ok = mid22_bad = 0
ab22_ok = ab22_bad = 0
for k, o in last.items():
    if "|2022-" not in k:
        continue
    if k.endswith("|mid"):
        if o.get("wheel_ok"):
            mid22_ok += 1
        else:
            mid22_bad += 1
    if k.endswith("|ask_bid"):
        if o.get("wheel_ok"):
            ab22_ok += 1
        else:
            ab22_bad += 1
print("last-win mid 2022 ok/bad", mid22_ok, mid22_bad)
print("last-win ask_bid 2022 ok/bad", ab22_ok, ab22_bad)
