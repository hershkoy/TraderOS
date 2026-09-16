import json
import os
from pathlib import Path

index_path = Path(r"D:\WORK\Projs\IB\backTraderTest\.cursor\hooks\state\continual-learning-index.json")
root = Path(r"C:\Users\Hezi\.cursor\projects\d-WORK-Projs-IB-backTraderTest\agent-transcripts")
idx = json.loads(index_path.read_text(encoding="utf-8"))
indexed = idx.get("transcripts", {})
indexed_norm = {os.path.normpath(k): v for k, v in indexed.items()}

files = list(root.rglob("*.jsonl"))
new_or_stale = []
current = {}
for f in files:
    p = os.path.normpath(str(f))
    mtime_ms = int(f.stat().st_mtime * 1000)
    current[p] = mtime_ms
    prev = indexed_norm.get(p)
    if prev is None:
        new_or_stale.append((p, mtime_ms, "NEW"))
    elif mtime_ms > prev.get("mtime_ms", 0):
        delta = mtime_ms - prev.get("mtime_ms")
        new_or_stale.append((p, mtime_ms, f"STALE prev={prev.get('mtime_ms')} delta={delta}"))

deleted = [k for k in indexed_norm if k not in current]
print(f"Total files: {len(files)}")
print(f"Indexed: {len(indexed)}")
print(f"New/stale: {len(new_or_stale)}")
print(f"Deleted: {len(deleted)}")
print("--- NEW/STALE ---")
for p, m, reason in sorted(new_or_stale, key=lambda x: x[1], reverse=True):
    print(f"{reason}\t{m}\t{p}")
print("--- DELETED ---")
for p in deleted:
    print(p)
