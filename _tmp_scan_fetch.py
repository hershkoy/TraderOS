from pathlib import Path

path = Path(r"D:/WORK/Projs/IB/backTraderTest/utils/data/fetch_data.py")
lines = path.read_text(encoding="utf-8").splitlines()
keys = [
    "barSizeSetting",
    "5 mins",
    "1 min",
    "15 mins",
    "def fetch_ib",
    "def ib_step",
    "def _prepare_ib",
    "bar_size",
    "durationStr",
    "whatToShow",
    "1m",
    "5m",
    "15m",
    "timeframe",
]
# Deduplicate noisy timeframe matches: only print if strong signal
strong = [
    "barSizeSetting",
    "5 mins",
    "1 min",
    "15 mins",
    "def fetch_ib",
    "def ib_step",
    "def _prepare_ib",
    "bar_size",
    "durationStr",
    "BAR_SIZE",
    "TIMEFRAME_TO",
]
for i, line in enumerate(lines, 1):
    if any(k in line for k in strong) or (
        ("1m" in line or "5m" in line or "15m" in line)
        and ("mins" in line or "bar" in line.lower() or "map" in line.lower() or ":" in line)
    ):
        print(f"{i}: {line}")

# Also dump function bodies for the key helpers
for fname in [
    "fetch_ib_historical_window",
    "ib_step_back_minutes",
    "_prepare_ib_duration_from_days",
]:
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {fname}"):
            start = i
            break
    if start is None:
        print(f"MISSING {fname}")
        continue
    end = start + 1
    while end < len(lines) and not (
        lines[end].startswith("def ") and end > start
    ):
        # stop at next top-level def
        if lines[end].startswith("def ") or (
            lines[end].startswith("class ") and end > start
        ):
            break
        end += 1
    # also stop if we hit another def at indent 0
    end2 = start + 1
    while end2 < len(lines):
        if lines[end2].startswith("def ") or lines[end2].startswith("class "):
            break
        end2 += 1
    print("=" * 80)
    print(f"FUNCTION {fname} lines {start+1}-{end2}")
    print("=" * 80)
    for i in range(start, min(end2, start + 120)):
        print(f"{i+1}: {lines[i]}")
