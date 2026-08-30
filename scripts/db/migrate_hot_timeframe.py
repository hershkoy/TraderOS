"""Apply /hot dashboard timeframe columns (stock, timeframe PK)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.scanning.channel_touch_candidates_store import ChannelTouchCandidatesStore


def main() -> int:
    store = ChannelTouchCandidatesStore()
    store.ensure_tables()
    settings = store.load_settings()
    rows = store.load_rows()
    print("timeframe_filter=%s as_of=%s as_of_1d=%s n_rows=%d" % (
        settings.get("timeframe_filter"),
        settings.get("as_of"),
        settings.get("as_of_1d"),
        len(rows),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
