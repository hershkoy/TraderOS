"""Load 15m watchlist JSON into the /hot TimescaleDB store (after a scan that skipped DB)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.scanning.channel_touch_candidates_store import ChannelTouchCandidatesStore

DEFAULT_JSON = ROOT / "reports" / "ascending_channels" / "channel_touch_15m_watchlist.json"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])
    store = ChannelTouchCandidatesStore()
    merged = store.replace_candidates(
        rows,
        meta={
            "as_of": payload.get("as_of"),
            "n_universe": int(payload.get("n_universe") or 0),
            "stale_warning": payload.get("stale_warning"),
        },
        timeframe="15m",
    )
    print("persisted %d 15m rows from %s as_of=%s" % (len(merged), path.name, payload.get("as_of")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
