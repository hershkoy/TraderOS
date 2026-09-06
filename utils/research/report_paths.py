"""Date-folder layout for ``reports/ascending_channels``.

Timestamped run artifacts go in ``YYYY-MM-DD`` subfolders. Live sidecars
(watchlist, coverage, alpaca lists, ``current_best``, ``1d_unrealistic``) stay at the folder root.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASCENDING_CHANNELS = ROOT / "reports" / "ascending_channels"

ISO_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Last 20YYMMDD or 20YYMMDD_HHMMSS in a filename.
STAMP_RE = re.compile(r"(20\d{6})(?:_(\d{6}))?")

KEEP_AT_ROOT_NAMES = frozenset(
    {
        "current_best",
        "1d_unrealistic",
        "alpaca_1d_missing_after_hang.txt",
        "alpaca_1d_need_update.txt",
        "alpaca_1d_symbols.txt",
        "channel_touch_15m_watchlist.json",
        "channel_touch_15m_fills_log.csv",
        "ib_15m_coverage.csv",
        "ib_5m_coverage.csv",
        "watchlist_channels_draw.json",
        "watchlist_trendline_alert_batch.json",
        "hto_channel_draw.json",
        "krys_channel_draw.json",
        "unit_channel_draw.json",
    }
)


def is_iso_date_dir_name(name: str) -> bool:
    if not ISO_DATE_DIR_RE.match(name or ""):
        return False
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def dated_outdir(
    base: Optional[Path] = None,
    when: Optional[datetime | date] = None,
) -> Path:
    """Return ``base/YYYY-MM-DD``, unless ``base`` is already a date folder."""
    base = Path(base) if base is not None else DEFAULT_ASCENDING_CHANNELS
    if when is None:
        when = datetime.now()
    day = when.date() if isinstance(when, datetime) else when
    out = base if is_iso_date_dir_name(base.name) else base / day.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_filename_date(name: str) -> Optional[date]:
    matches = list(STAMP_RE.finditer(name))
    if not matches:
        return None
    raw = matches[-1].group(1)
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def resolve_artifact(name: str, base: Optional[Path] = None) -> Path:
    """Locate ``name`` at ``base``, in its stamp-date folder, or any date folder."""
    base = Path(base) if base is not None else DEFAULT_ASCENDING_CHANNELS
    name = Path(name).name
    direct = base / name
    if direct.exists():
        return direct
    stamp_day = parse_filename_date(name)
    expected = (base / stamp_day.isoformat() / name) if stamp_day is not None else direct
    if expected.exists():
        return expected
    matches = [
        p
        for p in base.glob("*/" + name)
        if p.is_file() and is_iso_date_dir_name(p.parent.name)
    ]
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return expected


def iter_report_files(base: Path, pattern: str) -> Iterator[Path]:
    """Yield files matching ``pattern`` in ``base`` and one-level date subfolders.

    If ``base`` is already a ``YYYY-MM-DD`` folder, search its parent so a
    write-path dated outdir still finds artifacts from other days.
    """
    base = Path(base)
    if is_iso_date_dir_name(base.name):
        base = base.parent
    seen = set()
    candidates: Iterable[Path] = list(base.glob(pattern)) + list(base.glob("*/" + pattern))
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        if path.parent != base and not is_iso_date_dir_name(path.parent.name):
            continue
        seen.add(path)
        yield path


def artifact_sort_key(path: Path) -> tuple:
    """Sort key for 'latest' artifacts: filename stamp, then mtime."""
    matches = list(STAMP_RE.finditer(path.name))
    if matches:
        day = matches[-1].group(1)
        tod = matches[-1].group(2) or "000000"
        return (day + tod, path.stat().st_mtime)
    return ("", path.stat().st_mtime)


def latest_matching(
    base: Path,
    pattern: str,
    *,
    exclude_substr: Sequence[str] = (),
) -> Optional[Path]:
    files: List[Path] = []
    for path in iter_report_files(base, pattern):
        if any(token in path.name for token in exclude_substr):
            continue
        files.append(path)
    if not files:
        return None
    return max(files, key=artifact_sort_key)


def migrate_flat_reports(base: Optional[Path] = None) -> List[Tuple[Path, Path]]:
    """Move loose run artifacts into ``YYYY-MM-DD`` subfolders. Returns moves."""
    base = Path(base) if base is not None else DEFAULT_ASCENDING_CHANNELS
    moved: List[Tuple[Path, Path]] = []
    if not base.is_dir():
        return moved
    for item in list(base.iterdir()):
        if item.name in KEEP_AT_ROOT_NAMES:
            continue
        if item.is_dir() or not item.is_file():
            continue
        day = parse_filename_date(item.name)
        if day is None:
            day = datetime.fromtimestamp(item.stat().st_mtime).date()
        dest_dir = base / day.isoformat()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item.name
        if dest.exists():
            continue
        shutil.move(str(item), str(dest))
        moved.append((item, dest))
    return moved


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Move flat ascending-channel reports into date folders")
    ap.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_ASCENDING_CHANNELS,
        help="reports/ascending_channels root",
    )
    args = ap.parse_args(argv)
    moved = migrate_flat_reports(args.base)
    print("moved %d files under %s" % (len(moved), args.base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
