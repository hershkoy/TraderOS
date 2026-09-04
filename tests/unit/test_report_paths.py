"""Unit tests for dated report-folder helpers."""
from datetime import date, datetime
from pathlib import Path

from utils.research.report_paths import (
    dated_outdir,
    is_iso_date_dir_name,
    latest_matching,
    migrate_flat_reports,
    parse_filename_date,
    resolve_artifact,
)


def test_is_iso_date_dir_name():
    assert is_iso_date_dir_name("2026-09-04")
    assert not is_iso_date_dir_name("current_best")
    assert not is_iso_date_dir_name("2026-13-01")
    assert not is_iso_date_dir_name("20260904")


def test_parse_filename_date_stamp_suffix():
    assert parse_filename_date(
        "channel_touch_tv_report_interactive_rs_top1_default_fric0.10_15m_l3_wait1_realistic_20260904_114015.html"
    ) == date(2026, 9, 4)
    assert parse_filename_date("channel_touch_l3_opt_filter_ab_20260828.csv") == date(2026, 8, 28)
    assert parse_filename_date(
        "channel_touch_confidence_size_15m_20260829_115900_folds.csv"
    ) == date(2026, 8, 29)
    assert parse_filename_date("watchlist_channels_draw.json") is None


def test_dated_outdir_appends_day(tmp_path):
    when = datetime(2026, 9, 4, 12, 30, 0)
    out = dated_outdir(tmp_path, when=when)
    assert out == tmp_path / "2026-09-04"
    assert out.is_dir()


def test_dated_outdir_does_not_nest_date_folder(tmp_path):
    already = tmp_path / "2026-09-04"
    already.mkdir()
    out = dated_outdir(already, when=datetime(2026, 9, 5))
    assert out == already


def test_resolve_artifact_prefers_stamp_folder(tmp_path):
    name = "channel_touch_trades_20260828_194314.csv"
    dated = tmp_path / "2026-08-28"
    dated.mkdir()
    target = dated / name
    target.write_text("stock\n", encoding="utf-8")
    (tmp_path / "2026-09-01").mkdir()
    assert resolve_artifact(name, base=tmp_path) == target


def test_resolve_artifact_searches_date_dirs_without_stamp(tmp_path):
    name = "channel_touch_h2_break_span365.csv"
    dated = tmp_path / "2026-09-04"
    dated.mkdir()
    target = dated / name
    target.write_text("stock\n", encoding="utf-8")
    assert resolve_artifact(name, base=tmp_path) == target


def test_latest_matching_searches_date_subfolders(tmp_path):
    old_dir = tmp_path / "2026-08-28"
    new_dir = tmp_path / "2026-09-04"
    old_dir.mkdir()
    new_dir.mkdir()
    older = old_dir / "channel_touch_trades_20260828_194314.csv"
    newer = new_dir / "channel_touch_trades_20260904_113653.csv"
    raw = new_dir / "channel_touch_trades_raw_20260904_113653.csv"
    older.write_text("a\n", encoding="utf-8")
    newer.write_text("b\n", encoding="utf-8")
    raw.write_text("c\n", encoding="utf-8")
    found = latest_matching(
        tmp_path,
        "channel_touch_trades_*.csv",
        exclude_substr=("_trades_raw_",),
    )
    assert found == newer
    assert (
        latest_matching(new_dir, "channel_touch_trades_*.csv", exclude_substr=("_trades_raw_",))
        == newer
    )


def test_migrate_flat_reports_moves_stamped_keeps_live(tmp_path):
    stamp_name = "channel_touch_trades_20260904_113653.csv"
    (tmp_path / stamp_name).write_text("trades\n", encoding="utf-8")
    (tmp_path / "channel_touch_15m_watchlist.json").write_text("{}\n", encoding="utf-8")
    current_best = tmp_path / "current_best"
    current_best.mkdir()
    (current_best / "1d_channel_touch.html").write_text("<html></html>", encoding="utf-8")

    moved = migrate_flat_reports(tmp_path)
    assert len(moved) == 1
    dest = tmp_path / "2026-09-04" / stamp_name
    assert dest.exists()
    assert not (tmp_path / stamp_name).exists()
    assert (tmp_path / "channel_touch_15m_watchlist.json").exists()
    assert (current_best / "1d_channel_touch.html").exists()


def test_migrate_flat_reports_uses_mtime_without_stamp(tmp_path):
    loose = tmp_path / "channel_touch_h2_break_span365.csv"
    loose.write_text("x\n", encoding="utf-8")
    moved = migrate_flat_reports(tmp_path)
    assert len(moved) == 1
    dest = Path(moved[0][1])
    assert dest.parent.parent == tmp_path
    assert dest.name == loose.name
    assert dest.exists()
    assert not loose.exists()
