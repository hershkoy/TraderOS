"""CLI defaults for the 15m channel-touch scanner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "scanners"))

from datetime import datetime, timezone

from channel_touch_15m import (
    _should_rebuild_watchlist,
    _split_refresh_batches,
    build_arg_parser,
)
from utils.scanning.channel_touch_15m import FROZEN_OVERSHOOT_MIN, LIVE_15M_DEFAULTS  # noqa: E402


def test_15m_cli_defaults_match_h5_stack():
    ns = build_arg_parser().parse_args(["--dry-run"])
    assert ns.mode == "run"
    assert ns.provider == "IB"
    assert ns.timeframe == "15m"
    assert ns.min_l3_wait_bars == LIVE_15M_DEFAULTS["min_l3_wait_bars"] == 12
    assert ns.max_channel_span_days == 10.0
    assert ns.volume_rel_min == 2.0
    assert ns.overshoot_min == FROZEN_OVERSHOOT_MIN
    assert ns.proximity_below_pct == 0.0
    assert ns.lookback_sessions == 40
    assert ns.ib_client_id == 8826
    assert ns.skip_ib_refresh is False
    assert ns.refresh_below_pct == 0.5


def test_15m_cli_skip_ib_refresh():
    ns = build_arg_parser().parse_args(["--mode", "run", "--skip-ib-refresh"])
    assert ns.skip_ib_refresh is True


def test_15m_cli_proximity_mode():
    ns = build_arg_parser().parse_args(["--mode", "proximity", "--dry-run"])
    assert ns.mode == "proximity"


def test_crontab_has_proximity_job():
    import yaml

    data = yaml.safe_load((ROOT / "crons" / "crontab.yaml").read_text(encoding="utf-8"))
    names = [j["name"] for j in data["jobs"]]
    assert "channel_touch_15m_proximity" in names
    job = next(j for j in data["jobs"] if j["name"] == "channel_touch_15m_proximity")
    assert job["command"] == r"crons\channel_touch_15m_proximity.bat"
    assert "* 10-15 * * 1-5" in job["schedule"]
    assert job.get("timezone") == "America/New_York"


def test_rebuild_watchlist_on_prior_session():
    args = build_arg_parser().parse_args(["--mode", "run"])
    rows = [{"stock": "AAA", "timeframe": "15m", "status": "armed"}]
    assert _should_rebuild_watchlist(args, rows, "2026-09-01 19:45:00") is True
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    assert _should_rebuild_watchlist(args, rows, today) is False
    force = build_arg_parser().parse_args(["--mode", "run", "--rebuild-watchlist"])
    assert _should_rebuild_watchlist(force, rows, today) is True


def test_rebuild_watchlist_only_once_per_et_day():
    args = build_arg_parser().parse_args(["--mode", "run"])
    rows = [{"stock": "AAA", "timeframe": "15m", "status": "armed"}]
    now = datetime(2026, 9, 3, 14, 0, tzinfo=timezone.utc)
    already = {"watchlist_built_et": "2026-09-03"}
    assert (
        _should_rebuild_watchlist(
            args, rows, "2026-09-02 19:45:00", payload=already, now=now
        )
        is False
    )
    assert (
        _should_rebuild_watchlist(
            args, rows, "2026-09-02 19:45:00", payload={}, now=now
        )
        is True
    )


def test_split_refresh_batches_hot_first():
    rows = [
        {
            "stock": "FAR",
            "timeframe": "15m",
            "status": "armed",
            "hot": False,
            "resist": 100.0,
            "last_price": 90.0,
            "dist_live_pct": -10.0,
        },
        {
            "stock": "HOT",
            "timeframe": "15m",
            "status": "armed",
            "hot": True,
            "resist": 100.0,
            "last_price": 100.2,
            "dist_live_pct": 0.2,
        },
    ]
    first, rest = _split_refresh_batches(rows, refresh_below_pct=0.5)
    assert first == ["HOT"]
    assert rest == ["FAR"]
