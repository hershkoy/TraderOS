"""Unit tests for ThetaData OCC parse + first-marketable + NBBO debit walk."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from utils.research.pcs_thetadata_scale_status import (
    effective_state,
    public_status,
    write_status,
)
from utils.research.post_earnings_pcs_thetadata import (
    align_minute_legs,
    first_marketable_from_aligned,
    marketable_debit,
    parse_clock,
    parse_occ_ticker,
    vertical_credit_nbbo,
    walk_vertical_nbbo_debit,
)

ET = ZoneInfo("America/New_York")


def test_parse_occ():
    p = parse_occ_ticker("O:COST261030P00850000")
    assert p["symbol"] == "COST"
    assert p["expiration"] == date(2026, 10, 30)
    assert p["right"] == "put"
    assert p["strike"] == pytest.approx(850.0)


def test_vertical_credit_nbbo():
    c = vertical_credit_nbbo(2.0, 2.4, 0.8, 1.0)
    assert c["credit_bidask"] == pytest.approx(1.0)  # 2.0 - 1.0
    assert c["credit_mid"] == pytest.approx(1.3)  # 2.2 - 0.9


def test_parse_clock():
    assert parse_clock("10:00") == time(10, 0)
    assert parse_clock(time(9, 30)) == time(9, 30)


def test_marketable_debit():
    assert marketable_debit(1.5, 0.4) == pytest.approx(1.1)
    assert marketable_debit(0.3, 0.5) == pytest.approx(0.0)


def _mk_leg(minutes, bids, asks):
    day = date(2024, 6, 3)
    rows = []
    for m, b, a in zip(minutes, bids, asks):
        ts = datetime(day.year, day.month, day.day, m // 60, m % 60, tzinfo=ET)
        rows.append({"timestamp": ts, "bid": b, "ask": a})
    return pd.DataFrame(rows)


def test_align_and_first_meet_min():
    # 09:30 credit too small; 09:45 meets min 0.80 on $5
    short = _mk_leg([9 * 60 + 30, 9 * 60 + 45], [1.0, 1.5], [1.2, 1.7])
    long = _mk_leg([9 * 60 + 30, 9 * 60 + 45], [0.4, 0.5], [0.55, 0.65])
    aligned = align_minute_legs(short, long)
    assert len(aligned) == 2
    assert aligned.iloc[0]["credit_bidask"] == pytest.approx(0.45)  # 1.0-0.55
    assert aligned.iloc[1]["credit_bidask"] == pytest.approx(0.85)  # 1.5-0.65

    meet = first_marketable_from_aligned(aligned, min_credit=0.80, require_min=True)
    assert meet is not None
    assert meet["fill_clock"] == "09:45"
    assert meet["wait_minutes"] == 15
    assert meet["credit_bidask"] == pytest.approx(0.85)

    any_pos = first_marketable_from_aligned(aligned, min_credit=0.80, require_min=False)
    assert any_pos is not None
    assert any_pos["fill_clock"] == "09:30"
    assert any_pos["credit_bidask"] == pytest.approx(0.45)


def test_first_marketable_no_credit():
    short = _mk_leg([9 * 60 + 30], [0.5], [0.7])
    long = _mk_leg([9 * 60 + 30], [0.4], [0.6])
    aligned = align_minute_legs(short, long)
    assert first_marketable_from_aligned(aligned, min_credit=0.80, require_min=True) is None
    # Credit 0.5-0.6 = -0.1 → not any-positive either
    assert first_marketable_from_aligned(aligned, min_credit=0.80, require_min=False) is None


def test_walk_tp50():
    daily = pd.DataFrame(
        [
            {"date": date(2024, 6, 3), "debit": 1.2},
            {"date": date(2024, 6, 4), "debit": 0.35},  # <= 0.5 * 0.80
        ]
    )
    wr = walk_vertical_nbbo_debit(
        daily,
        entry_date=date(2024, 6, 3),
        credit=0.80,
        width=5.0,
        expiry=date(2024, 7, 5),
    )
    assert wr.status == "ok"
    assert wr.exit_reason == "tp50"
    assert wr.exit_date == date(2024, 6, 4)
    assert wr.pnl == pytest.approx(0.40)
    assert wr.ror_pct == pytest.approx(100.0 * 0.40 / 4.20)


def test_walk_sl_2x():
    daily = pd.DataFrame(
        [
            {"date": date(2024, 6, 3), "debit": 1.0},
            {"date": date(2024, 6, 5), "debit": 1.70},  # >= 2 * 0.80
        ]
    )
    wr = walk_vertical_nbbo_debit(
        daily,
        entry_date=date(2024, 6, 3),
        credit=0.80,
        width=5.0,
        expiry=date(2024, 7, 5),
    )
    assert wr.status == "ok"
    assert wr.exit_reason == "sl_2x"
    assert wr.exit_debit == pytest.approx(1.60)
    assert wr.pnl == pytest.approx(0.80 - 1.60)


def test_walk_expiry():
    daily = pd.DataFrame(
        [
            {"date": date(2024, 6, 3), "debit": 0.70},
            {"date": date(2024, 7, 5), "debit": 0.55},
        ]
    )
    wr = walk_vertical_nbbo_debit(
        daily,
        entry_date=date(2024, 6, 3),
        credit=0.80,
        width=5.0,
        expiry=date(2024, 7, 5),
    )
    assert wr.status == "ok"
    assert wr.exit_reason == "expiry"
    assert wr.pnl == pytest.approx(0.80 - 0.55)


def test_status_stale_and_public(tmp_path):
    path = tmp_path / "status.json"
    write_status(
        {
            "state": "running",
            "done": 3,
            "total": 10,
            "ok": 1,
            "no_credit": 2,
            "last_key": "AAPL|2024-01-02",
            "started_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2020-01-01T00:00:00+00:00",
        },
        path=path,
    )
    # write_status overwrites updated_at — force old stamp
    import json

    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["updated_at"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(obj), encoding="utf-8")

    assert effective_state(obj) == "stale"
    pub = public_status(path=path)
    assert pub is not None
    assert pub["state"] == "stale"
    assert pub["done"] == 3
    assert pub["total"] == 10
