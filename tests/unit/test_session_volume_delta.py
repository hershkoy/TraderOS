"""Buy/sell volume split from candle geometry (no Backtrader)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.research.session_volume_delta import (
    bar_volume_delta,
    consecutive_seller_sessions,
    session_volume_delta,
    sessions_from_by_day,
)

ET = ZoneInfo("America/New_York")


def _bar(ts, o, h, l, c, v):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_bar_volume_delta_close_near_high():
    d = bar_volume_delta(high=10.0, low=8.0, close=9.5, volume=100.0)
    assert d.buy_volume == 75.0
    assert d.sell_volume == 25.0
    assert d.buy_pct == 75.0
    assert d.sell_pct == 25.0
    assert d.delta == 50.0


def test_bar_volume_delta_close_at_low():
    d = bar_volume_delta(high=10.0, low=8.0, close=8.0, volume=80.0)
    assert d.buy_volume == 0.0
    assert d.sell_volume == 80.0
    assert d.sell_pct == 100.0


def test_session_volume_delta_sums_15m():
    d0 = datetime(2024, 2, 12, 9, 30, tzinfo=ET)
    d1 = datetime(2024, 2, 12, 15, 45, tzinfo=ET)
    # First bar: all buy (close=high). Second: all sell (close=low). Equal volume.
    bars = [
        _bar(d0, 10.0, 11.0, 10.0, 11.0, 100.0),
        _bar(d1, 11.0, 11.0, 10.0, 10.0, 100.0),
    ]
    sess = session_volume_delta(bars)
    assert sess is not None
    assert sess.session_date.isoformat() == "2024-02-12"
    assert sess.open == 10.0
    assert sess.high == 11.0
    assert sess.low == 10.0
    assert sess.close == 10.0
    assert sess.buy_volume == 100.0
    assert sess.sell_volume == 100.0
    assert sess.buy_pct == 50.0
    assert sess.sell_pct == 50.0
    assert sess.last_bar_ts == d1


def test_session_ohlc_volume_uses_day_range_not_15m_sum():
    """Close near the session high is mostly buy even if a 15m bar dumped."""
    d0 = datetime(2024, 2, 13, 9, 30, tzinfo=ET)
    d1 = datetime(2024, 2, 13, 15, 45, tzinfo=ET)
    bars = [
        _bar(d0, 13.70, 13.80, 13.54, 13.55, 100.0),
        _bar(d1, 13.55, 14.00, 13.55, 13.90, 100.0),
    ]
    summed = session_volume_delta(bars, volume_mode="15m_sum")
    daily = session_volume_delta(bars, volume_mode="session_ohlc")
    assert summed is not None and daily is not None
    assert summed.open == 13.70
    assert daily.close == 13.90
    # 15m-sum mixes a dump bar; session OHLC close is near the high -> more buy.
    assert daily.buy_pct > summed.buy_pct
    assert daily.buy_pct > 70.0


def test_consecutive_seller_sessions_skips_fill_day():
    fill = datetime(2024, 2, 9, 15, 45, tzinfo=ET).date()
    by_day = {
        fill: [_bar(datetime(2024, 2, 9, 15, 45, tzinfo=ET), 13.0, 14.2, 13.0, 14.1, 200.0)],
        datetime(2024, 2, 12, 15, 45, tzinfo=ET).date(): [
            _bar(datetime(2024, 2, 12, 15, 45, tzinfo=ET), 14.0, 14.1, 13.5, 13.55, 150.0)
        ],
        datetime(2024, 2, 13, 15, 45, tzinfo=ET).date(): [
            _bar(datetime(2024, 2, 13, 15, 45, tzinfo=ET), 13.5, 13.6, 13.0, 13.05, 150.0)
        ],
    }
    sessions = sessions_from_by_day(by_day)
    day = consecutive_seller_sessions(sessions, fill_day=fill, seller_pct_min=55.0, n_needed=2)
    assert day is not None
    assert day.isoformat() == "2024-02-13"


def test_session_ohlc_seller_streak_disagrees_with_15m_sum():
    """Daily close near the high is not a seller session even if 15m bars dump."""
    fill = datetime(2024, 2, 9, 15, 45, tzinfo=ET).date()
    d12 = datetime(2024, 2, 12, 9, 30, tzinfo=ET)
    d12c = datetime(2024, 2, 12, 15, 45, tzinfo=ET)
    d13 = datetime(2024, 2, 13, 9, 30, tzinfo=ET)
    d13c = datetime(2024, 2, 13, 15, 45, tzinfo=ET)
    by_day = {
        fill: [_bar(datetime(2024, 2, 9, 15, 45, tzinfo=ET), 13.0, 14.2, 13.0, 14.1, 200.0)],
        d12.date(): [
            _bar(d12, 14.00, 14.10, 13.50, 13.55, 1000.0),
            _bar(d12c, 13.55, 14.20, 13.50, 14.10, 50.0),
        ],
        d13.date(): [
            _bar(d13, 14.00, 14.10, 13.40, 13.45, 1000.0),
            _bar(d13c, 13.45, 14.30, 13.40, 14.20, 50.0),
        ],
    }
    sum_day = consecutive_seller_sessions(
        sessions_from_by_day(by_day, volume_mode="15m_sum"),
        fill_day=fill,
        seller_pct_min=55.0,
        n_needed=2,
    )
    ohlc_day = consecutive_seller_sessions(
        sessions_from_by_day(by_day, volume_mode="session_ohlc"),
        fill_day=fill,
        seller_pct_min=55.0,
        n_needed=2,
    )
    assert sum_day is not None
    assert ohlc_day is None


def test_consecutive_seller_sessions_breaks_on_buyer_day():
    fill = datetime(2024, 2, 9, 15, 45, tzinfo=ET).date()
    by_day = {
        datetime(2024, 2, 12, 15, 45, tzinfo=ET).date(): [
            _bar(datetime(2024, 2, 12, 15, 45, tzinfo=ET), 14.0, 14.1, 13.5, 13.55, 150.0)
        ],
        datetime(2024, 2, 13, 15, 45, tzinfo=ET).date(): [
            _bar(datetime(2024, 2, 13, 15, 45, tzinfo=ET), 13.5, 14.2, 13.5, 14.15, 150.0)
        ],
        datetime(2024, 2, 14, 15, 45, tzinfo=ET).date(): [
            _bar(datetime(2024, 2, 14, 15, 45, tzinfo=ET), 14.0, 14.1, 13.4, 13.45, 150.0)
        ],
    }
    sessions = sessions_from_by_day(by_day)
    day = consecutive_seller_sessions(sessions, fill_day=fill, seller_pct_min=55.0, n_needed=2)
    assert day is None
