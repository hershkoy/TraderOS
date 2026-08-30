"""Daily /hot watchlist helpers."""
from utils.scanning.channel_touch import LIVE_DEFAULTS
from utils.scanning.channel_touch_1d import TIMEFRAME_1D, daily_channel_kwargs


def test_daily_channel_kwargs_match_live_defaults():
    kw = daily_channel_kwargs()
    assert kw["error_pct"] == float(LIVE_DEFAULTS["error_pct"])
    assert kw["pivot_len"] == int(LIVE_DEFAULTS["pivot_len"])
    assert TIMEFRAME_1D == "1d"
