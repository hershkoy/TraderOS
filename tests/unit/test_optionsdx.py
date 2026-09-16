"""Unit tests for OptionsDX EOD CSV parsing."""

from datetime import datetime, timezone

from utils.options.optionsdx import (
    iter_csv_rows_quotes,
    normalize_header,
    parse_csv_text,
    quotes_from_row,
    underlying_from_archive_name,
)

HEADER = (
    "[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], "
    "[UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], [C_GAMMA], "
    "[C_VEGA], [C_THETA], [C_RHO], [C_IV], [C_VOLUME], [C_LAST], [C_SIZE], [C_BID], "
    "[C_ASK], [STRIKE], [P_BID], [P_ASK], [P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], "
    "[P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], [STRIKE_DISTANCE], "
    "[STRIKE_DISTANCE_PCT]"
)

ROW = (
    "1262638800, 2010-01-04 16:00, 2010-01-04, 16.000000, 20.090000, 2010-01-19, "
    "1263934800, 15.000000, 0.922150, 0.003930, 0.004730, -0.178400, 0.000310, "
    "12.045810, 12.000000, 13.000000, 262 x 583, 12.100000, 12.600000, 10.000000, "
    "0.000000, 0.040000, 0 x 11550, 0.030000, -0.020620, 0.016430, 0.001240, "
    "-0.004730, 0.000000, 0.977670, 0.000000, 10.100000, 0.502000"
)

EMPTY_IV_ROW = (
    "1262638800, 2010-01-04 16:00, 2010-01-04, 16.000000, 20.090000, 2010-01-19, "
    "1263934800, 15.000000, 0.743230, 0.016500, 0.010210, -0.201800, 0.001600, "
    "6.299850, 10.000000, 7.300000, 393 x 669, 7.100000, 7.400000, 15.000000, "
    "0.000000, 0.040000, 0 x 7671, 0.030000, -1.000000, 0.000000, 0.000000, "
    "0.000000, 0.000000, , 3.000000, 5.100000, 0.253000"
)


def test_underlying_from_archive_name():
    assert underlying_from_archive_name("spy_eod_2022q1-ww0cra.7z") == "SPY"
    assert underlying_from_archive_name("vix_eod_2010-0pjoap.7z") == "VIX"


def test_normalize_header_strips_brackets():
    assert normalize_header(" [QUOTE_UNIXTIME] ") == "QUOTE_UNIXTIME"


def test_parse_call_and_put():
    pairs = parse_csv_text(HEADER + "\n" + ROW + "\n", "VIX")
    assert len(pairs) == 2
    call_q, call_c = pairs[0]
    put_q, put_c = pairs[1]
    assert call_c["option_id"] == "VIX_2010-01-19_001000C"
    assert put_c["option_id"] == "VIX_2010-01-19_001000P"
    assert call_c["strike_cents"] == 1000
    assert call_q["bid"] == 12.1
    assert put_q["ask"] == 0.04
    assert call_q["volume"] == 12
    assert call_q["snapshot_type"] == "eod"
    ts = datetime.fromisoformat(call_q["ts"])
    assert ts == datetime(2010, 1, 4, 21, 0, tzinfo=timezone.utc)


def test_empty_put_iv_is_none():
    pairs = list(
        iter_csv_rows_quotes(
            (HEADER + "\n" + EMPTY_IV_ROW + "\n").splitlines(True), "VIX"
        )
    )
    put_q = pairs[1][0]
    assert put_q["iv"] is None
    assert put_q["delta"] == -1.0


def test_missing_strike_skipped():
    row = {"QUOTE_UNIXTIME": "1262638800", "EXPIRE_DATE": "2010-01-19", "STRIKE": ""}
    assert quotes_from_row(row, "SPY") == []
