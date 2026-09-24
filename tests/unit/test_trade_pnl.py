"""TPV paste parser and bar bucketing for the Charts pane."""
from __future__ import annotations

from utils.charting.trade_pnl import parse_tpv_trades, bucket_tpv, tpv_payload, trade_from_obj


def test_parse_compact_json():
    trades = parse_tpv_trades(
        '[{"s":"WGO","x":1561477500,"g":-5.73},{"s":"GNTX","x":1562682600,"g":-3.95}]'
    )
    assert len(trades) == 2
    assert trades[0]["s"] == "WGO"
    assert trades[0]["t_ms"] == 1561477500 * 1000
    assert trades[0]["g"] == -5.73


def test_parse_bulky_json_ex_ms():
    trades = parse_tpv_trades(
        '[{"sym":"WGO","enp":39.925,"exp":37.735,"en_ms":1560973500000,'
        '"ex_ms":1561477500000,"g":-5.7353}]'
    )
    assert trades[0]["s"] == "WGO"
    assert trades[0]["t_ms"] == 1561477500000
    assert abs(trades[0]["g"] + 5.7353) < 1e-9


def test_parse_csv_with_and_without_symbol():
    text = "WGO,1561477500,-5.73\n1562682600,-3.95\n"
    trades = parse_tpv_trades(text)
    assert trades[0]["s"] == "WGO"
    assert "s" not in trades[1]
    assert trades[1]["t_ms"] == 1562682600 * 1000


def test_parse_gain_from_prices():
    tr = trade_from_obj({"s": "AAA", "x": 1561477500, "enp": 100, "exp": 110})
    assert tr is not None
    assert abs(tr["g"] - 10.0) < 1e-9


def test_bucket_assigns_to_last_bar_at_or_before():
    trades = [{"s": "WGO", "t_ms": 1561477500000, "g": -5.0}]
    bars = [
        "2019-06-23 00:00:00",
        "2019-06-24 00:00:00",
        "2019-06-26 00:00:00",
    ]
    out = bucket_tpv(trades, bars)
    assert out["n_on"] == 1
    assert out["y"][0] is None
    assert out["y"][1] == -5.0
    assert out["colors"][1] == "#ef5350"


def test_this_sym_only_skips_other_names():
    trades = parse_tpv_trades("WGO,1561477500,-5.73\nGNTX,1561477500,1.2\n")
    bars = ["2019-06-24 00:00:00", "2019-06-26 00:00:00"]
    out = bucket_tpv(trades, bars, symbol="WGO", this_sym_only=True)
    assert out["n_on"] == 1
    assert out["n_skip"] == 1
    assert out["y"][0] == -5.73
    assert out["y"][1] is None


def test_tpv_payload_empty_raises():
    try:
        parse_tpv_trades("   ")
    except ValueError as exc:
        assert "Paste" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_tpv_payload_includes_bucket_when_bars():
    payload = tpv_payload(
        "WGO,1561477500,2.5",
        bar_times=["2019-06-24 00:00:00", "2019-06-26 00:00:00"],
    )
    assert payload["n_on"] == 1
    assert payload["y"][0] == 2.5
    assert payload["colors"][0] == "#26a69a"
