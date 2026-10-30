"""Unit test: iron-fly API bag uses BUY + negative credit (not TWS SELL/+credit)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.trading.spxw_iron_fly_combo import (
    _api_errors_have_code,
    build_bag,
    credit_limit_price,
    describe_short_iron_fly,
    is_short_if_credit_ticket,
    make_credit_bag_order,
    round_spxw_credit,
    validate_short_iron_fly,
)


class _Q:
    def __init__(self, conId, strike, right):
        self.conId = conId
        self.strike = strike
        self.right = right
        self.localSymbol = f"SPXW{strike}{right}"
        self.tradingClass = "SPXW"


def test_build_bag_open_close_is_open():
    ib = MagicMock()
    ib.qualifyContracts.side_effect = lambda *a: [
        _Q(4, 120, "C"),
        _Q(2, 90, "P"),
        _Q(3, 110, "C"),
        _Q(1, 100, "P"),
    ]

    sel = {
        "expiry": "20261027",
        "short_put": 100.0,
        "long_put": 90.0,
        "short_call": 110.0,
        "long_call": 120.0,
    }
    bag, detail = build_bag(ib, sel, bag_exchange="SMART")
    assert bag.secType == "BAG"
    assert all(leg.openClose == 1 for leg in bag.comboLegs)
    by_strike = {detail[k]["strike"]: detail[k]["action"] for k in detail}
    assert by_strike[100.0] == "SELL"
    assert by_strike[90.0] == "BUY"
    assert by_strike[110.0] == "SELL"
    assert by_strike[120.0] == "BUY"


def test_validate_rejects_crossed_strikes():
    sel = {
        "short_put": 100.0,
        "long_put": 110.0,
        "short_call": 105.0,
        "long_call": 120.0,
    }
    detail = {
        "short_put": {"strike": 100.0, "right": "P"},
        "long_put": {"strike": 110.0, "right": "P"},
        "short_call": {"strike": 105.0, "right": "C"},
        "long_call": {"strike": 120.0, "right": "C"},
    }
    with pytest.raises(RuntimeError, match="strike order"):
        validate_short_iron_fly(detail, sel)


def test_describe_buy_negative_is_credit():
    detail = {
        "short_put": {"strike": 7725.0, "right": "P", "conId": 1},
        "long_put": {"strike": 7710.0, "right": "P", "conId": 2},
        "short_call": {"strike": 7730.0, "right": "C", "conId": 3},
        "long_call": {"strike": 7745.0, "right": "C", "conId": 4},
    }
    text = describe_short_iron_fly(detail, order_action="BUY", lmt=-8.05)
    assert "CREDIT" in text
    assert "WRONG" not in text


def test_describe_flags_buy_positive_as_wrong():
    detail = {
        "short_put": {"strike": 100.0, "right": "P", "conId": 1},
        "long_put": {"strike": 90.0, "right": "P", "conId": 2},
        "short_call": {"strike": 110.0, "right": "C", "conId": 3},
        "long_call": {"strike": 120.0, "right": "C", "conId": 4},
    }
    text = describe_short_iron_fly(detail, order_action="BUY", lmt=7.95)
    assert "WRONG" in text


def test_round_spxw_credit_nickel_tick():
    # Mid 6.525 must not become 6.52 (Error 110)
    assert round_spxw_credit(6.5249999999999995) == 6.50
    assert round_spxw_credit(6.53) == 6.55
    assert round_spxw_credit(8.05) == 8.05


def test_credit_limit_price_api_default_buy_negative():
    assert credit_limit_price(8.05, "BUY") == -8.05
    assert credit_limit_price(6.5249999999999995, "BUY") == -6.50
    assert credit_limit_price(7.875, "SELL") == 7.90  # nearest $0.05
    assert is_short_if_credit_ticket("BUY", -8.65)
    assert is_short_if_credit_ticket("SELL", 8.05)
    assert not is_short_if_credit_ticket("BUY", 7.95)
    assert not is_short_if_credit_ticket("SELL", -7.95)


def test_make_credit_bag_order_defaults_buy_negative_lmt():
    order = make_credit_bag_order(credit=8.05, qty=1, account="DUR310354")
    assert order.action == "BUY"
    assert order.lmtPrice == -8.05
    assert order.totalQuantity == 1
    assert order.account == "DUR310354"
    assert order.transmit is True


def test_make_credit_bag_order_no_transmit_rounds_tick():
    order = make_credit_bag_order(credit=6.5249999999999995, transmit=False)
    assert order.transmit is False
    assert order.lmtPrice == -6.50


def test_make_credit_bag_order_sell_override_positive():
    order = make_credit_bag_order(credit=8.65, action="SELL")
    assert order.action == "SELL"
    assert order.lmtPrice == 8.65


def test_make_credit_bag_order_rejects_bad_action():
    with pytest.raises(ValueError):
        make_credit_bag_order(credit=1.0, action="HOLD")


def test_api_errors_detect_riskless_201():
    errs = [{"code": 201, "msg": "Riskless combination orders are not allowed."}]
    assert _api_errors_have_code(errs, 201)
    assert not _api_errors_have_code(errs, 10148)
