from datetime import date

from utils.research.tachun_long_book import (
    cagr,
    dcf_equity_value,
    fade_growth_path,
    haircut_betas,
    uptrend_ok,
)


def test_cagr_positive():
    g = cagr(100.0, 161.05, 5.0)
    assert g is not None
    assert abs(g - 0.10) < 0.001


def test_uptrend_last_above_first():
    pts = [
        (date(2010, 12, 31), 10.0),
        (date(2014, 12, 31), 15.0),
    ]
    assert uptrend_ok(pts, min_points=2)
    assert not uptrend_ok([(date(2010, 12, 31), 10.0), (date(2014, 12, 31), 9.0)], min_points=2)


def test_fade_growth_ends_near_terminal():
    path = fade_growth_path(0.12, years=20, g_term=0.02)
    assert len(path) == 20
    assert abs(path[-1] - 0.02) < 1e-9
    assert path[0] > path[-1]


def test_dcf_positive_and_higher_at_lower_discount():
    lo = dcf_equity_value(1_000_000.0, 0.08, 0.08)
    hi = dcf_equity_value(1_000_000.0, 0.08, 0.12)
    assert lo is not None and hi is not None
    assert lo > hi > 0


def test_haircut_betas_palantir_like():
    raw, opt, pess = haircut_betas(2.8)
    assert raw == 2.8
    assert opt <= 1.5
    assert pess > opt
    assert pess <= 3.0
