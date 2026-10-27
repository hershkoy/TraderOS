"""Unit tests for last-15m option-data filters. No network."""
from __future__ import annotations

from datetime import date, timedelta

import math

import pandas as pd

from utils.research.last_15m_option_filters import (
    attach_etf_features,
    choose_expiration,
    coalesce_strike_days,
    daily_surface,
    etf_for_label,
    etf_layer_wants_names,
    evaluate_skip,
    lock_first_rule,
    market_sector_panel,
    name_feature_frame,
    parse_greeks_csv,
    plan_strike_ranges,
    prior_session_values,
    put_strike_for_delta,
    quintile_table,
    sic_to_sector_etf,
    snap_strike,
    stamp_sector_atm,
    trailing_percentile,
)


def _sessions(n: int, start: date = date(2020, 1, 2)) -> list:
    out = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def test_sector_maps_labels_and_sic():
    assert etf_for_label("Technology") == "XLK"
    assert etf_for_label("Information Technology") == "XLK"
    assert etf_for_label("Materials") == "XLB"
    assert etf_for_label("Custom") is None
    assert etf_for_label("XLK") == "XLK"
    assert sic_to_sector_etf("7372") == "XLK"
    assert sic_to_sector_etf("6021") == "XLF"
    assert sic_to_sector_etf("4911") == "XLU"
    assert sic_to_sector_etf("1311") == "XLE"
    assert sic_to_sector_etf("2834") == "XLV"
    assert sic_to_sector_etf("") is None
    assert snap_strike(530.4) == 530.0
    assert snap_strike(12.2) == 12.0
    assert snap_strike(12.3) == 12.5
    day = date(2024, 5, 20)
    exps = [day + timedelta(days=d) for d in (10, 21, 32, 48)]
    assert choose_expiration(exps, day, 30, 18, 45) == day + timedelta(days=32)
    assert choose_expiration(exps, day, 14, 7, 21) == day + timedelta(days=10)
    strike = put_strike_for_delta(100.0, 30, 0.25, target_delta=-0.25)
    assert strike < 100.0
    assert strike == snap_strike(strike)


def test_parse_greeks_keeps_iv_and_drops_bad_rows():
    text = (
        "symbol,expiration,strike,timestamp,delta,implied_vol,volume\n"
        'SPY,2024-06-21,530,2024-05-20T16:00:00,-0.50,0.20,10\n'
        "SPY,2024-06-21,520,2024-05-20T16:00:00,-0.25,0.24,4\n"
        "SPY,2024-06-21,510,2024-05-20T16:00:00,-0.10,0,1\n"
    )
    got = parse_greeks_csv(text)
    assert len(got) == 2
    assert set(got["iv"]) == {0.20, 0.24}
    assert got["session"].iloc[0] == date(2024, 5, 20)


def test_parse_greeks_scales_percent_iv():
    text = (
        "timestamp,delta,implied_vol,strike\n"
        "2024-05-20T16:00:00,-0.50,20.0,100\n"
    )
    got = parse_greeks_csv(text)
    assert abs(float(got["iv"].iloc[0]) - 0.20) < 1e-9


def test_coalesce_merges_near_sessions_and_splits_gaps():
    rows = [
        {"symbol": "SPY", "expiration": date(2024, 6, 21), "strike": 530.0, "role": "atm30", "session": date(2024, 5, 20)},
        {"symbol": "SPY", "expiration": date(2024, 6, 21), "strike": 530.0, "role": "atm30", "session": date(2024, 5, 21)},
        {"symbol": "SPY", "expiration": date(2024, 6, 21), "strike": 530.0, "role": "atm30", "session": date(2024, 6, 20)},
    ]
    ranges = coalesce_strike_days(rows, max_gap_days=14)
    assert len(ranges) == 2
    assert ranges[0]["n_sessions"] == 2
    assert ranges[0]["end"] == date(2024, 5, 21)


def test_plan_uses_same_expiry_for_atm_and_put25():
    sessions = [date(2024, 5, 20), date(2024, 5, 21)]
    exps = [date(2024, 6, 21)]
    close = pd.Series([100.0, 100.0], index=pd.to_datetime(sessions))
    ranges = plan_strike_ranges(
        "XLK",
        close,
        exps,
        sessions,
        ["atm30", "put25"],
        iv_by_session={sessions[0]: 0.30, sessions[1]: 0.30},
    )
    roles = {r["role"] for r in ranges}
    assert roles == {"atm30", "put25"}
    atm = next(r for r in ranges if r["role"] == "atm30")
    put = next(r for r in ranges if r["role"] == "put25")
    assert atm["expiration"] == put["expiration"] == date(2024, 6, 21)
    assert atm["strike"] == 100.0
    assert put["strike"] < atm["strike"]


def test_surface_skew_and_prior_session_not_same_day():
    quotes = pd.DataFrame(
        [
            {"symbol": "SPY", "session": date(2024, 5, 17), "role": "atm30", "delta": -0.50, "iv": 0.20, "strike": 100, "expiration": date(2024, 6, 21)},
            {"symbol": "SPY", "session": date(2024, 5, 17), "role": "put25", "delta": -0.25, "iv": 0.28, "strike": 90, "expiration": date(2024, 6, 21)},
            {"symbol": "SPY", "session": date(2024, 5, 20), "role": "atm30", "delta": -0.51, "iv": 0.40, "strike": 101, "expiration": date(2024, 6, 21)},
            {"symbol": "SPY", "session": date(2024, 5, 20), "role": "put25", "delta": -0.24, "iv": 0.41, "strike": 91, "expiration": date(2024, 6, 21)},
            {"symbol": "SPY", "session": date(2024, 5, 17), "role": "atm30", "delta": -0.10, "iv": 0.90, "strike": 80, "expiration": date(2024, 6, 21)},
        ]
    )
    surf = daily_surface(quotes)
    row = surf.loc[surf["session"] == date(2024, 5, 17)].iloc[0]
    assert abs(float(row["atm_iv"]) - 0.20) < 1e-9
    assert abs(float(row["skew"]) - 0.08) < 1e-9
    panel = pd.DataFrame(
        {"spy_skew": [0.08, 0.01]},
        index=pd.to_datetime([date(2024, 5, 17), date(2024, 5, 20)]),
    )
    got = prior_session_values(panel, [date(2024, 5, 20), date(2024, 5, 17)], "spy_skew")
    assert abs(got[0] - 0.08) < 1e-9
    assert pd.isna(got[1])


def test_trailing_percentile_ignores_the_future():
    idx = pd.bdate_range("2020-01-01", periods=200)
    values = pd.Series(1.0, index=idx)
    values.iloc[-1] = 50.0
    pct = trailing_percentile(values, window=50, min_periods=20)
    assert math.isfinite(float(pct.iloc[100]))
    assert float(pct.iloc[-1]) == 1.0
    values2 = values.copy()
    values2.iloc[-1] = -50.0
    pct2 = trailing_percentile(values2, window=50, min_periods=20)
    assert abs(float(pct.iloc[100]) - float(pct2.iloc[100])) < 1e-12
    assert float(pct2.iloc[-1]) < 0.1


def test_dispersion_and_sector_join():
    sessions = _sessions(160)
    rows = []
    for i, day in enumerate(sessions):
        rows.append({"symbol": "SPY", "session": day, "atm_iv": 0.20 + (0.01 if i > 140 else 0.0), "skew": 0.04, "term": 0.0})
        for etf, level in (
            ("XLK", 0.40),
            ("XLF", 0.35),
            ("XLE", 0.36),
            ("XLV", 0.34),
            ("XLI", 0.33),
            ("XLY", 0.38),
        ):
            rows.append({"symbol": etf, "session": day, "atm_iv": level, "skew": 0.05, "term": 0.0})
    panel = market_sector_panel(pd.DataFrame(rows), sessions)
    assert panel["dispersion"].notna().sum() > 0
    assert float(panel["dispersion"].dropna().iloc[-1]) < 0
    trades = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "buy_date": [sessions[150], sessions[10]],
            "gain_pct": [1.0, -2.0],
            "adv_20": [1e6, 1e6],
        }
    )
    sectors = {"AAA": "Technology", "BBB": "Energy"}
    joined = attach_etf_features(trades, panel, sectors)
    assert joined.loc[0, "sector_etf"] == "XLK"
    assert joined.loc[1, "sector_etf"] == "XLE"
    assert pd.notna(joined.loc[0, "sector_iv_pct"])
    assert pd.isna(joined.loc[1, "spy_skew_pct"]) or joined.loc[1, "sector_iv_pct"] != joined.loc[1, "sector_iv_pct"]


def test_quintile_q5_is_the_top_tail():
    sessions = _sessions(40, date(2022, 6, 1))
    later = _sessions(20, date(2024, 6, 3))
    days = sessions + later
    feat = [0.1] * 30 + [0.9] * 10 + [0.5] * 20
    gain = [-1.0] * 30 + [-4.0] * 10 + [3.0] * 20
    # 2022 is the first 40 sessions (June 2022 weekdays may exceed 40 into July)
    trades = pd.DataFrame({"buy_date": days, "gain_pct": gain, "spy_skew_pct": feat})
    table = quintile_table(trades, "spy_skew_pct")
    q5 = table.loc[table["bucket"] == "Q5"].iloc[0]
    q1 = table.loc[table["bucket"] == "Q1"].iloc[0]
    assert int(q5["n"]) == 10
    assert float(q5["expectancy_pct"]) < float(q1["expectancy_pct"])


def test_lock_skips_a_loser_tail_and_rejects_a_winner_clip():
    days = _sessions(40, date(2021, 1, 4)) + _sessions(70, date(2022, 1, 3)) + _sessions(40, date(2024, 1, 2))
    n = len(days)
    feat = [0.10] * n
    gain = [1.0] * n
    for i in range(40, 90):
        feat[i] = 0.95
        gain[i] = -3.0
    for i in range(90, 110):
        feat[i] = 0.20
        gain[i] = 0.4
    trades = pd.DataFrame({"buy_date": days, "gain_pct": gain, "spy_skew_pct": feat, "stock": "X"})
    ev = evaluate_skip(trades, "spy_skew_pct")
    assert ev["separates"] is True
    assert ev["e_2022_kept"] > ev["e_2022_base"]
    locked, _scored = lock_first_rule(
        trades, (("spy_skew_pct", "top", 0.80), ("dispersion_pct", "top", 0.80))
    )
    assert locked == ("spy_skew_pct", "top", 0.80)

    feat2 = [0.10] * n
    gain2 = [-1.0] * n
    for i in range(0, 40):
        feat2[i] = 0.95
        gain2[i] = 8.0
    blunt = pd.DataFrame({"buy_date": days, "gain_pct": gain2, "spy_skew_pct": feat2})
    ev2 = evaluate_skip(blunt, "spy_skew_pct")
    assert ev2["separates"] is False


def test_name_layer_gate_and_name_minus_sector():
    assert etf_layer_wants_names(0, -0.2) is True
    assert etf_layer_wants_names(1, -2.1) is True
    assert etf_layer_wants_names(1, -0.4) is False

    sessions = _sessions(160)
    surface_rows = []
    panel_rows = []
    for i, day in enumerate(sessions):
        surface_rows.append(
            {
                "symbol": "AAA",
                "session": day,
                "atm_iv": 0.40 if i > 140 else 0.15,
                "skew": 0.02,
            }
        )
    panel = market_sector_panel(
        pd.DataFrame(
            [
                {
                    "symbol": etf,
                    "session": day,
                    "atm_iv": 0.18,
                    "skew": 0.04,
                    "term": 0.0,
                }
                for day in sessions
                for etf in ("SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY")
            ]
        ),
        sessions,
    )
    stamped = stamp_sector_atm(pd.DataFrame(surface_rows), panel, {"AAA": "Technology"})
    assert stamped["sector_atm_iv"].notna().all()
    trades = pd.DataFrame(
        {
            "stock": ["AAA", "AAA"],
            "buy_date": [sessions[150], sessions[20]],
            "gain_pct": [1.0, -1.0],
            "adv_20": [30_000_000, 1_000_000],
        }
    )
    named = name_feature_frame(stamped, trades, min_adv=25_000_000)
    assert pd.notna(named.loc[0, "name_iv_pct"])
    assert pd.isna(named.loc[1, "name_iv_pct"])
    assert float(named.loc[0, "name_vs_sector_pct"]) > 0.5
