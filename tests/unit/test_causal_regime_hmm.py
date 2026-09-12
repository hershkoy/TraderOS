"""Causal daily HMM + VIX backwardation skip helpers."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from utils.research.causal_regime_hmm import (
    CORE_FEATURE_COLS,
    attach_prior_session_regime,
    bull_state_index,
    build_regime_features,
    causal_bull_proba,
    fit_gaussian_hmm,
    keep_if_bull,
    keep_unless_backwardation,
    lookup_asof,
    match_states_to_templates,
    vix_backwardation_skip,
    wasserstein2_diag,
)
from utils.research.channel_touch_entry_features import completed_asof


def _two_regime_panel(n_bull: int = 180, n_bear: int = 180, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_bull + n_bear)
    r = np.concatenate(
        [
            rng.normal(0.0012, 0.007, size=n_bull),
            rng.normal(-0.0010, 0.018, size=n_bear),
        ]
    )
    spy = pd.Series(100.0 * np.cumprod(1.0 + r), index=idx)
    vix3m = pd.Series(18.0, index=idx)
    vix = pd.Series(np.where(np.arange(len(idx)) < n_bull, 14.0, 24.0), index=idx)
    return spy, vix, vix3m


def test_vix_backwardation_skip_only_when_curve_inverted():
    idx = pd.bdate_range("2020-01-02", periods=4)
    feat = pd.DataFrame(
        {
            "vix_curve": [0.8, 0.99, 1.01, 1.2],
            "vix": [12.0, 15.0, 16.0, 22.0],
        },
        index=idx,
    )
    skip = vix_backwardation_skip(feat, back_thresh=1.0)
    assert list(skip.astype(int)) == [0, 0, 1, 1]
    skip20 = vix_backwardation_skip(feat, back_thresh=1.0, min_vix=20.0)
    assert list(skip20.astype(int)) == [0, 0, 0, 1]


def test_last_15m_fill_uses_prior_session_asof():
    row = {
        "buy_time": datetime(2022, 3, 15, 15, 45),
        "buy_date": "2022-03-15",
    }
    asof = completed_asof(row, series_is_daily=True)
    assert asof is not None
    assert pd.Timestamp(asof).normalize() <= pd.Timestamp("2022-03-14")


def test_lookup_asof_does_not_see_next_session():
    idx = pd.bdate_range("2022-03-10", periods=5)
    s = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0], index=idx)
    assert lookup_asof(s, pd.Timestamp("2022-03-14")) == 0.6
    assert lookup_asof(s, pd.Timestamp("2022-03-12")) == 0.4


def test_attach_prior_session_does_not_use_fill_day_vix():
    idx = pd.bdate_range("2022-03-14", periods=3)
    features = pd.DataFrame(
        {"vix_curve": [0.9, 1.4, 1.5], "vix": [15.0, 30.0, 31.0]},
        index=idx,
    )
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2022-03-15",
                "buy_time": datetime(2022, 3, 15, 15, 45),
            }
        ]
    )
    out = attach_prior_session_regime(trades, features)
    assert abs(float(out.loc[0, "vix_curve_asof"]) - 0.9) < 1e-9
    keep = keep_unless_backwardation(out, back_thresh=1.0)
    assert bool(keep.iloc[0]) is True


def test_keep_if_bull_warmup_does_not_skip():
    trades = pd.DataFrame({"p_bull_asof": [np.nan, 0.8, 0.2]})
    keep = keep_if_bull(trades, min_p_bull=0.5)
    assert list(keep.astype(int)) == [1, 1, 0]


def test_wasserstein_match_undoes_state_permutation():
    rng = np.random.default_rng(3)
    x = np.vstack(
        [
            rng.normal(loc=[1.0, 0.0], scale=0.2, size=(80, 2)),
            rng.normal(loc=[-1.0, 0.0], scale=0.2, size=(80, 2)),
        ]
    )
    model = fit_gaussian_hmm(x, 2, n_iter=8, rng=rng)
    swapped = type(model)(
        n_states=2,
        mu=model.mu[[1, 0]],
        var=model.var[[1, 0]],
        log_A=model.log_A[np.ix_([1, 0], [1, 0])],
        log_pi=model.log_pi[[1, 0]],
    )
    matched = match_states_to_templates(swapped, model.mu, model.var)
    d00 = wasserstein2_diag(matched.mu[0], matched.var[0], model.mu[0], model.var[0])
    d01 = wasserstein2_diag(matched.mu[0], matched.var[0], model.mu[1], model.var[1])
    assert d00 < d01


def test_causal_hmm_p_bull_ignores_future_row():
    spy, vix, vix3m = _two_regime_panel()
    feat = build_regime_features(spy, vix, vix3m)
    p1 = causal_bull_proba(
        feat,
        n_states=2,
        feature_cols=CORE_FEATURE_COLS,
        min_train=80,
        refit_every=40,
        n_iter=6,
        seed=2,
    )
    feat2 = feat.copy()
    last = feat2.index[-1]
    feat2.loc[last, "spy_logret"] = 5.0
    p2 = causal_bull_proba(
        feat2,
        n_states=2,
        feature_cols=CORE_FEATURE_COLS,
        min_train=80,
        refit_every=40,
        n_iter=6,
        seed=2,
    )
    mid = p1.index[len(p1) // 2]
    assert pd.notna(p1.loc[mid, "p_bull"])
    assert abs(float(p1.loc[mid, "p_bull"]) - float(p2.loc[mid, "p_bull"])) < 1e-12


def test_causal_hmm_drops_p_bull_in_bear_half():
    spy, vix, vix3m = _two_regime_panel()
    feat = build_regime_features(spy, vix, vix3m)
    out = causal_bull_proba(
        feat,
        n_states=2,
        feature_cols=CORE_FEATURE_COLS,
        min_train=80,
        refit_every=30,
        n_iter=8,
        seed=4,
    )
    n = len(out)
    bull = out["p_bull"].iloc[int(n * 0.15) : int(n * 0.40)].mean()
    bear = out["p_bull"].iloc[int(n * 0.70) :].mean()
    assert pd.notna(bull) and pd.notna(bear)
    assert float(bull) > float(bear)


def test_bull_state_is_higher_spy_drift():
    mu = np.array([[0.002, 0.0], [-0.001, 0.0]])
    assert bull_state_index(mu) == 0
    mu2 = np.array([[-0.002, 0.0], [0.001, 0.0]])
    assert bull_state_index(mu2) == 1
