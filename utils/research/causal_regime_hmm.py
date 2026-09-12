"""Causal daily Gaussian HMM + VIX term-structure skip signals.

Last-15m fills are intra-session (15:45 ET). Daily SPY/VIX closes for that
calendar day are not known yet, so every join uses ``completed_asof``
(prior session). Expanding HMM fits never see t+1.

State identity: after each refit, match new Gaussian components to the first
fit's templates by 2-Wasserstein distance (Hungarian). Bull = template with
the highest SPY-log-return mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from utils.research.channel_touch_entry_features import completed_asof
from utils.research.vix_curve import align_series, curve_ratio

CORE_FEATURE_COLS: Tuple[str, ...] = (
    "spy_logret",
    "spy_vol_60",
    "spy_mean_20",
    "vix_logchg",
    "vix_curve",
)
MACRO_FEATURE_COLS: Tuple[str, ...] = (
    "tlt_logret",
    "gld_logret",
    "uup_logret",
)
VOL_WIN = 60
MEAN_WIN = 20
VAR_FLOOR = 1e-6
Z_CLIP = 8.0
STICKY = 0.90


@dataclass
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM. State 0 is not yet the bull label."""

    n_states: int
    mu: np.ndarray
    var: np.ndarray
    log_A: np.ndarray
    log_pi: np.ndarray


def _as_naive_close(s: pd.Series, name: str) -> pd.Series:
    x = s.astype(float).copy()
    x.index = pd.DatetimeIndex(x.index)
    if x.index.tz is not None:
        x.index = x.index.tz_convert(None)
    x.index = pd.DatetimeIndex(x.index).normalize()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    x.name = name
    return x


def build_regime_features(
    spy: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    macros: Optional[Mapping[str, pd.Series]] = None,
) -> pd.DataFrame:
    """Daily feature frame aligned on SPY ∩ VIX ∩ VIX3M. Macros optional."""
    spy_c = _as_naive_close(spy, "spy")
    vx = _as_naive_close(vix, "vix")
    vx3 = _as_naive_close(vix3m, "vix3m")
    spy_c, vx, vx3 = align_series(spy_c, vx, vx3)
    log_spy = np.log(spy_c.replace(0.0, np.nan))
    spy_logret = log_spy.diff()
    out = pd.DataFrame(
        {
            "spy_logret": spy_logret,
            "spy_vol_60": spy_logret.rolling(VOL_WIN, min_periods=20).std(),
            "spy_mean_20": spy_logret.rolling(MEAN_WIN, min_periods=10).mean(),
            "vix_logchg": np.log(vx.replace(0.0, np.nan)).diff(),
            "vix_curve": curve_ratio(vx, vx3),
            "vix": vx,
        },
        index=spy_c.index,
    )
    macros = macros or {}
    for col, key in (
        ("tlt_logret", "TLT"),
        ("gld_logret", "GLD"),
        ("uup_logret", "UUP"),
    ):
        raw = macros.get(key)
        if raw is None or raw.empty:
            out[col] = np.nan
            continue
        m = _as_naive_close(raw, key)
        m = m.reindex(out.index)
        out[col] = np.log(m.replace(0.0, np.nan)).diff()
    return out


def vix_backwardation_skip(
    features: pd.DataFrame,
    *,
    back_thresh: float = 1.0,
    min_vix: Optional[float] = None,
) -> pd.Series:
    """True = skip (risk-off). Prior-session curve; missing curve does not skip."""
    curve = pd.to_numeric(features["vix_curve"], errors="coerce")
    skip = curve >= float(back_thresh)
    if min_vix is not None:
        vx = pd.to_numeric(features["vix"], errors="coerce")
        skip = skip & (vx >= float(min_vix))
    return skip.fillna(False)


def _log_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """x (T,D) -> log p(x|k) for each state (T,K)."""
    # (T, 1, D) vs (K, D)
    diff = x[:, None, :] - mu[None, :, :]
    inv = 1.0 / var[None, :, :]
    quad = np.sum(diff * diff * inv, axis=2)
    log_det = np.sum(np.log(2.0 * np.pi * var), axis=1)
    return -0.5 * (quad + log_det[None, :])


def _forward_backward(
    log_emit: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_len, k = log_emit.shape
    log_alpha = np.empty((t_len, k), dtype=float)
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, t_len):
        log_alpha[t] = log_emit[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    log_beta = np.zeros((t_len, k), dtype=float)
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + log_emit[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
    return log_gamma, log_alpha


def _init_hmm(x: np.ndarray, n_states: int, rng: np.random.Generator) -> GaussianHMM:
    del rng
    t_len, dim = x.shape
    order = np.argsort(x[:, 0])
    mu = np.zeros((n_states, dim), dtype=float)
    var = np.zeros((n_states, dim), dtype=float)
    cuts = np.array_split(order, n_states)
    for k, idx in enumerate(cuts):
        sl = x[idx] if len(idx) else x
        mu[k] = sl.mean(axis=0)
        var[k] = sl.var(axis=0) + VAR_FLOOR
        if not np.all(np.isfinite(var[k])):
            var[k] = np.maximum(x.var(axis=0), VAR_FLOOR)
    a = np.full((n_states, n_states), (1.0 - STICKY) / max(n_states - 1, 1))
    np.fill_diagonal(a, STICKY)
    a = a / a.sum(axis=1, keepdims=True)
    pi = np.ones(n_states) / n_states
    return GaussianHMM(
        n_states=n_states,
        mu=mu,
        var=np.maximum(var, VAR_FLOOR),
        log_A=np.log(a),
        log_pi=np.log(pi),
    )


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int,
    *,
    n_iter: int = 15,
    rng: Optional[np.random.Generator] = None,
) -> GaussianHMM:
    """Baum-Welch on z-scored (T, D). Drops non-finite rows."""
    rng = rng or np.random.default_rng(0)
    ok = np.isfinite(x).all(axis=1)
    xw = x[ok]
    if xw.shape[0] < max(20, n_states * 10):
        raise ValueError("too few finite rows to fit HMM")
    model = _init_hmm(xw, n_states, rng)
    t_len = xw.shape[0]
    for _ in range(int(n_iter)):
        log_emit = _log_gaussian(xw, model.mu, model.var)
        log_gamma, _ = _forward_backward(log_emit, model.log_A, model.log_pi)
        gamma = np.exp(log_gamma)
        nk = gamma.sum(axis=0) + 1e-12
        mu = (gamma.T @ xw) / nk[:, None]
        var = np.empty_like(mu)
        for k in range(n_states):
            diff = xw - mu[k]
            var[k] = (gamma[:, k][:, None] * diff * diff).sum(axis=0) / nk[k]
        var = np.maximum(var, VAR_FLOOR)
        if t_len < 2:
            log_a = model.log_A
        else:
            pair = (
                log_gamma[:-1, :, None]
                + model.log_A[None, :, :]
                + log_emit[1:, None, :]
            )
            pair = pair - logsumexp(pair.reshape(t_len - 1, -1), axis=1)[:, None, None]
            log_xi_num = logsumexp(pair, axis=0)
            log_a = log_xi_num - logsumexp(log_xi_num, axis=1)[:, None]
        pi = gamma[0] / gamma[0].sum()
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(var)):
            break
        model = GaussianHMM(
            n_states=n_states,
            mu=mu,
            var=var,
            log_A=log_a,
            log_pi=np.log(np.maximum(pi, 1e-12)),
        )
    return model


def wasserstein2_diag(mu_a: np.ndarray, var_a: np.ndarray, mu_b: np.ndarray, var_b: np.ndarray) -> float:
    """Squared 2-Wasserstein between two diagonal Gaussians."""
    dmu = mu_a - mu_b
    ds = np.sqrt(np.maximum(var_a, 0.0)) - np.sqrt(np.maximum(var_b, 0.0))
    return float(np.sum(dmu * dmu + ds * ds))


def match_states_to_templates(
    model: GaussianHMM,
    templates_mu: np.ndarray,
    templates_var: np.ndarray,
) -> GaussianHMM:
    """Permute model states to templates (Hungarian on W2)."""
    k = model.n_states
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost[i, j] = wasserstein2_diag(
                model.mu[i], model.var[i], templates_mu[j], templates_var[j]
            )
    _row, col = linear_sum_assignment(cost)
    # row i assigned to template col[i]; we want new_state[template] = old
    inv = np.empty(k, dtype=int)
    for old, tmpl in enumerate(col):
        inv[tmpl] = old
    return GaussianHMM(
        n_states=k,
        mu=model.mu[inv],
        var=model.var[inv],
        log_A=model.log_A[np.ix_(inv, inv)],
        log_pi=model.log_pi[inv],
    )


def bull_state_index(mu: np.ndarray) -> int:
    """Bull = highest mean SPY log-return (column 0)."""
    return int(np.argmax(mu[:, 0]))


def _zscore(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (x - mean) / std
    z = np.clip(z, -Z_CLIP, Z_CLIP)
    z = np.where(np.isfinite(z), z, 0.0)
    return z, mean, std


def causal_bull_proba(
    features: pd.DataFrame,
    *,
    n_states: int = 2,
    feature_cols: Sequence[str] = CORE_FEATURE_COLS,
    min_train: int = 252,
    refit_every: int = 21,
    n_iter: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """Expanding HMM. ``p_bull[t]`` uses features through t only.

    Rows with any NaN in ``feature_cols`` keep p_bull null (caller must not skip).
    """
    cols = list(feature_cols)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError("missing feature columns: %s" % missing)
    idx = features.index
    x_raw = features[cols].to_numpy(dtype=float)
    t_len = len(idx)
    p_bull = np.full(t_len, np.nan, dtype=float)
    state = np.full(t_len, np.nan, dtype=float)
    rng = np.random.default_rng(int(seed))
    model: Optional[GaussianHMM] = None
    templates_mu: Optional[np.ndarray] = None
    templates_var: Optional[np.ndarray] = None
    mean = std = None
    log_alpha_prev: Optional[np.ndarray] = None
    last_fit = -10**9
    finite_rows = 0
    for t in range(t_len):
        if not np.isfinite(x_raw[t]).all():
            log_alpha_prev = None
            continue
        finite_rows += 1
        if finite_rows < int(min_train):
            continue
        need_fit = model is None or (t - last_fit) >= int(refit_every)
        if need_fit:
            window_ok = np.isfinite(x_raw[: t + 1]).all(axis=1)
            xw = x_raw[: t + 1][window_ok]
            try:
                z, mean, std = _zscore(xw)
                fitted = fit_gaussian_hmm(z, int(n_states), n_iter=n_iter, rng=rng)
            except ValueError:
                continue
            if templates_mu is None:
                bull = bull_state_index(fitted.mu)
                order = np.argsort(-fitted.mu[:, 0])
                # Put bull first so template 0 is always bull.
                if bull != 0:
                    fitted = GaussianHMM(
                        n_states=fitted.n_states,
                        mu=fitted.mu[order],
                        var=fitted.var[order],
                        log_A=fitted.log_A[np.ix_(order, order)],
                        log_pi=fitted.log_pi[order],
                    )
                templates_mu = fitted.mu.copy()
                templates_var = fitted.var.copy()
                model = fitted
            else:
                model = match_states_to_templates(fitted, templates_mu, templates_var)
            last_fit = t
            z_t = np.clip((x_raw[t] - mean) / std, -Z_CLIP, Z_CLIP)
            log_emit = _log_gaussian(z_t[None, :], model.mu, model.var)[0]
            if log_alpha_prev is None:
                log_alpha = model.log_pi + log_emit
            else:
                log_alpha = log_emit + logsumexp(log_alpha_prev[:, None] + model.log_A, axis=0)
            log_alpha_prev = log_alpha
            gamma = np.exp(log_alpha - logsumexp(log_alpha))
            p_bull[t] = float(gamma[0])
            state[t] = float(np.argmax(gamma))
            continue
        z_t = np.clip((x_raw[t] - mean) / std, -Z_CLIP, Z_CLIP)
        log_emit = _log_gaussian(z_t[None, :], model.mu, model.var)[0]
        if log_alpha_prev is None:
            log_alpha = model.log_pi + log_emit
        else:
            log_alpha = log_emit + logsumexp(log_alpha_prev[:, None] + model.log_A, axis=0)
        log_alpha_prev = log_alpha
        gamma = np.exp(log_alpha - logsumexp(log_alpha))
        p_bull[t] = float(gamma[0])
        state[t] = float(np.argmax(gamma))
    return pd.DataFrame({"p_bull": p_bull, "state": state}, index=idx)


def lookup_asof(series: pd.Series, asof: pd.Timestamp) -> Optional[float]:
    """Last finite value at or before ``asof`` (weekend asof uses Friday)."""
    if asof is None or series is None or series.empty:
        return None
    s = series.sort_index()
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    ts = ts.normalize() if ts.hour == 0 and ts.minute == 0 and ts.second == 0 else ts
    hist = s.loc[:ts].dropna()
    if hist.empty:
        return None
    val = float(hist.iloc[-1])
    if not np.isfinite(val):
        return None
    return val


def attach_prior_session_regime(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    p_bull: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Join prior-session VIX curve / VIX / optional p_bull onto each fill."""
    out = trades.copy()
    curve = pd.to_numeric(features["vix_curve"], errors="coerce") if "vix_curve" in features.columns else None
    vx = pd.to_numeric(features["vix"], errors="coerce") if "vix" in features.columns else None
    asofs = []
    curves = []
    vixes = []
    probs = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=True)
        asofs.append(asof)
        curves.append(None if curve is None or asof is None else lookup_asof(curve, asof))
        vixes.append(None if vx is None or asof is None else lookup_asof(vx, asof))
        probs.append(None if p_bull is None or asof is None else lookup_asof(p_bull, asof))
    out["regime_asof"] = asofs
    out["vix_curve_asof"] = curves
    out["vix_asof"] = vixes
    if p_bull is not None:
        out["p_bull_asof"] = probs
    return out


def keep_unless_backwardation(
    trades: pd.DataFrame,
    *,
    back_thresh: float = 1.0,
    min_vix: Optional[float] = None,
) -> pd.Series:
    """Keep when curve missing or in contango. Skip backwardation (and min VIX)."""
    curve = pd.to_numeric(trades["vix_curve_asof"], errors="coerce")
    skip = curve >= float(back_thresh)
    if min_vix is not None:
        vx = pd.to_numeric(trades["vix_asof"], errors="coerce")
        skip = skip & (vx >= float(min_vix))
    skip = skip.fillna(False)
    return ~skip


def keep_if_bull(
    trades: pd.DataFrame,
    *,
    min_p_bull: float,
) -> pd.Series:
    """Keep when p_bull missing (warmup) or p_bull >= threshold."""
    p = pd.to_numeric(trades["p_bull_asof"], errors="coerce")
    return p.isna() | (p >= float(min_p_bull))
