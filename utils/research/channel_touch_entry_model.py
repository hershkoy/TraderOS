"""Time-split entry scorer for channel-touch trades (no future features).

Train only on trades whose buy_date is before the test cutoff. Features are
the point-in-time entry snapshot. Labels are that trade's eventual net gain
(known only after the trade ends; never used as an input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MODEL_FEATURES: Sequence[str] = (
    "rsi_14",
    "dist_sma50_pct",
    "channel_pos",
    "atr_pct",
    "volume_rel_20",
    "squeeze_mom",
    "squeeze_mom_rising",
    "max_beyond_width",
    "rs_spy_126d",
    "rs_spy_21d",
    "spy_ret_20d",
    "spy_above_sma50",
    "close_loc",
    "room_to_resist_pct",
    "channel_width_pct",
    "range_pct",
)

# Same-bar OHLC of the fill bar is not known at a wick limit fill.
# After lagged snapshots (feature_asof=prior-bar), close_loc/RSI/%B on the
# trade CSV are the previous completed bar and may be used in MODEL_FEATURES.
NO_SAMEBAR_CLOSE_FEATURES: Sequence[str] = (
    "channel_pos",
    "atr_pct",
    "volume_rel_20",
    "squeeze_mom",
    "squeeze_mom_rising",
    "max_beyond_width",
    "rs_spy_126d",
    "rs_spy_21d",
    "spy_ret_20d",
    "spy_above_sma50",
    "room_to_resist_pct",
    "channel_width_pct",
)

DEFAULT_CUTOFF = "2023-01-01"
WALK_FOLDS: Sequence[Tuple[str, str, str]] = (
    ("train<=2021 test=2022", "2022-01-01", "2022-12-31"),
    ("train<=2022 test=2023", "2023-01-01", "2023-12-31"),
    ("train<=2023 test=2024-25", "2024-01-01", "2025-12-31"),
)


def _gain_col(df: pd.DataFrame) -> str:
    if "gain_pct_net" in df.columns:
        return "gain_pct_net"
    if "gain_pct" in df.columns:
        return "gain_pct"
    raise ValueError("need gain_pct_net or gain_pct")


def time_split(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if "buy_time" in work.columns:
        bt = pd.to_datetime(work["buy_time"], errors="coerce")
        bd = pd.to_datetime(work["buy_date"])
        ts = bt.fillna(bd)
    else:
        ts = pd.to_datetime(work["buy_date"])
    cut = pd.Timestamp(cutoff)
    return work.loc[ts < cut].copy(), work.loc[ts >= cut].copy()


def feature_matrix(
    df: pd.DataFrame,
    cols: Sequence[str] = MODEL_FEATURES,
    *,
    medians: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (n, f) with train medians filled; also the medians used."""
    blocks = []
    used_med = []
    for i, c in enumerate(cols):
        if c not in df.columns:
            s = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            s = pd.to_numeric(df[c], errors="coerce")
        arr = s.to_numpy(dtype=float)
        if medians is None:
            finite = arr[np.isfinite(arr)]
            med = float(np.median(finite)) if len(finite) else 0.0
        else:
            med = float(medians[i])
        used_med.append(med)
        arr = np.where(np.isfinite(arr), arr, med)
        blocks.append(arr)
    x = np.column_stack(blocks) if blocks else np.zeros((len(df), 0))
    return x, np.asarray(used_med, dtype=float)


def _standardize(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd, mu, sd


def trade_metrics(gains: np.ndarray) -> dict:
    g = np.asarray(gains, dtype=float)
    g = g[np.isfinite(g)]
    n = int(len(g))
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "median_pct": None,
        }
    wins = g[g > 0]
    losses = g[g <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float((-losses).sum()) if len(losses) else 0.0
    if gl > 1e-12:
        pf = gp / gl
    elif gp > 0:
        pf = float("inf")
    else:
        pf = 0.0
    return {
        "n_trades": n,
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "expectancy_pct": round(float(g.mean()), 4),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 4),
        "median_pct": round(float(np.median(g)), 4),
    }


class LogisticScorer:
    """L2 logistic regression via gradient descent. Fit on past trades only."""

    def __init__(self, l2: float = 0.5, lr: float = 0.05, epochs: int = 400):
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "LogisticScorer":
        n, f = x.shape
        w = np.zeros(f, dtype=float)
        b = 0.0
        sw = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw = np.clip(sw, 0.05, 20.0)
        sw = sw / (sw.mean() if sw.mean() > 0 else 1.0)
        for _ in range(self.epochs):
            z = x @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            err = (p - y) * sw
            w -= self.lr * ((x.T @ err) / n + self.l2 * w)
            b -= self.lr * float(err.mean())
        self.w = w
        self.b = b
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("not fit")
        z = x @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class TinyMLPScorer:
    """One-hidden-layer ReLU MLP. L2 + inner time-val early stop. No future rows."""

    def __init__(
        self,
        hidden: int = 8,
        l2: float = 0.2,
        lr: float = 0.03,
        epochs: int = 250,
        seed: int = 7,
    ):
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.w1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.w2: Optional[np.ndarray] = None
        self.b2: float = 0.0

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return h, p

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TinyMLPScorer":
        rng = np.random.default_rng(self.seed)
        n, f = x.shape
        h = self.hidden
        self.w1 = rng.normal(0.0, 0.2, size=(f, h))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0.0, 0.2, size=(h,))
        self.b2 = 0.0
        sw = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw = np.clip(sw, 0.05, 20.0)
        sw = sw / (sw.mean() if sw.mean() > 0 else 1.0)
        best_w1 = self.w1.copy()
        best_b1 = self.b1.copy()
        best_w2 = self.w2.copy()
        best_b2 = self.b2
        best_loss = 1e18
        patience = 0
        for _ in range(self.epochs):
            hid, p = self._forward(x)
            err = (p - y) * sw
            d_w2 = (hid.T @ err) / n + self.l2 * self.w2
            d_b2 = float(err.mean())
            relu_mask = (hid > 0).astype(float)
            d_h = np.outer(err, self.w2) * relu_mask
            d_w1 = (x.T @ d_h) / n + self.l2 * self.w1
            d_b1 = d_h.mean(axis=0)
            self.w1 -= self.lr * d_w1
            self.b1 -= self.lr * d_b1
            self.w2 -= self.lr * d_w2
            self.b2 -= self.lr * d_b2
            if x_val is not None and y_val is not None and len(x_val):
                _, pv = self._forward(x_val)
                pv = np.clip(pv, 1e-6, 1.0 - 1e-6)
                loss = float(-(y_val * np.log(pv) + (1.0 - y_val) * np.log(1.0 - pv)).mean())
                if loss < best_loss - 1e-5:
                    best_loss = loss
                    best_w1 = self.w1.copy()
                    best_b1 = self.b1.copy()
                    best_w2 = self.w2.copy()
                    best_b2 = self.b2
                    patience = 0
                else:
                    patience += 1
                    if patience >= 40:
                        break
        if x_val is not None:
            self.w1, self.b1, self.w2, self.b2 = best_w1, best_b1, best_w2, best_b2
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.w1 is None:
            raise RuntimeError("not fit")
        _, p = self._forward(x)
        return p


@dataclass
class ScoredSplit:
    name: str
    cutoff: str
    threshold: float
    train: dict
    train_kept: dict
    test: dict
    test_kept: dict
    n_features: int


def _choose_threshold(
    scores: np.ndarray,
    gains: np.ndarray,
    *,
    min_frac: float = 0.35,
    min_n: int = 40,
) -> float:
    """Max PF on this (inner) set; never looks at the outer test set."""
    best_thr = float(np.quantile(scores, 0.15)) if len(scores) else 0.5
    best_pf = -1.0
    floor = max(int(min_n), int(len(scores) * float(min_frac)))
    for q in np.linspace(0.10, 0.70, 13):
        thr = float(np.quantile(scores, q))
        kept = gains[scores >= thr]
        if len(kept) < floor:
            continue
        m = trade_metrics(kept)
        pf = m["profit_factor"]
        if pf is None:
            continue
        if pf > best_pf:
            best_pf = float(pf)
            best_thr = thr
    return best_thr


def fit_and_eval(
    df: pd.DataFrame,
    *,
    cutoff: str = DEFAULT_CUTOFF,
    model_kind: str = "mlp",
    min_frac: float = 0.35,
    feature_cols: Sequence[str] = MODEL_FEATURES,
) -> ScoredSplit:
    train_df, test_df = time_split(df, cutoff)
    gain_col = _gain_col(df)
    x_tr_raw, med = feature_matrix(train_df, feature_cols)
    x_te_raw, _ = feature_matrix(test_df, feature_cols, medians=med)
    x_tr, x_te, _, _ = _standardize(x_tr_raw, x_te_raw)
    y_tr = (pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float) > 0).astype(float)
    g_tr = pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float)
    g_te = pd.to_numeric(test_df[gain_col], errors="coerce").to_numpy(dtype=float)
    w_tr = np.clip(np.abs(g_tr), 0.1, 15.0)
    n_tr = len(x_tr)
    val_n = max(20, int(n_tr * 0.2)) if n_tr >= 80 else 0
    if val_n:
        x_fit, x_val = x_tr[:-val_n], x_tr[-val_n:]
        y_fit, y_val = y_tr[:-val_n], y_tr[-val_n:]
        w_fit = w_tr[:-val_n]
        g_val = g_tr[-val_n:]
    else:
        x_fit, y_fit, w_fit = x_tr, y_tr, w_tr
        x_val = y_val = g_val = None

    kind = (model_kind or "mlp").lower()
    if kind == "logistic":
        model = LogisticScorer()
        model.fit(x_fit, y_fit, sample_weight=w_fit)
    else:
        model = TinyMLPScorer()
        model.fit(x_fit, y_fit, x_val=x_val, y_val=y_val, sample_weight=w_fit)

    s_tr = model.score(x_tr)
    s_te = model.score(x_te) if len(x_te) else np.array([])
    if x_val is not None:
        thr = _choose_threshold(model.score(x_val), g_val, min_frac=min_frac, min_n=20)
    else:
        thr = _choose_threshold(s_tr, g_tr, min_frac=min_frac, min_n=20)
    return ScoredSplit(
        name=kind,
        cutoff=str(cutoff),
        threshold=float(thr),
        train=trade_metrics(g_tr),
        train_kept=trade_metrics(g_tr[s_tr >= thr]),
        test=trade_metrics(g_te) if len(g_te) else trade_metrics(np.array([])),
        test_kept=trade_metrics(g_te[s_te >= thr]) if len(g_te) else trade_metrics(np.array([])),
        n_features=int(x_tr.shape[1]),
    )


def walk_forward(
    df: pd.DataFrame,
    *,
    model_kind: str = "mlp",
    feature_cols: Sequence[str] = MODEL_FEATURES,
) -> List[ScoredSplit]:
    rows: List[ScoredSplit] = []
    bd = pd.to_datetime(df["buy_date"])
    for name, test_start, test_end in WALK_FOLDS:
        pool = df.loc[bd <= pd.Timestamp(test_end)].copy()
        if pool.empty:
            continue
        split = fit_and_eval(
            pool, cutoff=test_start, model_kind=model_kind, feature_cols=feature_cols
        )
        split.name = f"{model_kind} {name}"
        rows.append(split)
    return rows


def attach_scores(
    df: pd.DataFrame,
    *,
    cutoff: str = DEFAULT_CUTOFF,
    model_kind: str = "mlp",
    min_frac: float = 0.35,
    feature_cols: Sequence[str] = MODEL_FEATURES,
) -> Tuple[pd.DataFrame, float]:
    """Fit on buy_date < cutoff only; score every row. Threshold from train tail."""
    train_df, _ = time_split(df, cutoff)
    gain_col = _gain_col(df)
    x_tr_raw, med = feature_matrix(train_df, feature_cols)
    x_all_raw, _ = feature_matrix(df, feature_cols, medians=med)
    x_tr, x_all, _, _ = _standardize(x_tr_raw, x_all_raw)
    y_tr = (pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float) > 0).astype(float)
    g_tr = pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float)
    w_tr = np.clip(np.abs(g_tr), 0.1, 15.0)
    n_tr = len(x_tr)
    val_n = max(20, int(n_tr * 0.2)) if n_tr >= 80 else 0
    if val_n:
        x_fit, x_val = x_tr[:-val_n], x_tr[-val_n:]
        y_fit, y_val = y_tr[:-val_n], y_tr[-val_n:]
        w_fit = w_tr[:-val_n]
        g_val = g_tr[-val_n:]
    else:
        x_fit, y_fit, w_fit = x_tr, y_tr, w_tr
        x_val = y_val = g_val = None
    kind = (model_kind or "mlp").lower()
    if kind == "logistic":
        model = LogisticScorer()
        model.fit(x_fit, y_fit, sample_weight=w_fit)
    else:
        model = TinyMLPScorer()
        model.fit(x_fit, y_fit, x_val=x_val, y_val=y_val, sample_weight=w_fit)
    scores = model.score(x_all)
    if x_val is not None:
        thr = _choose_threshold(model.score(x_val), g_val, min_frac=min_frac, min_n=20)
    else:
        thr = _choose_threshold(model.score(x_tr), g_tr, min_frac=min_frac, min_n=20)
    out = df.copy()
    out["entry_score"] = scores
    out["entry_pass"] = scores >= thr
    return out, float(thr)


def scored_to_row(s: ScoredSplit) -> Dict[str, object]:
    def _flat(prefix: str, d: dict) -> dict:
        return {f"{prefix}_{k}": v for k, v in d.items()}

    return {
        "name": s.name,
        "cutoff": s.cutoff,
        "threshold": round(s.threshold, 4),
        "n_features": s.n_features,
        **_flat("train", s.train),
        **_flat("train_kept", s.train_kept),
        **_flat("test", s.test),
        **_flat("test_kept", s.test_kept),
    }


# ---------------------------------------------------------------------------
# Ridge P&L regression + confidence sizing (not a hard entry gate)
# ---------------------------------------------------------------------------

RIDGE_FEATURES_1D: Sequence[str] = NO_SAMEBAR_CLOSE_FEATURES
RIDGE_FEATURES_15M: Sequence[str] = MODEL_FEATURES

DEFAULT_EMBARGO_DAYS = 21
DEFAULT_RIDGE_L2 = 1.0
SIZE_MIN = 0.25
SIZE_MAX = 2.0
MIN_TRAIN_N = 80
MIN_TEST_N = 30


def resolve_ridge_features(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> List[str]:
    """Keep columns that exist; drop known clone pairs (keep the first of each)."""
    present = [c for c in cols if c in df.columns]
    if "channel_pos" in present and "room_to_resist_pct" in present:
        present = [c for c in present if c != "room_to_resist_pct"]
    if "channel_span_days" in present and "channel_age_at_buy_days" in present:
        present = [c for c in present if c != "channel_age_at_buy_days"]
    return present


def buy_timestamps(df: pd.DataFrame) -> pd.Series:
    if "buy_time" in df.columns:
        bt = pd.to_datetime(df["buy_time"], errors="coerce")
        bd = pd.to_datetime(df["buy_date"], errors="coerce")
        return bt.fillna(bd)
    return pd.to_datetime(df["buy_date"], errors="coerce")


def sell_timestamps(df: pd.DataFrame, *, fallback_days: int = 30) -> pd.Series:
    if "sell_time" in df.columns:
        st = pd.to_datetime(df["sell_time"], errors="coerce")
        sd = (
            pd.to_datetime(df["sell_date"], errors="coerce")
            if "sell_date" in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        ts = st.fillna(sd)
    elif "sell_date" in df.columns:
        ts = pd.to_datetime(df["sell_date"], errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    buy = buy_timestamps(df)
    out = ts.copy()
    missing = out.isna()
    out.loc[missing] = buy.loc[missing] + pd.Timedelta(days=int(fallback_days))
    return out


def winsorize_y(
    y: np.ndarray,
    *,
    lo_q: float = 0.01,
    hi_q: float = 0.99,
    abs_clip: float = 15.0,
) -> np.ndarray:
    """Clip train labels. Evaluation always uses uncapped realized P&L."""
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr)
    lo = float(np.quantile(finite, lo_q))
    hi = float(np.quantile(finite, hi_q))
    lo = max(lo, -float(abs_clip))
    hi = min(hi, float(abs_clip))
    if hi < lo:
        hi = lo
    return np.clip(arr, lo, hi)


class RidgePnlScorer:
    """Closed-form ridge with intercept. Features must already be z-scored on train."""

    def __init__(self, l2: float = DEFAULT_RIDGE_L2):
        self.l2 = float(l2)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePnlScorer":
        n, f = x.shape
        if n == 0 or f == 0:
            self.w = np.zeros(f, dtype=float)
            self.b = 0.0
            return self
        xx = np.column_stack([np.ones(n), x])
        reg = np.eye(f + 1) * self.l2
        reg[0, 0] = 0.0
        xtx = xx.T @ xx + reg
        xty = xx.T @ np.asarray(y, dtype=float)
        try:
            coef = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
        self.b = float(coef[0])
        self.w = np.asarray(coef[1:], dtype=float)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("not fit")
        if x.size == 0:
            return np.zeros(0, dtype=float)
        return x @ self.w + self.b


@dataclass
class SizeMap:
    """Train-only predicted-P&L percentiles mapped to size_min..size_max."""

    pred_lo: float
    pred_hi: float
    size_min: float = SIZE_MIN
    size_max: float = SIZE_MAX


def fit_size_map(
    pred_train: np.ndarray,
    *,
    size_min: float = SIZE_MIN,
    size_max: float = SIZE_MAX,
    lo_q: float = 0.10,
    hi_q: float = 0.90,
) -> SizeMap:
    p = np.asarray(pred_train, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return SizeMap(pred_lo=0.0, pred_hi=1.0, size_min=size_min, size_max=size_max)
    lo = float(np.quantile(p, lo_q))
    hi = float(np.quantile(p, hi_q))
    if hi <= lo:
        hi = lo + 1e-6
    return SizeMap(pred_lo=lo, pred_hi=hi, size_min=float(size_min), size_max=float(size_max))


def apply_size_map(
    pred: np.ndarray,
    smap: SizeMap,
    *,
    skip_negative: bool = False,
) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    span = smap.pred_hi - smap.pred_lo
    t = np.clip((p - smap.pred_lo) / span, 0.0, 1.0)
    size = smap.size_min + t * (smap.size_max - smap.size_min)
    size = np.where(np.isfinite(p), size, smap.size_min)
    if skip_negative:
        size = np.where(p < 0.0, 0.0, size)
    return size.astype(float)


def sized_trade_metrics(gains: np.ndarray, sizes: np.ndarray) -> dict:
    """P&L = size * gain. Zeros (skipped) are not wins or losses."""
    g = np.asarray(gains, dtype=float)
    s = np.asarray(sizes, dtype=float)
    n = int(len(g))
    if n == 0:
        return {
            "n_trades": 0,
            "n_active": 0,
            "win_rate_pct": None,
            "expectancy": None,
            "expectancy_active": None,
            "profit_factor": None,
            "median": None,
            "sum_pnl": None,
            "max_dd": None,
            "mean_size": None,
        }
    pnl = s * g
    finite = np.isfinite(pnl)
    pnl_f = pnl[finite]
    active = (s > 1e-12) & finite
    wins = pnl_f[pnl_f > 0]
    losses = pnl_f[pnl_f < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float((-losses).sum()) if len(losses) else 0.0
    if gl > 1e-12:
        pf = gp / gl
    elif gp > 0:
        pf = float("inf")
    else:
        pf = 0.0
    n_act = int(active.sum())
    wr = float((pnl_f[s[finite] > 1e-12] > 0).mean() * 100.0) if n_act else None
    eq = np.cumsum(np.where(finite, pnl, 0.0))
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    return {
        "n_trades": n,
        "n_active": n_act,
        "win_rate_pct": None if wr is None else round(wr, 2),
        "expectancy": round(float(np.nanmean(pnl)), 4),
        "expectancy_active": None if n_act == 0 else round(float(np.nanmean(pnl[active])), 4),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 4),
        "median": round(float(np.nanmedian(pnl)), 4),
        "sum_pnl": round(float(np.nansum(pnl)), 4),
        "max_dd": round(max_dd, 4),
        "mean_size": round(float(np.nanmean(s)), 4),
    }


def drop_top_n_sized_metrics(gains: np.ndarray, sizes: np.ndarray, n: int = 3) -> dict:
    g = np.asarray(gains, dtype=float)
    s = np.asarray(sizes, dtype=float)
    pnl = s * g
    if len(pnl) <= n:
        return sized_trade_metrics(np.array([]), np.array([]))
    order = np.argsort(pnl)[::-1]
    keep = np.ones(len(pnl), dtype=bool)
    keep[order[:n]] = False
    return sized_trade_metrics(g[keep], s[keep])


def spearman_pred_actual(pred: np.ndarray, actual: np.ndarray) -> Optional[float]:
    p = pd.Series(np.asarray(pred, dtype=float))
    a = pd.Series(np.asarray(actual, dtype=float))
    mask = p.notna() & a.notna() & np.isfinite(p) & np.isfinite(a)
    if int(mask.sum()) < 20:
        return None
    rho = float(p[mask].corr(a[mask], method="spearman"))
    if not np.isfinite(rho):
        return None
    return round(rho, 4)


def pred_quintile_table(pred: np.ndarray, actual: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "pred": np.asarray(pred, dtype=float),
            "gain": np.asarray(actual, dtype=float),
        }
    )
    work = work.loc[np.isfinite(work["pred"]) & np.isfinite(work["gain"])]
    if work.empty:
        return pd.DataFrame()
    try:
        work["bucket"] = pd.qcut(work["pred"], q=int(n_bins), duplicates="drop")
    except (ValueError, TypeError):
        return pd.DataFrame()
    rows = []
    for key, part in work.groupby("bucket", observed=False):
        m = trade_metrics(part["gain"].to_numpy())
        rows.append({"pred_bucket": str(key), **m})
    return pd.DataFrame(rows)


def purged_embargo_split(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train sells before test_start - embargo; test buys in [test_start, test_end]."""
    buy = buy_timestamps(df)
    sell = sell_timestamps(df)
    t0 = pd.Timestamp(test_start)
    t1 = pd.Timestamp(test_end)
    train_cut = t0 - pd.Timedelta(days=int(embargo_days))
    train_mask = sell < train_cut
    test_mask = (buy >= t0) & (buy <= t1)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def complementary_purged_split(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold diagnostic: train = all non-test rows whose hold does not overlap the test window."""
    buy = buy_timestamps(df)
    sell = sell_timestamps(df)
    t0 = pd.Timestamp(test_start)
    t1 = pd.Timestamp(test_end)
    embargo = pd.Timedelta(days=int(embargo_days))
    test_mask = (buy >= t0) & (buy <= t1)
    overlap = (buy <= t1 + embargo) & (sell >= t0 - embargo)
    train_mask = (~test_mask) & (~overlap)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def expanding_year_windows(
    buy: pd.Series,
    *,
    min_train_frac: float = 0.60,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """First test fold starts at the min_train_frac quantile of buy dates, then calendar years."""
    b = pd.to_datetime(buy).dropna().sort_values()
    if b.empty:
        return []
    t60 = pd.Timestamp(b.quantile(float(min_train_frac)))
    last = pd.Timestamp(b.max())
    start = t60.normalize()
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    while start <= last:
        year_end = pd.Timestamp(year=start.year, month=12, day=31)
        end = min(year_end, last)
        if end >= start:
            windows.append((start, end))
        start = pd.Timestamp(year=start.year + 1, month=1, day=1)
    return windows


def time_quantile_windows(buy: pd.Series, n_splits: int = 5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    b = pd.to_datetime(buy).dropna().sort_values()
    if b.empty or int(n_splits) < 2:
        return []
    qs = np.linspace(0.0, 1.0, int(n_splits) + 1)
    edges = [pd.Timestamp(b.quantile(q)) for q in qs]
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(int(n_splits)):
        lo = edges[i]
        hi = edges[i + 1]
        if i < int(n_splits) - 1:
            hi = hi - pd.Timedelta(milliseconds=1)
        if hi >= lo:
            out.append((lo, hi))
    return out


def fit_ridge_size_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    l2: float = DEFAULT_RIDGE_L2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SizeMap, int]:
    """Fit on train (winsorized y); return test pred, scale-all size, skip-neg size, map, n_features."""
    cols = resolve_ridge_features(train_df, feature_cols)
    gain_col = _gain_col(train_df)
    x_tr_raw, med = feature_matrix(train_df, cols)
    x_te_raw, _ = feature_matrix(test_df, cols, medians=med)
    x_tr, x_te, _, _ = _standardize(x_tr_raw, x_te_raw)
    y_raw = pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float)
    y_fit = winsorize_y(y_raw)
    ok = np.isfinite(y_fit)
    model = RidgePnlScorer(l2=l2)
    model.fit(x_tr[ok], y_fit[ok])
    pred_tr = model.predict(x_tr)
    pred_te = model.predict(x_te) if len(test_df) else np.zeros(0, dtype=float)
    smap = fit_size_map(pred_tr[ok])
    size_all = apply_size_map(pred_te, smap, skip_negative=False)
    size_skip = apply_size_map(pred_te, smap, skip_negative=True)
    return pred_te, size_all, size_skip, smap, int(x_tr.shape[1])


def _attach_fold_sizes(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    size_all: np.ndarray,
    size_skip: np.ndarray,
    *,
    fold: str,
) -> pd.DataFrame:
    out = test_df.copy()
    out["pred_pnl"] = pred
    out["size_equal"] = 1.0
    out["size_scale_all"] = size_all
    out["size_skip_neg"] = size_skip
    mean_s = float(np.nanmean(size_all)) if len(size_all) else 1.0
    if not np.isfinite(mean_s) or mean_s <= 1e-12:
        mean_s = 1.0
    out["size_scale_all_norm"] = size_all / mean_s
    out["fold"] = fold
    return out


def _fold_metric_row(
    name: str,
    test_start: str,
    test_end: str,
    n_train: int,
    n_features: int,
    smap: SizeMap,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    size_all: np.ndarray,
    size_skip: np.ndarray,
) -> dict:
    gain_col = _gain_col(test_df)
    g = pd.to_numeric(test_df[gain_col], errors="coerce").to_numpy(dtype=float)
    mean_s = float(np.nanmean(size_all)) if len(size_all) else 1.0
    if not np.isfinite(mean_s) or mean_s <= 1e-12:
        mean_s = 1.0
    size_norm = size_all / mean_s
    equal = sized_trade_metrics(g, np.ones(len(g), dtype=float))
    scale = sized_trade_metrics(g, size_all)
    skip = sized_trade_metrics(g, size_skip)
    norm = sized_trade_metrics(g, size_norm)
    drop3 = drop_top_n_sized_metrics(g, size_all, n=3)
    rho = spearman_pred_actual(pred, g)
    return {
        "name": name,
        "test_start": str(test_start)[:10],
        "test_end": str(test_end)[:10],
        "n_train": int(n_train),
        "n_test": int(len(test_df)),
        "n_features": int(n_features),
        "pred_lo": round(smap.pred_lo, 4),
        "pred_hi": round(smap.pred_hi, 4),
        "spearman": rho,
        "equal_n": equal["n_trades"],
        "equal_E": equal["expectancy"],
        "equal_PF": equal["profit_factor"],
        "equal_MDD": equal["max_dd"],
        "equal_sum": equal["sum_pnl"],
        "scale_n_active": scale["n_active"],
        "scale_E": scale["expectancy"],
        "scale_PF": scale["profit_factor"],
        "scale_MDD": scale["max_dd"],
        "scale_sum": scale["sum_pnl"],
        "scale_mean_size": scale["mean_size"],
        "scale_drop3_PF": drop3["profit_factor"],
        "scale_drop3_E": drop3["expectancy"],
        "skip_n_active": skip["n_active"],
        "skip_E": skip["expectancy"],
        "skip_PF": skip["profit_factor"],
        "skip_MDD": skip["max_dd"],
        "skip_sum": skip["sum_pnl"],
        "norm_E": norm["expectancy"],
        "norm_PF": norm["profit_factor"],
        "norm_MDD": norm["max_dd"],
        "norm_sum": norm["sum_pnl"],
    }


def expanding_ridge_walk_forward(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    min_train_frac: float = 0.60,
    min_train_n: int = MIN_TRAIN_N,
    min_test_n: int = MIN_TEST_N,
    l2: float = DEFAULT_RIDGE_L2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding yearly OOS folds. Returns (fold_metrics, stitched_oos_trades)."""
    buy = buy_timestamps(df)
    windows = expanding_year_windows(buy, min_train_frac=min_train_frac)
    rows: List[dict] = []
    parts: List[pd.DataFrame] = []
    for start, end in windows:
        train_df, test_df = purged_embargo_split(
            df, str(start), str(end), embargo_days=embargo_days
        )
        if len(train_df) < int(min_train_n) or len(test_df) < int(min_test_n):
            continue
        pred, size_all, size_skip, smap, n_feat = fit_ridge_size_fold(
            train_df, test_df, feature_cols, l2=l2
        )
        name = f"wf {str(start)[:10]}..{str(end)[:10]}"
        rows.append(
            _fold_metric_row(
                name, str(start), str(end), len(train_df), n_feat, smap,
                test_df, pred, size_all, size_skip,
            )
        )
        parts.append(_attach_fold_sizes(test_df, pred, size_all, size_skip, fold=name))
    oos = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    return pd.DataFrame(rows), oos


def purged_kfold_ridge(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_splits: int = 5,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    min_train_n: int = MIN_TRAIN_N,
    min_test_n: int = MIN_TEST_N,
    l2: float = DEFAULT_RIDGE_L2,
) -> pd.DataFrame:
    """Diagnostic only: complementary time folds with purge (may use future regimes)."""
    buy = buy_timestamps(df)
    windows = time_quantile_windows(buy, n_splits=n_splits)
    rows: List[dict] = []
    for i, (start, end) in enumerate(windows):
        train_df, test_df = complementary_purged_split(
            df, str(start), str(end), embargo_days=embargo_days
        )
        if len(train_df) < int(min_train_n) or len(test_df) < int(min_test_n):
            continue
        pred, size_all, size_skip, smap, n_feat = fit_ridge_size_fold(
            train_df, test_df, feature_cols, l2=l2
        )
        name = f"kfold{i + 1} {str(start)[:10]}..{str(end)[:10]}"
        rows.append(
            _fold_metric_row(
                name, str(start), str(end), len(train_df), n_feat, smap,
                test_df, pred, size_all, size_skip,
            )
        )
    return pd.DataFrame(rows)


def stitched_oos_metrics(oos: pd.DataFrame) -> dict:
    if oos.empty:
        empty = sized_trade_metrics(np.array([]), np.array([]))
        return {
            "equal": empty,
            "scale_all": empty,
            "skip_neg": empty,
            "scale_all_norm": empty,
            "scale_drop3": empty,
            "spearman": None,
        }
    gain_col = _gain_col(oos)
    g = pd.to_numeric(oos[gain_col], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(oos["pred_pnl"], errors="coerce").to_numpy(dtype=float)
    size_all = pd.to_numeric(oos["size_scale_all"], errors="coerce").to_numpy(dtype=float)
    size_skip = pd.to_numeric(oos["size_skip_neg"], errors="coerce").to_numpy(dtype=float)
    size_norm = pd.to_numeric(oos["size_scale_all_norm"], errors="coerce").to_numpy(dtype=float)
    ones = np.ones(len(g), dtype=float)
    return {
        "equal": sized_trade_metrics(g, ones),
        "scale_all": sized_trade_metrics(g, size_all),
        "skip_neg": sized_trade_metrics(g, size_skip),
        "scale_all_norm": sized_trade_metrics(g, size_norm),
        "scale_drop3": drop_top_n_sized_metrics(g, size_all, n=3),
        "spearman": spearman_pred_actual(pred, g),
    }


def confidence_size_verdict(stitched: dict, fold_df: pd.DataFrame) -> str:
    """Promote-to-research only if OOS scale-all beats equal on PF and E, not one fold, drop-top-3 PF>1."""
    eq = stitched["equal"]
    sc = stitched["scale_all"]
    d3 = stitched["scale_drop3"]
    if not eq["n_trades"] or sc["expectancy"] is None or eq["expectancy"] is None:
        return "no_promote"
    pf_eq = eq["profit_factor"]
    pf_sc = sc["profit_factor"]
    if pf_eq is None or pf_sc is None:
        return "no_promote"
    beat_e = sc["expectancy"] > eq["expectancy"]
    beat_pf = pf_sc > pf_eq
    drop_ok = d3["profit_factor"] is not None and d3["profit_factor"] > 1.0
    fold_ok = True
    if fold_df is not None and not fold_df.empty and "scale_E" in fold_df.columns:
        better = 0
        n_f = 0
        for _, row in fold_df.iterrows():
            if row.get("equal_E") is None or row.get("scale_E") is None:
                continue
            n_f += 1
            if float(row["scale_E"]) > float(row["equal_E"]) and (
                row.get("scale_PF") is None
                or row.get("equal_PF") is None
                or float(row["scale_PF"]) >= float(row["equal_PF"])
            ):
                better += 1
        if n_f >= 2 and better < 2:
            fold_ok = False
    if beat_e and beat_pf and drop_ok and fold_ok:
        return "research_only"
    return "no_promote"


# ---------------------------------------------------------------------------
# Hard keep/skip filter (ridge P&L or logistic win-prob) with purged WF
# ---------------------------------------------------------------------------

# Known at a 15m next-mid fill: rails, wait, calendar, extra-vs-parent.
# Fill-day daily close / RSI / %B / range / SMA distance / same-day volume
# are not known until the session ends (current_best 1d is next-mid).
HONEST_1D_NEXTMID_FEATURES: Sequence[str] = (
    "channel_pos",
    "room_to_resist_pct",
    "channel_width_pct",
    "slope_pct_per_bar",
    "channel_span_days",
    "channel_age_at_buy_days",
    "wait_bars",
    "is_extra",
    "spy_ret_20d",
    "spy_above_sma50",
    "spy_atr_pct",
    "dow",
    "month",
)

HONEST_1D_GEOM_FEATURES: Sequence[str] = (
    "channel_pos",
    "room_to_resist_pct",
    "channel_width_pct",
    "slope_pct_per_bar",
    "channel_span_days",
    "channel_age_at_buy_days",
    "wait_bars",
    "dow",
    "month",
)

LEAKY_1D_CLOSE_FEATURES: Sequence[str] = MODEL_FEATURES


def add_loser_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_extra (shakeout re-break vs first H2 fill). Known at the extra fill."""
    out = df.copy()
    if "shakeout_breakout" in out.columns:
        extra = pd.to_numeric(out["shakeout_breakout"], errors="coerce")
        out["is_extra"] = extra.fillna(0.0).astype(float)
    elif "parent_exit_reason" in out.columns:
        out["is_extra"] = out["parent_exit_reason"].notna().astype(float)
    else:
        out["is_extra"] = 0.0
    return out


def _sort_by_buy(df: pd.DataFrame) -> pd.DataFrame:
    ts = buy_timestamps(df)
    order = ts.argsort(kind="mergesort")
    return df.iloc[order].copy()


def fit_keep_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    kind: str = "ridge",
    l2: float = DEFAULT_RIDGE_L2,
    min_frac: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Fit on train only. Return test scores, keep_skip0, keep_thr, threshold, n_features.

    Ridge scores are predicted gain_pct; skip0 keeps pred>=0; thr keeps pred>=train
    PF-max quantile (same rule as logistic). Logistic scores are P(win).
    """
    cols = resolve_ridge_features(train_df, feature_cols)
    train_df = _sort_by_buy(train_df)
    test_df = test_df.copy()
    gain_col = _gain_col(train_df)
    x_tr_raw, med = feature_matrix(train_df, cols)
    x_te_raw, _ = feature_matrix(test_df, cols, medians=med)
    x_tr, x_te, _, _ = _standardize(x_tr_raw, x_te_raw)
    g_tr = pd.to_numeric(train_df[gain_col], errors="coerce").to_numpy(dtype=float)
    n_feat = int(x_tr.shape[1])
    kind_l = (kind or "ridge").lower()
    n_tr = len(x_tr)
    val_n = max(20, int(n_tr * 0.2)) if n_tr >= 80 else 0

    if kind_l == "logistic":
        y_tr = (g_tr > 0).astype(float)
        w_tr = np.clip(np.abs(g_tr), 0.1, 15.0)
        if val_n:
            x_fit, x_val = x_tr[:-val_n], x_tr[-val_n:]
            y_fit, y_val = y_tr[:-val_n], y_tr[-val_n:]
            w_fit = w_tr[:-val_n]
            g_val = g_tr[-val_n:]
        else:
            x_fit, y_fit, w_fit = x_tr, y_tr, w_tr
            x_val = y_val = g_val = None
        model = LogisticScorer()
        model.fit(x_fit, y_fit, sample_weight=w_fit)
        s_tr = model.score(x_tr)
        s_te = model.score(x_te) if len(x_te) else np.zeros(0, dtype=float)
        if x_val is not None:
            thr = _choose_threshold(model.score(x_val), g_val, min_frac=min_frac, min_n=20)
        else:
            thr = _choose_threshold(s_tr, g_tr, min_frac=min_frac, min_n=20)
        keep_thr = s_te >= thr if len(s_te) else np.zeros(0, dtype=bool)
        keep_skip0 = s_te >= 0.5 if len(s_te) else np.zeros(0, dtype=bool)
        return s_te, keep_skip0, keep_thr, float(thr), n_feat

    y_fit = winsorize_y(g_tr)
    ok = np.isfinite(y_fit)
    model = RidgePnlScorer(l2=l2)
    model.fit(x_tr[ok], y_fit[ok])
    s_tr = model.predict(x_tr)
    s_te = model.predict(x_te) if len(x_te) else np.zeros(0, dtype=float)
    if val_n:
        thr = _choose_threshold(s_tr[-val_n:], g_tr[-val_n:], min_frac=min_frac, min_n=20)
    else:
        thr = _choose_threshold(s_tr[ok], g_tr[ok], min_frac=min_frac, min_n=20)
    keep_thr = s_te >= thr if len(s_te) else np.zeros(0, dtype=bool)
    keep_skip0 = s_te >= 0.0 if len(s_te) else np.zeros(0, dtype=bool)
    return s_te, keep_skip0, keep_thr, float(thr), n_feat


def _keep_fold_row(
    name: str,
    test_start: str,
    test_end: str,
    n_train: int,
    n_features: int,
    threshold: float,
    test_df: pd.DataFrame,
    scores: np.ndarray,
    keep: np.ndarray,
    rule: str,
) -> dict:
    gain_col = _gain_col(test_df)
    g = pd.to_numeric(test_df[gain_col], errors="coerce").to_numpy(dtype=float)
    all_m = trade_metrics(g)
    kept_g = g[np.asarray(keep, dtype=bool)] if len(g) else np.array([])
    kept_m = trade_metrics(kept_g)
    drop3 = drop_top_n_sized_metrics(
        kept_g, np.ones(len(kept_g), dtype=float), n=3
    ) if len(kept_g) else trade_metrics(np.array([]))
    rho = spearman_pred_actual(scores, g)
    return {
        "name": name,
        "rule": rule,
        "test_start": str(test_start)[:10],
        "test_end": str(test_end)[:10],
        "n_train": int(n_train),
        "n_test": int(len(test_df)),
        "n_kept": int(np.asarray(keep, dtype=bool).sum()) if len(keep) else 0,
        "n_features": int(n_features),
        "threshold": round(float(threshold), 4),
        "spearman": rho,
        "all_n": all_m["n_trades"],
        "all_E": all_m["expectancy_pct"],
        "all_PF": all_m["profit_factor"],
        "all_WR": all_m["win_rate_pct"],
        "kept_n": kept_m["n_trades"],
        "kept_E": kept_m["expectancy_pct"],
        "kept_PF": kept_m["profit_factor"],
        "kept_WR": kept_m["win_rate_pct"],
        "kept_med": kept_m["median_pct"],
        "kept_drop3_E": drop3.get("expectancy") if isinstance(drop3, dict) else None,
        "kept_drop3_PF": drop3.get("profit_factor") if isinstance(drop3, dict) else None,
    }


def expanding_keep_walk_forward(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    kind: str = "ridge",
    rule: str = "skip0",
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    min_train_frac: float = 0.60,
    min_train_n: int = MIN_TRAIN_N,
    min_test_n: int = MIN_TEST_N,
    l2: float = DEFAULT_RIDGE_L2,
    min_frac: float = 0.35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding yearly OOS keep/skip. rule=skip0 (pred>=0 or p>=0.5) or thr (train PF)."""
    buy = buy_timestamps(df)
    windows = expanding_year_windows(buy, min_train_frac=min_train_frac)
    rows: List[dict] = []
    parts: List[pd.DataFrame] = []
    rule_l = (rule or "skip0").lower()
    for start, end in windows:
        train_df, test_df = purged_embargo_split(
            df, str(start), str(end), embargo_days=embargo_days
        )
        if len(train_df) < int(min_train_n) or len(test_df) < int(min_test_n):
            continue
        scores, keep0, keep_thr, thr, n_feat = fit_keep_fold(
            train_df, test_df, feature_cols, kind=kind, l2=l2, min_frac=min_frac
        )
        keep = keep0 if rule_l == "skip0" else keep_thr
        name = f"{kind} {rule_l} {str(start)[:10]}..{str(end)[:10]}"
        rows.append(
            _keep_fold_row(
                name, str(start), str(end), len(train_df), n_feat, thr,
                test_df, scores, keep, rule_l,
            )
        )
        part = test_df.copy()
        part["pred_score"] = scores
        part["keep"] = np.asarray(keep, dtype=bool)
        part["fold"] = name
        parts.append(part)
    oos = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    return pd.DataFrame(rows), oos


def stitched_keep_metrics(oos: pd.DataFrame) -> dict:
    if oos.empty or "keep" not in oos.columns:
        empty = trade_metrics(np.array([]))
        return {"all": empty, "kept": empty, "dropped": empty, "kept_drop3": empty, "spearman": None}
    gain_col = _gain_col(oos)
    g = pd.to_numeric(oos[gain_col], errors="coerce").to_numpy(dtype=float)
    keep = oos["keep"].to_numpy(dtype=bool)
    pred = (
        pd.to_numeric(oos["pred_score"], errors="coerce").to_numpy(dtype=float)
        if "pred_score" in oos.columns
        else np.full(len(g), np.nan)
    )
    kept_g = g[keep]
    drop3 = drop_top_n_sized_metrics(kept_g, np.ones(len(kept_g), dtype=float), n=3)
    return {
        "all": trade_metrics(g),
        "kept": trade_metrics(kept_g),
        "dropped": trade_metrics(g[~keep]),
        "kept_drop3": drop3,
        "spearman": spearman_pred_actual(pred, g),
        "n_kept": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "keep_frac": round(float(keep.mean()), 4) if len(keep) else None,
    }


def keep_filter_verdict(stitched: dict, fold_df: pd.DataFrame) -> str:
    """Promote-to-research if OOS kept beats all on E and PF, >=2 folds, drop-top-3 PF>1."""
    all_m = stitched.get("all") or {}
    kept = stitched.get("kept") or {}
    d3 = stitched.get("kept_drop3") or {}
    if not all_m.get("n_trades") or not kept.get("n_trades"):
        return "no_promote"
    if kept.get("expectancy_pct") is None or all_m.get("expectancy_pct") is None:
        return "no_promote"
    pf_all = all_m.get("profit_factor")
    pf_kept = kept.get("profit_factor")
    if pf_all is None or pf_kept is None:
        return "no_promote"
    beat_e = float(kept["expectancy_pct"]) > float(all_m["expectancy_pct"])
    beat_pf = float(pf_kept) > float(pf_all)
    drop_ok = d3.get("profit_factor") is not None and float(d3["profit_factor"]) > 1.0
    fold_ok = True
    if fold_df is not None and not fold_df.empty and "kept_E" in fold_df.columns:
        better = 0
        n_f = 0
        for _, row in fold_df.iterrows():
            if row.get("all_E") is None or row.get("kept_E") is None:
                continue
            n_f += 1
            pf_k = row.get("kept_PF")
            pf_a = row.get("all_PF")
            pf_ge = pf_k is None or pf_a is None or float(pf_k) >= float(pf_a)
            if float(row["kept_E"]) > float(row["all_E"]) and pf_ge:
                better += 1
        if n_f >= 2 and better < 2:
            fold_ok = False
    if beat_e and beat_pf and drop_ok and fold_ok:
        return "research_only"
    return "no_promote"
