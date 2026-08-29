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
