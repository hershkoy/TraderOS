"""Portfolio performance metrics and Phase 5 promotion gates."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

TRADING_DAYS = 252
IS_START = "2018-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-11-26"
EVAL_START = "2018-01-01"
EVAL_END = "2025-11-26"


@dataclass
class PerfStats:
    name: str
    start: str
    end: str
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    pct_time_invested: float
    excess_cagr_vs_spy: float
    excess_sharpe_vs_spy: float
    n_obs: int
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    notes: str = ""
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        d = asdict(self)
        return d


def format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def clip_index(obj: pd.Series | pd.DataFrame, start: str, end: str) -> pd.Series | pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return obj.loc[(obj.index >= s) & (obj.index <= e)]


def equity_from_returns(returns: pd.Series, start_equity: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0)
    return (1.0 + r).cumprod() * start_equity


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annual_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily equity curve."""
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    rets = eq.pct_change().fillna(0.0)
    grouped = (1.0 + rets).groupby(rets.index.year).prod() - 1.0
    grouped.index.name = "year"
    grouped.name = "return"
    return grouped


def trade_stats(trade_rets: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in trade_rets if x is not None and pd.notna(x)]
    n = len(vals)
    if n == 0:
        return {"n_trades": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]
    gross_win = float(sum(wins)) if wins else 0.0
    gross_loss = float(-sum(losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_win > 0 else 0.0)
    wr = len(wins) / n
    exp = float(sum(vals) / n)
    return {
        "n_trades": float(n),
        "win_rate": float(wr),
        "profit_factor": float(pf) if math.isfinite(pf) else 99.0,
        "expectancy": exp,
    }


def perf_stats(
    name: str,
    equity: pd.Series,
    invested: Optional[pd.Series] = None,
    spy_cagr: Optional[float] = None,
    spy_sharpe: Optional[float] = None,
    notes: str = "",
    trade_rets: Optional[Sequence[float]] = None,
) -> PerfStats:
    eq = equity.dropna()
    empty = PerfStats(
        name=name,
        start="",
        end="",
        total_return=0.0,
        cagr=0.0,
        ann_vol=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        pct_time_invested=0.0,
        excess_cagr_vs_spy=0.0,
        excess_sharpe_vs_spy=0.0,
        n_obs=0,
        notes=notes or "insufficient data",
    )
    if len(eq) < 2:
        return empty

    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else 0.0
    ann_vol = float(rets.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(rets) else 0.0
    sharpe = float(cagr / ann_vol) if ann_vol > 1e-12 else 0.0
    downside = rets.clip(upper=0.0)
    ann_down = float(downside.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(downside) else 0.0
    sortino = float(cagr / ann_down) if ann_down > 1e-12 else 0.0
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
    if invested is None:
        pct_inv = 1.0
    else:
        pct_inv = float(invested.reindex(eq.index).fillna(0.0).mean())

    ts = trade_stats(trade_rets or [])
    return PerfStats(
        name=name,
        start=str(eq.index[0].date()),
        end=str(eq.index[-1].date()),
        total_return=total_ret,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        pct_time_invested=pct_inv,
        excess_cagr_vs_spy=(cagr - spy_cagr) if spy_cagr is not None else 0.0,
        excess_sharpe_vs_spy=(sharpe - spy_sharpe) if spy_sharpe is not None else 0.0,
        n_obs=int(len(eq)),
        n_trades=int(ts["n_trades"]),
        win_rate=float(ts["win_rate"]),
        profit_factor=float(ts["profit_factor"]),
        expectancy=float(ts["expectancy"]),
        notes=notes,
    )


def rebase_equity(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if eq.empty or float(eq.iloc[0]) == 0:
        return eq
    return eq / float(eq.iloc[0])


def window_equity(equity: pd.Series, invested: Optional[pd.Series], start: str, end: str) -> Tuple[pd.Series, Optional[pd.Series]]:
    eq = clip_index(equity, start, end)
    eq = rebase_equity(eq)
    inv = None
    if invested is not None:
        inv = clip_index(invested, start, end).reindex(eq.index).fillna(0.0)
    return eq, inv


def passes_phase5_gates(stats: PerfStats, spy: PerfStats) -> Tuple[bool, str]:
    """
    Stricter than Phase 1-4: Sharpe > 1.0 AND Sharpe >= SPY AND MDD better than SPY.
    """
    sharpe_abs = stats.sharpe > 1.0
    sharpe_vs = stats.sharpe + 1e-12 >= spy.sharpe
    mdd_ok = abs(stats.max_drawdown) + 1e-12 < abs(spy.max_drawdown)
    if sharpe_abs and sharpe_vs and mdd_ok:
        return True, "PASS (Sharpe>1 + Sharpe>=SPY + MDD)"
    reasons = []
    if not sharpe_abs:
        reasons.append("Sharpe<=1.0")
    if not sharpe_vs:
        reasons.append("Sharpe<SPY")
    if not mdd_ok:
        reasons.append("MDD>=SPY")
    return False, "FAIL (" + ", ".join(reasons) + ")"


def format_table(rows: Sequence[PerfStats], spy: PerfStats) -> str:
    headers = [
        "name",
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "Calmar",
        "Vol",
        "TotRet",
        "Invested%",
        "xsCAGR",
        "xsSharpe",
        "trades",
        "WR",
        "PF",
        "gate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        if r.name == spy.name or r.name.startswith("SPY_buy_hold"):
            gate = "BENCHMARK"
        else:
            _, gate = passes_phase5_gates(r, spy)
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    f"{r.cagr:.2%}",
                    f"{r.sharpe:.2f}",
                    f"{r.sortino:.2f}",
                    f"{r.max_drawdown:.2%}",
                    f"{r.calmar:.2f}",
                    f"{r.ann_vol:.2%}",
                    f"{r.total_return:.2%}",
                    f"{r.pct_time_invested:.0%}",
                    f"{r.excess_cagr_vs_spy:+.2%}",
                    f"{r.excess_sharpe_vs_spy:+.2f}",
                    str(r.n_trades),
                    f"{r.win_rate:.0%}" if r.n_trades else "-",
                    f"{r.profit_factor:.2f}" if r.n_trades else "-",
                    gate,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_annual_table(annual_map: Dict[str, pd.Series]) -> str:
    years: List[int] = []
    for s in annual_map.values():
        years.extend(int(y) for y in s.index.tolist())
    years = sorted(set(years))
    if not years:
        return "_No annual returns._"
    names = list(annual_map.keys())
    header = ["year"] + names
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for y in years:
        cells = [str(y)]
        for name in names:
            s = annual_map[name]
            if y in s.index:
                cells.append(f"{float(s.loc[y]):.2%}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
