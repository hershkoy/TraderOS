"""
Moving Average Indicators
"""
import pandas as pd
import numpy as np


def SMA(data: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=period).mean()


def EMA(data: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=period).mean()


def WMA(data: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
