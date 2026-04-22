from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_return(price: pd.Series) -> pd.Series:
    series = pd.to_numeric(price, errors="coerce")
    return np.log(series).diff()


def compute_drawdown(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    running_max = price.cummax()
    return price / running_max - 1.0


def max_drawdown(drawdown: pd.Series) -> float:
    if drawdown.empty:
        return np.nan
    return float(drawdown.min())


def annualized_volatility(returns: pd.Series, periods_per_year: int = 365) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def mean_absolute_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    return float(returns.abs().mean())


def contiguous_segments(index: pd.DatetimeIndex, labels: pd.Series):
    """Tra ve tung doan lien tiep co cung nhan regime."""
    if len(index) == 0:
        return
    start = index[0]
    prev = index[0]
    current_label = labels.iloc[0]
    for ts, label in zip(index[1:], labels.iloc[1:]):
        if label != current_label:
            yield start, prev, current_label
            start = ts
            current_label = label
        prev = ts
    yield start, prev, current_label
