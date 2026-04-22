from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REGIME_LOOKBACK


def classify_regimes(volatility: pd.Series, lookback: int = REGIME_LOOKBACK) -> pd.Series:
    # Dung vol hien tai lam gia tri can phan loai.
    # Cua so tham chieu gom "lookback" gia tri Vol30 gan nhat va co bao gom ngay hien tai.
    q1 = volatility.rolling(lookback, min_periods=lookback).quantile(0.25)
    q2 = volatility.rolling(lookback, min_periods=lookback).quantile(0.50)
    q3 = volatility.rolling(lookback, min_periods=lookback).quantile(0.75)

    def label(v, q1_value, q2_value, q3_value):
        # Chua du lich su de tinh cac moc quantile thi chua gan regime.
        if pd.isna(v) or pd.isna(q1_value) or pd.isna(q2_value) or pd.isna(q3_value):
            return np.nan
        if v < q1_value:
            return "Deep Calm"
        if v < q2_value:
            return "Calm"
        if v < q3_value:
            return "Turbulent"
        return "Stress Turbulent"

    # Tra ve mot regime cho tung ngay va giu nguyen index de merge vao bang chinh.
    return pd.Series(
        (label(v, q1_value, q2_value, q3_value) for v, q1_value, q2_value, q3_value in zip(volatility, q1, q2, q3)),
        index=volatility.index,
        dtype="object",
    )
