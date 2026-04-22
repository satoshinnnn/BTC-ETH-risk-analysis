from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REGIME_ORDER
from .utils import annualized_volatility, max_drawdown, mean_absolute_return

ASSET_CONFIG = {
    "BTC": {
        "return_col": "btc_return",
        "drawdown_col": "btc_drawdown",
        "volume_col": "btc_volume",
    },
    "ETH": {
        "return_col": "eth_return",
        "drawdown_col": "eth_drawdown",
        "volume_col": "eth_volume",
    },
}

METRIC_ROWS = [
    ("Mean Absolute Return", "Mean Absolute Return"),
    ("Volatility (Annualized)", "Volatility"),
    ("Max Drawdown", "Max Drawdown"),
    # Them volume de so sanh muc thanh khoan giua cac regime.
    ("Average Daily Volume", "Average Daily Volume"),
]


def _iter_regime_subsets(merged: pd.DataFrame):
    yield "Overall", merged
    for regime in REGIME_ORDER:
        yield regime, merged[merged["regime"] == regime]


def _asset_metrics(df: pd.DataFrame, return_col: str, drawdown_col: str, volume_col: str) -> dict[str, float]:
    returns = df[return_col].dropna()
    drawdown = pd.to_numeric(df[drawdown_col], errors="coerce").dropna()
    volume = pd.to_numeric(df[volume_col], errors="coerce").dropna()
    # MDD theo regime lay truc tiep tu full-path drawdown roi loc theo regime
    # de bang metrics khop voi chuoi drawdown dang duoc ve tren bieu do.
    return {
        "Mean Absolute Return": mean_absolute_return(returns),
        "Volatility": annualized_volatility(returns),
        "Max Drawdown": max_drawdown(drawdown),
        # Lay trung binh volume ngay trong tung subset regime.
        "Average Daily Volume": float(volume.mean()) if not volume.empty else np.nan,
    }


def compute_regime_summary(merged: pd.DataFrame) -> dict[str, dict[str, dict[str, float] | float]]:
    summary: dict[str, dict[str, dict[str, float] | float]] = {}

    for regime, subset in _iter_regime_subsets(merged):
        regime_summary: dict[str, dict[str, float] | float] = {}
        for asset, config in ASSET_CONFIG.items():
            regime_summary[asset] = _asset_metrics(
                subset,
                config["return_col"],
                config["drawdown_col"],
                config["volume_col"],
            )

        # Bang correlation dung mot gia tri Pearson correlation cho ca subset regime, khong dung rolling corr.
        valid = subset[["btc_return", "eth_return"]].dropna()
        regime_summary["Correlation"] = float(valid.corr().iloc[0, 1]) if len(valid) >= 2 else np.nan
        summary[regime] = regime_summary

    return summary


def compute_metrics_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = compute_regime_summary(merged)
    columns = [
        "Metric",
        "BTC Overall",
        "ETH Overall",
        *[f"BTC {regime}" for regime in REGIME_ORDER],
        *[f"ETH {regime}" for regime in REGIME_ORDER],
    ]

    records = []
    for label, field in METRIC_ROWS:
        row = {"Metric": label}
        row["BTC Overall"] = summary["Overall"]["BTC"][field]
        row["ETH Overall"] = summary["Overall"]["ETH"][field]
        for regime in REGIME_ORDER:
            row[f"BTC {regime}"] = summary[regime]["BTC"][field]
            row[f"ETH {regime}"] = summary[regime]["ETH"][field]
        records.append(row)

    return pd.DataFrame(records, columns=columns)


def compute_correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = compute_regime_summary(merged)
    row = {"Metric": "BTC-ETH Correlation", "Overall": summary["Overall"]["Correlation"]}
    for regime in REGIME_ORDER:
        row[regime] = summary[regime]["Correlation"]
    return pd.DataFrame([row], columns=["Metric", "Overall", *REGIME_ORDER])


def table_to_console(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    out = df.copy()

    if "Metric" in out.columns:
        # MAR, Volatility va MDD duoc in theo %, con correlation giu dang so thap phan.
        percent_metrics = {"Mean Absolute Return", "Volatility (Annualized)", "Max Drawdown"}
        metric_series = out["Metric"]
        for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            formatted = []
            for idx, value in out[col].items():
                if pd.isna(value):
                    formatted.append("NaN")
                    continue
                metric_name = metric_series.loc[idx]
                if metric_name in percent_metrics:
                    formatted.append(f"{value * 100:.2f}%")
                elif metric_name == "Average Daily Volume":
                    # Volume de nguyen don vi goc, khong doi sang %.
                    formatted.append(f"{value:,.2f}")
                else:
                    formatted.append(float_fmt.format(value))
            out[col] = formatted
    else:
        for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            out[col] = out[col].map(lambda x: "NaN" if pd.isna(x) else float_fmt.format(x))

    return out.to_string(index=False)
