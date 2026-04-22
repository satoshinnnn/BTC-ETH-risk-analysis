from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import (
    BTC_COLOR,
    CORR_COLOR,
    ETH_COLOR,
    PLOTS_DIR,
    REGIME_COLORS,
    REGIME_LOOKBACK,
    ROLLING_WINDOW,
    VOL_COLOR,
)
from .utils import contiguous_segments

plt.style.use("seaborn-v0_8-darkgrid")

VOLATILITY_TITLE = (
    f"BTC & ETH Rolling Volatility ({ROLLING_WINDOW}D, Annualized) "
    f"with BTC Regime Shading (Vol {ROLLING_WINDOW}D Signal, Trailing {REGIME_LOOKBACK}D Quantiles)"
)
DRAWNDOWN_TITLE = (
    f"BTC & ETH Drawdown with BTC Regime Shading "
    f"(Vol {ROLLING_WINDOW}D Signal, Trailing {REGIME_LOOKBACK}D Quantiles)"
)
CORRELATION_TITLE = (
    f"BTC-ETH Rolling Correlation ({ROLLING_WINDOW}D) "
    f"with BTC Regime Shading (Vol {ROLLING_WINDOW}D Signal, Trailing {REGIME_LOOKBACK}D Quantiles)"
)
BTC_PRICE_TITLE = (
    f"BTC Price and Volume with BTC Regime Shading "
    f"(Vol {ROLLING_WINDOW}D Signal, Trailing {REGIME_LOOKBACK}D Quantiles)"
)
ETH_PRICE_TITLE = (
    f"ETH Price and Volume with BTC Regime Shading "
    f"(Vol {ROLLING_WINDOW}D Signal, Trailing {REGIME_LOOKBACK}D Quantiles)"
)

REGIME_LEGEND_ITEMS = [
    ("Deep Calm (< 25th percentile)", "Deep Calm"),
    ("Calm (25th to < 50th percentile)", "Calm"),
    ("Turbulent (50th to < 75th percentile)", "Turbulent"),
    ("Stress Turbulent (>= 75th percentile)", "Stress Turbulent"),
]


def _apply_regime_shading(ax, df: pd.DataFrame):
    if "regime" not in df.columns:
        return

    labels = df["regime"].fillna("Unknown")
    for start, end, label in contiguous_segments(pd.DatetimeIndex(df["timestamp"]), labels):
        if label == "Unknown":
            continue
        # Extend shading through the end of each contiguous regime block.
        ax.axvspan(start, end + pd.Timedelta(days=1), color=REGIME_COLORS.get(label, "#dddddd"), alpha=0.4, linewidth=0)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _format_percent_axis(ax):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def _format_volume_axis(ax, asset: str):
    # Binance volume here is base-asset daily volume: BTC/day or ETH/day.
    ax.set_ylabel(f"Daily Volume ({asset})")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def _format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_xlabel("Time")
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3, color="#6b7280")
    ax.grid(which="major", axis="x", alpha=0.22)
    ax.grid(which="minor", axis="x", alpha=0.08)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")


def _regime_patches():
    return [
        Patch(facecolor=REGIME_COLORS[key], edgecolor="black", linewidth=0.8, alpha=0.4, label=label)
        for label, key in REGIME_LEGEND_ITEMS
    ]


def _series_handles(asset: str | None = None, include_volume: bool = False):
    handles = []
    if asset in (None, "BTC"):
        handles.append(Line2D([0], [0], color=BTC_COLOR, linewidth=2, label="BTC"))
    if asset in (None, "ETH"):
        handles.append(Line2D([0], [0], color=ETH_COLOR, linewidth=2, label="ETH"))
    if include_volume:
        handles.append(Patch(facecolor=VOL_COLOR, edgecolor="black", alpha=0.28, label="Volume"))
    return handles


def _combined_legend_handles(asset: str | None = None, include_volume: bool = False, include_corr: bool = False):
    handles = []
    if include_corr:
        handles.append(Line2D([0], [0], color=CORR_COLOR, linewidth=2, label="BTC-ETH Correlation"))
    else:
        handles.extend(_series_handles(asset=asset, include_volume=include_volume))
    handles.extend(_regime_patches())

    # Deduplicate labels so the legend stays compact even when handles are composed from multiple groups.
    unique_handles = []
    seen_labels = set()
    for handle in handles:
        label = handle.get_label()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_handles.append(handle)

    return unique_handles


def _add_combined_legend(
    ax,
    loc: str = "upper left",
    bbox_to_anchor=None,
    include_volume: bool = False,
    asset: str | None = None,
    include_corr: bool = False,
):
    legend = ax.legend(
        handles=_combined_legend_handles(asset=asset, include_volume=include_volume, include_corr=include_corr),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=1,
        frameon=False,
    )
    ax.add_artist(legend)


def plot_volatility(merged: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged["timestamp"], merged["btc_rolling_vol"], color=BTC_COLOR)
    ax.plot(merged["timestamp"], merged["eth_rolling_vol"], color=ETH_COLOR)
    _apply_regime_shading(ax, merged)
    ax.set_title(VOLATILITY_TITLE)
    ax.set_ylabel("Annualized Volatility (%)")
    _format_percent_axis(ax)
    _add_combined_legend(ax, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    _format_time_axis(ax)
    path = PLOTS_DIR / "plot1_volatility.png"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_drawdown(merged: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged["timestamp"], merged["btc_drawdown"], color=BTC_COLOR)
    ax.plot(merged["timestamp"], merged["eth_drawdown"], color=ETH_COLOR)
    _apply_regime_shading(ax, merged)
    ax.set_title(DRAWNDOWN_TITLE)
    ax.set_ylabel("Drawdown (%)")
    _format_percent_axis(ax)
    _add_combined_legend(ax, loc="lower right")
    _format_time_axis(ax)
    path = PLOTS_DIR / "plot2_drawdown.png"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_correlation(merged: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged["timestamp"], merged["rolling_corr"], color=CORR_COLOR)
    _apply_regime_shading(ax, merged)
    ax.set_title(CORRELATION_TITLE)
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    _add_combined_legend(ax, loc="lower left", include_corr=True)
    _format_time_axis(ax)
    path = PLOTS_DIR / "plot3_correlation.png"
    _save(fig, path)
    plt.close(fig)
    return path


def plot_price_volume(merged: pd.DataFrame, asset: str) -> Path:
    prefix = asset.lower()
    price_col = f"{prefix}_close"
    volume_col = f"{prefix}_volume"
    price_color = BTC_COLOR if asset == "BTC" else ETH_COLOR

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        merged["timestamp"],
        merged[price_col],
        linewidth=1.5,
        color=price_color,
        zorder=3,
    )
    _apply_regime_shading(ax1, merged)
    ax1.set_title(BTC_PRICE_TITLE if asset == "BTC" else ETH_PRICE_TITLE)
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, alpha=0.22)

    ax2 = ax1.twinx()
    ax2.bar(
        merged["timestamp"],
        merged[volume_col],
        width=1.0,
        alpha=0.28,
        color=VOL_COLOR,
        zorder=1,
    )
    _format_volume_axis(ax2, asset)

    if asset == "BTC":
        _add_combined_legend(ax1, loc="upper left", bbox_to_anchor=(0.0, 1.02), include_volume=True, asset="BTC")
    else:
        _add_combined_legend(ax1, loc="upper left", bbox_to_anchor=(0.0, 1.02), include_volume=True, asset="ETH")
    _format_time_axis(ax1)

    path = PLOTS_DIR / f"plot4_{prefix}_price_volume.png" if asset == "BTC" else PLOTS_DIR / f"plot5_{prefix}_price_volume.png"
    _save(fig, path)
    plt.close(fig)
    return path


def create_dashboard(merged: pd.DataFrame) -> Path:
    def _price_volume_panel(ax, merged_df: pd.DataFrame, asset: str):
        prefix = asset.lower()
        price_col = f"{prefix}_close"
        volume_col = f"{prefix}_volume"
        price_color = BTC_COLOR if asset == "BTC" else ETH_COLOR

        ax.plot(
            merged_df["timestamp"],
            merged_df[price_col],
            linewidth=1.4,
            color=price_color,
            zorder=3,
        )
        _apply_regime_shading(ax, merged_df)
        ax.set_ylabel("Price (USDT)")
        ax.grid(True, alpha=0.22)

        ax2 = ax.twinx()
        ax2.bar(
            merged_df["timestamp"],
            merged_df[volume_col],
            width=1.0,
            alpha=0.3,
            color=VOL_COLOR,
            zorder=1,
        )
        _format_volume_axis(ax2, asset)
        ax2.grid(False)
        _add_combined_legend(ax, loc="upper left", bbox_to_anchor=(0.0, 1.02), include_volume=True, asset=asset)
        _format_time_axis(ax)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)

    ax_vol = axes[0, 0]
    ax_dd = axes[0, 1]
    ax_corr = axes[0, 2]
    ax_btc = axes[1, 0]
    ax_eth = axes[1, 1]
    ax_note = axes[1, 2]

    ax_vol.plot(merged["timestamp"], merged["btc_rolling_vol"], color=BTC_COLOR)
    ax_vol.plot(merged["timestamp"], merged["eth_rolling_vol"], color=ETH_COLOR)
    _apply_regime_shading(ax_vol, merged)
    ax_vol.set_title(VOLATILITY_TITLE)
    ax_vol.set_ylabel("Annualized Volatility (%)")
    _format_percent_axis(ax_vol)
    _add_combined_legend(ax_vol, loc="lower left")
    ax_vol.grid(True, alpha=0.22)
    _format_time_axis(ax_vol)

    ax_dd.plot(merged["timestamp"], merged["btc_drawdown"], color=BTC_COLOR)
    ax_dd.plot(merged["timestamp"], merged["eth_drawdown"], color=ETH_COLOR)
    _apply_regime_shading(ax_dd, merged)
    ax_dd.set_title(DRAWNDOWN_TITLE)
    ax_dd.set_ylabel("Drawdown (%)")
    _format_percent_axis(ax_dd)
    _add_combined_legend(ax_dd, loc="lower left")
    ax_dd.grid(True, alpha=0.22)
    _format_time_axis(ax_dd)

    ax_corr.plot(merged["timestamp"], merged["rolling_corr"], color=CORR_COLOR)
    _apply_regime_shading(ax_corr, merged)
    ax_corr.set_title(CORRELATION_TITLE)
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_ylim(-1.05, 1.05)
    _add_combined_legend(ax_corr, loc="lower left", include_corr=True)
    ax_corr.grid(True, alpha=0.22)
    _format_time_axis(ax_corr)

    _price_volume_panel(ax_btc, merged, "BTC")
    ax_btc.set_title(BTC_PRICE_TITLE)

    _price_volume_panel(ax_eth, merged, "ETH")
    ax_eth.set_title(ETH_PRICE_TITLE)

    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.98,
        "BTC Regime Shading\n"
        f"Signal: BTC {ROLLING_WINDOW}D annualized volatility\n"
        f"Reference: trailing {REGIME_LOOKBACK}D Vol30 window including the current day\n"
        "Regime is assigned at the close of day t using closed daily candles\n"
        "- Deep Calm: < 25th percentile\n"
        "- Calm: 25th to < 50th percentile\n"
        "- Turbulent: 50th to < 75th percentile\n"
        "- Stress Turbulent: >= 75th percentile",
        va="top",
        ha="left",
        fontsize=11,
    )
    ax_note.legend(
        handles=_combined_legend_handles(include_volume=True),
        loc="lower left",
        frameon=False,
    )

    path = PLOTS_DIR.parent / "dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
