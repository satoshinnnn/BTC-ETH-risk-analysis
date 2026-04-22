# Crypto Risk Metrics and Dynamic Regime Analysis

This project analyzes BTC and ETH daily spot data from Binance and produces:

- processed analysis-ready datasets
- finance-style risk tables
- regime-aware plots
- a combined dashboard

## Overview

The pipeline fetches daily BTC and ETH market data, computes core risk indicators, assigns dynamic BTC volatility regimes, and summarizes asset behavior under different market states.

### Core outputs

- daily log returns
- 30-day rolling annualized volatility
- drawdown series
- 30-day rolling BTC-ETH correlation
- dynamic BTC regime labels
- metrics tables by overall sample and by regime
- standalone plots and a combined dashboard

## Regime Framework

Regimes are based on **BTC 30-day annualized rolling volatility**.

- Signal window: `30 days`
- Regime reference window: `60 days`
- Regime reference set at time `t`: trailing `Vol30` values over the most recent 60-day window including `t`
- Current day `t` is included because the analysis uses closed daily candles

Regime labels:

- `Deep Calm`
- `Calm`
- `Turbulent`
- `Stress Turbulent`

This design treats the regime at `t` as an end-of-day market state observed at the close of day `t`.

## Metrics

The main metrics table includes:

- `Mean Absolute Return`
- `Volatility (Annualized)`
- `Max Drawdown`
- `Average Daily Volume`

The correlation table is exported separately as:

- `BTC-ETH Correlation`

## Plots

The project exports:

- `BTC & ETH Rolling Volatility`
- `BTC & ETH Drawdown`
- `BTC-ETH Rolling Correlation`
- `BTC Price and Volume`
- `ETH Price and Volume`
- `Dashboard`

All plots use the same BTC regime shading context.

## Project Structure

```text
crypto-risk-analysis/
├── data/
│   ├── btc.csv
│   └── eth.csv
├── output/
│   ├── plots/
│   ├── tables/
│   └── dashboard.png
├── src/
│   ├── config.py
│   ├── data_fetcher.py
│   ├── metrics.py
│   ├── plots.py
│   ├── processor.py
│   ├── regimes.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with the default lookback:

```bash
python main.py
```

Run with a custom lookback:

```bash
python main.py --lookback_years 4
python main.py --lookback_years 6
```

Run with a custom date range:

```bash
python main.py --start 2018-01-01 --end 2026-04-10
python main.py --start 2022-01-01 --end 2026-04-10
```

## Outputs

### Processed data

- `data/btc.csv`
- `data/eth.csv`

These files contain analysis-ready columns such as:

- `timestamp`
- close
- volume
- return
- drawdown
- rolling volatility
- regime

### Tables

- `output/tables/metrics_table_1.csv`
- `output/tables/correlation_table_2.csv`
- `output/tables/metrics_table_1_pretty.txt`
- `output/tables/correlation_table_2_pretty.txt`

### Figures

- `output/plots/plot1_volatility.png`
- `output/plots/plot2_drawdown.png`
- `output/plots/plot3_correlation.png`
- `output/plots/plot4_btc_price_volume.png`
- `output/plots/plot5_eth_price_volume.png`
- `output/dashboard.png`

## Notes

- Only closed daily candles are used.
- `--end YYYY-MM-DD` is treated as inclusive.
- If `--end` exceeds the latest closed UTC day, it is clamped automatically.
- BTC defines the market regime context used across all metrics and plots.
- BTC and ETH are aligned on overlapping timestamps before analysis.
- Volume is base-asset daily volume (`BTC/day` or `ETH/day`).

## Next Step

The current codebase focuses on descriptive analysis. Forecasting can be added later as a separate extension module without changing the descriptive pipeline.
