# TSFM-Eval

This repository contains the benchmark dataset `forecast_results.parquet`, generated for the analysis and evaluation presented in our paper:

> **Evaluating Accuracy, Calibration, and Efficiency in Zero-Shot Time Series Foundation Models**

submitted to ADBIS 2026.

The repository also includes the Jupyter notebook (`jupyter_code.ipynb`) used to reproduce the plots and aggregation analyses presented in the paper. Additionally, aggregation tables can be derived directly from the released dataset to validate the findings reported in the paper or support further exploratory analysis.

Below you will find the experimental characteristics, evaluation setup, and dataset schema.

# Evaluated Models

## Statistical Baselines

- ARIMA
- Running Average (RA)
- Random Walk with Drift (RWD)

## Supervised Deep Learning

- PatchTST

## Time Series Foundation Models (TSFMs)

- Chronos-2
- TiRex
- Moirai-2.0
- Sundial
- TimesFM
- Toto


# Evaluation Datasets

## EdgeTraffic

Discrete high-frequency traffic counts collected from a road intersection in Iași, Romania. The dataset is highly volatile, integer-valued, and strongly periodic, making it suitable for evaluating high-frequency forecasting robustness.

## MixGridPL

Continuous electricity load measurements from the Polish national power grid. The dataset exhibits strong seasonality, long-range temporal dependencies, and smooth continuous dynamics.

## SPY

Daily closing prices of the SPDR S&P 500 ETF. The series is stochastic and non-stationary, exhibiting long-term trends and short-term volatility.


# Experimental Scenarios

## Fixed Context (FixedC)

A fixed historical context length is used while varying the forecast horizon. This scenario evaluates long-range forecasting behavior, calibration degradation at increasing horizons, and inference latency scaling.

## Fixed Horizon (FixedH)

A fixed prediction horizon is used while varying the historical context length. This setup is designed to study context sensitivity, long-context reasoning, and context saturation effects.


# Evaluation Metrics

## sMAPE

Symmetric Mean Absolute Percentage Error (sMAPE) is used for point forecast evaluation. The metric is scale-independent, symmetric, and robust across heterogeneous datasets.

## ICE

Interval Coverage Error (ICE) evaluates probabilistic calibration by measuring the deviation between empirical and nominal interval coverage. For the benchmark's 80% prediction intervals, the ideal ICE value is 0, with lower values indicating better calibration.

## IMAE

Interval Mean Absolute Error (IMAE) measures the magnitude of interval violations. The metric is zero when all observations fall inside the prediction interval and increases proportionally to the severity of violations.

While ICE measures how often intervals fail, IMAE measures how severe those failures are.

**Note:** For additional metric definitions and methodological details, please refer to the paper.

# Dataset Schema

| Column | Description |
|---|---|
| `dataset_name (category)` | Name of evaluated dataset |
| `model_name (category)` | Forecasting model |
| `context_length (int32)` | Historical context window size |
| `horizon_length (int32)` | Forecast horizon length |
| `data_scenario (category)` | Experimental scenario (`FixedC`, `FixedH`) |
| `window_idx (int32)` | Sequential ID of the rolling evaluation window. Groups all step-ahead forecasts generated from a single historical context block |
| `horizon_idx (int32)` | Absolute index of the target time step in the source dataset. Maps the prediction directly to its corresponding ground-truth value (`target_true`) |
| `target_true (float32)` | Ground-truth target value |
| `p50 (float32)` | Median forecast (50th percentile) |
| `p10 (float32)` | Lower quantile forecast (10th percentile) |
| `p90 (float32)` | Upper quantile forecast (90th percentile) |
| `fit_wall_time (float32)` | Model fitting/training wall-clock time (seconds) |
| `predict_wall_time (float32)` | Inference wall-clock time (seconds) |
| `sape (float32)` | Symmetric Absolute Percentage Error contribution (designated for sMAPE metric) |
| `interval_violation (bool)` | Boolean indicator for interval violation (designated for ICE metric) |
| `interval_absolute_error (float32)` | Magnitude of interval violation (designated for IMAE metric) |
| `interval_width (float32)` | Width of prediction interval |
| `residual (float32)` | Forecast residual |
| `absolute_error (float32)` | Absolute point forecast error |

In addition to the metrics presented in the paper, the released dataset includes several complementary derived features intended to support downstream exploratory analysis. Specifically:

- `interval_width` enables prediction interval sharpness analysis
- `residual` captures signed forecasting bias
- `absolute_error` facilitates fine-grained point forecasting diagnostics

# Code Execution Instructions

## Clone the repository

```bash
git clone https://github.com/unic-ailab/TSFM-Eval.git
```

## Open the `jupyter_code.ipynb` notebook file

```bash
jupyter notebook jupyter_code.ipynb
```

## Reproduce Plots

To reproduce the plots, please execute the notebook cells in order, from top to bottom.

# License

This repository is released under the Apache License 2.0.
