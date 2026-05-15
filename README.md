# TSFM-Eval

Benchmark dataset and evaluation framework accompanying the paper:

> **Evaluating Accuracy, Calibration, and Efficiency in Zero-Shot Time Series Foundation Models**

Submitted to ADBIS 2026.

---

## Overview

TSFM-Eval is a comprehensive benchmark for evaluating zero-shot Time Series Foundation Models (TSFMs) with respect to point forecasting accuracy, probabilistic calibration, and computational efficiency.

The benchmark includes statistical baselines, supervised deep learning models, and modern TSFMs evaluated on traffic forecasting, energy load forecasting, and financial time series forecasting tasks.

The repository provides benchmarking datasets, probabilistic forecast outputs, latency measurements, evaluation results, and reproducibility resources.

---

# Benchmark Scope

The released dataset contains more than **3.6 million forecasting records** across **3 datasets**, **9 forecasting models**, and multiple experimental configurations involving varying context lengths and forecast horizons.

Key characteristics:
- **3,638,930 forecasting records**
- **17 columns**
- Apache Parquet format
- Optimized in-memory footprint of approximately **195 MB**

---

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

---

# Datasets

## EdgeTraffic

Discrete high-frequency traffic counts collected from a road intersection in Iași, Romania. The dataset is highly volatile, integer-valued, and strongly periodic, making it suitable for evaluating high-frequency forecasting robustness.

## MixGridPL

Continuous electricity load measurements from the Polish national power grid. The dataset exhibits strong seasonality, long-range temporal dependencies, and smooth continuous dynamics.

## SPY

Daily closing prices of the SPDR S&P 500 ETF. The series is stochastic and non-stationary, combining long-term trend structure with short-term volatility.

---

# Experimental Scenarios

## Fixed Context (FixedC)

A fixed historical context length is used while varying the forecast horizon. This scenario evaluates long-range forecasting behavior, calibration degradation at increasing horizons, and inference latency scaling.

## Fixed Horizon (FixedH)

A fixed prediction horizon is used while varying the historical context length. This setup is designed to study context sensitivity, long-context reasoning, and context saturation effects.

---

# Dataset Schema

| Column | Description |
|---|---|
| `dataset_name (category)` | Name of evaluated dataset |
| `model_name (category)` | Forecasting model |
| `context_length (int32)` | Historical context window size |
| `horizon_length (int32)` | Forecast horizon length |
| `data_scenario (category)` | Experimental scenario (`FixedC`, `FixedH`) |
| `target_true (float32)` | Ground-truth target value |
| `p50 (float32)` | Median forecast (50th percentile) |
| `p10 (float32)` | Lower quantile forecast (10th percentile) |
| `p90 (float32)` | Upper quantile forecast (90th percentile) |
| `fit_wall_time (float32)` | Model fitting/training wall-clock time (seconds) |
| `predict_wall_time (float32)` | Inference wall-clock time (seconds) |
| `sape (float32)` | Symmetric Absolute Percentage Error contribution |
| `interval_violation (bool)` | Boolean indicator for interval violation for ICE metric |
| `interval_absolute_error (float32)` | Magnitude of interval violation for IMAE metric |
| `interval_width (float32)` | Width of prediction interval (`p90 - p10`) |
| `residual (float32)` | Forecast residual (`target_true - p50`) |
| `absolute_error (float32)` | Absolute point forecast error |

In addition to the metrics presented in the paper, the released dataset includes several complementary derived features intended to support downstream exploratory analysis. Specifically, `interval_width` enables prediction interval sharpness analysis, `residual` captures signed forecasting bias, and `absolute_error` facilitates fine-grained point forecasting diagnostics. Together, these additions support calibration-versus-width studies, uncertainty analysis, residual diagnostics, and forecasting reliability evaluation without requiring additional post-processing.

---

# Evaluation Metrics

## sMAPE

Symmetric Mean Absolute Percentage Error (sMAPE) is used for point forecast evaluation. The metric is scale-independent, symmetric, and robust across heterogeneous datasets.

## ICE

Interval Coverage Error (ICE) evaluates probabilistic calibration by measuring the deviation between empirical and nominal interval coverage. For the benchmark's 80% prediction intervals, the ideal ICE value is `0`, with lower values indicating better calibration.

## IMAE

Interval Mean Absolute Error (IMAE) measures the magnitude of interval violations. The metric is zero when all observations fall inside the prediction interval and increases proportionally to the severity of violations.

While ICE measures how often intervals fail, IMAE measures how severe those failures are.


---

# Data Types

Optimized schema:

```text
bool(1), category(3), float32(11), int32(2)
```

---

# Code

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Read Dataset

```python
import pandas as pd

df = pd.read_parquet("tsfm_eval.parquet")

print(df.head())
```

---

# Example Analyses

The benchmark supports a broad range of analyses, including forecasting accuracy benchmarking, probabilistic calibration evaluation, sharpness-versus-calibration tradeoff analysis, context sensitivity studies, horizon scaling experiments, latency benchmarking, residual diagnostics, uncertainty quantification research, and deployment-oriented TSFM evaluation.

---

# Key Findings

The benchmark reveals several notable observations. TSFMs consistently outperform traditional statistical baselines, while transformer-based TSFMs achieve extremely low inference latency. TiRex demonstrates the most stable probabilistic calibration, whereas patch-based transformers exhibit calibration degradation at long forecast horizons. The experiments also show that increasing context length improves performance only up to dataset-dependent saturation points, highlighting that point accuracy alone is insufficient for evaluating forecasting reliability.

---

# Reproducibility

The benchmark was implemented in Python using PyTorch and Hugging Face model implementations.

All TSFMs were evaluated in strict zero-shot mode without fine-tuning and using pretrained default configurations.
