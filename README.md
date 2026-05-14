# TSFM-Eval

Benchmark dataset and evaluation framework accompanying the paper:

> **Evaluating Accuracy, Calibration, and Efficiency in Zero-Shot Time Series Foundation Models**

Submitted to ADBIS 2026.

---

## Overview

TSFM-Eval is a comprehensive benchmark for evaluating zero-shot Time Series Foundation Models (TSFMs) across:

- point forecasting accuracy,
- probabilistic calibration,
- and computational efficiency.

The benchmark evaluates statistical baselines, supervised deep learning models, and modern TSFMs on datasets spanning:

- traffic forecasting,
- energy load forecasting,
- and financial time series forecasting.

The repository provides:
- benchmarking datasets,
- evaluation outputs,
- probabilistic forecast intervals,
- latency measurements,
- and reproducibility resources.

---

# Benchmark Scope

The released dataset contains:

- **3,638,930 forecasting records**
- **17 columns**
- **3 datasets**
- **9 forecasting models**
- Multiple:
  - context lengths,
  - forecast horizons,
  - and evaluation scenarios

Format:
- Apache Parquet

Optimized memory footprint:
- ~195 MB in-memory using optimized dtypes

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
Discrete high-frequency traffic counts collected from a road intersection in Iași, Romania.

Characteristics:
- highly volatile,
- integer-valued,
- periodic,
- high-frequency traffic observations.

---

## MixGridPL
Continuous electricity load measurements for the Polish national power grid.

Characteristics:
- strong seasonality,
- long-range temporal structure,
- smooth continuous dynamics.

---

## SPY
Daily closing prices of the SPDR S&P 500 ETF.

Characteristics:
- stochastic,
- non-stationary,
- long-term trend with short-term volatility.

---

# Experimental Scenarios

## Fixed Context (FixedC)

A fixed historical context length is used while varying the prediction horizon.

Purpose:
- evaluate long-range forecasting,
- latency scaling,
- and calibration degradation over long horizons.

---

## Fixed Horizon (FixedH)

A fixed prediction horizon is used while varying the historical context length.

Purpose:
- study context sensitivity,
- identify context saturation points,
- and evaluate long-context reasoning.

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

The released dataset additionally includes several derived features for complementary exploratory analysis beyond the metrics presented in the paper:

- `interval_width` for prediction interval sharpness analysis,
- `residual` for signed forecasting bias diagnostics,
- `absolute_error` for fine-grained point forecasting error analysis.

These features support:
- residual diagnostics,
- calibration-vs-width tradeoff studies,
- uncertainty sharpness analysis,
- outlier sensitivity evaluation,
- and forecasting reliability exploration without requiring additional post-processing.

---

# Evaluation Metrics

## sMAPE

Symmetric Mean Absolute Percentage Error used for point forecast evaluation.

Properties:
- scale-independent,
- symmetric,
- robust across heterogeneous datasets.

---

## ICE

Interval Coverage Error evaluates probabilistic calibration.

For an 80% prediction interval:
- ideal value: `0`
- measures deviation between empirical and nominal coverage.

Lower values indicate better calibration.

---

## IMAE

Interval Mean Absolute Error measures the magnitude of interval violations.

Properties:
- zero when all observations fall inside the interval,
- increases proportionally to violation severity.

IMAE complements ICE:
- ICE measures **how often** intervals fail,
- IMAE measures **how severe** failures are.

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

The dataset supports analyses such as:

- forecasting accuracy benchmarking,
- probabilistic calibration evaluation,
- sharpness vs calibration trade-offs,
- context sensitivity analysis,
- horizon scaling analysis,
- latency benchmarking,
- residual diagnostics,
- uncertainty quantification research,
- and deployment-oriented TSFM evaluation.

---

# Key Findings

The benchmark reveals several important observations:

- TSFMs consistently outperform traditional statistical baselines.
- Transformer-based TSFMs achieve extremely low inference latency.
- TiRex provides the most stable probabilistic calibration.
- Patch-based transformers suffer calibration degradation at long horizons.
- Longer context windows improve performance only up to saturation points.
- Point accuracy alone is insufficient for evaluating forecasting reliability.

---

# Reproducibility

The benchmark was implemented using:

- Python
- PyTorch
- Hugging Face model implementations

All TSFMs were evaluated:

- in zero-shot mode,
- without fine-tuning,
- using pretrained default configurations.
