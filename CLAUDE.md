# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Python-based finance portfolio data analysis tool covering data ingestion, cleaning, exploratory analysis, metrics computation, time-series forecasting, and Monte Carlo simulation. Scenario parameters are driven by CSV config, and outputs (plots, reports) are written to `outputs/`.

## Common Commands

```bash
# Run the full pipeline
python main.py

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

## Architecture

### Entry Point & Config
- `main.py` — orchestrates the end-to-end pipeline by calling modules in order
- `config.py` — centralises all configurable values (file paths, model params, date ranges, etc.); all other modules import from here rather than hardcoding

### `src/` Modules (called in pipeline order)
| Module | Responsibility |
|---|---|
| `data_ingestion.py` | Load raw data from `data/raw/` (CSV, API, etc.) |
| `data_cleaning.py` | Normalise, fill gaps, handle outliers; write to `data/processed/` |
| `eda.py` | Generate exploratory charts/stats; saves figures to `outputs/plots/` |
| `metrics.py` | Compute portfolio metrics (returns, Sharpe, drawdown, etc.) |
| `forecasting.py` | Time-series forecasting models (e.g. ARIMA, Prophet) |
| `monte_carlo.py` | Monte Carlo simulation using scenario params from `scenario_params/scenarios.csv` |
| `export.py` | Write final reports/data to `outputs/reports/` and `data/exports/` |

### Data Flow
```
data/raw/ → data_ingestion → data_cleaning → data/processed/
                                                   ↓
                                    eda / metrics / forecasting / monte_carlo
                                                   ↓
                                           outputs/ & data/exports/
```

### Scenario Parameters
`scenario_params/scenarios.csv` drives Monte Carlo and forecasting runs. Each row is one named scenario with its parameter overrides. `monte_carlo.py` and `forecasting.py` iterate over rows from this file rather than accepting hardcoded params.
