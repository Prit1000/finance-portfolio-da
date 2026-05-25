# Finance Portfolio Analysis Pipeline

A Python pipeline that fetches multi-asset portfolio data from Yahoo Finance, cleans it, runs exploratory analysis, computes financial metrics, generates time-series forecasts, and runs Monte Carlo simulations — all driven by a central config and scenario CSV.

---

## Pipeline Overview

```
Yahoo Finance
     │
     ▼ Step 1 — Data Ingestion        data/raw/prices_raw.csv
                                       data/raw/metadata.json
     ▼ Step 2 — Data Cleaning         data/processed/
     ▼ Step 3 — EDA                   outputs/plots/
     ▼ Step 4 — Metrics               (returns, Sharpe, drawdown …)
     ▼ Step 5 — Forecasting           (ARIMA / Prophet per scenario)
     ▼ Step 6 — Monte Carlo           (simulation per scenario)
     ▼ Step 7 — Export                outputs/reports/  data/exports/
```

All 7 steps are fully operational — module summaries are written to the pipeline log, not printed to console.

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/Prit1000/finance-portfolio-da.git
cd finance-portfolio-da

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your portfolio in config.py (tickers, date range, paths)

# 5. Run the pipeline
python main.py
```

### Output after Step 1

```
data/raw/prices_raw.csv   — long-format OHLCV, one row per (date, ticker)
data/raw/metadata.json    — sector, market cap, beta, P/E, etc. per ticker
logs/pipeline_YYYY-MM-DD.log
```

### Output after Step 2

```
data/processed/prices_clean.parquet    — cleaned & validated OHLCV (snappy Parquet)
data/processed/returns_daily.parquet   — simple_return, log_return per (date, ticker)
data/processed/cleaning_report.json   — audit counts (duplicates, gaps, outliers)
```

### Output after Step 3

```
outputs/plots/01_price_trends/         — close price + rolling MA; volume bar chart per ticker
outputs/plots/02_return_distributions/ — histogram + KDE, Q-Q plot per ticker; combined boxplot
outputs/plots/03_volatility/           — rolling annualized vol per ticker; monthly vol heatmap
outputs/plots/04_correlations/         — correlation matrix; top-pair scatter plots; sector heatmap
outputs/reports/eda_summary.json       — distribution stats, monthly vol, correlation matrix, outlier report
```

### Output after Step 4

```
data/processed/metrics_per_ticker.parquet  — long format: (ticker, metric_name, value, category)
data/processed/portfolio_metrics.parquet   — single-row wide format: Sharpe, VaR, drawdown, beta, etc.
data/processed/rolling_metrics.parquet     — long format: (date, ticker, metric_name, value)
data/processed/drawdown_series.parquet     — per (date, ticker): close, running_peak, drawdown_pct
outputs/reports/metrics_summary.json       — per-ticker and portfolio metrics with config snapshot
```

### Output after Step 5

```
data/processed/forecasts.parquet           — long format: (scenario_name, ticker, model_type, target, forecast_date, forecast, lower_ci, upper_ci, confidence_level)
data/processed/forecast_metrics.parquet    — long format: (scenario_name, ticker, model_type, metric_name, value); metrics: rmse, mae, mape, directional_accuracy, coverage_rate, etc.
data/processed/stationarity_tests.parquet  — ADF test results per (ticker, series_type); series_type ∈ {returns, log_prices}
outputs/reports/forecasting_summary.json   — per-scenario stats, best model per ticker, stationarity counts, config snapshot
```

### Output after Step 6

```
data/processed/mc_paths_summary.parquet         — long format: (scenario_name, ticker, method, day_offset, date, percentile, value); P1–P99 per simulated day
data/processed/mc_terminal_distribution.parquet — per (scenario, ticker, method): s0, terminal mean/std/skew/kurtosis, terminal_pN and return_pN per percentile
data/processed/mc_metrics.parquet               — long format: (scenario_name, ticker, method, metric_name, value); metrics: var_95, var_99, cvar_95, prob_loss, etc.
data/processed/mc_drawdown_distribution.parquet — per (scenario, ticker, method): mean/median/p5/p1 max drawdown and prob_drawdown_exceeds_20pct
data/exports/mc_paths_full.parquet              — full simulation paths (simulation_id, day_offset, value); only written if MC_SAVE_FULL_PATHS=True
outputs/reports/monte_carlo_summary.json        — run timestamp, GBM params, correlation matrix health, per-scenario stats, method comparison, config snapshot
```

### Output after Step 7

```
data/exports/prices_clean.csv        — CSV copy of prices_clean.parquet
data/exports/returns_daily.csv       — CSV copy of returns_daily.parquet
data/exports/metrics_per_ticker.csv  — CSV copy of metrics_per_ticker.parquet
data/exports/portfolio_metrics.csv   — CSV copy of portfolio_metrics.parquet
data/exports/forecasts.csv           — CSV copy of forecasts.parquet
data/exports/mc_metrics.csv          — CSV copy of mc_metrics.parquet
data/exports/portfolio_report.xlsx   — Excel workbook: Portfolio Metrics, Per-Ticker Metrics (pivoted),
                                        Drawdown Summary, Forecasts, MC Risk (pivoted), Config
outputs/reports/pipeline_summary.json — master summary: run timestamp, pipeline_version, tickers,
                                         date range, embedded step summaries (Steps 2–6), config snapshot
```

---

## Configuration (`config.py`)

All behaviour is controlled here — no hardcoded values anywhere in `src/`.

| Variable | Default | Description |
|---|---|---|
| `TICKERS` | `["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT"]` | Portfolio symbols |
| `DATE_START` | `"2022-01-01"` | Fetch start date (YYYY-MM-DD) |
| `DATE_END` | `"2024-12-31"` | Fetch end date (YYYY-MM-DD) |
| `FETCH_INTERVAL` | `"1d"` | Bar interval (daily only in v1) |
| `MAX_RETRIES` | `3` | Retry attempts for flaky API calls |
| `RAW_DATA_DIR` | `Path("data/raw")` | Raw output directory |
| `LOG_DIR` | `Path("logs")` | Log file directory |
| `PLOTS_DIR` | `Path("outputs/plots")` | EDA chart output directory |
| `REPORTS_DIR` | `Path("outputs/reports")` | EDA/export report directory |
| `EDA_PLOT_DPI` | `300` | PNG resolution (print-quality) |
| `EDA_ROLLING_WINDOWS` | `[20, 50]` | Rolling MA windows for price trend charts |
| `EDA_VOL_WINDOW` | `30` | Rolling volatility window (days) |
| `EDA_TRADING_DAYS_PER_YEAR` | `252` | Annualization factor |
| `EDA_TOP_N_CORRELATIONS` | `3` | Number of top correlated pairs to scatter-plot |
| `EDA_TOP_N_MOVES` | `10` | Top single-day moves per ticker in outlier report |
| `EDA_MIN_QQ_ROWS` | `30` | Minimum rows required to generate a Q-Q plot |
| `EDA_PLOT_STYLE` | `"seaborn-v0_8-whitegrid"` | Matplotlib style |
| `RISK_FREE_RATE` | `0.04` | Annual risk-free rate for Sharpe/Sortino |
| `BENCHMARK_TICKER` | `None` | Ticker for beta calculation; must be in `TICKERS` if set |
| `PORTFOLIO_WEIGHTS` | `None` | Custom weights dict (must sum to 1.0); `None` → equal-weighted |
| `VAR_CONFIDENCE_LEVELS` | `[0.95, 0.99]` | Historical VaR confidence levels |
| `CVAR_CONFIDENCE_LEVELS` | `[0.95]` | CVaR confidence levels |
| `ROLLING_SHARPE_WINDOW` | `90` | Rolling Sharpe window (trading days) |
| `ROLLING_BETA_WINDOW` | `60` | Rolling beta window (trading days) |
| `ROLLING_CORR_WINDOW` | `60` | Rolling correlation window (trading days) |
| `METRICS_TRADING_DAYS_PER_YEAR` | `252` | Annualization factor for metrics |
| `EXCLUDE_BENCHMARK_FROM_PORTFOLIO` | `True` | Drop benchmark ticker from portfolio aggregation |
| `SCENARIOS_CSV` | `Path("scenario_params/scenarios.csv")` | Scenario parameter file for Step 5 |
| `FORECAST_HORIZON_DAYS` | `30` | Default forecast horizon (trading days) |
| `TRAIN_INITIAL_DAYS` | `252` | Initial training window for walk-forward backtest |
| `WALK_FORWARD_STEP_DAYS` | `30` | Roll-forward step size (days) |
| `WALK_FORWARD_EXPANDING` | `True` | `True` = expanding window; `False` = fixed rolling window |
| `FORECAST_CONFIDENCE_LEVEL` | `0.95` | Default CI width for forecasts |
| `ARIMA_MAX_P` / `ARIMA_MAX_Q` / `ARIMA_MAX_D` | `5` / `5` / `2` | `auto_arima` search bounds |
| `ARIMA_SEASONAL` | `False` | Enable SARIMA search |
| `PROPHET_YEARLY_SEASONALITY` | `True` | Prophet yearly component |
| `PROPHET_WEEKLY_SEASONALITY` | `True` | Prophet weekly component |
| `PROPHET_DAILY_SEASONALITY` | `False` | Prophet daily component |
| `MIN_OBSERVATIONS_FOR_FORECAST` | `100` | Skip ticker if series is shorter than this |
| `RANDOM_SEED` | `42` | Seed set at forecasting entry for reproducibility |
| `FORECAST_COVERAGE_WARN_BELOW` | `0.70` | Log warning if CI coverage drops below this |
| `FORECAST_COVERAGE_WARN_ABOVE` | `0.99` | Log info if CI coverage exceeds this |
| `MC_SCENARIOS_CSV` | `Path("scenario_params/mc_scenarios.csv")` | Scenario parameter file for Step 6 |
| `MC_DEFAULT_N_SIMULATIONS` | `10000` | Default simulation count per scenario |
| `MC_DEFAULT_HORIZON_DAYS` | `30` | Default simulation horizon (trading days) |
| `MC_DEFAULT_BLOCK_SIZE` | `10` | Default block size for block bootstrap |
| `MC_RANDOM_SEED` | `42` | Base seed; per-scenario seed derived via `md5(scenario_name)` |
| `MC_PERCENTILES` | `[1,5,25,50,75,95,99]` | Percentile bands computed for path summaries and terminal distribution |
| `MC_VAR_LEVELS` | `[0.95, 0.99]` | Confidence levels for simulated VaR |
| `MC_CVAR_LEVELS` | `[0.95]` | Confidence levels for simulated CVaR |
| `MC_SAVE_FULL_PATHS` | `False` | Write all simulation paths to `data/exports/mc_paths_full.parquet` |
| `MC_USE_CORRELATION` | `True` | Use Cholesky-correlated GBM for portfolio simulation |
| `MC_DRIFT_METHOD` | `"historical"` | `"historical"` = use mean log-return as drift; `"zero"` = risk-neutral |
| `MC_TRADING_DAYS_PER_YEAR` | `252` | Annualization factor for GBM parameter estimation |
| `MC_PROBABILITY_THRESHOLDS` | `[-0.20,-0.10,0.0,0.10,0.20]` | Return thresholds for loss/gain probability computation |
| `EXPORT_CSV` | `True` | Write CSV copies of processed Parquet files to `data/exports/` |
| `EXPORT_EXCEL` | `True` | Write `portfolio_report.xlsx` workbook to `data/exports/` |
| `PIPELINE_VERSION` | `"1.0"` | Version string embedded in `pipeline_summary.json` |

---

## Project Structure

```
finance-portfolio-da/
├── main.py                   # Pipeline orchestrator
├── config.py                 # All configurable values
├── requirements.txt
├── src/
│   ├── data_ingestion.py     # Step 1 — Yahoo Finance fetch (DONE)
│   ├── data_cleaning.py      # Step 2 — Normalise & fill gaps (DONE)
│   ├── schemas.py            # Pandera schemas for pipeline data contracts
│   ├── eda.py                # Step 3 — Exploratory analysis (DONE)
│   ├── metrics.py            # Step 4 — Portfolio metrics (DONE)
│   ├── forecasting.py        # Step 5 — ARIMA / Prophet (DONE)
│   ├── monte_carlo.py        # Step 6 — Monte Carlo simulation (DONE)
│   └── export.py             # Step 7 — Reports & exports (DONE)
├── scenario_params/
│   ├── scenarios.csv         # Parameter rows for Step 5 (Forecasting)
│   └── mc_scenarios.csv      # Parameter rows for Step 6 (Monte Carlo)
├── tests/
│   ├── conftest.py           # Shared fixtures (sample_prices_raw, tmp dirs)
│   └── unit/                 # Per-module unit tests
├── data/
│   ├── raw/                  # prices_raw.csv, metadata.json
│   └── processed/            # prices_clean.parquet, returns_daily.parquet, cleaning_report.json
├── outputs/
│   ├── plots/                # EDA charts
│   └── reports/              # Final reports
├── notebooks/
│   ├── 01_eda_exploration.ipynb
│   └── 02_model_experiments.ipynb
└── logs/                     # pipeline_YYYY-MM-DD.log
```

---

## Testing

```bash
# Run the full test suite
pytest tests/

# Run tests for a single module
pytest tests/unit/test_02-data-cleaning.py -v
```

Tests are spec-driven — each test file is derived from the module's spec in `.claude/specs/`, not from reading the implementation. Shared fixtures live in `tests/conftest.py`.

---

## Data Contracts

### `data/raw/prices_raw.csv`

Long-format, split- and dividend-adjusted (`auto_adjust=True`).

| Column | dtype | Description |
|---|---|---|
| `date` | datetime | Trading day |
| `ticker` | string | Uppercase symbol |
| `open` / `high` / `low` / `close` | float64 | Adjusted OHLC |
| `volume` | Int64 | Daily volume |

### `data/processed/` (Step 2 outputs)

| File | Format | Contents |
|---|---|---|
| `prices_clean.parquet` | Parquet (snappy) | Cleaned OHLCV — same schema as raw, validated by `prices_clean_schema` |
| `returns_daily.parquet` | Parquet (snappy) | `simple_return`, `log_return` per (date, ticker) |
| `cleaning_report.json` | JSON | Audit counts: duplicates removed, rows dropped, gaps filled, outliers flagged |

### `data/processed/` (Step 4 outputs)

| File | Format | Contents |
|---|---|---|
| `metrics_per_ticker.parquet` | Parquet (snappy) | Long format: `ticker`, `metric_name`, `value`, `category` (`return`/`risk`/`risk_adjusted`) |
| `portfolio_metrics.parquet` | Parquet (snappy) | Single-row wide: Sharpe, Sortino, Calmar, VaR/CVaR, max drawdown, beta, diversification ratio |
| `rolling_metrics.parquet` | Parquet (snappy) | Long format: `date`, `ticker`, `metric_name`, `value` — rolling Sharpe, vol, beta, correlation |
| `drawdown_series.parquet` | Parquet (snappy) | Per `(date, ticker)`: `close`, `running_peak`, `drawdown_pct` (always ≤ 0) |

Conventions: VaR/CVaR are **positive loss values**; max drawdown is a **negative value** (−0.20 = 20% loss).

### `data/processed/` (Step 5 outputs)

| File | Format | Contents |
|---|---|---|
| `forecasts.parquet` | Parquet (snappy) | Out-of-sample forecasts: `scenario_name`, `ticker`, `model_type`, `target`, `forecast_date`, `forecast`, `lower_ci`, `upper_ci`, `confidence_level` |
| `forecast_metrics.parquet` | Parquet (snappy) | Walk-forward backtest metrics per (scenario, ticker, model): rmse, mae, mape, directional_accuracy, coverage_rate, mean_interval_width, n_folds, n_predictions |
| `stationarity_tests.parquet` | Parquet (snappy) | ADF test results per (ticker, series_type): adf_statistic, p_value, is_stationary, critical_1pct, critical_5pct |
| `forecasting_summary.json` | JSON | Per-scenario stats, best model per ticker (by RMSE), stationarity summary, config snapshot, run timestamp |

Valid models: `arima_returns`, `arima_log_prices`, `prophet`, `naive_random_walk`, `naive_drift`, `naive_mean`. On model convergence failure, output falls back to `naive_random_walk`.

### `data/processed/` (Step 6 outputs)

| File | Format | Contents |
|---|---|---|
| `mc_paths_summary.parquet` | Parquet (snappy) | Long format: `scenario_name`, `ticker`, `method`, `day_offset`, `date`, `percentile`, `value` — P1–P99 per simulated day |
| `mc_terminal_distribution.parquet` | Parquet (snappy) | Per (scenario, ticker, method): `s0`, terminal mean/std/skew/kurtosis, `terminal_pN` and `return_pN` per configured percentile |
| `mc_metrics.parquet` | Parquet (snappy) | Long format: `scenario_name`, `ticker`, `method`, `metric_name`, `value` — `var_95`, `var_99`, `cvar_95`, `prob_loss`, probability thresholds |
| `mc_drawdown_distribution.parquet` | Parquet (snappy) | Per (scenario, ticker, method): mean/median/p5/p1 max drawdown (all ≤ 0), `prob_drawdown_exceeds_20pct` |

Conventions: VaR/CVaR are **positive loss values**; max drawdown is a **negative value** — matching Step 4. Portfolio paths start at 1.0 (normalised). Full simulation paths are only written if `MC_SAVE_FULL_PATHS=True` (to `data/exports/mc_paths_full.parquet`).

### `data/raw/metadata.json`

```json
{
  "AAPL": {
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 4424418721792,
    "currency": "USD",
    "beta": 1.065,
    "trailingPE": 36.5,
    "fiftyTwoWeekHigh": 303.2,
    "fiftyTwoWeekLow": 193.46
  }
}
```

---

## Scenario Parameters

Two separate CSV files drive the scenario-driven steps:

- **`scenario_params/scenarios.csv`** — Step 5 (Forecasting). Required columns: `scenario_name`, `target` (`returns` or `prices`), `tickers` (`all` or comma-separated). Optional: `horizon_days`, `models`, `confidence_level`.
- **`scenario_params/mc_scenarios.csv`** — Step 6 (Monte Carlo). Required columns: `scenario_name`, `method` (`gbm`, `bootstrap`, `block_bootstrap`), `horizon_days`, `n_simulations`. Optional: `block_size`, `drift_method` (`historical` or `zero`), `tickers`, `simulate_portfolio`.

Add rows to test multiple market assumptions in a single run. Each module iterates over its CSV rather than accepting hardcoded parameters.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Market data (Step 1 only — no other module imports it) |
| `pandas` / `numpy` | Data manipulation |
| `pandera` | Schema validation at module boundaries (`src/schemas.py`) |
| `pandas_market_calendars` | Trading calendar for gap-filling in Step 2 |
| `pyarrow` | Parquet read/write for processed data |
| `loguru` | Structured logging to console + file |
| `tenacity` | Retry / exponential backoff for flaky `.info` calls (`ConnectionError`, `TimeoutError`, `OSError`) |
| `matplotlib` / `seaborn` / `plotly` | Visualisation |
| `statsmodels` | ADF stationarity test |
| `pmdarima` | `auto_arima` model selection for ARIMA forecasting |
| `prophet` | Prophet time-series forecasting |
| `scipy` | Statistical metrics |
| `openpyxl` | Excel workbook generation in Step 7 |
| `pytest` | Test suite |
| `jupyter` | Exploratory notebooks |
