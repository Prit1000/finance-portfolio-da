# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Python-based finance portfolio analysis pipeline: fetch → clean → EDA → metrics → forecasting → Monte Carlo → export. Seven sequential modules, each called as `run_*()` from `main.py`. Outputs (plots, reports) land in `outputs/`.

## Common Commands

```bash
# Run the full pipeline
python main.py

# Run a single module in isolation (smoke test pattern)
python -c "from src import data_ingestion; print(data_ingestion.run_ingestion())"
python -c "from src import metrics; print(metrics.run_metrics())"
python -c "from src import forecasting; print(forecasting.run_forecasting())"

# Install dependencies
pip install -r requirements.txt

# Run the full test suite
pytest tests/

# Run tests for a single module
pytest tests/unit/test_04-metrics.py -v

# Run a single test by name
pytest tests/unit/test_04-metrics.py::test_load_data_raises_on_missing_file -v

# Launch notebooks
jupyter notebook notebooks/
```

## Architecture

### Entry Point & Config
- `main.py` — calls each module's `run_*()` function in order; Steps 1–5 are wired; Steps 6–7 imports are commented out pending implementation
- `config.py` — single source of truth for all values (`TICKERS`, `DATE_START`, `DATE_END`, `FETCH_INTERVAL`, `MAX_RETRIES`, and all `Path` constants); every module imports from here, nothing is hardcoded elsewhere

### Pipeline Module Pattern

Each `src/` module exposes one public orchestrator:

```python
def run_<step>() -> dict:   # called by main.py; returns a summary dict
```

Internal helpers are prefixed with `_`. All logging uses `loguru.logger`. `print()` is banned in all `src/` modules — the end-of-run console summary belongs in `main.py` only.

### `src/` Modules (pipeline order)

| # | Module | Status | Responsibility |
|---|---|---|---|
| 1 | `data_ingestion.py` | **Done** | Fetch OHLCV + metadata from Yahoo Finance; write to `data/raw/` |
| 2 | `data_cleaning.py` | **Done** | Normalise, fill gaps, handle outliers; write to `data/processed/` |
| 3 | `eda.py` | **Done** | Exploratory charts/stats; save to `outputs/plots/` and `outputs/reports/` |
| 4 | `metrics.py` | **Done** | Portfolio metrics (returns, Sharpe, drawdown, VaR, rolling); write to `data/processed/` and `outputs/reports/` |
| 5 | `forecasting.py` | **Done** | ARIMA / Prophet / naive baselines; walk-forward backtest; iterates over `scenario_params/scenarios.csv` |
| 6 | `monte_carlo.py` | Pending | Monte Carlo simulation; iterates over `scenario_params/scenarios.csv` |
| 7 | `export.py` | Pending | Write final reports to `outputs/reports/` and `data/exports/` |

Stub files exist for steps 6–7 but contain no implementation yet.

### Data Flow

```
Yahoo Finance
     ↓  (data_ingestion — network calls ONLY here)
data/raw/prices_raw.csv   data/raw/metadata.json
     ↓  (data_cleaning)
data/processed/
     ↓
eda / metrics / forecasting / monte_carlo
     ↓
outputs/plots/   outputs/reports/   data/exports/
```

### Key Architectural Constraints

- **`data_ingestion.py` is the only module that may import `yfinance` or make network calls.** All other modules read from `data/raw/` or `data/processed/`.
- **`config.py` is the only place for configurable values.** No hardcoded tickers, dates, paths, or thresholds in any `src/` file.
- **Long-format DataFrames only.** Wide-format breaks when new tickers are added. Wide pivots are permitted as transient local variables inside a single function but must never cross function boundaries or be persisted.
- **`scenario_params/scenarios.csv`** drives `forecasting.py` and `monte_carlo.py` — each row is one named scenario with parameter overrides; modules iterate over rows rather than accepting hardcoded params.
- **`src/schemas.py`** holds Pandera schemas enforcing dtype/uniqueness contracts at module boundaries. Import and validate at the start of each downstream step rather than repeating inline checks. Current schemas: `prices_clean_schema`, `returns_schema` (Steps 1–2); `metrics_per_ticker_schema`, `rolling_metrics_schema`, `drawdown_schema` (Step 4); `forecasts_schema`, `forecast_metrics_schema`, `stationarity_schema` (Step 5).

### Spec-Driven Development

Each module has a spec in `.claude/specs/<step>.md` (e.g. `03-eda.md`). Tests are generated from the spec, not by reading the implementation. When adding a new module, write the spec first. Test files are named `tests/unit/test_0N-<module-name>.py`.

### Data Contracts (Step 1 outputs, consumed by all downstream steps)

**`data/raw/prices_raw.csv`** — long format, one row per (date, ticker):

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | Trading day |
| `ticker` | `string` | Uppercase symbol |
| `open` / `high` / `low` / `close` | `float64` | Split- and dividend-adjusted (`auto_adjust=True`) |
| `volume` | `Int64` (nullable) | Daily volume |

**`data/raw/metadata.json`** — keyed by ticker, 9 fields each:
`shortName`, `sector`, `industry`, `marketCap`, `currency`, `beta`, `trailingPE`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`. Missing fields stored as `null`.

**Step 2 outputs (`data/processed/`)** — written as Parquet (snappy, via pyarrow):

| File | Contents |
|---|---|
| `prices_clean.parquet` | Cleaned OHLCV, same schema as `prices_raw.csv`; validated by `prices_clean_schema` |
| `returns_daily.parquet` | Per-(date, ticker): `simple_return`, `log_return` — validated by `returns_schema` |
| `cleaning_report.json` | Audit counts: duplicates removed, rows dropped, gaps filled, outliers flagged |

**Step 3 outputs (`outputs/`)** — PNG plots + one JSON summary:

| Path | Contents |
|---|---|
| `outputs/plots/01_price_trends/` | Close price + rolling MA per ticker; volume bar chart per ticker |
| `outputs/plots/02_return_distributions/` | Histogram + KDE, Q-Q plot per ticker; combined boxplot |
| `outputs/plots/03_volatility/` | Rolling annualized vol per ticker; monthly vol heatmap |
| `outputs/plots/04_correlations/` | Correlation matrix heatmap; top-pair scatter plots; sector heatmap |
| `outputs/reports/eda_summary.json` | Distribution stats, monthly vol, correlation matrix, outlier report, plot counts |

**Step 4 outputs (`data/processed/` + `outputs/reports/`)** — Parquet files + one JSON summary:

| File | Contents |
|---|---|
| `data/processed/metrics_per_ticker.parquet` | Long format (ticker, metric_name, value, category); categories: `return`, `risk`, `risk_adjusted` |
| `data/processed/portfolio_metrics.parquet` | Single-row wide format: Sharpe, Sortino, Calmar, VaR/CVaR, beta, diversification ratio |
| `data/processed/rolling_metrics.parquet` | Long format (date, ticker, metric_name, value); metrics: `rolling_sharpe_90`, `rolling_volatility_90`, `rolling_beta_60`, `rolling_corr_60` |
| `data/processed/drawdown_series.parquet` | Per (date, ticker): `close`, `running_peak`, `drawdown_pct` (always ≤ 0) |
| `outputs/reports/metrics_summary.json` | Per-ticker and portfolio metrics, config snapshot, run timestamp |

Key conventions enforced in Step 4:
- VaR and CVaR are **positive loss values**; max drawdown is a **negative value** (−0.20 = 20% loss)
- `BENCHMARK_TICKER` (if set) must already be in `TICKERS`; no network calls allowed here
- `PORTFOLIO_WEIGHTS` must sum to 1.0 (within 1e-6) or `ValueError` is raised
- Historical (empirical) VaR/CVaR only — no parametric or Monte Carlo variants in v1

**Step 5 outputs (`data/processed/` + `outputs/reports/`)** — Parquet files + one JSON summary:

| File | Contents |
|---|---|
| `data/processed/forecasts.parquet` | Long format (scenario_name, ticker, model_type, target, forecast_date, forecast, lower_ci, upper_ci, confidence_level) |
| `data/processed/forecast_metrics.parquet` | Long format (scenario_name, ticker, model_type, metric_name, value); metrics: rmse, mae, mape, directional_accuracy, coverage_rate, mean_interval_width, n_folds, n_predictions |
| `data/processed/stationarity_tests.parquet` | ADF test results per (ticker, series_type); series_type ∈ {returns, log_prices} |
| `outputs/reports/forecasting_summary.json` | Per-scenario stats, best model per ticker, stationarity counts, config snapshot, run timestamp |

Key conventions enforced in Step 5:
- Models: `arima_returns`, `arima_log_prices`, `prophet`, `naive_random_walk`, `naive_drift`, `naive_mean`
- `target` in scenarios.csv must be `"returns"` or `"prices"`; `tickers` column accepts `"all"` or comma-separated uppercase symbols
- Walk-forward backtesting uses `TRAIN_INITIAL_DAYS` (default 252) initial window, rolling by `WALK_FORWARD_STEP_DAYS` (default 30); `WALK_FORWARD_EXPANDING=True` grows the window
- On convergence failure, falls back to `naive_random_walk` silently
- Prophet requires `train_dates` to be passed; synthesised dates are used as a fallback with a warning
- `RANDOM_SEED` is set at `run_forecasting()` entry for reproducibility

### Logging

`loguru` writes structured logs to `logs/pipeline_YYYY-MM-DD.log` (daily rotation, created by `main.py` on startup). Individual module log files (e.g. `logs/eda_YYYY-MM-DD.log`) are added per-module inside each `run_*()` function and removed in the `finally` block.

### Known yfinance Behaviour (v1.x)

`yf.download()` with multiple tickers returns a `MultiIndex` DataFrame where **level 0 = ticker symbol, level 1 = field name** (title-cased: `Open`, `High`, etc.). The ingestion module detects this dynamically. Do not change to per-ticker loop — the batch call is ~10× faster.

**Removed v1.x params — do not pass these to `yf.download()`:**
- `group_by` — removed in v1.x; column ordering is now fixed (ticker, field)
- `threads` — removed in v1.x; threading is handled internally by yfinance

**Retry scope:** `_fetch_single_metadata` retries only on `(ConnectionError, TimeoutError, OSError)` — do not widen to bare `Exception`, as that would retry programming errors and add unnecessary delay.
