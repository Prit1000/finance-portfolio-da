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

Steps 1–4 are fully operational — module summaries are written to the pipeline log, not printed to console. Steps 5–7 are pending implementation.

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

---

## Configuration (`config.py`)

All behaviour is controlled here — no hardcoded values anywhere in `src/`.

| Variable | Default | Description |
|---|---|---|
| `TICKERS` | `["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT"]` | Portfolio symbols |
| `DATE_START` | `"2023-01-01"` | Fetch start date (YYYY-MM-DD) |
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
│   ├── forecasting.py        # Step 5 — ARIMA / Prophet
│   ├── monte_carlo.py        # Step 6 — Simulation
│   └── export.py             # Step 7 — Reports & exports
├── scenario_params/
│   └── scenarios.csv         # Parameter rows for Steps 5 & 6
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

`scenario_params/scenarios.csv` drives Steps 5 (Forecasting) and 6 (Monte Carlo). Each row is one named scenario with parameter overrides. Add rows to test multiple market assumptions in a single run.

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
| `statsmodels` | ARIMA forecasting |
| `prophet` | Prophet time-series forecasting |
| `scipy` | Statistical metrics |
| `fpdf2` / `jinja2` | PDF and HTML report generation |
| `pytest` | Test suite |
| `jupyter` | Exploratory notebooks |
