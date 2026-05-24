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

Steps 3–7 are pending implementation. Steps 1–2 are fully operational — module summaries are written to the pipeline log, not printed to console.

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
data/processed/prices_clean.parquet   — cleaned & validated OHLCV (snappy Parquet)
data/processed/returns.parquet        — simple_return, log_return per (date, ticker)
data/processed/cleaning_report.json  — audit counts (duplicates, gaps, outliers)
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
│   ├── eda.py                # Step 3 — Exploratory analysis
│   ├── metrics.py            # Step 4 — Portfolio metrics
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
│   └── processed/            # prices_clean.parquet, returns.parquet, cleaning_report.json
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
| `returns.parquet` | Parquet (snappy) | `simple_return`, `log_return` per (date, ticker) |
| `cleaning_report.json` | JSON | Audit counts: duplicates removed, rows dropped, gaps filled, outliers flagged |

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
