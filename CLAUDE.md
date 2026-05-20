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

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

## Architecture

### Entry Point & Config
- `main.py` — calls each module's `run_*()` function in order; Steps 2–7 are currently commented out pending implementation
- `config.py` — single source of truth for all values (`TICKERS`, `DATE_START`, `DATE_END`, `FETCH_INTERVAL`, `MAX_RETRIES`, and all `Path` constants); every module imports from here, nothing is hardcoded elsewhere

### Pipeline Module Pattern

Each `src/` module exposes one public orchestrator:

```python
def run_<step>() -> dict:   # called by main.py; returns a summary dict
```

Internal helpers are prefixed with `_`. All logging uses `loguru.logger`; `print()` is only for the end-of-run console summary.

### `src/` Modules (pipeline order)

| # | Module | Status | Responsibility |
|---|---|---|---|
| 1 | `data_ingestion.py` | **Done** | Fetch OHLCV + metadata from Yahoo Finance; write to `data/raw/` |
| 2 | `data_cleaning.py` | Pending | Normalise, fill gaps, handle outliers; write to `data/processed/` |
| 3 | `eda.py` | Pending | Exploratory charts/stats; save to `outputs/plots/` |
| 4 | `metrics.py` | Pending | Portfolio metrics (returns, Sharpe, drawdown, etc.) |
| 5 | `forecasting.py` | Pending | ARIMA / Prophet; iterates over `scenario_params/scenarios.csv` |
| 6 | `monte_carlo.py` | Pending | Monte Carlo simulation; iterates over `scenario_params/scenarios.csv` |
| 7 | `export.py` | Pending | Write final reports to `outputs/reports/` and `data/exports/` |

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
- **`config.py` is the only place for configurable values.** No hardcoded tickers, dates, or paths in any `src/` file.
- **Long-format DataFrames only.** Wide-format breaks when new tickers are added.
- **`scenario_params/scenarios.csv`** drives `forecasting.py` and `monte_carlo.py` — each row is one named scenario with parameter overrides; modules iterate over rows rather than accepting hardcoded params.

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

### Logging

`loguru` writes structured logs to `logs/pipeline_YYYY-MM-DD.log` (daily rotation, created by `main.py` on startup). Individual module log files (e.g. `logs/ingestion_YYYY-MM-DD.log`) may also be added per-module.

### Known yfinance Behaviour (v1.x)

`yf.download()` with multiple tickers returns a `MultiIndex` DataFrame where **level 0 = ticker symbol, level 1 = field name** (title-cased: `Open`, `High`, etc.). The ingestion module detects this dynamically. Do not change to per-ticker loop — the batch call is ~10× faster.
