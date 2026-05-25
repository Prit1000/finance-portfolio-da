# Step 7 — Export

## Purpose
Consolidates all pipeline outputs into portable CSV exports, an optional
multi-sheet Excel workbook, and a master `pipeline_summary.json` that
aggregates the per-step JSON reports from Steps 2–6.

## Public API

```python
def run_export() -> dict:
```

Returns:

```python
{
    "csv_files_written": int,
    "excel_written": bool,
    "pipeline_summary_written": bool,
    "outputs_written": list[str],
    "duration_sec": float,
}
```

## Config knobs (add to config.py)

| Key | Default | Meaning |
|---|---|---|
| `EXPORT_CSV` | `True` | Write CSV snapshots of key datasets |
| `EXPORT_EXCEL` | `True` | Write multi-sheet Excel workbook (requires openpyxl) |

## Inputs (read-only)

**`data/processed/`** — parquet files written by Steps 2–6:

| File | Used for |
|---|---|
| `prices_clean.parquet` | CSV + Excel |
| `returns_daily.parquet` | CSV + Excel |
| `metrics_per_ticker.parquet` | CSV + Excel (pivoted wide) |
| `portfolio_metrics.parquet` | CSV + Excel |
| `rolling_metrics.parquet` | Excel only |
| `drawdown_series.parquet` | Excel (drawdown summary) |
| `forecasts.parquet` | CSV + Excel |
| `forecast_metrics.parquet` | Excel only |
| `mc_metrics.parquet` | CSV + Excel (pivoted wide) |
| `mc_terminal_distribution.parquet` | Excel only |
| `cleaning_report.json` | pipeline_summary.json |

**`outputs/reports/`** — JSON summaries written by Steps 3–6:

| File | Used for |
|---|---|
| `eda_summary.json` | pipeline_summary.json |
| `metrics_summary.json` | pipeline_summary.json |
| `forecasting_summary.json` | pipeline_summary.json |
| `monte_carlo_summary.json` | pipeline_summary.json |

If any upstream file is missing, log a warning and skip that export — do not raise.

## Outputs

### `data/exports/` — CSV snapshots (if `EXPORT_CSV=True`)

| File | Source |
|---|---|
| `prices_clean.csv` | `prices_clean.parquet` |
| `returns_daily.csv` | `returns_daily.parquet` |
| `metrics_per_ticker.csv` | `metrics_per_ticker.parquet` |
| `portfolio_metrics.csv` | `portfolio_metrics.parquet` |
| `forecasts.csv` | `forecasts.parquet` |
| `mc_metrics.csv` | `mc_metrics.parquet` |

### `data/exports/portfolio_report.xlsx` (if `EXPORT_EXCEL=True` and openpyxl available)

Sheets written in order:

| Sheet name | Contents |
|---|---|
| `Portfolio Metrics` | `portfolio_metrics.parquet` as-is |
| `Per-Ticker Metrics` | `metrics_per_ticker.parquet` pivoted wide (index=ticker, columns=metric_name) |
| `Drawdown Summary` | Per-ticker max/mean `drawdown_pct` derived from `drawdown_series.parquet` |
| `Forecasts` | `forecasts.parquet` as-is |
| `MC Risk` | `mc_metrics.parquet` pivoted wide (index=scenario_name+ticker+method, columns=metric_name) |
| `Config` | Key config values as two-column `parameter / value` table |

If a pivot fails, fall back to writing the long-format DataFrame into that sheet and log a warning.
If openpyxl is not installed, log a warning and skip the Excel export entirely.

### `outputs/reports/pipeline_summary.json`

Top-level keys:

```json
{
  "run_timestamp": "2024-01-01T00:00:00Z",
  "pipeline_version": "1.0",
  "date_range": ["2022-01-01", "2024-12-31"],
  "tickers": ["AAPL", ...],
  "steps": {
    "step_2_cleaning":   { ...cleaning_report.json contents... },
    "step_3_eda":        { ...eda_summary.json contents... },
    "step_4_metrics":    { ...metrics_summary.json contents... },
    "step_5_forecasting":{ ...forecasting_summary.json contents... },
    "step_6_monte_carlo":{ ...monte_carlo_summary.json contents... }
  },
  "config_snapshot": {
    "tickers": [...],
    "date_start": "...",
    "date_end": "...",
    "risk_free_rate": 0.04,
    "benchmark_ticker": null,
    "portfolio_weights": null
  },
  "export": {
    "csv_files_written": 6,
    "excel_written": true,
    "duration_sec": 1.23
  }
}
```

## Conventions

- No network calls; no yfinance import.
- All logging via `loguru.logger`; module log file added/removed in try/finally.
- `print()` is banned.
- Missing upstream files are skipped with a warning, not a raised exception.