# SPEC: Forecasting Module (Multi-Asset Portfolio)

**Module:** `src/forecasting.py`
**Pipeline Position:** 5 of 7
**Owner:** Quantitative Research Layer
**Status:** Not started
**Depends on:** `src/data_cleaning.py` (Step 2) — consumes `data/processed/prices_clean.parquet` and `data/processed/returns_daily.parquet`. Also reads `scenario_params/scenarios.csv`.

---
## 1. Problem Statement

Build the fifth module of the pipeline. This module must read cleaned price and return data, fit a configurable set of time-series forecasting models (ARIMA, Prophet, naive baselines) per ticker, run walk-forward backtesting, and persist forecasts and accuracy metrics to `data/processed/` and `outputs/reports/`.

**Why this matters:**
- Forecasting is the first **predictive** module — everything before this described what happened; this projects what might happen
- The pipeline must produce a documented baseline for the Monte Carlo module (Step 6) to compare against
- Walk-forward validation is the industry-standard methodology for time-series model evaluation — single train/test splits cause look-ahead bias and inflate apparent accuracy

**Pain point being solved:** Without a centralised forecasting module, every analyst would fit models ad-hoc in notebooks with inconsistent validation methodology. Worse, most public forecasting code uses naive train/test splits that leak future information. Centralisation enforces correct methodology (walk-forward) and reproducible results.

**Philosophical stance:** **The confidence interval is the forecast.** Point estimates on financial time series are nearly worthless due to weak-form market efficiency. The width of the CI, the coverage rate, and the comparison to naive baselines are the actual deliverables. Any forecast that cannot beat a random walk out-of-sample is documented as such — we do not hide weak models.

---

## 2. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Load `prices_clean.parquet`, `returns_daily.parquet`, and `scenario_params/scenarios.csv` | Must |
| FR2 | Validate inputs against `prices_clean_schema` and `returns_schema` before fitting | Must |
| FR3 | Run Augmented Dickey-Fuller stationarity test per ticker; log results | Must |
| FR4 | Fit ARIMA model on returns using `pmdarima.auto_arima` with AIC selection | Must |
| FR5 | Fit ARIMA model on log-prices with differencing as needed | Must |
| FR6 | Fit Prophet model with configurable seasonality components | Must |
| FR7 | Fit three naive baselines: random walk, drift model, historical mean | Must |
| FR8 | Iterate over each row in `scenarios.csv`, running the specified model with overrides | Must |
| FR9 | Generate point forecasts + confidence intervals at `FORECAST_CONFIDENCE_LEVEL` | Must |
| FR10 | Run walk-forward validation with `TRAIN_INITIAL_DAYS` and `WALK_FORWARD_STEP_DAYS` | Must |
| FR11 | Compute per-fold backtest metrics: RMSE, MAE, MAPE, directional accuracy, coverage rate | Must |
| FR12 | Persist forecasts to `data/processed/forecasts.parquet` | Must |
| FR13 | Persist backtest metrics to `data/processed/forecast_metrics.parquet` | Must |
| FR14 | Persist stationarity test results to `data/processed/stationarity_tests.parquet` | Must |
| FR15 | Write structured `outputs/reports/forecasting_summary.json` with best-model-per-ticker | Must |
| FR16 | Log per-scenario per-ticker actions to `logs/forecasting_YYYY-MM-DD.log` | Must |
| FR17 | Idempotent: re-running with `RANDOM_SEED` set must produce identical outputs | Must |
| FR18 | Skip tickers with fewer than `MIN_OBSERVATIONS_FOR_FORECAST` rows; log warning | Must |
| FR19 | Handle model convergence failures gracefully — fall back to naive baseline | Must |

---

## 3. API Contracts

### 3.1 Inputs (from `config.py`)

```python
# Existing
PROCESSED_DATA_DIR: pathlib.Path        # Path("data/processed")
REPORTS_DIR: pathlib.Path               # Path("outputs/reports")
LOG_DIR: pathlib.Path                   # Path("logs")

# New for Step 5
SCENARIOS_CSV: pathlib.Path             # Path("scenario_params/scenarios.csv")
FORECAST_HORIZON_DAYS: int              # 30 — default; overridable per scenario
TRAIN_INITIAL_DAYS: int                 # 252 — min training window (1 trading year)
WALK_FORWARD_STEP_DAYS: int             # 30 — roll-forward step
WALK_FORWARD_EXPANDING: bool            # True — expanding window vs rolling
FORECAST_CONFIDENCE_LEVEL: float        # 0.95
ARIMA_MAX_P: int                        # 5 — auto_arima search bound
ARIMA_MAX_Q: int                        # 5
ARIMA_MAX_D: int                        # 2
ARIMA_SEASONAL: bool                    # False — daily financial data rarely has reliable seasonality
PROPHET_YEARLY_SEASONALITY: bool        # True
PROPHET_WEEKLY_SEASONALITY: bool        # True
PROPHET_DAILY_SEASONALITY: bool         # False
MIN_OBSERVATIONS_FOR_FORECAST: int      # 100
RANDOM_SEED: int                        # 42 — for reproducibility where stochasticity exists
FORECAST_TRADING_DAYS_PER_YEAR: int     # 252 — annualization factor (reuse from Step 4)
```

### 3.2 Scenario CSV Schema (`scenario_params/scenarios.csv`)

| Column | dtype | Required | Example | Description |
|---|---|---|---|---|
| `scenario_name` | string | Yes | `arima_returns_30d` | Unique identifier |
| `model` | string | Yes | `arima_returns`, `arima_log_prices`, `prophet`, `naive_random_walk`, `naive_drift`, `naive_mean` | Model type |
| `target` | string | Yes | `returns` or `prices` | Forecast target |
| `horizon_days` | int | Yes | `30` | Forecast horizon |
| `confidence_level` | float | Yes | `0.95` | CI level |
| `tickers` | string | No | `"all"` or `"AAPL,MSFT"` | Subset filter; `"all"` if missing |
| `notes` | string | No | `"baseline scenario"` | Free text |

At least one row per supported model type required for minimal coverage.

### 3.3 Public Functions

```python
def load_data(
    processed_dir: Path,
    scenarios_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read prices_clean.parquet, returns_daily.parquet, and scenarios.csv.
    Validate pandera schemas. Validate scenario CSV column presence.
    Raise FileNotFoundError if any file is missing.
    Returns (prices, returns, scenarios).
    """
```

---

```python
def check_stationarity(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test per ticker on simple_return.
    Returns DataFrame: [ticker, adf_statistic, p_value, is_stationary, critical_1pct, critical_5pct].
    is_stationary = True if p_value < 0.05.
    """
```

---

```python
def fit_arima(
    series: pd.Series,
    max_p: int,
    max_q: int,
    max_d: int,
    seasonal: bool,
    horizon: int,
    confidence_level: float,
) -> dict:
    """
    Fit ARIMA via pmdarima.auto_arima with AIC criterion.
    Forecast `horizon` steps ahead with confidence intervals.
    Returns:
      {
        "order": (p, d, q),
        "aic": float,
        "forecast": np.ndarray,
        "lower_ci": np.ndarray,
        "upper_ci": np.ndarray,
        "converged": bool
      }
    On convergence failure, return dict with converged=False and fall back triggered upstream.
    """
```

---

```python
def fit_prophet(
    series: pd.Series,
    dates: pd.Series,
    yearly: bool,
    weekly: bool,
    daily: bool,
    horizon: int,
    confidence_level: float,
) -> dict:
    """
    Fit Prophet model. Expects a series indexed by date.
    Returns:
      {
        "forecast": np.ndarray,
        "lower_ci": np.ndarray,
        "upper_ci": np.ndarray,
        "trend": np.ndarray,
        "yearly_component": np.ndarray | None,
        "weekly_component": np.ndarray | None,
        "converged": bool
      }
    """
```

---

```python
def fit_naive(
    series: pd.Series,
    method: str,
    horizon: int,
    confidence_level: float,
) -> dict:
    """
    Fit naive baseline. method ∈ {"random_walk", "drift", "mean"}.
      - random_walk: forecast = last_observed_value (constant)
      - drift: forecast = last + mean_change * step
      - mean: forecast = historical mean (constant)

    CIs computed from historical residual std × sqrt(horizon_step) for random_walk/drift,
    or historical std for mean.

    Returns same dict structure as fit_arima for consistency.
    """
```

---

```python
def walk_forward_validate(
    series: pd.Series,
    dates: pd.Series,
    model_fn: Callable,
    model_kwargs: dict,
    train_initial_days: int,
    step_days: int,
    horizon: int,
    expanding: bool,
) -> pd.DataFrame:
    """
    Walk-forward backtest:
      - Start with `train_initial_days` of training data
      - Forecast `horizon` days ahead
      - Roll forward by `step_days`, refit, repeat until end of series
      - If expanding=True, training window grows; else fixed size

    Returns long-format DataFrame:
      [fold, fold_start_date, fold_end_date, forecast_date, actual, predicted, lower_ci, upper_ci]
    """
```

---

```python
def compute_forecast_metrics(backtest_results: pd.DataFrame) -> dict:
    """
    Compute aggregate metrics across all folds:
      - rmse: sqrt(mean((actual - predicted)^2))
      - mae: mean(|actual - predicted|)
      - mape: mean(|actual - predicted| / |actual|) * 100 (skip rows where actual≈0)
      - directional_accuracy: % of times sign(predicted_change) == sign(actual_change)
      - coverage_rate: % of actuals inside [lower_ci, upper_ci]
      - mean_interval_width: mean(upper_ci - lower_ci)
      - n_folds: int
      - n_predictions: int

    Returns a flat dict for easy DataFrame conversion.
    """
```

---

```python
def run_scenario(
    scenario_row: pd.Series,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config_overrides: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute one scenario row end-to-end:
      - Filter to scenario's tickers (or all)
      - Dispatch to correct fit_* function based on model column
      - Run walk_forward_validate
      - Compute metrics
      - Generate final out-of-sample forecast from full data

    Returns (forecasts_df, metrics_df) both tagged with scenario_name.
    forecasts_df schema: [scenario_name, ticker, forecast_date, forecast, lower_ci, upper_ci, model_type]
    metrics_df schema: [scenario_name, ticker, metric_name, value]
    """
```

---

```python
def save_forecasts(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    stationarity: pd.DataFrame,
    summary_dict: dict,
    processed_dir: Path,
    reports_dir: Path,
) -> None:
    """
    Write:
      - data/processed/forecasts.parquet
      - data/processed/forecast_metrics.parquet
      - data/processed/stationarity_tests.parquet
      - outputs/reports/forecasting_summary.json (indent=2)

    Create directories if missing. Overwrite without prompting.
    """
```

---

```python
def run_forecasting() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_data → validate schemas
      → check_stationarity
      → for each scenario in scenarios.csv:
          → for each ticker in scenario's ticker list:
              → run_scenario → walk_forward_validate → compute_forecast_metrics
      → aggregate results
      → save_forecasts

    Returns a summary dict:
      {
        "scenarios_run": int,
        "tickers_forecasted": int,
        "scenarios_failed": list[str],
        "best_model_per_ticker": dict[str, str],  # by RMSE
        "total_forecasts": int,
        "duration_sec": float
      }
    """
```

---

## 4. Output Contracts

### 4.1 `data/processed/forecasts.parquet`

Long format. One row per (scenario, ticker, forecast_date).

| Column | dtype | Notes |
|---|---|---|
| `scenario_name` | `string` | From scenarios.csv |
| `ticker` | `string` | Uppercase |
| `model_type` | `string` | e.g. `"arima_returns"` |
| `target` | `string` | `"returns"` or `"prices"` |
| `forecast_date` | `datetime64[ns]` | Future date being predicted |
| `forecast` | `float64` | Point estimate |
| `lower_ci` | `float64` | CI lower bound |
| `upper_ci` | `float64` | CI upper bound |
| `confidence_level` | `float64` | e.g. 0.95 |

### 4.2 `data/processed/forecast_metrics.parquet`

Long format. One row per (scenario, ticker, metric_name).

| Column | dtype | Notes |
|---|---|---|
| `scenario_name` | `string` | |
| `ticker` | `string` | |
| `model_type` | `string` | |
| `metric_name` | `string` | `rmse`, `mae`, `mape`, `directional_accuracy`, `coverage_rate`, `mean_interval_width`, `n_folds` |
| `value` | `float64` | |

### 4.3 `data/processed/stationarity_tests.parquet`

| Column | dtype | Notes |
|---|---|---|
| `ticker` | `string` | |
| `series_type` | `string` | `"returns"` or `"log_prices"` |
| `adf_statistic` | `float64` | |
| `p_value` | `float64` | |
| `is_stationary` | `bool` | p_value < 0.05 |
| `critical_1pct` | `float64` | |
| `critical_5pct` | `float64` | |

### 4.4 `outputs/reports/forecasting_summary.json`

```json
{
  "run_timestamp": "2026-05-25T14:22:11Z",
  "input": {
    "prices_rows": 9750,
    "returns_rows": 9743,
    "tickers": 7,
    "scenarios_loaded": 6,
    "date_range": ["2023-01-03", "2024-12-30"]
  },
  "stationarity": {
    "returns_stationary_count": 7,
    "log_prices_stationary_count": 0
  },
  "scenarios_run": {
    "arima_returns_30d": {
      "tickers_succeeded": 7,
      "tickers_failed": [],
      "avg_rmse": 0.0182,
      "avg_coverage_rate": 0.93
    },
    "prophet_30d_seasonal": {
      "tickers_succeeded": 7,
      "tickers_failed": [],
      "avg_rmse": 0.0214,
      "avg_coverage_rate": 0.91
    },
    "naive_random_walk": {
      "tickers_succeeded": 7,
      "tickers_failed": [],
      "avg_rmse": 0.0179,
      "avg_coverage_rate": 0.94
    }
  },
  "best_model_per_ticker": {
    "AAPL": {"model": "arima_returns", "rmse": 0.0171, "beats_random_walk": true},
    "MSFT": {"model": "naive_random_walk", "rmse": 0.0162, "beats_random_walk": null},
    "GOOGL": {"model": "arima_returns", "rmse": 0.0188, "beats_random_walk": false}
  },
  "config_used": {
    "train_initial_days": 252,
    "walk_forward_step_days": 30,
    "expanding_window": true,
    "confidence_level": 0.95,
    "random_seed": 42
  },
  "duration_sec": 47.83
}
```

---

## 5. Constraints

| Constraint | Reason |
|---|---|
| Do NOT import `yfinance` or make network calls | Only `data_ingestion.py` may touch the network |
| Do NOT recompute returns — read from `returns_daily.parquet` | Single source of truth |
| Do NOT modify input DataFrames | Defensive copy at module entry |
| Use **walk-forward validation**, not train/test split | Standard methodology; avoids look-ahead bias |
| Use `pmdarima.auto_arima` for ARIMA fitting | Standard AIC-based selection; faster than manual grid search |
| Forecast log-prices, never raw prices, when target=prices | Prevents negative-price forecasts; stabilizes variance |
| Set `numpy.random.seed(RANDOM_SEED)` at module start | Reproducibility |
| Long-format Parquet for all time-series outputs | Consistent with rest of pipeline |
| `loguru` for logging; no `print()` in `src/forecasting.py` | Persistent logs, structured output |
| Idempotent: same inputs + seed → byte-identical Parquet | Reproducibility |
| Validate inputs via pandera schemas at module entry | Catch contract violations early |
| ARIMA convergence failures must fall back to naive baseline | Pipeline robustness; never crash on one bad ticker |
| Coverage rate must be reported even if poor | Honest reporting; the gap is the insight |
| Always include at least one naive baseline scenario | Required null hypothesis for model evaluation |

---

## 6. Edge Cases & Error Handling

| Case | Expected Behavior |
|---|---|
| `prices_clean.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `returns_daily.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `scenarios.csv` missing | Raise `FileNotFoundError` with hint to create scenario file |
| `scenarios.csv` has no rows | Raise `ValueError` — nothing to forecast |
| `scenarios.csv` has unknown model name | Raise `ValueError` listing the row and invalid value |
| `scenarios.csv` has `tickers="AAPL,FAKE"` where FAKE not in data | Log warning, skip FAKE, continue with AAPL |
| Ticker has fewer rows than `MIN_OBSERVATIONS_FOR_FORECAST` | Skip ticker, log warning, record in `tickers_failed` |
| Ticker has fewer rows than `TRAIN_INITIAL_DAYS + horizon` | Skip walk-forward, fit final forecast only, log warning |
| ARIMA `auto_arima` returns None or raises | Catch, log error, fall back to naive_random_walk for that ticker-scenario |
| Prophet fit raises (e.g. cmdstanpy backend issue) | Catch, log error, mark scenario as failed for that ticker |
| ADF test returns NaN (constant series) | Mark `is_stationary=False`, log warning |
| MAPE with actual ≈ 0 (e.g. returns near zero) | Skip those rows in MAPE calc; use small epsilon guard |
| Coverage rate very low (<70%) | Log warning — model underestimates uncertainty |
| Coverage rate very high (>99%) | Log info — CI may be too wide / overestimated uncertainty |
| All scenarios fail for a ticker | Log critical error; ticker excluded from `best_model_per_ticker` |
| Existing files in `data/processed/` from prior run | Overwrite without prompting |
| `RANDOM_SEED` not set | Use system default; log warning about non-reproducibility |
| Prophet dependency not installed | Raise `ImportError` with install hint; do not silently skip |
| Forecast horizon > data length | Raise `ValueError` — nonsensical request |

---

## 7. Acceptance Criteria

The module is considered complete when **all** the following are true:

- [ ] `python main.py` executes `run_forecasting()` as Step 5 after Step 4
- [ ] `data/processed/forecasts.parquet` exists with all scenarios × tickers × forecast_dates
- [ ] `data/processed/forecast_metrics.parquet` exists with RMSE, MAE, coverage per scenario-ticker
- [ ] `data/processed/stationarity_tests.parquet` exists with ADF results per ticker
- [ ] `outputs/reports/forecasting_summary.json` exists with all fields from §4.4
- [ ] At least one naive baseline scenario exists in `scenarios.csv`
- [ ] `lower_ci ≤ forecast ≤ upper_ci` for every row in forecasts.parquet — invariant
- [ ] `coverage_rate` is between 0 and 1 for every scenario-ticker
- [ ] `n_folds ≥ 1` for every scenario-ticker that succeeded
- [ ] Walk-forward folds have no overlap between train and forecast windows (verified in tests)
- [ ] Re-running `main.py` with same `RANDOM_SEED` produces byte-identical Parquet files
- [ ] Setting `MIN_OBSERVATIONS_FOR_FORECAST` to a very high value cleanly skips all tickers
- [ ] Removing Prophet rows from scenarios.csv produces a complete run without Prophet
- [ ] `logs/forecasting_YYYY-MM-DD.log` shows per-scenario per-ticker actions
- [ ] All public functions have type hints and docstrings
- [ ] No hardcoded thresholds, tickers, model parameters, or paths in `src/forecasting.py`
- [ ] At least 10 unit tests in `tests/unit/test_forecasting.py` pass
- [ ] No `print()` calls in `src/forecasting.py`
- [ ] Model convergence failure for one ticker does not crash the entire run

---

## 8. File Deliverables

| File | Purpose |
|---|---|
| `src/forecasting.py` | Module implementation |
| `src/schemas.py` | Add `forecasts_schema`, `forecast_metrics_schema`, `stationarity_schema` |
| `config.py` | Add the new variables listed in §3.1 |
| `main.py` | Uncomment & integrate Step 5 |
| `scenario_params/scenarios.csv` | Define at least 4 scenarios (incl. one naive baseline) |
| `tests/unit/test_forecasting.py` | Unit tests (see §10) |
| `requirements.txt` | Add `pmdarima`, `prophet`, `statsmodels` (if not present) |

---

## 9. `main.py` Integration

```python
# ── Step 5: Forecasting ───────────────────────────────────────────────
logger.info("STEP 5/7 — Forecasting")
forecasting_summary = forecasting.run_forecasting()
logger.info(f"Forecasting summary: {forecasting_summary}")
```

Uncomment the `from src import forecasting` import at the top.

---

## 10. Unit Test Plan

`tests/unit/test_forecasting.py` — minimum tests:

| # | Test | What it proves |
|---|---|---|
| 1 | `test_load_data_raises_on_missing_scenarios_csv` | FileNotFoundError raised with hint |
| 2 | `test_check_stationarity_returns_expected_columns` | ADF result DataFrame has correct schema |
| 3 | `test_fit_arima_recovers_known_ar1_params` | Synthetic AR(1) series with known phi → auto_arima recovers within tolerance |
| 4 | `test_fit_naive_random_walk_forecast_is_constant` | Random walk forecast equals last observed value, repeated |
| 5 | `test_fit_naive_drift_extrapolates_trend` | Drift model produces linearly increasing forecast for upward trending data |
| 6 | `test_walk_forward_no_train_test_leakage` | Train end_date < test start_date in every fold |
| 7 | `test_walk_forward_expanding_window_grows` | Each fold's training window ≥ previous fold's |
| 8 | `test_compute_metrics_coverage_in_zero_one_range` | Coverage rate always in [0, 1] |
| 9 | `test_compute_metrics_rmse_on_perfect_forecast_is_zero` | If predicted == actual, RMSE = 0 |
| 10 | `test_compute_metrics_directional_accuracy_on_synthetic` | Hand-computed expected value matches |
| 11 | `test_arima_failure_falls_back_to_naive` | Forcing ARIMA failure → naive forecast returned, scenario not marked as failed |
| 12 | `test_invariant_lower_ci_leq_forecast_leq_upper_ci` | Forecast bounds ordered correctly |
| 13 | `test_idempotent_rerun_with_seed` | Two runs with same seed produce identical output |
| 14 | `test_unknown_model_raises_value_error` | Unknown model in scenarios.csv raises clear error |
| 15 | `test_min_observations_skips_short_ticker` | Ticker with < MIN_OBSERVATIONS skipped cleanly |

Use small synthetic time series (white noise, AR(1), trending) as fixtures in `conftest.py`. Hand-compute expected values for at least 3 tests. Do NOT load real Yahoo data in unit tests — fits would be too slow.

---

## 11. Manual Test Plan (After Implementation)

```bash
# 1. Fresh run after Step 4
rm -f data/processed/forecasts.parquet data/processed/forecast_metrics.parquet
rm -f data/processed/stationarity_tests.parquet
rm -f outputs/reports/forecasting_summary.json
python main.py

# 2. Verify outputs exist
ls data/processed/ | grep -E "(forecast|stationarity)"
# Expected: forecasts.parquet, forecast_metrics.parquet, stationarity_tests.parquet

ls outputs/reports/
# Expected: forecasting_summary.json

# 3. Inspect stationarity results
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/stationarity_tests.parquet')
print(df)
print('Returns stationary:', df[df.series_type=='returns'].is_stationary.sum(), '/', len(df[df.series_type=='returns']))
"

# 4. Inspect forecast metrics — does any model beat random walk?
python -c "
import pandas as pd
m = pd.read_parquet('data/processed/forecast_metrics.parquet')
rmse = m[m.metric_name=='rmse'].pivot(index='ticker', columns='scenario_name', values='value')
print(rmse.round(4))
print()
print('Best scenario per ticker:')
print(rmse.idxmin(axis=1))
"

# 5. Verify coverage rates are sensible
python -c "
import pandas as pd
m = pd.read_parquet('data/processed/forecast_metrics.parquet')
cov = m[m.metric_name=='coverage_rate']
print(cov.pivot(index='ticker', columns='scenario_name', values='value').round(2))
print()
print('Expected coverage at 95% CI: ~0.95')
"

# 6. Check invariant: lower_ci <= forecast <= upper_ci
python -c "
import pandas as pd
f = pd.read_parquet('data/processed/forecasts.parquet')
assert (f.lower_ci <= f.forecast).all(), 'lower_ci > forecast violation'
assert (f.forecast <= f.upper_ci).all(), 'forecast > upper_ci violation'
print('CI invariant OK ✓')
"

# 7. View summary
cat outputs/reports/forecasting_summary.json | python -m json.tool | head -60

# 8. Idempotency — checksum before and after rerun should match
md5sum data/processed/forecasts.parquet
python main.py
md5sum data/processed/forecasts.parquet  # must match (requires RANDOM_SEED set)

# 9. Resilience — corrupt one ticker, rerun, confirm others still produce forecasts
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/returns_daily.parquet')
df.loc[df.ticker=='AAPL', 'simple_return'] = float('nan')  # break AAPL only
df.to_parquet('data/processed/returns_daily.parquet')
"
python main.py
# AAPL should appear in tickers_failed; other tickers complete normally
# (restore returns_daily.parquet via re-running Step 2 after)
```

---

## 12. Out of Scope (Explicit Non-Goals)

- ❌ Monte Carlo simulation (Step 6 handles that — distinct methodology)
- ❌ Visualization / fan charts (Step 7 export)
- ❌ Machine learning models (LSTM, XGBoost, Transformer) — v2 scope
- ❌ Multivariate models (VAR, VECM) — v2 scope
- ❌ GARCH / EGARCH volatility forecasting — v2 (high value but adds complexity)
- ❌ Exogenous regressors (macro features, sentiment) — v2
- ❌ Ensemble / model averaging — v2
- ❌ Bayesian forecasting (PyMC, Stan) — v2
- ❌ Backtesting trading strategies (forecast ≠ trading signal)
- ❌ Rolling-window vs expanding-window comparison studies — pick one via config
- ❌ Hyperparameter tuning beyond auto_arima's AIC search
- ❌ Cross-ticker forecasts (e.g. "AAPL given MSFT moves")
- ❌ Intraday forecasting
- ❌ Probabilistic forecasts beyond a single CI level

---

## 13. Notes for Claude Code

- Implement functions in the order listed in §3.3 — each is independently testable
- After each function, write its unit test before moving to the next (TDD-lite)
- For ARIMA: `pmdarima.auto_arima(y, max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D, seasonal=ARIMA_SEASONAL, suppress_warnings=True, error_action="ignore")`
- For Prophet: rename columns to `ds` (datetime) and `y` (value) — Prophet requires this
- For walk-forward: use `pd.DataFrame.iloc[:train_end]` slicing; never index by date in a way that could leak
- For coverage rate: `(lower_ci <= actual) & (actual <= upper_ci)` then `.mean()`
- For directional accuracy: compare `sign(predicted - last_known)` vs `sign(actual - last_known)`
- For Parquet writes use `df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")`
- Use `loguru.logger` everywhere; no `print()`
- Keep `run_forecasting()` thin — it should read like a table of contents
- Set `numpy.random.seed(config.RANDOM_SEED)` at the top of `run_forecasting()`
- Wrap each model fit in try/except — log the error, fall back to naive baseline, never let the pipeline crash on one ticker-scenario
- For Prophet logging: silence cmdstanpy's verbose output via `logging.getLogger("cmdstanpy").setLevel(logging.WARNING)`
- pmdarima has known compatibility issues with NumPy 2.x — if installation fails, lock NumPy to <2.0
- Document the formula for each metric in the function docstring (e.g. "RMSE = sqrt(mean((actual - predicted)^2))")
- If something seems missing from this spec, flag it before adding silently
- Do NOT add forecasting methods not listed here (LSTM, XGBoost, GARCH, ensemble) — flag as scope discussion items first
- The forecasting output is consumed by Step 7 (export) for fan-chart visualization; preserve the wide CI columns rather than splitting into separate files