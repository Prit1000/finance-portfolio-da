# SPEC: Monte Carlo Simulation Module

**Module:** `src/monte_carlo.py` | **Pipeline Position:** 6 of 7
**Depends on:** Step 2 outputs — `data/processed/prices_clean.parquet`, `data/processed/returns_daily.parquet`. Also reads `scenario_params/mc_scenarios.csv`.

---

## 1. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Load `prices_clean.parquet`, `returns_daily.parquet`, `mc_scenarios.csv` | Must |
| FR2 | Validate inputs against pandera schemas before simulating | Must |
| FR3 | Estimate GBM params (μ, σ) per ticker from log-returns | Must |
| FR4 | GBM with Itô correction: `S(t+1) = S(t) * exp((μ - σ²/2) + σZ)` | Must |
| FR5 | Historical bootstrap (resample with replacement) | Must |
| FR6 | Block bootstrap (resample consecutive blocks, circular) | Must |
| FR7 | Correlated portfolio simulation via Cholesky decomposition | Must |
| FR8 | Fall back to nearest PD matrix if correlation matrix is non-PD | Must |
| FR9 | Iterate over each `mc_scenarios.csv` row, run specified method | Must |
| FR10 | Per scenario: path summaries — P1/P5/P25/P50/P75/P95/P99 per day | Must |
| FR11 | Per scenario: terminal distribution — percentiles, mean, std, skew, kurtosis | Must |
| FR12 | Per scenario: simulated VaR (95%, 99%) and CVaR (95%) | Must |
| FR13 | Per scenario: P(loss), P(gain ≥ 10%), P(loss ≥ 20%) | Must |
| FR14 | Per scenario: max drawdown distribution across paths | Must |
| FR15–18 | Persist all four outputs to `data/processed/mc_*.parquet` | Must |
| FR19 | Persist full paths to `data/exports/mc_paths_full.parquet` if `MC_SAVE_FULL_PATHS=True` | Must |
| FR20 | Write `outputs/reports/monte_carlo_summary.json` | Must |
| FR21 | Log per-scenario per-ticker actions to `logs/monte_carlo_YYYY-MM-DD.log` | Must |
| FR22 | Idempotent: same `MC_RANDOM_SEED` → byte-identical Parquet | Must |
| FR23 | Per-scenario seed = `MC_RANDOM_SEED + hash(scenario_name) % 2**32` | Must |
| FR24 | Validate VaR ≤ CVaR and VaR_99 ≥ VaR_95 before saving | Must |

---

## 2. Config Variables (`config.py`)

```python
MC_SCENARIOS_CSV: Path              # Path("scenario_params/mc_scenarios.csv")
MC_DEFAULT_N_SIMULATIONS: int       # 10000
MC_DEFAULT_HORIZON_DAYS: int        # 30
MC_DEFAULT_BLOCK_SIZE: int          # 10
MC_RANDOM_SEED: int                 # 42
MC_PERCENTILES: list[float]         # [1, 5, 25, 50, 75, 95, 99]
MC_VAR_LEVELS: list[float]          # [0.95, 0.99]
MC_CVAR_LEVELS: list[float]         # [0.95]
MC_SAVE_FULL_PATHS: bool            # False
MC_USE_CORRELATION: bool            # True
MC_DRIFT_METHOD: str                # "historical" or "zero"
MC_TRADING_DAYS_PER_YEAR: int       # 252
MC_PROBABILITY_THRESHOLDS: list[float]  # [-0.20, -0.10, 0.0, 0.10, 0.20]
```

Existing: `PROCESSED_DATA_DIR`, `REPORTS_DIR`, `LOG_DIR`, `EXPORTS_DIR`, `PORTFOLIO_WEIGHTS`, `BENCHMARK_TICKER`, `EXCLUDE_BENCHMARK_FROM_PORTFOLIO`.

---

## 3. Scenario CSV Schema (`scenario_params/mc_scenarios.csv`)

| Column | dtype | Required | Example |
|---|---|---|---|
| `scenario_name` | string | Yes | `gbm_30d_10k` |
| `method` | string | Yes | `gbm`, `bootstrap`, `block_bootstrap` |
| `horizon_days` | int | Yes | `30` |
| `n_simulations` | int | Yes | `10000` |
| `block_size` | int | No | `10` |
| `drift_method` | string | No | `historical` or `zero` |
| `tickers` | string | No | `"all"` or `"AAPL,MSFT"` |
| `simulate_portfolio` | bool | No | `true` |
| `notes` | string | No | free text |

Minimum 3 rows — one per method.

---

## 4. Public API

```python
def load_data(processed_dir: Path, scenarios_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read prices_clean.parquet, returns_daily.parquet, mc_scenarios.csv.
    Validate pandera schemas. Raise FileNotFoundError if missing. Returns (prices, returns, scenarios)."""

def estimate_gbm_params(returns: pd.DataFrame, trading_days_per_year: int, drift_method: str) -> pd.DataFrame:
    """Per ticker on log returns:
      mu_daily = mean(log_return) if drift_method=="historical" else 0.0
      sigma_daily = std(log_return)
      mu_annual = mu_daily * trading_days_per_year
      sigma_annual = sigma_daily * sqrt(trading_days_per_year)
    Returns DataFrame: [ticker, mu_daily, sigma_daily, mu_annual, sigma_annual, n_observations]."""

def simulate_gbm(s0: float, mu_daily: float, sigma_daily: float, horizon: int,
                 n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """GBM: S(t+1) = S(t) * exp((mu - sigma**2/2) + sigma * Z). dt=1 baked into daily params.
    Returns shape (n_simulations, horizon + 1). Column 0 = s0."""

def simulate_bootstrap(s0: float, historical_returns: np.ndarray, horizon: int,
                       n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """Resample simple returns with replacement. Compound to build paths.
    Returns shape (n_simulations, horizon + 1)."""

def simulate_block_bootstrap(s0: float, historical_returns: np.ndarray, horizon: int,
                              n_simulations: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample blocks of consecutive returns (circular). Returns shape (n_simulations, horizon + 1)."""

def simulate_portfolio_correlated(initial_prices: dict[str, float], gbm_params: pd.DataFrame,
                                   correlation_matrix: pd.DataFrame, weights: dict[str, float],
                                   horizon: int, n_simulations: int, rng: np.random.Generator,
                                   ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Correlated GBM via Cholesky: L = chol(corr), correlated_Z = L @ iid_Z.
    Non-PD input: clip negative eigenvalues to 1e-8, reconstruct, log warning.
    Returns (per_ticker_paths {ticker: (n_sim, horizon+1)}, portfolio_paths (n_sim, horizon+1))."""

def compute_path_summary(paths: np.ndarray, percentiles: list[float],
                          dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-day percentiles across paths. Returns long-format: [day_offset, date, percentile, value]."""

def compute_terminal_distribution(paths: np.ndarray, s0: float, percentiles: list[float]) -> dict:
    """Stats on paths[:, -1]. Returns:
      {terminal_mean, terminal_std, terminal_skew, terminal_kurtosis,
       percentiles: {p: value}, return_percentiles: {p: (value/s0 - 1)}}"""

def compute_simulated_var_cvar(paths: np.ndarray, s0: float,
                                var_levels: list[float], cvar_levels: list[float]) -> dict:
    """Terminal return = paths[:,-1]/s0 - 1. VaR and CVaR reported as positive loss values.
    Returns {var_95, var_99, cvar_95, ...}"""

def compute_loss_probabilities(paths: np.ndarray, s0: float, thresholds: list[float]) -> dict:
    """P(terminal_return < t) per threshold. Returns {threshold: probability}."""

def compute_path_drawdowns(paths: np.ndarray) -> dict:
    """Max drawdown per path via cummax. Returns:
      {mean_max_drawdown, median_max_drawdown, p5_max_drawdown, p1_max_drawdown,
       p_drawdown_exceeds_20pct}"""

def run_scenario(scenario_row: pd.Series, prices: pd.DataFrame, returns: pd.DataFrame,
                 gbm_params: pd.DataFrame, correlation_matrix: pd.DataFrame,
                 config_overrides: dict, seed: int,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """One scenario end-to-end: filter tickers → dispatch simulation → compute metrics → tag outputs.
    Returns (path_summary_df, terminal_dist_df, metrics_df, drawdown_dist_df, full_paths_or_None)."""

def save_simulations(path_summaries, terminal_distributions, mc_metrics, drawdown_distributions,
                     full_paths, summary_dict, processed_dir, exports_dir, reports_dir,
                     save_full_paths: bool) -> None:
    """Write all 4 parquet files + optional exports + summary JSON. Create dirs. Overwrite."""

def run_monte_carlo() -> dict:
    """Orchestrator. Pipeline:
      load_data → validate → estimate_gbm_params → build correlation matrix
      → for each scenario: derive seed, run_scenario → aggregate → save_simulations
    Returns {scenarios_run, tickers_simulated, total_paths_generated, scenarios_failed,
             correlation_matrix_was_non_pd, outputs_written, full_paths_saved, duration_sec}"""
```

---

## 5. Output Schemas

### `mc_paths_summary.parquet` (long format)

| Column | dtype |
|---|---|
| `scenario_name` | string |
| `ticker` | string — includes `"PORTFOLIO"` |
| `method` | string |
| `day_offset` | Int64 |
| `date` | datetime64[ns] |
| `percentile` | float64 |
| `value` | float64 |

### `mc_terminal_distribution.parquet`

| Column | dtype |
|---|---|
| `scenario_name`, `ticker`, `method` | string |
| `s0` | float64 |
| `terminal_mean`, `terminal_std`, `terminal_skew`, `terminal_kurtosis` | float64 |
| `terminal_p1` … `terminal_p99` | float64 — one per percentile |
| `return_p1` … `return_p99` | float64 |

### `mc_metrics.parquet` (long format)

| Column | dtype |
|---|---|
| `scenario_name`, `ticker`, `method` | string |
| `metric_name` | string — `var_95`, `var_99`, `cvar_95`, `prob_loss`, etc. |
| `value` | float64 |

### `mc_drawdown_distribution.parquet`

| Column | dtype |
|---|---|
| `scenario_name`, `ticker`, `method` | string |
| `mean_max_drawdown`, `median_max_drawdown`, `p5_max_drawdown`, `p1_max_drawdown` | float64 |
| `prob_drawdown_exceeds_20pct` | float64 |

### `mc_paths_full.parquet` (optional, long format)

| Column | dtype |
|---|---|
| `scenario_name`, `ticker` | string |
| `simulation_id` | Int64 |
| `day_offset` | Int64 |
| `value` | float64 |

### `monte_carlo_summary.json` — required top-level keys

`run_timestamp`, `input` (prices_rows, returns_rows, tickers, scenarios_loaded, date_range), `gbm_parameters` (per ticker: mu_annual, sigma_annual), `correlation_matrix_health` (is_positive_definite, min_eigenvalue, condition_number), `scenarios_run` (per scenario: method, n_simulations, horizon_days, tickers_simulated, portfolio_simulated, duration_sec), `method_comparison` (portfolio var_95/cvar_95 by method), `config_used`, `duration_sec`.

---

## 6. Constraints

| Constraint | Reason |
|---|---|
| No `yfinance` or network calls | Only `data_ingestion.py` touches network |
| Read returns from parquet, don't recompute | Single source of truth |
| Defensive copy of input DataFrames | No mutations |
| GBM params estimated on log returns | Mathematically correct |
| Apply Itô correction `(μ - σ²/2)` | Without it, expected returns overstated |
| `np.random.default_rng(seed)` only | Modern NumPy API |
| Per-scenario seed = `(MC_RANDOM_SEED + hash(scenario_name)) % 2**32` | Reproducible + varied |
| Eigenvalue clipping for non-PD fallback | Real corr matrices can be near-singular |
| VaR/CVaR as positive loss values | Matches Step 4 convention |
| Max drawdown as negative values | Matches Step 4 convention |
| Long-format Parquet for time-series outputs | Pipeline consistency |
| `loguru` only — no `print()` | Persistent structured logs |
| Vectorize with NumPy — no Python loops over paths | 100× speedup required |
| Parquet writes: `engine="pyarrow", index=False, compression="snappy"` | Pipeline standard |

---

## 7. Edge Cases

| Case | Behaviour |
|---|---|
| `prices_clean.parquet` or `returns_daily.parquet` missing | `FileNotFoundError` — "Run Step 2 first" |
| `mc_scenarios.csv` missing | `FileNotFoundError` with create-file hint |
| `mc_scenarios.csv` empty | `ValueError` |
| Unknown `method` | `ValueError` with row number |
| Correlation matrix non-PD | Clip eigenvalues, log warning, set flag |
| Correlation matrix has NaN | `ValueError` |
| `n_simulations` < 100 | Log warning |
| `n_simulations` > 1,000,000 | Log warning |
| Horizon > 2 years | Log warning |
| `block_size` ≥ len(historical returns) | `ValueError` |
| `block_size` = 1 | Allow, log info (equivalent to simple bootstrap) |
| Ticker σ = 0 | GBM produces deterministic path, log warning |
| Ticker has 1 return | Skip bootstrap methods, allow GBM |
| `MC_RANDOM_SEED` not set | Use entropy, log critical warning |
| All scenarios fail | Log critical, raise `RuntimeError` |
| `simulate_portfolio=true`, no `PORTFOLIO_WEIGHTS` | Equal weights, log info |
| `PORTFOLIO_WEIGHTS` not summing to 1 | `ValueError` |
| `MC_SAVE_FULL_PATHS=True`, size > 1 GB | Log critical warning before writing |
| `drift_method = "zero"` | μ = 0 regardless of historical |

---

## 8. Unit Tests (`tests/unit/test_monte_carlo.py`)

| # | Test |
|---|---|
| 1 | `test_load_data_raises_on_missing_scenarios_csv` |
| 2 | `test_estimate_gbm_params_matches_hand_calc` |
| 3 | `test_gbm_zero_volatility_produces_deterministic_path` |
| 4 | `test_gbm_mean_converges_to_expected` (N=50k, horizon=5) |
| 5 | `test_bootstrap_uses_with_replacement` |
| 6 | `test_block_bootstrap_preserves_serial_correlation` (AR(1) input) |
| 7 | `test_cholesky_on_identity_equals_independent` |
| 8 | `test_cholesky_handles_non_pd_matrix` |
| 9 | `test_var_99_geq_var_95` |
| 10 | `test_cvar_geq_var` |
| 11 | `test_path_drawdowns_all_non_positive` |
| 12 | `test_percentile_bands_monotonic` |
| 13 | `test_idempotent_rerun_with_seed` |
| 14 | `test_per_scenario_seed_produces_different_paths` |
| 15 | `test_unknown_method_raises_value_error` |
| 16 | `test_save_full_paths_flag_respected` |

Use synthetic white-noise and AR(1) fixtures in `conftest.py`.

---

## 9. Acceptance Criteria

- All 4 `mc_*.parquet` files exist with correct schemas
- `monte_carlo_summary.json` has all required keys
- `var_99 ≥ var_95` for every (scenario, ticker)
- `cvar_95 ≥ var_95` for every (scenario, ticker)
- All max drawdown values ≤ 0
- Percentile bands monotonic (P5 ≤ P25 ≤ P50 ≤ P75 ≤ P95)
- Same `MC_RANDOM_SEED` → byte-identical Parquet on rerun
- Non-PD correlation matrix projected and logged
- Method comparison present in summary JSON
- `MC_SAVE_FULL_PATHS=False` → no `mc_paths_full.parquet`
- Log file written to `logs/monte_carlo_YYYY-MM-DD.log`
- All public functions have type hints and docstrings
- No `print()` in `src/monte_carlo.py`
- No hardcoded tickers, paths, or parameters
- All 16 unit tests pass

---

## 10. `main.py` Integration

```python
from src import monte_carlo  # uncomment

# Step 6
logger.info("STEP 6/7 — Monte Carlo Simulation")
mc_summary = monte_carlo.run_monte_carlo()
logger.info(f"Monte Carlo summary: {mc_summary}")
```

---

## 11. File Deliverables

| File | Action |
|---|---|
| `src/monte_carlo.py` | Create |
| `src/schemas.py` | Add `mc_paths_summary_schema`, `mc_terminal_schema`, `mc_metrics_schema` |
| `config.py` | Add §2 variables |
| `main.py` | Uncomment Step 6 |
| `scenario_params/mc_scenarios.csv` | 3 rows minimum, one per method |
| `tests/unit/test_monte_carlo.py` | 16 tests |
| `data/exports/.gitkeep` | Create |