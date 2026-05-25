# SPEC: Portfolio Metrics Module (Multi-Asset Portfolio)

**Module:** `src/metrics.py`
**Pipeline Position:** 4 of 7
**Owner:** Quantitative Analytics Layer
**Status:** Not started
**Depends on:** `src/data_cleaning.py` (Step 2) — consumes `data/processed/prices_clean.parquet` and `data/processed/returns_daily.parquet`

---

## 1. Problem Statement

Build the fourth module of the pipeline. This module must read cleaned price and return data, compute a deterministic set of quantitative portfolio metrics, and persist them to `data/processed/` and `outputs/reports/`.

**Why this matters:**
- Metrics are the **quantitative core** of the pipeline — everything before this was data engineering, everything after (forecasting, Monte Carlo) consumes these as baselines
- Investment committees, risk managers, and LPs all consume the same standard metrics (Sharpe, Sortino, max drawdown, VaR) — computing them inconsistently across reports erodes trust
- Risk metrics drive regulatory reporting (Basel III VaR, Solvency II CVaR) — they must be computed once, correctly, with a documented methodology

**Pain point being solved:** Without a centralised metrics module, every report, forecast, and simulation would recompute Sharpe and VaR independently, often with subtly different conventions (annualization, risk-free rate, VaR sign). Centralisation enforces one definition of "the truth" per metric.

**Philosophical stance:** **One number per definition.** Each metric is computed once, with an explicit formula documented in code. No "Sharpe-style" or "modified Sharpe" variants in v1.

---

## 2. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Load `prices_clean.parquet` and `returns_daily.parquet` | Must |
| FR2 | Validate inputs against `prices_clean_schema` and `returns_schema` before computing | Must |
| FR3 | Compute per-ticker return metrics: total return, CAGR, mean daily/monthly/annual return, best/worst day, % positive days | Must |
| FR4 | Compute per-ticker risk metrics: daily vol, annualized vol, downside deviation, VaR (95%, 99%), CVaR (95%), max drawdown, drawdown duration | Must |
| FR5 | Compute per-ticker risk-adjusted metrics: Sharpe, Sortino, Calmar | Must |
| FR6 | Compute portfolio-level metrics using equal weights (default) or `PORTFOLIO_WEIGHTS` from config | Must |
| FR7 | Compute beta of each ticker and portfolio vs `BENCHMARK_TICKER` if configured | Must |
| FR8 | Compute diversification ratio (weighted avg ticker vol / portfolio vol) | Must |
| FR9 | Compute rolling Sharpe (90-day), rolling beta (60-day), rolling correlation to benchmark | Must |
| FR10 | Compute per-ticker drawdown time series (date × ticker × drawdown_pct) | Must |
| FR11 | Persist per-ticker metrics to `data/processed/metrics_per_ticker.parquet` | Must |
| FR12 | Persist portfolio-level metrics to `data/processed/portfolio_metrics.parquet` | Must |
| FR13 | Persist rolling metrics to `data/processed/rolling_metrics.parquet` | Must |
| FR14 | Persist drawdown time series to `data/processed/drawdown_series.parquet` | Must |
| FR15 | Write human-readable `outputs/reports/metrics_summary.json` | Must |
| FR16 | Log per-block computation actions to `logs/metrics_YYYY-MM-DD.log` | Must |
| FR17 | Idempotent: re-running on identical input must produce identical outputs | Must |
| FR18 | Validate that VaR ≤ CVaR (in magnitude) and Sharpe is finite before saving | Must |
| FR19 | Use risk-free rate from config; annualize correctly (√252 for vol, /252 for rate) | Must |

---

## 3. API Contracts

### 3.1 Inputs (from `config.py`)

```python
# Existing
PROCESSED_DATA_DIR: pathlib.Path      # Path("data/processed")
REPORTS_DIR: pathlib.Path             # Path("outputs/reports")
LOG_DIR: pathlib.Path                 # Path("logs")

# New for Step 4
RISK_FREE_RATE: float                 # 0.04 — annual; e.g. US 10Y T-bill yield
BENCHMARK_TICKER: str | None          # "^GSPC" for S&P 500; None to skip beta
PORTFOLIO_WEIGHTS: dict[str, float] | None  # None → equal-weighted; else must sum to 1.0
VAR_CONFIDENCE_LEVELS: list[float]    # [0.95, 0.99]
CVAR_CONFIDENCE_LEVELS: list[float]   # [0.95]
ROLLING_SHARPE_WINDOW: int            # 90 — trading days
ROLLING_BETA_WINDOW: int              # 60 — trading days
ROLLING_CORR_WINDOW: int              # 60 — trading days
METRICS_TRADING_DAYS_PER_YEAR: int    # 252 — annualization factor
EXCLUDE_BENCHMARK_FROM_PORTFOLIO: bool # True — benchmark ticker not part of portfolio aggregation
```

**Pre-condition for benchmark:** `BENCHMARK_TICKER` (if set) must be present in `TICKERS` in Step 1, so it's already in `prices_clean.parquet`. No network calls in this module.

### 3.2 Public Functions

```python
def load_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read prices_clean.parquet and returns_daily.parquet.
    Validate against pandera schemas before returning.
    Raise FileNotFoundError if either file is missing with hint to run Step 2.
    Returns (prices, returns).
    """
```

---

```python
def compute_return_metrics(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - total_return: cumprod(1 + simple_return) - 1
      - cagr: (1 + total_return) ** (trading_days_per_year / n_days) - 1
      - mean_daily_return, mean_monthly_return, mean_annual_return
      - best_day_return, worst_day_return
      - pct_positive_days

    Returns long-format DataFrame: [ticker, metric_name, value].
    """
```

---

```python
def compute_risk_metrics(
    returns: pd.DataFrame,
    trading_days_per_year: int,
    var_levels: list[float],
    cvar_levels: list[float],
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - daily_vol: returns.std()
      - annual_vol: daily_vol * sqrt(trading_days_per_year)
      - downside_deviation: std of returns where return < 0, annualized
      - var_95, var_99: historical method, returned as positive loss values
      - cvar_95: mean of returns below var_95 threshold, returned as positive
      - max_drawdown: minimum of running drawdown series (negative value)
      - max_drawdown_duration_days: longest peak-to-recovery span

    Returns long-format DataFrame: [ticker, metric_name, value].
    """
```

---

```python
def compute_drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Per ticker, compute drawdown_pct = (close / running_max_close) - 1.
    Running max is cumulative — does NOT reset on partial recovery.

    Returns long-format DataFrame: [date, ticker, close, running_peak, drawdown_pct].
    """
```

---

```python
def compute_risk_adjusted(
    return_metrics: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    risk_free_rate: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - sharpe: (annual_return - risk_free_rate) / annual_vol
      - sortino: (annual_return - risk_free_rate) / downside_deviation
      - calmar: cagr / abs(max_drawdown)

    Risk-free rate is annual. Returns must already be annualized.
    Returns long-format DataFrame: [ticker, metric_name, value].
    """
```

---

```python
def compute_portfolio_metrics(
    returns: pd.DataFrame,
    weights: dict[str, float] | None,
    benchmark_ticker: str | None,
    exclude_benchmark: bool,
    risk_free_rate: float,
    trading_days_per_year: int,
    var_levels: list[float],
    cvar_levels: list[float],
) -> pd.DataFrame:
    """
    Construct portfolio return series as weighted sum of ticker returns.
    If weights is None, use equal weights across all non-benchmark tickers.
    If benchmark_ticker and exclude_benchmark, drop benchmark from portfolio.

    Compute the same set of metrics as per-ticker, plus:
      - beta_vs_benchmark (if benchmark configured)
      - diversification_ratio = sum(w_i * vol_i) / portfolio_vol

    Returns single-row DataFrame with columns = metric names.
    """
```

---

```python
def compute_rolling_metrics(
    returns: pd.DataFrame,
    benchmark_ticker: str | None,
    risk_free_rate: float,
    trading_days_per_year: int,
    sharpe_window: int,
    beta_window: int,
    corr_window: int,
) -> pd.DataFrame:
    """
    Per ticker, compute time series of:
      - rolling_sharpe: (rolling_mean - rf_daily) / rolling_std * sqrt(252)
      - rolling_volatility: rolling_std * sqrt(252)
      - rolling_beta_vs_benchmark (if benchmark configured)
      - rolling_corr_vs_benchmark (if benchmark configured)

    NaN rows from initial window are dropped.
    Returns long-format DataFrame: [date, ticker, metric_name, value].
    """
```

---

```python
def save_metrics(
    per_ticker: pd.DataFrame,
    portfolio: pd.DataFrame,
    rolling: pd.DataFrame,
    drawdowns: pd.DataFrame,
    summary_dict: dict,
    processed_dir: Path,
    reports_dir: Path,
) -> None:
    """
    Write:
      - data/processed/metrics_per_ticker.parquet
      - data/processed/portfolio_metrics.parquet
      - data/processed/rolling_metrics.parquet
      - data/processed/drawdown_series.parquet
      - outputs/reports/metrics_summary.json (indent=2)

    Create directories if missing. Overwrite without prompting.
    """
```

---

```python
def run_metrics() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_data → validate schemas
      → compute_return_metrics → compute_risk_metrics → compute_drawdown_series
      → compute_risk_adjusted → compute_portfolio_metrics → compute_rolling_metrics
      → save_metrics

    Returns a summary dict:
      {
        "tickers_analyzed": int,
        "portfolio_sharpe": float,
        "portfolio_max_drawdown": float,
        "benchmark_used": str | None,
        "weights_strategy": "equal" | "custom",
        "outputs_written": list[str],
        "duration_sec": float
      }
    """
```

---

## 4. Output Contracts

### 4.1 `data/processed/metrics_per_ticker.parquet`

Long format. One row per (ticker, metric_name).

| Column | dtype | Notes |
|---|---|---|
| `ticker` | `string` | Uppercase |
| `metric_name` | `string` | e.g. `"sharpe"`, `"var_95"`, `"max_drawdown"` |
| `value` | `float64` | Numeric value of the metric |
| `category` | `string` | One of: `"return"`, `"risk"`, `"risk_adjusted"` |

### 4.2 `data/processed/portfolio_metrics.parquet`

Single-row DataFrame (wide format is acceptable here since it's one row).

| Column | dtype | Notes |
|---|---|---|
| `total_return` | `float64` | |
| `cagr` | `float64` | |
| `annual_vol` | `float64` | |
| `sharpe` | `float64` | |
| `sortino` | `float64` | |
| `calmar` | `float64` | |
| `max_drawdown` | `float64` | Negative value |
| `var_95`, `var_99` | `float64` | Positive loss values |
| `cvar_95` | `float64` | Positive loss value |
| `beta_vs_benchmark` | `float64` | NaN if no benchmark |
| `diversification_ratio` | `float64` | > 1 means diversification benefit |
| `weights_strategy` | `string` | `"equal"` or `"custom"` |

### 4.3 `data/processed/rolling_metrics.parquet`

Long format. One row per (date, ticker, metric_name).

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | |
| `ticker` | `string` | |
| `metric_name` | `string` | e.g. `"rolling_sharpe_90"` |
| `value` | `float64` | |

### 4.4 `data/processed/drawdown_series.parquet`

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | |
| `ticker` | `string` | |
| `close` | `float64` | |
| `running_peak` | `float64` | |
| `drawdown_pct` | `float64` | Always ≤ 0 |

### 4.5 `outputs/reports/metrics_summary.json`

```json
{
  "run_timestamp": "2026-05-24T12:15:42Z",
  "input": {
    "prices_rows": 9750,
    "returns_rows": 9743,
    "tickers": 7,
    "benchmark": "^GSPC",
    "date_range": ["2023-01-03", "2024-12-30"]
  },
  "per_ticker": {
    "AAPL": {
      "total_return": 0.342,
      "cagr": 0.156,
      "annual_vol": 0.231,
      "sharpe": 0.502,
      "sortino": 0.731,
      "calmar": 0.92,
      "max_drawdown": -0.169,
      "max_drawdown_duration_days": 47,
      "var_95": 0.024,
      "var_99": 0.041,
      "cvar_95": 0.032,
      "beta_vs_benchmark": 1.18
    }
  },
  "portfolio": {
    "weights_strategy": "equal",
    "weights": {"AAPL": 0.143, "MSFT": 0.143, "...": "..."},
    "total_return": 0.281,
    "cagr": 0.132,
    "annual_vol": 0.182,
    "sharpe": 0.508,
    "sortino": 0.742,
    "calmar": 1.04,
    "max_drawdown": -0.127,
    "var_95": 0.019,
    "cvar_95": 0.025,
    "beta_vs_benchmark": 0.96,
    "diversification_ratio": 1.24
  },
  "config_used": {
    "risk_free_rate": 0.04,
    "benchmark_ticker": "^GSPC",
    "trading_days_per_year": 252,
    "var_confidence_levels": [0.95, 0.99],
    "rolling_sharpe_window": 90,
    "rolling_beta_window": 60
  },
  "duration_sec": 2.41
}
```

---

## 5. Constraints

| Constraint | Reason |
|---|---|
| Do NOT import `yfinance` or make network calls | Only `data_ingestion.py` may touch the network |
| Do NOT recompute returns — read from `returns_daily.parquet` | Single source of truth; recomputing risks inconsistency |
| Do NOT modify input DataFrames | Defensive copy at module entry |
| Use **historical method** for VaR/CVaR in v1 (not parametric, not Monte Carlo) | Simplest, no distributional assumption; MC version is Step 6 |
| Annualize vol with `sqrt(252)`, NOT `*252` | Vol scales with square root of time |
| Convert annual risk-free rate to daily as `rf / 252` for Sharpe inputs | Match frequencies |
| VaR and CVaR reported as **positive loss values** | Industry convention; documented clearly in code |
| Max drawdown reported as **negative value** | Convention: -0.20 means 20% drawdown |
| Long-format Parquet for time-series outputs | Consistent with rest of pipeline |
| `loguru` for logging; no `print()` in `src/metrics.py` | Persistent logs, structured output |
| Idempotent: same inputs → byte-identical Parquet | Reproducibility |
| Validate inputs via pandera schemas at module entry | Catch contract violations early |
| Portfolio weights must sum to 1.0 (within 1e-6) | Defensive check; raise ValueError otherwise |
| No transaction costs, no rebalancing logic in v1 | Out of scope; portfolio assumes daily rebalancing implicitly via weight × return |

---

## 6. Edge Cases & Error Handling

| Case | Expected Behavior |
|---|---|
| `prices_clean.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `returns_daily.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `BENCHMARK_TICKER` set but not in TICKERS / prices | Log warning, set beta-related metrics to NaN, continue |
| `BENCHMARK_TICKER` is None | Skip beta and rolling beta entirely; no error |
| `PORTFOLIO_WEIGHTS` doesn't sum to 1.0 | Raise `ValueError` with the actual sum |
| `PORTFOLIO_WEIGHTS` has tickers not in dataset | Raise `ValueError` listing missing tickers |
| `PORTFOLIO_WEIGHTS` missing tickers from dataset | Raise `ValueError` listing extra tickers |
| All returns zero for a ticker (flat-lined) | Sharpe/Sortino = NaN (division by zero); log warning, do not crash |
| Ticker has fewer rows than `ROLLING_SHARPE_WINDOW` | Skip rolling metrics for that ticker, log warning |
| `RISK_FREE_RATE` < 0 | Allowed (some currencies have negative rates); log info |
| `RISK_FREE_RATE` > 0.5 | Log warning (likely config error) |
| Max drawdown ticker that never recovered to peak | Duration = days from peak to end of data; flag `"recovered": false` in summary |
| Single ticker in dataset | Portfolio metrics = ticker metrics; diversification_ratio = 1.0 |
| Returns DataFrame has NaN values | Raise `ValueError` — should have been cleaned in Step 2 |
| `VAR_CONFIDENCE_LEVELS` contains value ≤ 0 or ≥ 1 | Raise `ValueError` |
| Existing files in `data/processed/` from prior metrics run | Overwrite without prompting |
| Beta computation with constant benchmark returns | Returns NaN, log warning |

---

## 7. Acceptance Criteria

The module is considered complete when **all** the following are true:

- [ ] `python main.py` executes `run_metrics()` as Step 4 after Step 3
- [ ] `data/processed/metrics_per_ticker.parquet` exists with all expected metrics per ticker
- [ ] `data/processed/portfolio_metrics.parquet` exists with a single row of portfolio metrics
- [ ] `data/processed/rolling_metrics.parquet` exists with date × ticker × metric long format
- [ ] `data/processed/drawdown_series.parquet` exists with running peak and drawdown_pct
- [ ] `outputs/reports/metrics_summary.json` exists with all fields from §4.5
- [ ] Sharpe values are finite for all tickers with non-zero volatility
- [ ] VaR_95 ≤ VaR_99 (in magnitude) for every ticker — invariant check
- [ ] CVaR_95 ≥ VaR_95 for every ticker — invariant check
- [ ] Max drawdown values are ≤ 0 for every ticker
- [ ] Portfolio Sharpe is reasonable (typically between -1 and 3) — flag in log if outside
- [ ] Re-running `main.py` produces byte-identical Parquet files (idempotent)
- [ ] Setting `BENCHMARK_TICKER = None` produces all NaN beta columns without crashing
- [ ] Setting custom `PORTFOLIO_WEIGHTS` that don't sum to 1.0 raises `ValueError`
- [ ] `logs/metrics_YYYY-MM-DD.log` shows per-block computation actions
- [ ] All public functions have type hints and docstrings
- [ ] No hardcoded thresholds, tickers, weights, or paths in `src/metrics.py`
- [ ] At least 8 unit tests in `tests/unit/test_metrics.py` pass
- [ ] No `print()` calls in `src/metrics.py`

---

## 8. File Deliverables

| File | Purpose |
|---|---|
| `src/metrics.py` | Module implementation |
| `src/schemas.py` | Add `metrics_per_ticker_schema`, `rolling_metrics_schema`, `drawdown_schema` |
| `config.py` | Add the new variables listed in §3.1 |
| `main.py` | Uncomment & integrate Step 4 |
| `tests/unit/test_metrics.py` | Unit tests (see §10) |
| `requirements.txt` | Confirm `scipy` is present (used for some statistical functions) |

---

## 9. `main.py` Integration

```python
# ── Step 4: Metrics ───────────────────────────────────────────────────
logger.info("STEP 4/7 — Portfolio Metrics")
metrics_summary = metrics.run_metrics()
logger.info(f"Metrics summary: {metrics_summary}")
```

Uncomment the `from src import metrics` import at the top.

---

## 10. Unit Test Plan

`tests/unit/test_metrics.py` — minimum tests:

| # | Test | What it proves |
|---|---|---|
| 1 | `test_load_data_raises_on_missing_file` | FileNotFoundError raised with hint when parquet missing |
| 2 | `test_compute_return_metrics_cagr_matches_hand_calc` | CAGR formula correct on synthetic constant-return series |
| 3 | `test_compute_risk_metrics_var_is_positive` | VaR returned as positive loss value |
| 4 | `test_compute_risk_metrics_cvar_geq_var` | CVaR always ≥ VaR in magnitude (invariant) |
| 5 | `test_compute_drawdown_running_peak_monotonic` | Running peak is non-decreasing over time |
| 6 | `test_compute_drawdown_all_negative_or_zero` | Drawdown_pct values are always ≤ 0 |
| 7 | `test_sharpe_zero_vol_returns_nan` | Flat-lined ticker → Sharpe NaN, not crash |
| 8 | `test_portfolio_weights_must_sum_to_one` | ValueError raised when weights don't sum to 1.0 |
| 9 | `test_portfolio_equal_weights_default` | None weights → equal weights across tickers |
| 10 | `test_diversification_ratio_geq_one` | Diversification ratio ≥ 1 for uncorrelated assets |
| 11 | `test_rolling_sharpe_window_drops_initial_nans` | First `window-1` rows excluded from rolling output |
| 12 | `test_idempotent_rerun` | Running metrics twice produces identical Parquet output |
| 13 | `test_no_modifications_to_input_dataframes` | Input prices/returns DataFrames unchanged after run |
| 14 | `test_beta_skipped_when_benchmark_none` | BENCHMARK_TICKER=None → beta columns all NaN, no crash |

Use small synthetic DataFrames as fixtures in `conftest.py`. Hand-compute expected values for at least 2 tests so the math is verified end-to-end.

---

## 11. Manual Test Plan (After Implementation)

```bash
# 1. Fresh run after Step 3
rm -f data/processed/metrics_*.parquet data/processed/portfolio_metrics.parquet
rm -f data/processed/rolling_metrics.parquet data/processed/drawdown_series.parquet
rm -f outputs/reports/metrics_summary.json
python main.py

# 2. Verify outputs exist
ls data/processed/ | grep -E "(metrics|drawdown)"
# Expected: metrics_per_ticker.parquet, portfolio_metrics.parquet,
#           rolling_metrics.parquet, drawdown_series.parquet

ls outputs/reports/
# Expected: metrics_summary.json

# 3. Inspect per-ticker metrics
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/metrics_per_ticker.parquet')
print('Shape:', df.shape)
print(df.pivot(index='ticker', columns='metric_name', values='value').round(4))
"

# 4. Verify invariants
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/metrics_per_ticker.parquet')
p = df.pivot(index='ticker', columns='metric_name', values='value')
assert (p['var_95'] <= p['var_99']).all(), 'VaR_95 must be <= VaR_99'
assert (p['cvar_95'] >= p['var_95']).all(), 'CVaR_95 must be >= VaR_95'
assert (p['max_drawdown'] <= 0).all(), 'Max drawdown must be <= 0'
print('All invariants pass ✓')
"

# 5. Inspect portfolio metrics
python -c "
import pandas as pd
print(pd.read_parquet('data/processed/portfolio_metrics.parquet').T)
"

# 6. Inspect rolling metrics (sample)
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/rolling_metrics.parquet')
print(df.head(20))
print('Unique metrics:', df['metric_name'].unique())
"

# 7. View summary
cat outputs/reports/metrics_summary.json | python -m json.tool | head -60

# 8. Idempotency — checksum before and after rerun should match
md5sum data/processed/metrics_per_ticker.parquet
python main.py
md5sum data/processed/metrics_per_ticker.parquet  # must match

# 9. Resilience — set BENCHMARK_TICKER=None in config, rerun
# Confirm beta-related fields are NaN but pipeline completes
```

---

## 12. Out of Scope (Explicit Non-Goals)

- ❌ Parametric VaR (assumes normality — wrong for fat-tailed returns)
- ❌ Monte Carlo VaR (overlaps with Step 6 simulation)
- ❌ Portfolio optimization (mean-variance, risk parity, Black-Litterman) — v2
- ❌ Factor models (Fama-French, CAPM decomposition) — v2
- ❌ Transaction cost modeling — out of scope
- ❌ Tax-adjusted returns — out of scope
- ❌ Multi-currency portfolios (FX-adjusted returns) — out of scope
- ❌ Attribution analysis (sector / style attribution) — v2
- ❌ Backtest engine — separate concern, not a "metric"
- ❌ Plotting (Step 7 export handles visualization)
- ❌ Information ratio, tracking error — could be added in v2 if benchmark expanded
- ❌ Skewness / kurtosis (already in Step 3 EDA — not duplicated here)

---

## 13. Notes for Claude Code

- Implement functions in the order listed in §3.2 — each is independently testable
- After each function, write its unit test before moving to the next (TDD-lite)
- Hand-compute expected values for 2 tests minimum to verify formulas end-to-end
- For VaR: `np.quantile(returns, 1 - confidence_level)` then take absolute value
- For CVaR: mean of returns ≤ VaR threshold, then take absolute value
- For max drawdown: use `(price / price.cummax() - 1).min()`
- For Sharpe: `(annual_return - risk_free_rate) / annual_vol` — both numerator and denominator must be annual
- For rolling Sharpe: convert risk-free rate to daily (`rf / 252`) when working with daily returns
- For beta: `cov(asset_returns, benchmark_returns) / var(benchmark_returns)`
- For Parquet writes use `df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")`
- Use `loguru.logger` everywhere; no `print()`
- Keep `run_metrics()` thin — it should read like a table of contents
- Document every formula in the function docstring — Sharpe especially has 3+ "industry standard" variants; pick one and state it
- If `BENCHMARK_TICKER` is in `TICKERS`, exclude it from portfolio aggregation when `EXCLUDE_BENCHMARK_FROM_PORTFOLIO=True`
- Defensive copy input DataFrames at function entry (`df = df.copy()`) — never mutate
- Validate weights sum to 1.0 with `math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)`
- If something seems missing from this spec, flag it before adding silently
- Do NOT add metrics not listed here (Treynor ratio, Information ratio, Omega ratio, etc.) — flag as scope discussion items first