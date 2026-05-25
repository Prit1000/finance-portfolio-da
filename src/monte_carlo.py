"""
src/monte_carlo.py — Monte Carlo Simulation Module (Step 6 of 7).

Simulates probabilistic price/return paths per ticker and portfolio using
GBM, historical bootstrap, and block bootstrap methods. Outputs distributional
risk metrics (VaR, CVaR, drawdown) to data/processed/ and outputs/reports/.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

import config
from src.schemas import prices_clean_schema, returns_schema


# ── Private helpers ───────────────────────────────────────────────────────────

def _threshold_to_metric_name(t: float) -> str:
    """Convert a return threshold to a human-readable metric name."""
    pct = int(round(t * 100))
    if pct < 0:
        return f"prob_loss_{abs(pct)}pct"
    elif pct == 0:
        return "prob_loss"
    else:
        return f"prob_gain_lt_{pct}pct"


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _make_positive_definite(corr: np.ndarray) -> np.ndarray:
    """Project to nearest PD correlation matrix via eigenvalue clipping."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_pd))
    return corr_pd / np.outer(d, d)


def _resolve_portfolio_weights(
    tickers: list[str],
    config_weights: dict[str, float] | None,
) -> dict[str, float]:
    if config_weights is None:
        logger.info("No PORTFOLIO_WEIGHTS configured — using equal weights.")
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}
    weight_sum = sum(config_weights.get(t, 0.0) for t in tickers)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"PORTFOLIO_WEIGHTS don't sum to 1.0 for tickers {tickers} (sum={weight_sum:.6f})."
        )
    return {t: config_weights.get(t, 0.0) for t in tickers}


_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")


def _build_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    # Returns a (n_tickers × n_tickers) correlation matrix — wide format by necessity.
    # Never persisted; sliced to numpy immediately at every use site.
    wide = returns.pivot(index="date", columns="ticker", values="log_return")
    return wide.corr()


def _stable_scenario_seed(base_seed: int, scenario_name: str) -> int:
    """Derive a per-scenario RNG seed stable across Python processes.

    Uses hashlib.md5 instead of built-in hash() to avoid PYTHONHASHSEED
    randomisation, satisfying FR22 (same MC_RANDOM_SEED → identical outputs).
    """
    name_hash = int(hashlib.md5(scenario_name.encode()).hexdigest(), 16)
    return (base_seed + name_hash) % (2**32)


def _concat_dfs(lst: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames, skipping empty ones."""
    valid = [df for df in lst if not df.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def _compute_method_comparison(mc_metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Compare portfolio VaR/CVaR across simulation methods for the summary JSON."""
    result: dict[str, Any] = {}
    if mc_metrics_df.empty or "PORTFOLIO" not in mc_metrics_df["ticker"].unique():
        return result

    port = mc_metrics_df[mc_metrics_df["ticker"] == "PORTFOLIO"]
    for metric in ["var_95", "cvar_95"]:
        sub = port[port["metric_name"] == metric]
        if not sub.empty:
            result[f"portfolio_{metric}"] = sub.set_index("scenario_name")["value"].to_dict()

    if "portfolio_var_95" in result:
        vals = result["portfolio_var_95"]
        gbm_v = vals.get(next((k for k in vals if "gbm" in k), ""), None)
        boot_v = vals.get(
            next((k for k in vals if "bootstrap" in k and "block" not in k), ""), None
        )
        if gbm_v and boot_v and gbm_v > 0:
            pct_diff = (boot_v - gbm_v) / gbm_v * 100
            result["interpretation"] = (
                f"Bootstrap VaR {'>' if boot_v > gbm_v else '<='} GBM VaR "
                f"by {abs(pct_diff):.1f}% — "
                f"{'suggests fat tails in historical returns' if boot_v > gbm_v else 'GBM captures tail risk adequately'}."
            )

    return result


def _parse_bool_field(value: Any, default: bool) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _validate_var_cvar_invariants(mc_metrics: pd.DataFrame) -> None:
    """Log errors (without raising) for any violated VaR/CVaR invariants."""
    if mc_metrics.empty:
        return

    def _pivot(name: str) -> pd.DataFrame:
        return mc_metrics[mc_metrics["metric_name"] == name][
            ["scenario_name", "ticker", "value"]
        ].set_index(["scenario_name", "ticker"])

    var95 = _pivot("var_95")
    var99 = _pivot("var_99")
    cvar95 = _pivot("cvar_95")

    if not var95.empty and not var99.empty:
        merged = var95.join(var99, lsuffix="_95", rsuffix="_99", how="inner")
        violations = merged[merged["value_99"] < merged["value_95"]]
        if not violations.empty:
            logger.error(f"VaR invariant violated (var_99 < var_95) for: {violations.index.tolist()}")

    if not var95.empty and not cvar95.empty:
        merged = var95.join(cvar95, lsuffix="_var", rsuffix="_cvar", how="inner")
        violations = merged[merged["value_cvar"] < merged["value_var"]]
        if not violations.empty:
            logger.error(f"CVaR invariant violated (cvar_95 < var_95) for: {violations.index.tolist()}")


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(
    processed_dir: Path,
    scenarios_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read prices_clean.parquet, returns_daily.parquet, and mc_scenarios.csv.

    Validates pandera schemas and scenario column presence.

    Returns:
        (prices, returns, scenarios) as DataFrames (defensive copies).

    Raises:
        FileNotFoundError: if any required file is missing.
        ValueError: if scenarios CSV is empty or contains unknown methods.
    """
    prices_path = processed_dir / "prices_clean.parquet"
    returns_path = processed_dir / "returns_daily.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(f"{prices_path} not found. Run Step 2 first.")
    if not returns_path.exists():
        raise FileNotFoundError(f"{returns_path} not found. Run Step 2 first.")
    if not scenarios_csv.exists():
        raise FileNotFoundError(
            f"{scenarios_csv} not found. "
            "Create scenario_params/mc_scenarios.csv with columns: "
            "scenario_name, method, horizon_days, n_simulations."
        )

    prices = pd.read_parquet(prices_path)
    returns = pd.read_parquet(returns_path)
    scenarios = pd.read_csv(scenarios_csv)

    prices_clean_schema.validate(prices)
    returns_schema.validate(returns)

    required_cols = {"scenario_name", "method", "horizon_days", "n_simulations"}
    missing = required_cols - set(scenarios.columns)
    if missing:
        raise ValueError(f"mc_scenarios.csv missing required columns: {missing}")

    if len(scenarios) == 0:
        raise ValueError("mc_scenarios.csv has no rows — nothing to simulate.")

    valid_methods = {"gbm", "bootstrap", "block_bootstrap"}
    invalid = scenarios[~scenarios["method"].isin(valid_methods)]
    if not invalid.empty:
        raise ValueError(
            f"Unknown method(s) {invalid['method'].tolist()} in mc_scenarios.csv "
            f"rows {invalid.index.tolist()}. Valid methods: {valid_methods}"
        )

    if "drift_method" in scenarios.columns:
        valid_drift = {"historical", "zero"}
        bad_drift = scenarios["drift_method"].dropna()
        bad_drift = bad_drift[~bad_drift.isin(valid_drift)]
        if not bad_drift.empty:
            raise ValueError(
                f"Unknown drift_method value(s) {bad_drift.tolist()} in mc_scenarios.csv. "
                f"Valid values: {valid_drift}"
            )

    return prices.copy(), returns.copy(), scenarios.copy()


def estimate_gbm_params(
    returns: pd.DataFrame,
    trading_days_per_year: int,
    drift_method: str,
) -> pd.DataFrame:
    """Estimate GBM parameters per ticker from log returns.

    Formula:
        mu_daily    = mean(log_return)  if drift_method == "historical" else 0.0
        sigma_daily = std(log_return)   (sample std, ddof=1)
        mu_annual   = mu_daily * trading_days_per_year
        sigma_annual= sigma_daily * sqrt(trading_days_per_year)

    Returns:
        DataFrame with columns [ticker, mu_daily, sigma_daily, mu_annual,
        sigma_annual, n_observations].
    """
    rows = []
    for ticker, grp in returns.groupby("ticker"):
        log_ret = grp["log_return"].dropna().values
        n_obs = len(log_ret)

        mu_daily = float(np.mean(log_ret)) if drift_method == "historical" else 0.0
        sigma_daily = float(np.std(log_ret, ddof=1)) if n_obs > 1 else 0.0

        if sigma_daily == 0.0:
            logger.warning(
                f"{ticker}: zero variance in returns — GBM will produce a deterministic path."
            )

        rows.append({
            "ticker": ticker,
            "mu_daily": mu_daily,
            "sigma_daily": sigma_daily,
            "mu_annual": mu_daily * trading_days_per_year,
            "sigma_annual": sigma_daily * np.sqrt(trading_days_per_year),
            "n_observations": n_obs,
        })

    return pd.DataFrame(rows)


def simulate_gbm(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    horizon: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Geometric Brownian Motion simulation with Itô correction.

    Formula: S(t+1) = S(t) * exp((mu - sigma²/2) + sigma * Z)
    dt=1 day is baked into daily parameters.

    Returns:
        Array of shape (n_simulations, horizon + 1); column 0 = s0.
    """
    Z = rng.standard_normal((n_simulations, horizon))
    # Itô correction: drift term is (mu - 0.5 * sigma**2) to preserve expected return
    log_returns = (mu_daily - 0.5 * sigma_daily ** 2) + sigma_daily * Z
    cum_log = np.concatenate(
        [np.zeros((n_simulations, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return s0 * np.exp(cum_log)


def simulate_bootstrap(
    s0: float,
    historical_returns: np.ndarray,
    horizon: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Historical bootstrap: resample simple returns with replacement.

    Compounds resampled returns to construct price paths starting at s0.

    Returns:
        Array of shape (n_simulations, horizon + 1).
    """
    if len(historical_returns) <= 1:
        logger.warning("Only 1 historical return available — bootstrap returns flat paths.")
        return np.full((n_simulations, horizon + 1), s0)

    sampled = rng.choice(historical_returns, size=(n_simulations, horizon), replace=True)
    cum_returns = np.concatenate(
        [np.ones((n_simulations, 1)), np.cumprod(1.0 + sampled, axis=1)], axis=1
    )
    return s0 * cum_returns


def simulate_block_bootstrap(
    s0: float,
    historical_returns: np.ndarray,
    horizon: int,
    n_simulations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap: resample blocks of consecutive returns (circular).

    Circular indexing wraps at the series end to handle edge effects.
    Vectorised via numpy broadcasting — no Python loops over simulations.

    Returns:
        Array of shape (n_simulations, horizon + 1).

    Raises:
        ValueError: if block_size >= len(historical_returns).
    """
    n_hist = len(historical_returns)
    if block_size >= n_hist:
        raise ValueError(
            f"block_size ({block_size}) must be less than the length of "
            f"historical returns ({n_hist})."
        )
    if block_size == 1:
        logger.info("block_size=1 is equivalent to simple bootstrap.")

    n_blocks = int(np.ceil(horizon / block_size))
    # block_starts: (n_simulations, n_blocks)
    block_starts = rng.integers(0, n_hist, size=(n_simulations, n_blocks))

    # indices: (n_simulations, n_blocks, block_size) via broadcasting
    offsets = np.arange(block_size)
    indices = (block_starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % n_hist
    # block_returns: (n_simulations, n_blocks, block_size) → flatten → trim
    sampled = historical_returns[indices].reshape(n_simulations, -1)[:, :horizon]

    cum_returns = np.concatenate(
        [np.ones((n_simulations, 1)), np.cumprod(1.0 + sampled, axis=1)], axis=1
    )
    return s0 * cum_returns


def simulate_portfolio_correlated(
    initial_prices: dict[str, float],
    gbm_params: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    weights: dict[str, float],
    horizon: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Correlated GBM simulation via Cholesky decomposition.

    For each step: correlated_Z = L @ iid_Z, where L = chol(correlation_matrix).
    If correlation_matrix is not positive-definite, clips negative eigenvalues
    to 1e-8, reconstructs, and logs a warning.

    Returns:
        per_ticker_paths: {ticker: ndarray of shape (n_simulations, horizon + 1)}
        portfolio_paths:  ndarray of shape (n_simulations, horizon + 1);
                          portfolio value is normalised to start at 1.0.

    Raises:
        ValueError: if correlation_matrix contains NaN.
    """
    tickers = list(initial_prices.keys())

    corr = correlation_matrix.loc[tickers, tickers].values.astype(float)

    if np.any(np.isnan(corr)):
        raise ValueError("Correlation matrix contains NaN values — check cleaned returns.")

    if not _is_positive_definite(corr):
        logger.warning(
            "Correlation matrix is not positive-definite. "
            "Projecting to nearest PD via eigenvalue clipping."
        )
        corr = _make_positive_definite(corr)

    L = np.linalg.cholesky(corr)

    params_idx = gbm_params.set_index("ticker")
    mu = np.array([params_idx.loc[t, "mu_daily"] for t in tickers])
    sigma = np.array([params_idx.loc[t, "sigma_daily"] for t in tickers])
    s0_arr = np.array([initial_prices[t] for t in tickers])
    w_arr = np.array([weights[t] for t in tickers])

    n_tickers = len(tickers)
    # Z_iid: (n_tickers, n_simulations, horizon)
    Z_iid = rng.standard_normal((n_tickers, n_simulations, horizon))
    # Correlate along ticker axis: L @ Z_iid for each (sim, step)
    corr_Z = np.einsum("ij,jst->ist", L, Z_iid)  # (n_tickers, n_simulations, horizon)

    # GBM per ticker with Itô correction
    drift = mu - 0.5 * sigma ** 2  # (n_tickers,)
    log_returns = (
        drift[:, np.newaxis, np.newaxis] + sigma[:, np.newaxis, np.newaxis] * corr_Z
    )
    zeros = np.zeros((n_tickers, n_simulations, 1))
    cum_log = np.concatenate([zeros, np.cumsum(log_returns, axis=2)], axis=2)
    # all_paths: (n_tickers, n_simulations, horizon + 1)
    all_paths = s0_arr[:, np.newaxis, np.newaxis] * np.exp(cum_log)

    per_ticker_paths = {t: all_paths[i] for i, t in enumerate(tickers)}

    # Portfolio: weighted sum of normalised per-ticker paths; starts at 1.0
    normalized = all_paths / s0_arr[:, np.newaxis, np.newaxis]
    portfolio_paths = np.einsum("i,ist->st", w_arr, normalized)

    return per_ticker_paths, portfolio_paths


def compute_path_summary(
    paths: np.ndarray,
    percentiles: list[float],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute per-day percentiles across simulation paths.

    Args:
        paths:      ndarray (n_simulations, horizon + 1)
        percentiles: list of percentile values, e.g. [1, 5, 25, 50, 75, 95, 99]
        dates:      DatetimeIndex of length horizon + 1 (day 0 = simulation start)

    Returns:
        Long-format DataFrame: [day_offset, date, percentile, value].
    """
    # pct_values: (n_percentiles, n_days)
    pct_values = np.percentile(paths, q=percentiles, axis=0)
    n_pct, n_days = pct_values.shape

    # Vectorised construction via meshgrid
    pct_idx_grid, day_grid = np.meshgrid(np.arange(n_pct), np.arange(n_days), indexing="ij")

    return pd.DataFrame({
        "day_offset": pd.array(day_grid.ravel(), dtype=pd.Int64Dtype()),
        "date": pd.DatetimeIndex(dates[day_grid.ravel()]),
        "percentile": [float(percentiles[p]) for p in pct_idx_grid.ravel()],
        "value": pct_values.ravel(),
    })


def compute_terminal_distribution(
    paths: np.ndarray,
    s0: float,
    percentiles: list[float],
) -> dict:
    """Compute distributional statistics of terminal values (paths[:, -1]).

    Returns:
        {
          "terminal_mean": float,
          "terminal_std":  float,
          "terminal_skew": float,
          "terminal_kurtosis": float,
          "percentiles": {p: terminal_value},
          "return_percentiles": {p: (terminal_value / s0 - 1)}
        }
    """
    terminal = paths[:, -1]
    terminal_returns = terminal / s0 - 1.0

    return {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)),
        "terminal_skew": float(stats.skew(terminal)),
        "terminal_kurtosis": float(stats.kurtosis(terminal)),
        "percentiles": {p: float(np.percentile(terminal, p)) for p in percentiles},
        "return_percentiles": {p: float(np.percentile(terminal_returns, p)) for p in percentiles},
    }


def compute_simulated_var_cvar(
    paths: np.ndarray,
    s0: float,
    var_levels: list[float],
    cvar_levels: list[float],
) -> dict:
    """Compute VaR and CVaR from the terminal return distribution.

    VaR convention: positive loss value (e.g. 0.05 = 5% loss).
    CVaR is the mean of returns strictly in the tail beyond VaR — also positive.

    Returns:
        {"var_95": float, "var_99": float, "cvar_95": float, ...}
    """
    terminal_returns = paths[:, -1] / s0 - 1.0
    result: dict[str, float] = {}

    for level in var_levels:
        var = float(-np.percentile(terminal_returns, (1.0 - level) * 100.0))
        result[f"var_{int(round(level * 100))}"] = var

    for level in cvar_levels:
        threshold = np.percentile(terminal_returns, (1.0 - level) * 100.0)
        tail = terminal_returns[terminal_returns <= threshold]
        cvar = float(-np.mean(tail)) if len(tail) > 0 else 0.0
        result[f"cvar_{int(round(level * 100))}"] = cvar

    return result


def compute_loss_probabilities(
    paths: np.ndarray,
    s0: float,
    thresholds: list[float],
) -> dict:
    """Compute P(terminal_return < t) for each threshold t.

    Negative threshold: probability of loss of |t| or more.
    Zero threshold:     probability of any loss.
    Positive threshold: probability that gain is less than t.

    Returns:
        {metric_name: probability}  where keys are human-readable (see _threshold_to_metric_name).
    """
    terminal_returns = paths[:, -1] / s0 - 1.0
    return {
        _threshold_to_metric_name(t): float(np.mean(terminal_returns < t))
        for t in thresholds
    }


def compute_path_drawdowns(paths: np.ndarray) -> dict:
    """Compute max drawdown distribution across simulation paths.

    Uses vectorised cumulative-maximum: drawdown[t] = price[t] / cummax[t] - 1.
    Max drawdown is the minimum drawdown per path — always ≤ 0.

    Returns:
        {
          "mean_max_drawdown":            float (≤ 0),
          "median_max_drawdown":          float (≤ 0),
          "p5_max_drawdown":              float (worst 5% of paths),
          "p1_max_drawdown":              float (worst 1% of paths),
          "prob_drawdown_exceeds_20pct":  float (probability max drawdown < -0.20),
        }
    """
    cum_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / cum_max - 1.0
    max_drawdowns = np.min(drawdowns, axis=1)  # most negative value per path

    return {
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p5_max_drawdown": float(np.percentile(max_drawdowns, 5)),
        "p1_max_drawdown": float(np.percentile(max_drawdowns, 1)),
        "prob_drawdown_exceeds_20pct": float(np.mean(max_drawdowns < -0.20)),
    }


def run_scenario(
    scenario_row: pd.Series,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    gbm_params: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    config_overrides: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict | None]:
    """Execute one scenario row end-to-end.

    Pipeline per scenario:
        filter tickers → dispatch simulation → compute metrics → tag outputs.
    If simulate_portfolio=True, also runs portfolio simulation.

    Args:
        scenario_row:     one row from mc_scenarios.csv as a pd.Series
        config_overrides: dict of config values (MC_PERCENTILES, etc.)
        seed:             per-scenario reproducible seed

    Returns:
        (path_summary_df, terminal_dist_df, metrics_df, drawdown_dist_df,
         full_paths_dict | None)
        full_paths_dict: {ticker: paths_array} when MC_SAVE_FULL_PATHS=True, else None.
    """
    scenario_name = str(scenario_row["scenario_name"])
    method = str(scenario_row["method"])
    horizon = int(scenario_row["horizon_days"])
    n_sim = int(scenario_row["n_simulations"])

    block_size_raw = scenario_row.get("block_size", np.nan)
    block_size = (
        int(block_size_raw)
        if pd.notna(block_size_raw)
        else config_overrides["MC_DEFAULT_BLOCK_SIZE"]
    )

    drift_override = scenario_row.get("drift_method", np.nan)
    drift_method = (
        str(drift_override)
        if pd.notna(drift_override)
        else config_overrides["MC_DRIFT_METHOD"]
    )

    tickers_field = scenario_row.get("tickers", "all")
    if pd.isna(tickers_field):
        tickers_field = "all"
    simulate_portfolio = _parse_bool_field(
        scenario_row.get("simulate_portfolio", False), default=False
    )

    all_tickers = sorted(prices["ticker"].unique().tolist())
    tickers: list[str] = (
        all_tickers
        if str(tickers_field).strip() == "all"
        else [t.strip() for t in str(tickers_field).split(",")]
    )
    for t in tickers:
        if not _TICKER_RE.fullmatch(t):
            raise ValueError(
                f"[{scenario_name}] Invalid ticker '{t}' in mc_scenarios.csv — "
                "expected uppercase letters, digits, '.', or '-', max 15 chars."
            )

    if n_sim < 100:
        logger.warning(f"[{scenario_name}] n_simulations={n_sim} < 100 — estimates will be noisy.")
    if n_sim > 1_000_000:
        logger.warning(
            f"[{scenario_name}] n_simulations={n_sim} > 1,000,000 — memory/disk implications."
        )
    trading_days_per_year = config_overrides["MC_TRADING_DAYS_PER_YEAR"]
    if horizon > 2 * trading_days_per_year:
        logger.warning(f"[{scenario_name}] horizon={horizon} > 2 years — extrapolation risk.")

    rng = np.random.default_rng(seed)

    last_prices_s = prices.sort_values("date").groupby("ticker")["close"].last()
    last_date = prices["date"].max()
    # future_dates[0] = last_date (day_offset=0 = s0); [horizon] = end of simulation
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)

    all_path_summaries: list[pd.DataFrame] = []
    all_terminal_dists: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_drawdowns: list[pd.DataFrame] = []
    per_ticker_paths_store: dict[str, np.ndarray] = {}

    save_full = config_overrides["MC_SAVE_FULL_PATHS"]
    full_paths_dict: dict[str, np.ndarray] | None = {} if save_full else None

    percentiles = config_overrides["MC_PERCENTILES"]
    var_levels = config_overrides["MC_VAR_LEVELS"]
    cvar_levels = config_overrides["MC_CVAR_LEVELS"]
    prob_thresholds = config_overrides["MC_PROBABILITY_THRESHOLDS"]

    for ticker in tickers:
        if ticker not in last_prices_s.index:
            logger.warning(f"[{scenario_name}] {ticker} not in prices — skipping.")
            continue

        s0 = float(last_prices_s[ticker])
        ticker_simple = returns[returns["ticker"] == ticker]["simple_return"].dropna().values

        logger.info(
            f"[{scenario_name}] Simulating {ticker} via {method} "
            f"(n_sim={n_sim}, horizon={horizon}, drift={drift_method})"
        )

        if method == "gbm":
            params_row = gbm_params[gbm_params["ticker"] == ticker].iloc[0]
            mu_d = float(params_row["mu_daily"]) if drift_method == "historical" else 0.0
            paths = simulate_gbm(s0, mu_d, float(params_row["sigma_daily"]), horizon, n_sim, rng)
        elif method == "bootstrap":
            if len(ticker_simple) <= 1:
                logger.warning(
                    f"[{scenario_name}] {ticker} has only {len(ticker_simple)} return(s) "
                    "— skipping bootstrap."
                )
                continue
            paths = simulate_bootstrap(s0, ticker_simple, horizon, n_sim, rng)
        else:  # block_bootstrap
            if len(ticker_simple) <= 1:
                logger.warning(
                    f"[{scenario_name}] {ticker} has only {len(ticker_simple)} return(s) "
                    "— skipping block_bootstrap."
                )
                continue
            paths = simulate_block_bootstrap(s0, ticker_simple, horizon, n_sim, block_size, rng)

        per_ticker_paths_store[ticker] = paths

        summary = compute_path_summary(paths, percentiles, future_dates)
        summary["scenario_name"] = scenario_name
        summary["ticker"] = ticker
        summary["method"] = method
        all_path_summaries.append(summary)

        td = compute_terminal_distribution(paths, s0, percentiles)
        terminal_row: dict[str, Any] = {
            "scenario_name": scenario_name,
            "ticker": ticker,
            "method": method,
            "s0": s0,
            "terminal_mean": td["terminal_mean"],
            "terminal_std": td["terminal_std"],
            "terminal_skew": td["terminal_skew"],
            "terminal_kurtosis": td["terminal_kurtosis"],
        }
        for p, v in td["percentiles"].items():
            terminal_row[f"terminal_p{int(p)}"] = v
        for p, v in td["return_percentiles"].items():
            terminal_row[f"return_p{int(p)}"] = v
        all_terminal_dists.append(pd.DataFrame([terminal_row]))

        var_cvar = compute_simulated_var_cvar(paths, s0, var_levels, cvar_levels)
        loss_probs = compute_loss_probabilities(paths, s0, prob_thresholds)
        all_metrics.append(pd.DataFrame([
            {
                "scenario_name": scenario_name,
                "ticker": ticker,
                "method": method,
                "metric_name": k,
                "value": float(v),
            }
            for k, v in {**var_cvar, **loss_probs}.items()
        ]))

        dd = compute_path_drawdowns(paths)
        all_drawdowns.append(pd.DataFrame([{
            "scenario_name": scenario_name,
            "ticker": ticker,
            "method": method,
            **dd,
        }]))

        if save_full:
            full_paths_dict[ticker] = paths  # type: ignore[index]

    # ── Portfolio simulation ──────────────────────────────────────────────────
    if simulate_portfolio and len(per_ticker_paths_store) >= 2:
        portfolio_tickers = list(per_ticker_paths_store.keys())
        s0_dict = {t: float(last_prices_s[t]) for t in portfolio_tickers}
        weights = _resolve_portfolio_weights(
            portfolio_tickers, config_overrides.get("PORTFOLIO_WEIGHTS")
        )

        logger.info(f"[{scenario_name}] Simulating PORTFOLIO via {method}.")

        if method == "gbm" and config_overrides.get("MC_USE_CORRELATION", True):
            corr_sub = correlation_matrix.loc[portfolio_tickers, portfolio_tickers]
            gbm_sub = gbm_params[gbm_params["ticker"].isin(portfolio_tickers)].copy()
            if drift_method == "zero":
                gbm_sub["mu_daily"] = 0.0
            _, portfolio_paths = simulate_portfolio_correlated(
                s0_dict, gbm_sub, corr_sub, weights, horizon, n_sim, rng
            )
        else:
            # Weighted sum of independently simulated normalised paths
            ref_shape = next(iter(per_ticker_paths_store.values())).shape
            portfolio_paths = np.zeros(ref_shape)
            for t, w in weights.items():
                portfolio_paths += w * (per_ticker_paths_store[t] / s0_dict[t])

        p_s0 = 1.0  # portfolio normalised to start at 1.0

        summary = compute_path_summary(portfolio_paths, percentiles, future_dates)
        summary["scenario_name"] = scenario_name
        summary["ticker"] = "PORTFOLIO"
        summary["method"] = method
        all_path_summaries.append(summary)

        td = compute_terminal_distribution(portfolio_paths, p_s0, percentiles)
        terminal_row = {
            "scenario_name": scenario_name,
            "ticker": "PORTFOLIO",
            "method": method,
            "s0": p_s0,
            "terminal_mean": td["terminal_mean"],
            "terminal_std": td["terminal_std"],
            "terminal_skew": td["terminal_skew"],
            "terminal_kurtosis": td["terminal_kurtosis"],
        }
        for p, v in td["percentiles"].items():
            terminal_row[f"terminal_p{int(p)}"] = v
        for p, v in td["return_percentiles"].items():
            terminal_row[f"return_p{int(p)}"] = v
        all_terminal_dists.append(pd.DataFrame([terminal_row]))

        var_cvar = compute_simulated_var_cvar(portfolio_paths, p_s0, var_levels, cvar_levels)
        loss_probs = compute_loss_probabilities(portfolio_paths, p_s0, prob_thresholds)
        all_metrics.append(pd.DataFrame([
            {
                "scenario_name": scenario_name,
                "ticker": "PORTFOLIO",
                "method": method,
                "metric_name": k,
                "value": float(v),
            }
            for k, v in {**var_cvar, **loss_probs}.items()
        ]))

        dd = compute_path_drawdowns(portfolio_paths)
        all_drawdowns.append(pd.DataFrame([{
            "scenario_name": scenario_name,
            "ticker": "PORTFOLIO",
            "method": method,
            **dd,
        }]))

        if save_full:
            full_paths_dict["PORTFOLIO"] = portfolio_paths  # type: ignore[index]

    return (
        _concat_dfs(all_path_summaries),
        _concat_dfs(all_terminal_dists),
        _concat_dfs(all_metrics),
        _concat_dfs(all_drawdowns),
        full_paths_dict,
    )


def save_simulations(
    path_summaries: pd.DataFrame,
    terminal_distributions: pd.DataFrame,
    mc_metrics: pd.DataFrame,
    drawdown_distributions: pd.DataFrame,
    full_paths: list[dict] | None,
    summary_dict: dict,
    processed_dir: Path,
    exports_dir: Path,
    reports_dir: Path,
    save_full_paths: bool,
) -> None:
    """Persist all Monte Carlo outputs to disk.

    Writes:
        data/processed/mc_paths_summary.parquet
        data/processed/mc_terminal_distribution.parquet
        data/processed/mc_metrics.parquet
        data/processed/mc_drawdown_distribution.parquet
        data/exports/mc_paths_full.parquet  (only if save_full_paths=True)
        outputs/reports/monte_carlo_summary.json

    Validates VaR/CVaR invariants before writing. Overwrites existing files.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _validate_var_cvar_invariants(mc_metrics)

    def _write_parquet(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            logger.warning(f"Writing empty DataFrame to {path} — no rows produced.")
        df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
        logger.info(f"Wrote {len(df)} rows → {path}")

    _write_parquet(path_summaries, processed_dir / "mc_paths_summary.parquet")
    _write_parquet(terminal_distributions, processed_dir / "mc_terminal_distribution.parquet")
    _write_parquet(mc_metrics, processed_dir / "mc_metrics.parquet")
    _write_parquet(drawdown_distributions, processed_dir / "mc_drawdown_distribution.parquet")

    if save_full_paths and full_paths:
        total_elements = sum(r["paths"].size for r in full_paths)
        estimated_gb = total_elements * 8 / 1e9
        if estimated_gb > 1.0:
            logger.critical(
                f"Full paths estimated size: {estimated_gb:.2f} GB — writing to disk."
            )

        pieces = []
        for r in full_paths:
            sn: str = r["scenario_name"]
            t: str = r["ticker"]
            paths: np.ndarray = r["paths"]
            n_sim, n_days = paths.shape
            sim_ids = np.repeat(np.arange(n_sim), n_days)
            day_offsets = np.tile(np.arange(n_days), n_sim)
            pieces.append(pd.DataFrame({
                "scenario_name": pd.array([sn] * len(sim_ids), dtype="string"),
                "ticker": pd.array([t] * len(sim_ids), dtype="string"),
                "simulation_id": pd.array(sim_ids, dtype=pd.Int64Dtype()),
                "day_offset": pd.array(day_offsets, dtype=pd.Int64Dtype()),
                "value": paths.ravel(),
            }))

        full_df = pd.concat(pieces, ignore_index=True)
        full_path = exports_dir / "mc_paths_full.parquet"
        _write_parquet(full_df, full_path)
        logger.info(f"Full paths file: {full_path.stat().st_size / 1e6:.1f} MB")

    json_path = reports_dir / "monte_carlo_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)
    logger.info(f"Wrote summary → {json_path}")


def run_monte_carlo() -> dict:
    """Orchestrate the Monte Carlo simulation pipeline (Step 6).

    Pipeline:
        load_data → validate schemas
        → estimate_gbm_params
        → build correlation matrix from returns
        → for each scenario: derive seed → run_scenario
        → aggregate results
        → save_simulations

    Returns:
        {
          "scenarios_run": int,
          "tickers_simulated": int,
          "total_paths_generated": int,
          "scenarios_failed": list[str],
          "correlation_matrix_was_non_pd": bool,
          "outputs_written": list[str],
          "full_paths_saved": bool,
          "duration_sec": float
        }
    """
    t_start = time.time()

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = config.LOG_DIR / f"monte_carlo_{datetime.now().strftime('%Y-%m-%d')}.log"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_id = logger.add(log_path, level="INFO", rotation="1 day")

    try:
        if not hasattr(config, "MC_RANDOM_SEED") or config.MC_RANDOM_SEED is None:
            logger.critical(
                "MC_RANDOM_SEED is not set — using system entropy. "
                "Results will NOT be reproducible."
            )
            base_seed = int(np.random.default_rng().integers(0, 2**32))
        else:
            base_seed = int(config.MC_RANDOM_SEED)

        # ── Load & validate inputs ────────────────────────────────────────────
        logger.info("Loading data and validating schemas.")
        prices, returns, scenarios = load_data(
            config.PROCESSED_DATA_DIR, config.MC_SCENARIOS_CSV
        )

        n_tickers = prices["ticker"].nunique()
        date_range = [
            str(prices["date"].min().date()),
            str(prices["date"].max().date()),
        ]
        logger.info(
            f"Loaded {len(prices)} price rows, {len(returns)} return rows, "
            f"{len(scenarios)} scenarios, {n_tickers} tickers."
        )

        # ── GBM parameter estimation ──────────────────────────────────────────
        logger.info(f"Estimating GBM params (drift_method={config.MC_DRIFT_METHOD}).")
        gbm_params = estimate_gbm_params(
            returns, config.MC_TRADING_DAYS_PER_YEAR, config.MC_DRIFT_METHOD
        )

        # ── Correlation matrix ────────────────────────────────────────────────
        logger.info("Building correlation matrix from log returns.")
        corr_matrix = _build_correlation_matrix(returns)

        corr_arr = corr_matrix.values
        corr_was_non_pd = not _is_positive_definite(corr_arr)
        min_eigval = float(np.linalg.eigvalsh(corr_arr).min())
        cond_num = float(np.linalg.cond(corr_arr))
        if corr_was_non_pd:
            logger.warning("Global correlation matrix is not positive-definite.")

        # ── Config overrides passed to run_scenario ───────────────────────────
        config_overrides: dict[str, Any] = {
            "MC_DEFAULT_N_SIMULATIONS": config.MC_DEFAULT_N_SIMULATIONS,
            "MC_DEFAULT_HORIZON_DAYS": config.MC_DEFAULT_HORIZON_DAYS,
            "MC_DEFAULT_BLOCK_SIZE": config.MC_DEFAULT_BLOCK_SIZE,
            "MC_DRIFT_METHOD": config.MC_DRIFT_METHOD,
            "MC_PERCENTILES": config.MC_PERCENTILES,
            "MC_VAR_LEVELS": config.MC_VAR_LEVELS,
            "MC_CVAR_LEVELS": config.MC_CVAR_LEVELS,
            "MC_PROBABILITY_THRESHOLDS": config.MC_PROBABILITY_THRESHOLDS,
            "MC_USE_CORRELATION": config.MC_USE_CORRELATION,
            "MC_SAVE_FULL_PATHS": config.MC_SAVE_FULL_PATHS,
            "MC_TRADING_DAYS_PER_YEAR": config.MC_TRADING_DAYS_PER_YEAR,
            "PORTFOLIO_WEIGHTS": config.PORTFOLIO_WEIGHTS,
        }

        # ── Scenario loop ─────────────────────────────────────────────────────
        all_path_summaries: list[pd.DataFrame] = []
        all_terminal_dists: list[pd.DataFrame] = []
        all_mc_metrics: list[pd.DataFrame] = []
        all_drawdowns: list[pd.DataFrame] = []
        all_full_paths: list[dict] = []

        scenarios_failed: list[str] = []
        scenarios_run_info: dict[str, dict] = {}
        total_paths = 0

        for _, scenario_row in scenarios.iterrows():
            sname = str(scenario_row["scenario_name"])
            scenario_seed = _stable_scenario_seed(base_seed, sname)
            t_scen = time.time()

            logger.info(f"Running scenario: {sname} (seed={scenario_seed})")
            try:
                ps_df, td_df, met_df, dd_df, fp_dict = run_scenario(
                    scenario_row, prices, returns, gbm_params, corr_matrix,
                    config_overrides, scenario_seed
                )

                all_path_summaries.append(ps_df)
                all_terminal_dists.append(td_df)
                all_mc_metrics.append(met_df)
                all_drawdowns.append(dd_df)

                if fp_dict:
                    for ticker, paths_arr in fp_dict.items():
                        all_full_paths.append({
                            "scenario_name": sname,
                            "ticker": ticker,
                            "paths": paths_arr,
                        })

                n_tickers_sim = int(met_df["ticker"].nunique()) if not met_df.empty else 0
                portfolio_simulated = "PORTFOLIO" in (
                    met_df["ticker"].unique() if not met_df.empty else []
                )
                n_sim = int(scenario_row["n_simulations"])
                total_paths += n_sim * n_tickers_sim
                scen_dur = round(time.time() - t_scen, 2)
                scenarios_run_info[sname] = {
                    "method": str(scenario_row["method"]),
                    "n_simulations": n_sim,
                    "horizon_days": int(scenario_row["horizon_days"]),
                    "tickers_simulated": n_tickers_sim,
                    "portfolio_simulated": portfolio_simulated,
                    "duration_sec": scen_dur,
                }
                logger.info(f"Scenario {sname} completed in {scen_dur}s.")

            except Exception as exc:
                logger.error(f"Scenario {sname} failed: {exc}")
                scenarios_failed.append(sname)

        if not scenarios_run_info:
            logger.critical("All scenarios failed.")
            raise RuntimeError(
                f"All Monte Carlo scenarios failed: {scenarios_failed}"
            )

        # ── Aggregate ─────────────────────────────────────────────────────────
        path_summaries_df = _concat_dfs(all_path_summaries)
        terminal_dists_df = _concat_dfs(all_terminal_dists)
        mc_metrics_df = _concat_dfs(all_mc_metrics)
        drawdowns_df = _concat_dfs(all_drawdowns)

        # ── Cast string columns to StringDtype ────────────────────────────────
        str_cols = ["scenario_name", "ticker", "method"]
        for df in [path_summaries_df, terminal_dists_df, mc_metrics_df, drawdowns_df]:
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].astype(pd.StringDtype())

        if "metric_name" in mc_metrics_df.columns:
            mc_metrics_df["metric_name"] = mc_metrics_df["metric_name"].astype(pd.StringDtype())

        # ── Method comparison (portfolio VaR by method) ───────────────────────
        method_comparison = _compute_method_comparison(mc_metrics_df)

        # ── GBM parameters for summary JSON ──────────────────────────────────
        gbm_params_summary = (
            gbm_params.set_index("ticker")[["mu_annual", "sigma_annual"]]
            .round(6)
            .to_dict(orient="index")
        )

        # ── Summary JSON ──────────────────────────────────────────────────────
        duration = round(time.time() - t_start, 2)
        summary_dict: dict[str, Any] = {
            "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {
                "prices_rows": len(prices),
                "returns_rows": len(returns),
                "tickers": n_tickers,
                "scenarios_loaded": len(scenarios),
                "date_range": date_range,
            },
            "gbm_parameters": gbm_params_summary,
            "correlation_matrix_health": {
                "is_positive_definite": not corr_was_non_pd,
                "min_eigenvalue": round(min_eigval, 6),
                "condition_number": round(cond_num, 4),
            },
            "scenarios_run": scenarios_run_info,
            "method_comparison": method_comparison,
            "config_used": {
                "random_seed": base_seed,
                "trading_days_per_year": config.MC_TRADING_DAYS_PER_YEAR,
                "save_full_paths": config.MC_SAVE_FULL_PATHS,
                "use_correlation": config.MC_USE_CORRELATION,
                "drift_method": config.MC_DRIFT_METHOD,
            },
            "duration_sec": duration,
        }

        outputs_written = [
            str(config.PROCESSED_DATA_DIR / "mc_paths_summary.parquet"),
            str(config.PROCESSED_DATA_DIR / "mc_terminal_distribution.parquet"),
            str(config.PROCESSED_DATA_DIR / "mc_metrics.parquet"),
            str(config.PROCESSED_DATA_DIR / "mc_drawdown_distribution.parquet"),
            str(config.REPORTS_DIR / "monte_carlo_summary.json"),
        ]
        if config.MC_SAVE_FULL_PATHS:
            outputs_written.append(str(config.EXPORTS_DATA_DIR / "mc_paths_full.parquet"))

        save_simulations(
            path_summaries=path_summaries_df,
            terminal_distributions=terminal_dists_df,
            mc_metrics=mc_metrics_df,
            drawdown_distributions=drawdowns_df,
            full_paths=all_full_paths if all_full_paths else None,
            summary_dict=summary_dict,
            processed_dir=config.PROCESSED_DATA_DIR,
            exports_dir=config.EXPORTS_DATA_DIR,
            reports_dir=config.REPORTS_DIR,
            save_full_paths=config.MC_SAVE_FULL_PATHS,
        )

        return_dict: dict[str, Any] = {
            "scenarios_run": len(scenarios_run_info),
            "tickers_simulated": n_tickers,
            "total_paths_generated": total_paths,
            "scenarios_failed": scenarios_failed,
            "correlation_matrix_was_non_pd": corr_was_non_pd,
            "outputs_written": outputs_written,
            "full_paths_saved": config.MC_SAVE_FULL_PATHS and bool(all_full_paths),
            "duration_sec": duration,
        }
        logger.info(f"Monte Carlo complete in {duration}s.")
        return return_dict

    finally:
        logger.remove(log_id)
