"""
tests/unit/test_06-monte-carlo.py — Unit tests for src/monte_carlo.py (Step 6).

All test logic is derived exclusively from the spec at
.claude/specs/06-monte-carlo.md. No implementation details are assumed
beyond public function signatures and the output contracts documented
in the spec.

Covers all 16 spec §8 tests plus additional tests for full coverage of the
§9 acceptance criteria and §7 edge cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    compute_loss_probabilities,
    compute_path_drawdowns,
    compute_path_summary,
    compute_simulated_var_cvar,
    compute_terminal_distribution,
    estimate_gbm_params,
    load_data,
    run_scenario,
    save_simulations,
    simulate_block_bootstrap,
    simulate_bootstrap,
    simulate_gbm,
    simulate_portfolio_correlated,
)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_returns_df(
    tickers: list[str],
    n: int = 200,
    seed: int = 0,
    mu: float = 0.001,
    sigma: float = 0.015,
) -> pd.DataFrame:
    """White-noise returns DataFrame in the long format expected by the module."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    rows = []
    for ticker in tickers:
        ret = rng.normal(mu, sigma, size=n)
        for d, r in zip(dates, ret):
            rows.append({
                "date": d,
                "ticker": ticker,
                "simple_return": float(r),
                "log_return": float(np.log1p(r)),
            })
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_prices_df(
    tickers: list[str],
    n: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimal prices DataFrame in long format (matches prices_clean_schema)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    rows = []
    for ticker in tickers:
        close = 100.0
        for d in dates:
            r = float(rng.normal(0.001, 0.02))
            close = max(close * (1.0 + r), 0.01)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": round(close * 0.99, 4),
                "high": round(close * 1.02, 4),
                "low": round(close * 0.98, 4),
                "close": round(close, 4),
                "volume": 1_000_000,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["volume"] = df["volume"].astype(pd.Int64Dtype())
    return df


def _make_gbm_params(
    tickers: list[str],
    mu: float = 0.001,
    sigma: float = 0.02,
) -> pd.DataFrame:
    """Minimal GBM params DataFrame (matches estimate_gbm_params output)."""
    return pd.DataFrame([
        {
            "ticker": t,
            "mu_daily": mu,
            "sigma_daily": sigma,
            "mu_annual": mu * 252,
            "sigma_annual": sigma * np.sqrt(252),
            "n_observations": 200,
        }
        for t in tickers
    ])


def _make_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    """Identity correlation matrix for a given ticker list."""
    n = len(tickers)
    return pd.DataFrame(np.eye(n), index=tickers, columns=tickers)


def _make_scenario_row(
    scenario_name: str = "test_gbm",
    method: str = "gbm",
    horizon_days: int = 10,
    n_simulations: int = 300,
    block_size: int = 5,
    drift_method: str = "historical",
    tickers: str = "all",
    simulate_portfolio: bool = False,
) -> pd.Series:
    """Build a minimal scenario pd.Series to pass to run_scenario."""
    return pd.Series({
        "scenario_name": scenario_name,
        "method": method,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "block_size": block_size,
        "drift_method": drift_method,
        "tickers": tickers,
        "simulate_portfolio": simulate_portfolio,
        "notes": "unit test",
    })


def _make_config_overrides(
    percentiles: list[float] | None = None,
    save_full: bool = False,
) -> dict:
    """Minimal config_overrides dict for passing to run_scenario."""
    return {
        "MC_DEFAULT_N_SIMULATIONS": 500,
        "MC_DEFAULT_HORIZON_DAYS": 10,
        "MC_DEFAULT_BLOCK_SIZE": 5,
        "MC_DRIFT_METHOD": "historical",
        "MC_PERCENTILES": percentiles or [5.0, 25.0, 50.0, 75.0, 95.0],
        "MC_VAR_LEVELS": [0.95, 0.99],
        "MC_CVAR_LEVELS": [0.95],
        "MC_PROBABILITY_THRESHOLDS": [-0.20, -0.10, 0.0, 0.10, 0.20],
        "MC_USE_CORRELATION": False,
        "MC_SAVE_FULL_PATHS": save_full,
        "MC_TRADING_DAYS_PER_YEAR": 252,
        "PORTFOLIO_WEIGHTS": None,
    }


# ── Class: TestLoadData ───────────────────────────────────────────────────────

class TestLoadData:
    """Tests for load_data() — FR1, FR2, §7 edge cases."""

    # §8 Test 1
    def test_load_data_raises_on_missing_scenarios_csv(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, tmp_path
    ):
        """FileNotFoundError when mc_scenarios.csv is absent — §7, §8 Test 1."""
        missing = tmp_path / "no_such_mc_scenarios.csv"
        with pytest.raises(FileNotFoundError, match="mc_scenarios"):
            load_data(tmp_processed_dir, missing)

    def test_load_data_raises_on_missing_prices_parquet(
        self, tmp_processed_dir, mc_scenarios_csv
    ):
        """FileNotFoundError when prices_clean.parquet is absent — §7."""
        with pytest.raises(FileNotFoundError, match="Run Step 2 first"):
            load_data(tmp_processed_dir, mc_scenarios_csv)

    def test_load_data_raises_on_missing_returns_parquet(
        self, tmp_processed_dir, mc_prices_parquet, mc_scenarios_csv
    ):
        """FileNotFoundError when returns_daily.parquet is absent — §7.
        prices_clean.parquet exists; returns_daily.parquet does not."""
        returns_path = tmp_processed_dir / "returns_daily.parquet"
        assert not returns_path.exists(), "Fixture should not have written returns."
        with pytest.raises(FileNotFoundError, match="Run Step 2 first"):
            load_data(tmp_processed_dir, mc_scenarios_csv)

    def test_load_data_raises_on_empty_scenarios_csv(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, tmp_path
    ):
        """ValueError when mc_scenarios.csv has no data rows — §7."""
        empty_csv = tmp_path / "mc_scenarios.csv"
        pd.DataFrame(columns=["scenario_name", "method", "horizon_days", "n_simulations"]).to_csv(
            empty_csv, index=False
        )
        with pytest.raises(ValueError, match="no rows"):
            load_data(tmp_processed_dir, empty_csv)

    # §8 Test 15
    def test_load_data_raises_on_unknown_method(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, tmp_path
    ):
        """ValueError for unknown method in scenarios CSV — §7, §8 Test 15."""
        bad_csv = tmp_path / "bad_mc_scenarios.csv"
        pd.DataFrame([{
            "scenario_name": "bad",
            "method": "lstm_transformer",
            "horizon_days": 30,
            "n_simulations": 200,
        }]).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Unknown method"):
            load_data(tmp_processed_dir, bad_csv)

    def test_load_data_happy_path_returns_three_dataframes(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, mc_scenarios_csv
    ):
        """load_data returns (prices, returns, scenarios) with correct types — FR1."""
        prices, returns, scenarios = load_data(tmp_processed_dir, mc_scenarios_csv)
        assert isinstance(prices, pd.DataFrame), "prices must be a DataFrame"
        assert isinstance(returns, pd.DataFrame), "returns must be a DataFrame"
        assert isinstance(scenarios, pd.DataFrame), "scenarios must be a DataFrame"

    def test_load_data_returns_defensive_copies(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, mc_scenarios_csv
    ):
        """Mutating the returned DataFrames must not corrupt a second load — §6."""
        prices1, returns1, _ = load_data(tmp_processed_dir, mc_scenarios_csv)
        prices1["close"] = -999.0  # mutate
        prices2, returns2, _ = load_data(tmp_processed_dir, mc_scenarios_csv)
        assert (prices2["close"] > 0).all(), (
            "Mutation of returned DataFrame corrupted the source data."
        )

    def test_load_data_prices_has_required_columns(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, mc_scenarios_csv
    ):
        """prices DataFrame must have all OHLCV columns — FR1."""
        prices, _, _ = load_data(tmp_processed_dir, mc_scenarios_csv)
        for col in ("date", "ticker", "open", "high", "low", "close", "volume"):
            assert col in prices.columns, f"Missing required column: {col}"

    def test_load_data_scenarios_required_columns_present(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, mc_scenarios_csv
    ):
        """scenarios DataFrame must have the 4 required columns — FR1."""
        _, _, scenarios = load_data(tmp_processed_dir, mc_scenarios_csv)
        for col in ("scenario_name", "method", "horizon_days", "n_simulations"):
            assert col in scenarios.columns, f"Required scenario column missing: {col}"


# ── Class: TestEstimateGbmParams ─────────────────────────────────────────────

class TestEstimateGbmParams:
    """Tests for estimate_gbm_params() — FR3."""

    # §8 Test 2
    def test_estimate_gbm_params_matches_hand_calc(self):
        """mu_daily and sigma_daily must match sample mean/std of log returns — §8 Test 2."""
        rng = np.random.default_rng(7)
        n = 2000
        true_mu = 0.002
        true_sigma = 0.018
        log_rets = rng.normal(true_mu, true_sigma, size=n)
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=n),
            "ticker": pd.array(["TEST"] * n, dtype="string"),
            "simple_return": log_rets,
            "log_return": log_rets,
        })

        params = estimate_gbm_params(df, 252, "historical")
        row = params[params["ticker"] == "TEST"].iloc[0]

        hand_mu = float(np.mean(log_rets))
        hand_sigma = float(np.std(log_rets, ddof=1))

        assert abs(row["mu_daily"] - hand_mu) < 1e-10, (
            f"mu_daily {row['mu_daily']} != hand-calc {hand_mu}"
        )
        assert abs(row["sigma_daily"] - hand_sigma) < 1e-10, (
            f"sigma_daily {row['sigma_daily']} != hand-calc {hand_sigma}"
        )

    def test_estimate_gbm_params_zero_drift_method_sets_mu_to_zero(self):
        """drift_method='zero' must produce mu_daily=0.0 — §7, §6."""
        df = _make_returns_df(["AAPL"], n=200, mu=0.005)
        params = estimate_gbm_params(df, 252, "zero")
        mu_val = float(params[params["ticker"] == "AAPL"].iloc[0]["mu_daily"])
        assert mu_val == pytest.approx(0.0), (
            f"Expected mu_daily=0 with drift_method='zero', got {mu_val}"
        )

    def test_estimate_gbm_params_annualised_values(self):
        """mu_annual = mu_daily * 252; sigma_annual = sigma_daily * sqrt(252) — FR3."""
        df = _make_returns_df(["MSFT"], n=500)
        params = estimate_gbm_params(df, 252, "historical")
        row = params[params["ticker"] == "MSFT"].iloc[0]
        assert row["mu_annual"] == pytest.approx(row["mu_daily"] * 252, rel=1e-9), (
            "mu_annual != mu_daily * 252"
        )
        assert row["sigma_annual"] == pytest.approx(row["sigma_daily"] * np.sqrt(252), rel=1e-9), (
            "sigma_annual != sigma_daily * sqrt(252)"
        )

    def test_estimate_gbm_params_columns_present(self):
        """Output must have all six required columns — FR3."""
        df = _make_returns_df(["A", "B"])
        params = estimate_gbm_params(df, 252, "historical")
        for col in ("ticker", "mu_daily", "sigma_daily", "mu_annual", "sigma_annual", "n_observations"):
            assert col in params.columns, f"Missing column: {col}"

    def test_estimate_gbm_params_one_row_per_ticker(self):
        """One row per ticker, no duplicates — FR3."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        df = _make_returns_df(tickers, n=100)
        params = estimate_gbm_params(df, 252, "historical")
        assert len(params) == len(tickers), (
            f"Expected {len(tickers)} rows, got {len(params)}"
        )
        assert params["ticker"].nunique() == len(tickers), "Duplicate ticker rows in GBM params"

    def test_estimate_gbm_params_single_return_gives_zero_sigma(self):
        """Ticker with 1 return → sigma_daily=0.0 (ddof=1 with n=1) — §7."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")],
            "ticker": pd.array(["SOLO"], dtype="string"),
            "simple_return": [0.01],
            "log_return": [0.00995],
        })
        params = estimate_gbm_params(df, 252, "historical")
        assert params.iloc[0]["sigma_daily"] == pytest.approx(0.0), (
            "sigma_daily should be 0.0 when n_observations=1"
        )


# ── Class: TestSimulateGbm ────────────────────────────────────────────────────

class TestSimulateGbm:
    """Tests for simulate_gbm() — FR4."""

    def test_gbm_output_shape(self):
        """Shape must be (n_simulations, horizon + 1) — FR4."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=20, n_simulations=150, rng=rng)
        assert paths.shape == (150, 21), (
            f"Expected (150, 21), got {paths.shape}"
        )

    def test_gbm_col0_equals_s0(self):
        """Column 0 of every path must equal s0 — FR4 spec."""
        rng = np.random.default_rng(1)
        s0 = 123.45
        paths = simulate_gbm(s0, 0.001, 0.02, horizon=10, n_simulations=100, rng=rng)
        assert np.allclose(paths[:, 0], s0), (
            "Column 0 must equal s0 for all paths."
        )

    def test_gbm_no_nan_or_negative_values(self):
        """GBM paths must never contain NaN or negative values — FR4."""
        rng = np.random.default_rng(2)
        paths = simulate_gbm(50.0, -0.005, 0.03, horizon=30, n_simulations=500, rng=rng)
        assert not np.any(np.isnan(paths)), "GBM paths contain NaN"
        assert np.all(paths > 0), "GBM paths contain non-positive values"

    # §8 Test 3
    def test_gbm_zero_volatility_produces_deterministic_path(self):
        """sigma=0 → all paths identical, following S0 * exp(mu * t) — §8 Test 3."""
        rng = np.random.default_rng(42)
        mu = 0.001
        s0 = 100.0
        horizon = 10
        paths = simulate_gbm(s0, mu_daily=mu, sigma_daily=0.0, horizon=horizon,
                              n_simulations=200, rng=rng)
        # All paths identical to each other
        assert np.allclose(paths, paths[0]), (
            "Zero-volatility GBM must produce identical paths across simulations."
        )
        # Exact deterministic formula: S(t) = S0 * exp(mu * t) when sigma=0
        expected = s0 * np.exp(mu * np.arange(horizon + 1))
        assert np.allclose(paths[0], expected, rtol=1e-10), (
            "Zero-volatility path does not match S0 * exp(mu * t)."
        )

    # §8 Test 4
    def test_gbm_mean_converges_to_expected(self):
        """E[S_T] = S0 * exp(mu * T) with Itô correction — §8 Test 4, N=50k, horizon=5."""
        rng = np.random.default_rng(0)
        mu, sigma, s0, horizon, n = 0.001, 0.02, 100.0, 5, 50_000
        paths = simulate_gbm(s0, mu, sigma, horizon, n, rng)
        terminal_mean = np.mean(paths[:, -1])
        expected_mean = s0 * np.exp(mu * horizon)
        relative_error = abs(terminal_mean / expected_mean - 1.0)
        assert relative_error < 0.01, (
            f"GBM terminal mean {terminal_mean:.4f} is more than 1% from "
            f"expected {expected_mean:.4f} — Itô correction may be missing."
        )

    def test_gbm_ito_correction_is_applied(self):
        """Without Itô correction, mean would be biased upward — FR4 (FR4 spec formula)."""
        # With Itô: drift = mu - sigma^2/2; so log-mean should track mu*T exactly
        rng = np.random.default_rng(99)
        mu, sigma, s0, horizon, n = 0.0, 0.05, 100.0, 252, 20_000
        paths = simulate_gbm(s0, mu, sigma, horizon, n, rng)
        # With Itô: E[S_T] = S0 * exp(mu*T) = S0 (since mu=0)
        terminal_mean = float(np.mean(paths[:, -1]))
        assert abs(terminal_mean - s0) / s0 < 0.02, (
            f"With mu=0 and Itô correction, E[S_T] should be ≈ S0={s0}, "
            f"got {terminal_mean:.2f}. Itô correction may be wrong."
        )

    @pytest.mark.parametrize("horizon,n_sim", [(5, 100), (30, 500), (252, 200)])
    def test_gbm_parametrized_shapes(self, horizon, n_sim):
        """Shape is correct for various horizons and n_simulations — FR4."""
        rng = np.random.default_rng(42)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon, n_sim, rng)
        assert paths.shape == (n_sim, horizon + 1), (
            f"Expected ({n_sim}, {horizon + 1}), got {paths.shape}"
        )


# ── Class: TestSimulateBootstrap ─────────────────────────────────────────────

class TestSimulateBootstrap:
    """Tests for simulate_bootstrap() — FR5."""

    # §8 Test 5
    def test_bootstrap_output_shape_and_s0_col(self):
        """Shape (n_sim, horizon+1), column 0 = s0 — FR5, §8 Test 5."""
        hist = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
        rng = np.random.default_rng(99)
        paths = simulate_bootstrap(s0=100.0, historical_returns=hist,
                                   horizon=20, n_simulations=500, rng=rng)
        assert paths.shape == (500, 21), f"Expected (500, 21), got {paths.shape}"
        assert np.allclose(paths[:, 0], 100.0), "Column 0 must equal s0"

    def test_bootstrap_no_nan_or_negative(self):
        """Bootstrap paths must not contain NaN or non-positive values — FR5."""
        hist = np.linspace(0.005, 0.015, 100)
        rng = np.random.default_rng(10)
        paths = simulate_bootstrap(100.0, hist, horizon=30, n_simulations=200, rng=rng)
        assert not np.any(np.isnan(paths)), "Bootstrap paths contain NaN"
        assert np.all(paths > 0), "Bootstrap paths contain non-positive values"

    def test_bootstrap_draws_only_from_historical(self):
        """Sampled returns must only come from historical set — FR5 (with replacement)."""
        # Use extreme values that are easy to detect
        hist = np.array([0.10, -0.10])
        rng = np.random.default_rng(7)
        paths = simulate_bootstrap(100.0, hist, horizon=50, n_simulations=1000, rng=rng)
        # Ratios between consecutive days: should be either 1.10 or 0.90
        ratios = paths[:, 1:] / paths[:, :-1]
        unique_ratios = np.unique(np.round(ratios, 8))
        assert set(np.round(unique_ratios, 5)) == {round(1.10, 5), round(0.90, 5)}, (
            "Bootstrap sampled returns outside the historical set."
        )

    def test_bootstrap_single_return_returns_flat_paths(self):
        """With only 1 historical return, bootstrap must return flat paths — §7."""
        hist = np.array([0.05])
        rng = np.random.default_rng(0)
        paths = simulate_bootstrap(100.0, hist, horizon=10, n_simulations=100, rng=rng)
        assert paths.shape == (100, 11), f"Unexpected shape: {paths.shape}"
        assert np.allclose(paths, 100.0), "Single-return bootstrap must return flat paths at s0"


# ── Class: TestSimulateBlockBootstrap ────────────────────────────────────────

class TestSimulateBlockBootstrap:
    """Tests for simulate_block_bootstrap() — FR6."""

    def test_block_bootstrap_output_shape(self):
        """Shape must be (n_simulations, horizon + 1) — FR6."""
        hist = np.random.default_rng(0).normal(0.001, 0.015, 200)
        rng = np.random.default_rng(0)
        paths = simulate_block_bootstrap(100.0, hist, horizon=30, n_simulations=250, block_size=10, rng=rng)
        assert paths.shape == (250, 31), f"Expected (250, 31), got {paths.shape}"

    def test_block_bootstrap_col0_equals_s0(self):
        """Column 0 must equal s0 — FR6."""
        hist = np.random.default_rng(1).normal(0.001, 0.015, 100)
        rng = np.random.default_rng(1)
        s0 = 75.0
        paths = simulate_block_bootstrap(s0, hist, horizon=20, n_simulations=100, block_size=5, rng=rng)
        assert np.allclose(paths[:, 0], s0), "Column 0 must equal s0"

    def test_block_bootstrap_block_size_geq_len_raises(self):
        """block_size >= len(historical_returns) must raise ValueError — §7."""
        hist = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # len=5
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="block_size"):
            simulate_block_bootstrap(100.0, hist, horizon=10, n_simulations=100,
                                     block_size=5, rng=rng)

    def test_block_bootstrap_block_size_1_is_allowed(self):
        """block_size=1 must run without error (equivalent to simple bootstrap) — §7."""
        hist = np.random.default_rng(2).normal(0.001, 0.015, 100)
        rng = np.random.default_rng(2)
        # Should not raise
        paths = simulate_block_bootstrap(100.0, hist, horizon=15, n_simulations=100,
                                         block_size=1, rng=rng)
        assert paths.shape == (100, 16), f"Unexpected shape: {paths.shape}"

    def test_block_bootstrap_no_nan_or_negative(self):
        """Block bootstrap paths must not contain NaN or non-positive values — FR6."""
        hist = np.random.default_rng(3).normal(0.001, 0.015, 150)
        rng = np.random.default_rng(3)
        paths = simulate_block_bootstrap(100.0, hist, horizon=30, n_simulations=200,
                                         block_size=10, rng=rng)
        assert not np.any(np.isnan(paths)), "Block bootstrap paths contain NaN"
        assert np.all(paths > 0), "Block bootstrap paths contain non-positive values"

    # §8 Test 6
    def test_block_bootstrap_preserves_serial_correlation(self):
        """Block bootstrap preserves more autocorrelation than IID bootstrap — §8 Test 6."""
        # AR(1) process: r[t] = 0.6 * r[t-1] + noise
        rng_gen = np.random.default_rng(1)
        n = 600
        ar1 = np.zeros(n)
        for i in range(1, n):
            ar1[i] = 0.6 * ar1[i - 1] + rng_gen.standard_normal() * 0.01

        def mean_lag1_autocorr(paths: np.ndarray, n_paths: int = 100) -> float:
            """Compute mean lag-1 autocorrelation of log-differences across paths."""
            log_returns = np.diff(np.log(np.clip(paths[:n_paths], 1e-10, None)), axis=1)
            ac = []
            for r in log_returns:
                if np.std(r) > 1e-12:
                    ac.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
            return float(np.nanmean(ac)) if ac else 0.0

        rng1 = np.random.default_rng(42)
        simple_paths = simulate_bootstrap(100.0, ar1, horizon=80, n_simulations=200, rng=rng1)
        rng2 = np.random.default_rng(42)
        block_paths = simulate_block_bootstrap(100.0, ar1, horizon=80, n_simulations=200,
                                               block_size=15, rng=rng2)

        simple_ac = mean_lag1_autocorr(simple_paths)
        block_ac = mean_lag1_autocorr(block_paths)

        assert block_ac > simple_ac, (
            f"Block bootstrap lag-1 autocorr ({block_ac:.4f}) should exceed "
            f"IID bootstrap ({simple_ac:.4f}) for AR(1) input."
        )


# ── Class: TestSimulatePortfolioCorrelated ────────────────────────────────────

class TestSimulatePortfolioCorrelated:
    """Tests for simulate_portfolio_correlated() — FR7, FR8."""

    # §8 Test 7
    def test_cholesky_on_identity_equals_independent(self):
        """Identity correlation → near-zero cross-ticker correlation — §8 Test 7."""
        tickers = ["A", "B"]
        s0 = {"A": 100.0, "B": 200.0}
        gbm_params = _make_gbm_params(tickers)
        identity_corr = _make_correlation_matrix(tickers)
        weights = {"A": 0.5, "B": 0.5}

        rng = np.random.default_rng(0)
        per_ticker, portfolio = simulate_portfolio_correlated(
            s0, gbm_params, identity_corr, weights, 30, 3000, rng
        )

        assert set(per_ticker.keys()) == set(tickers), "Missing tickers in per_ticker output"
        assert per_ticker["A"].shape == (3000, 31), f"Wrong shape: {per_ticker['A'].shape}"
        assert portfolio.shape == (3000, 31), f"Wrong portfolio shape: {portfolio.shape}"

        # With identity correlation, terminal returns should be near-uncorrelated
        ret_A = per_ticker["A"][:, -1] / 100.0 - 1
        ret_B = per_ticker["B"][:, -1] / 200.0 - 1
        corr_val = float(np.corrcoef(ret_A, ret_B)[0, 1])
        assert abs(corr_val) < 0.15, (
            f"Identity correlation should produce near-0 cross-ticker corr, got {corr_val:.4f}"
        )

    # §8 Test 8
    def test_cholesky_handles_non_pd_matrix(self):
        """Non-PD correlation matrix is projected without raising — §8 Test 8, FR8."""
        tickers = ["A", "B", "C"]
        s0 = {t: 100.0 for t in tickers}
        gbm_params = _make_gbm_params(tickers)
        weights = {t: 1 / 3 for t in tickers}

        # Singular (non-PD): rows 0 and 1 are identical
        bad_vals = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bad_corr = pd.DataFrame(bad_vals, index=tickers, columns=tickers)

        rng = np.random.default_rng(5)
        # Must not raise; should produce valid paths after eigenvalue clipping
        per_ticker, portfolio = simulate_portfolio_correlated(
            s0, gbm_params, bad_corr, weights, 10, 200, rng
        )
        assert portfolio.shape == (200, 11), f"Unexpected portfolio shape: {portfolio.shape}"
        assert not np.any(np.isnan(portfolio)), "Portfolio paths contain NaN after PD projection"
        assert np.all(portfolio > 0), "Portfolio paths are non-positive after PD projection"

    def test_portfolio_paths_start_at_1(self):
        """Portfolio paths must be normalised to start at 1.0 — FR7 spec."""
        tickers = ["X", "Y"]
        s0 = {"X": 50.0, "Y": 300.0}
        gbm_params = _make_gbm_params(tickers)
        corr = _make_correlation_matrix(tickers)
        weights = {"X": 0.4, "Y": 0.6}

        rng = np.random.default_rng(11)
        _, portfolio = simulate_portfolio_correlated(
            s0, gbm_params, corr, weights, 15, 200, rng
        )
        assert np.allclose(portfolio[:, 0], 1.0), (
            "Portfolio paths must start at 1.0 (normalised)."
        )

    def test_portfolio_correlated_nan_in_corr_raises_value_error(self):
        """NaN in correlation matrix must raise ValueError — §7."""
        tickers = ["A", "B"]
        s0 = {"A": 100.0, "B": 100.0}
        gbm_params = _make_gbm_params(tickers)
        weights = {"A": 0.5, "B": 0.5}

        nan_vals = np.array([[1.0, np.nan], [np.nan, 1.0]])
        nan_corr = pd.DataFrame(nan_vals, index=tickers, columns=tickers)

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="NaN"):
            simulate_portfolio_correlated(s0, gbm_params, nan_corr, weights, 10, 100, rng)

    def test_portfolio_correlated_all_paths_positive(self):
        """All per-ticker and portfolio paths must be positive — FR7."""
        tickers = ["P", "Q"]
        s0 = {"P": 100.0, "Q": 100.0}
        gbm_params = _make_gbm_params(tickers, mu=0.001, sigma=0.04)
        corr = _make_correlation_matrix(tickers)
        weights = {"P": 0.5, "Q": 0.5}

        rng = np.random.default_rng(7)
        per_ticker, portfolio = simulate_portfolio_correlated(
            s0, gbm_params, corr, weights, 30, 500, rng
        )
        for t, paths in per_ticker.items():
            assert np.all(paths > 0), f"Non-positive values in per-ticker paths for {t}"
        assert np.all(portfolio > 0), "Non-positive values in portfolio paths"


# ── Class: TestComputePathSummary ─────────────────────────────────────────────

class TestComputePathSummary:
    """Tests for compute_path_summary() — FR10."""

    def test_path_summary_output_columns(self):
        """Output must contain [day_offset, date, percentile, value] — FR10, §5."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=500, rng=rng)
        dates = pd.bdate_range("2025-01-01", periods=11)
        summary = compute_path_summary(paths, [5.0, 50.0, 95.0], dates)
        for col in ("day_offset", "date", "percentile", "value"):
            assert col in summary.columns, f"Missing column: {col}"

    def test_path_summary_row_count(self):
        """Row count = (horizon + 1) * len(percentiles) — FR10."""
        rng = np.random.default_rng(1)
        horizon = 15
        percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=horizon, n_simulations=300, rng=rng)
        dates = pd.bdate_range("2025-01-01", periods=horizon + 1)
        summary = compute_path_summary(paths, percentiles, dates)
        expected_rows = (horizon + 1) * len(percentiles)
        assert len(summary) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(summary)}"
        )

    # §8 Test 12 / AC §9
    def test_percentile_bands_monotonic(self):
        """Percentile bands must be monotonically ordered at each day — §8 Test 12, AC §9."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=5_000, rng=rng)
        dates = pd.bdate_range("2025-01-01", periods=31)
        percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]
        summary = compute_path_summary(paths, percentiles, dates)

        pivot = summary.pivot_table(index="day_offset", columns="percentile", values="value")
        for p_lo, p_hi in [(5.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 95.0)]:
            violations = (pivot[p_lo] > pivot[p_hi]).sum()
            assert violations == 0, (
                f"Monotonicity violated at {violations} days: P{int(p_lo)} > P{int(p_hi)}"
            )

    def test_path_summary_day0_value_equals_s0(self):
        """At day_offset=0, all percentiles must equal s0 — FR10 (paths[:,0] = s0)."""
        s0 = 150.0
        rng = np.random.default_rng(5)
        paths = simulate_gbm(s0, 0.001, 0.02, horizon=20, n_simulations=500, rng=rng)
        dates = pd.bdate_range("2025-01-01", periods=21)
        summary = compute_path_summary(paths, [5.0, 50.0, 95.0], dates)
        day0 = summary[summary["day_offset"] == 0]["value"]
        assert np.allclose(day0.values, s0), (
            f"All day-0 percentiles should equal s0={s0}, got: {day0.values}"
        )


# ── Class: TestComputeTerminalDistribution ────────────────────────────────────

class TestComputeTerminalDistribution:
    """Tests for compute_terminal_distribution() — FR11."""

    def test_terminal_distribution_required_keys_present(self):
        """Result must contain all four scalar stats and both percentile dicts — FR11."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=2000, rng=rng)
        result = compute_terminal_distribution(paths, s0=100.0, percentiles=[5.0, 50.0, 95.0])
        for key in ("terminal_mean", "terminal_std", "terminal_skew", "terminal_kurtosis",
                    "percentiles", "return_percentiles"):
            assert key in result, f"Missing key: {key}"

    def test_terminal_distribution_percentile_counts(self):
        """Percentile dicts must have one entry per requested percentile — FR11."""
        rng = np.random.default_rng(1)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=20, n_simulations=1000, rng=rng)
        percentiles = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
        result = compute_terminal_distribution(paths, s0=100.0, percentiles=percentiles)
        assert len(result["percentiles"]) == len(percentiles), (
            f"Expected {len(percentiles)} percentile entries, got {len(result['percentiles'])}"
        )
        assert len(result["return_percentiles"]) == len(percentiles), (
            f"Expected {len(percentiles)} return_percentile entries"
        )

    def test_terminal_distribution_return_percentile_formula(self):
        """return_percentile[p] = percentile[p] / s0 - 1 — FR11."""
        rng = np.random.default_rng(2)
        s0 = 80.0
        paths = simulate_gbm(s0, 0.001, 0.02, horizon=30, n_simulations=2000, rng=rng)
        percentiles = [25.0, 50.0, 75.0]
        result = compute_terminal_distribution(paths, s0=s0, percentiles=percentiles)
        for p in percentiles:
            expected_return = result["percentiles"][p] / s0 - 1.0
            assert result["return_percentiles"][p] == pytest.approx(expected_return, rel=1e-9), (
                f"return_percentile[{p}] = {result['return_percentiles'][p]:.6f} "
                f"!= {expected_return:.6f}"
            )

    def test_terminal_distribution_std_non_negative(self):
        """terminal_std must be non-negative — FR11."""
        rng = np.random.default_rng(3)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=20, n_simulations=1000, rng=rng)
        result = compute_terminal_distribution(paths, s0=100.0, percentiles=[50.0])
        assert result["terminal_std"] >= 0.0, (
            f"terminal_std must be non-negative, got {result['terminal_std']}"
        )


# ── Class: TestComputeSimulatedVarCvar ────────────────────────────────────────

class TestComputeSimulatedVarCvar:
    """Tests for compute_simulated_var_cvar() — FR12, FR24, AC §9."""

    # §8 Test 9
    def test_var_99_geq_var_95(self):
        """var_99 must be ≥ var_95 — §8 Test 9, FR24, AC §9."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=10_000, rng=rng)
        result = compute_simulated_var_cvar(paths, 100.0, [0.95, 0.99], [0.95])
        assert result["var_99"] >= result["var_95"], (
            f"var_99={result['var_99']:.4f} < var_95={result['var_95']:.4f} — invariant violated."
        )

    # §8 Test 10
    def test_cvar_geq_var(self):
        """cvar_95 must be ≥ var_95 — §8 Test 10, FR24, AC §9."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=10_000, rng=rng)
        result = compute_simulated_var_cvar(paths, 100.0, [0.95], [0.95])
        assert result["cvar_95"] >= result["var_95"], (
            f"cvar_95={result['cvar_95']:.4f} < var_95={result['var_95']:.4f} — invariant violated."
        )

    def test_var_cvar_are_positive_loss_values(self):
        """VaR and CVaR must be reported as positive loss values — §6, FR12."""
        rng = np.random.default_rng(42)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=5_000, rng=rng)
        result = compute_simulated_var_cvar(paths, 100.0, [0.95, 0.99], [0.95])
        for key in ("var_95", "var_99", "cvar_95"):
            assert result[key] >= 0.0, (
                f"{key}={result[key]:.4f} is negative — VaR/CVaR must be positive loss values."
            )

    def test_var_keys_match_levels(self):
        """Keys must be 'var_95', 'var_99', 'cvar_95' — FR12."""
        rng = np.random.default_rng(1)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=1000, rng=rng)
        result = compute_simulated_var_cvar(paths, 100.0, [0.95, 0.99], [0.95])
        assert "var_95" in result, "Key 'var_95' missing from VaR/CVaR result"
        assert "var_99" in result, "Key 'var_99' missing from VaR/CVaR result"
        assert "cvar_95" in result, "Key 'cvar_95' missing from VaR/CVaR result"

    @pytest.mark.parametrize("method_fn,method_kwargs", [
        ("gbm", {"mu_daily": 0.001, "sigma_daily": 0.02}),
    ])
    def test_var_invariant_holds_across_methods(self, method_fn, method_kwargs):
        """var_99 ≥ var_95 must hold regardless of simulation method — FR24."""
        rng = np.random.default_rng(55)
        paths = simulate_gbm(100.0, horizon=30, n_simulations=5_000, rng=rng, **method_kwargs)
        result = compute_simulated_var_cvar(paths, 100.0, [0.95, 0.99], [0.95])
        assert result["var_99"] >= result["var_95"], (
            f"var_99 < var_95 for {method_fn}: {result}"
        )


# ── Class: TestComputeLossProbabilities ───────────────────────────────────────

class TestComputeLossProbabilities:
    """Tests for compute_loss_probabilities() — FR13."""

    def test_loss_probabilities_values_between_0_and_1(self):
        """All probability values must be in [0, 1] — FR13."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=2000, rng=rng)
        result = compute_loss_probabilities(paths, s0=100.0, thresholds=[-0.20, -0.10, 0.0, 0.10, 0.20])
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"P({key})={val:.4f} is outside [0, 1]"

    def test_loss_probabilities_zero_threshold_is_prob_loss(self):
        """Threshold=0.0 gives the probability of any loss — FR13."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=5000, rng=rng)
        result = compute_loss_probabilities(paths, s0=100.0, thresholds=[0.0])
        assert "prob_loss" in result, "'prob_loss' key missing for threshold=0.0"
        assert 0.0 <= result["prob_loss"] <= 1.0, (
            f"prob_loss={result['prob_loss']:.4f} out of [0, 1]"
        )

    def test_loss_probabilities_monotone_in_thresholds(self):
        """P(return < t) must be non-decreasing as t increases — FR13."""
        rng = np.random.default_rng(2)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=5000, rng=rng)
        thresholds = sorted([-0.30, -0.20, -0.10, 0.0, 0.10])
        # Compute probabilities directly from terminal returns to verify monotonicity
        terminal_returns = paths[:, -1] / 100.0 - 1.0
        probs_by_threshold = [float(np.mean(terminal_returns < t)) for t in thresholds]
        for i in range(len(thresholds) - 1):
            assert probs_by_threshold[i] <= probs_by_threshold[i + 1], (
                f"P(return < {thresholds[i]}) > P(return < {thresholds[i+1]}) — monotonicity violated."
            )

    def test_loss_probabilities_key_names(self):
        """Key names must follow the human-readable metric name convention — FR13."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=500, rng=rng)
        result = compute_loss_probabilities(paths, s0=100.0, thresholds=[-0.20, 0.0])
        assert "prob_loss_20pct" in result, (
            f"Expected 'prob_loss_20pct' key for threshold=-0.20, got: {list(result.keys())}"
        )
        assert "prob_loss" in result, (
            f"Expected 'prob_loss' key for threshold=0.0, got: {list(result.keys())}"
        )


# ── Class: TestComputePathDrawdowns ──────────────────────────────────────────

class TestComputePathDrawdowns:
    """Tests for compute_path_drawdowns() — FR14, AC §9."""

    # §8 Test 11
    def test_path_drawdowns_all_non_positive(self):
        """All max-drawdown summary statistics must be ≤ 0 — §8 Test 11, AC §9."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=30, n_simulations=2000, rng=rng)
        result = compute_path_drawdowns(paths)
        for key in ("mean_max_drawdown", "median_max_drawdown", "p5_max_drawdown", "p1_max_drawdown"):
            assert result[key] <= 0.0, (
                f"{key}={result[key]:.4f} is positive — max drawdown must be ≤ 0."
            )

    def test_path_drawdowns_monotone_increasing_gives_zero(self):
        """Monotonically increasing paths must have max drawdown = 0 — FR14."""
        # Perfectly increasing paths: no drawdown possible
        n_sim, horizon = 50, 20
        t = np.arange(horizon + 1)
        paths = 100.0 * np.exp(0.005 * t)[np.newaxis, :] * np.ones((n_sim, 1))
        result = compute_path_drawdowns(paths)
        assert result["mean_max_drawdown"] == pytest.approx(0.0, abs=1e-10), (
            "Monotone increasing paths must have mean_max_drawdown=0."
        )
        assert result["median_max_drawdown"] == pytest.approx(0.0, abs=1e-10), (
            "Monotone increasing paths must have median_max_drawdown=0."
        )

    def test_path_drawdowns_required_keys_present(self):
        """Result must contain all five required keys — FR14."""
        rng = np.random.default_rng(1)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=20, n_simulations=500, rng=rng)
        result = compute_path_drawdowns(paths)
        for key in ("mean_max_drawdown", "median_max_drawdown", "p5_max_drawdown",
                    "p1_max_drawdown", "prob_drawdown_exceeds_20pct"):
            assert key in result, f"Missing key: {key}"

    def test_path_drawdowns_p1_le_p5(self):
        """p1_max_drawdown must be ≤ p5_max_drawdown (p1 is the worst tail) — FR14."""
        rng = np.random.default_rng(2)
        paths = simulate_gbm(100.0, 0.001, 0.03, horizon=60, n_simulations=3000, rng=rng)
        result = compute_path_drawdowns(paths)
        assert result["p1_max_drawdown"] <= result["p5_max_drawdown"], (
            f"p1_max_drawdown={result['p1_max_drawdown']:.4f} > "
            f"p5_max_drawdown={result['p5_max_drawdown']:.4f}"
        )

    def test_path_drawdowns_exceeds_prob_in_unit_interval(self):
        """prob_drawdown_exceeds_20pct must be in [0, 1] — FR14."""
        rng = np.random.default_rng(3)
        paths = simulate_gbm(100.0, 0.001, 0.03, horizon=30, n_simulations=1000, rng=rng)
        result = compute_path_drawdowns(paths)
        assert 0.0 <= result["prob_drawdown_exceeds_20pct"] <= 1.0, (
            f"prob_drawdown_exceeds_20pct={result['prob_drawdown_exceeds_20pct']:.4f} out of [0, 1]"
        )


# ── Class: TestIdempotencyAndSeeding ─────────────────────────────────────────

class TestIdempotencyAndSeeding:
    """Idempotency and per-scenario seed tests — FR22, FR23, AC §9."""

    # §8 Test 13
    def test_idempotent_rerun_with_seed(self):
        """Same RNG seed → byte-identical paths — §8 Test 13, FR22, AC §9."""
        rng1 = np.random.default_rng(42)
        paths1 = simulate_gbm(100.0, 0.001, 0.02, horizon=15, n_simulations=1_000, rng=rng1)

        rng2 = np.random.default_rng(42)
        paths2 = simulate_gbm(100.0, 0.001, 0.02, horizon=15, n_simulations=1_000, rng=rng2)

        assert np.array_equal(paths1, paths2), (
            "Paths differ with the same seed — RNG is not deterministic."
        )

    # §8 Test 14
    def test_per_scenario_seed_produces_different_paths(self):
        """Different scenario names → different seeds → different paths — §8 Test 14, FR23."""
        base_seed = 42
        seed_a = (base_seed + hash("scenario_alpha")) % (2 ** 32)
        seed_b = (base_seed + hash("scenario_beta")) % (2 ** 32)

        assert seed_a != seed_b, (
            "Scenario seeds collided — two distinct names produced the same seed."
        )

        rng_a = np.random.default_rng(seed_a)
        rng_b = np.random.default_rng(seed_b)
        paths_a = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=200, rng=rng_a)
        paths_b = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=200, rng=rng_b)

        assert not np.array_equal(paths_a, paths_b), (
            "Different scenario seeds produced byte-identical paths — seeding is broken."
        )

    def test_per_scenario_seed_formula(self):
        """Per-scenario seed = (base_seed + hash(name)) % 2**32 — FR23."""
        base_seed = 42
        for name in ("gbm_30d", "bootstrap_60d", "block_90d"):
            expected_seed = (base_seed + hash(name)) % (2 ** 32)
            assert 0 <= expected_seed < 2 ** 32, (
                f"Seed {expected_seed} for scenario '{name}' is outside [0, 2^32)."
            )

    def test_bootstrap_idempotent_with_seed(self):
        """Bootstrap paths are also deterministic with the same seed — FR22."""
        hist = np.random.default_rng(0).normal(0.001, 0.015, 100)
        rng1 = np.random.default_rng(7)
        paths1 = simulate_bootstrap(100.0, hist, horizon=20, n_simulations=300, rng=rng1)
        rng2 = np.random.default_rng(7)
        paths2 = simulate_bootstrap(100.0, hist, horizon=20, n_simulations=300, rng=rng2)
        assert np.array_equal(paths1, paths2), (
            "Bootstrap paths differ with the same seed — not idempotent."
        )


# ── Class: TestRunScenario ────────────────────────────────────────────────────

class TestRunScenario:
    """Tests for run_scenario() — FR9, FR10–FR14."""

    def test_run_scenario_gbm_returns_four_dataframes(self):
        """run_scenario returns a 5-tuple; first 4 are DataFrames — FR9."""
        prices = _make_prices_df(["AAPL", "MSFT"], n=100)
        returns = _make_returns_df(["AAPL", "MSFT"], n=99)
        gbm_params = _make_gbm_params(["AAPL", "MSFT"])
        corr = _make_correlation_matrix(["AAPL", "MSFT"])
        scenario = _make_scenario_row(method="gbm", horizon_days=5, n_simulations=200)
        cfg = _make_config_overrides()

        result = run_scenario(scenario, prices, returns, gbm_params, corr, cfg, seed=42)

        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
        path_summary, terminal_dist, metrics, drawdown_dist, full_paths = result
        assert isinstance(path_summary, pd.DataFrame), "path_summary must be a DataFrame"
        assert isinstance(terminal_dist, pd.DataFrame), "terminal_dist must be a DataFrame"
        assert isinstance(metrics, pd.DataFrame), "metrics must be a DataFrame"
        assert isinstance(drawdown_dist, pd.DataFrame), "drawdown_dist must be a DataFrame"

    def test_run_scenario_gbm_path_summary_has_required_columns(self):
        """path_summary must have scenario_name, ticker, method, day_offset, date, percentile, value — FR10."""
        prices = _make_prices_df(["AAPL"], n=50)
        returns = _make_returns_df(["AAPL"], n=49)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=5, n_simulations=100)
        cfg = _make_config_overrides()

        path_summary, _, _, _, _ = run_scenario(scenario, prices, returns, gbm_params, corr, cfg, seed=1)

        for col in ("scenario_name", "ticker", "method", "day_offset", "date", "percentile", "value"):
            assert col in path_summary.columns, f"Missing column in path_summary: {col}"

    def test_run_scenario_metrics_has_var_and_cvar(self):
        """metrics DataFrame must contain var_95, var_99, cvar_95 rows — FR12."""
        prices = _make_prices_df(["AAPL"], n=60)
        returns = _make_returns_df(["AAPL"], n=59)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=5, n_simulations=200)
        cfg = _make_config_overrides()

        _, _, metrics, _, _ = run_scenario(scenario, prices, returns, gbm_params, corr, cfg, seed=1)

        metric_names = set(metrics["metric_name"].tolist())
        for expected in ("var_95", "var_99", "cvar_95"):
            assert expected in metric_names, f"'{expected}' missing from metrics"

    def test_run_scenario_bootstrap_output_shape(self):
        """run_scenario with method='bootstrap' returns valid DataFrames — FR9."""
        prices = _make_prices_df(["AAPL", "MSFT"], n=80)
        returns = _make_returns_df(["AAPL", "MSFT"], n=79)
        gbm_params = _make_gbm_params(["AAPL", "MSFT"])
        corr = _make_correlation_matrix(["AAPL", "MSFT"])
        scenario = _make_scenario_row(
            scenario_name="bootstrap_test",
            method="bootstrap",
            horizon_days=5,
            n_simulations=150,
        )
        cfg = _make_config_overrides()

        path_summary, terminal_dist, metrics, drawdown_dist, _ = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=10
        )
        assert len(path_summary) > 0, "path_summary must not be empty for bootstrap"
        assert len(metrics) > 0, "metrics must not be empty for bootstrap"

    def test_run_scenario_block_bootstrap_output_shape(self):
        """run_scenario with method='block_bootstrap' returns valid DataFrames — FR9."""
        prices = _make_prices_df(["AAPL", "MSFT"], n=100)
        returns = _make_returns_df(["AAPL", "MSFT"], n=99)
        gbm_params = _make_gbm_params(["AAPL", "MSFT"])
        corr = _make_correlation_matrix(["AAPL", "MSFT"])
        scenario = _make_scenario_row(
            scenario_name="block_test",
            method="block_bootstrap",
            horizon_days=5,
            n_simulations=150,
            block_size=5,
        )
        cfg = _make_config_overrides()

        path_summary, _, metrics, _, _ = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=20
        )
        assert len(path_summary) > 0, "path_summary must not be empty for block_bootstrap"
        assert len(metrics) > 0, "metrics must not be empty for block_bootstrap"

    def test_run_scenario_full_paths_none_when_save_false(self):
        """full_paths is None when MC_SAVE_FULL_PATHS=False — FR19 (negation)."""
        prices = _make_prices_df(["AAPL"], n=60)
        returns = _make_returns_df(["AAPL"], n=59)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=5, n_simulations=100)
        cfg = _make_config_overrides(save_full=False)

        _, _, _, _, full_paths = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=0
        )
        assert full_paths is None, (
            f"full_paths should be None when MC_SAVE_FULL_PATHS=False, got {type(full_paths)}"
        )

    def test_run_scenario_full_paths_dict_when_save_true(self):
        """full_paths is a non-None dict when MC_SAVE_FULL_PATHS=True — FR19."""
        prices = _make_prices_df(["AAPL"], n=60)
        returns = _make_returns_df(["AAPL"], n=59)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=5, n_simulations=100)
        cfg = _make_config_overrides(save_full=True)

        _, _, _, _, full_paths = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=0
        )
        assert full_paths is not None, "full_paths should be a dict when MC_SAVE_FULL_PATHS=True"
        assert isinstance(full_paths, dict), f"full_paths must be a dict, got {type(full_paths)}"

    def test_run_scenario_drawdown_values_non_positive(self):
        """drawdown_dist mean_max_drawdown must be ≤ 0 — AC §9."""
        prices = _make_prices_df(["AAPL"], n=80)
        returns = _make_returns_df(["AAPL"], n=79)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=20, n_simulations=500)
        cfg = _make_config_overrides()

        _, _, _, drawdown_dist, _ = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=5
        )
        mean_dd = float(drawdown_dist["mean_max_drawdown"].iloc[0])
        assert mean_dd <= 0.0, (
            f"mean_max_drawdown={mean_dd:.4f} is positive — drawdowns must be ≤ 0."
        )

    def test_run_scenario_var_99_geq_var_95_in_output(self):
        """var_99 ≥ var_95 in the scenario metrics output — FR24, AC §9."""
        prices = _make_prices_df(["AAPL"], n=80)
        returns = _make_returns_df(["AAPL"], n=79)
        gbm_params = _make_gbm_params(["AAPL"])
        corr = _make_correlation_matrix(["AAPL"])
        scenario = _make_scenario_row(method="gbm", horizon_days=30, n_simulations=2000)
        cfg = _make_config_overrides()

        _, _, metrics, _, _ = run_scenario(
            scenario, prices, returns, gbm_params, corr, cfg, seed=7
        )
        for ticker in metrics["ticker"].unique():
            sub = metrics[metrics["ticker"] == ticker].set_index("metric_name")["value"]
            if "var_95" in sub.index and "var_99" in sub.index:
                assert sub["var_99"] >= sub["var_95"], (
                    f"[{ticker}] var_99={sub['var_99']:.4f} < var_95={sub['var_95']:.4f}"
                )


# ── Class: TestSaveSimulations ────────────────────────────────────────────────

class TestSaveSimulations:
    """Tests for save_simulations() — FR15–FR20, AC §9."""

    def _make_minimal_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Minimal DataFrames satisfying output schema column requirements."""
        path_summary = pd.DataFrame({
            "scenario_name": pd.array(["s1"], dtype="string"),
            "ticker": pd.array(["AAPL"], dtype="string"),
            "method": pd.array(["gbm"], dtype="string"),
            "day_offset": pd.array([0], dtype=pd.Int64Dtype()),
            "date": [pd.Timestamp("2025-01-01")],
            "percentile": [50.0],
            "value": [100.0],
        })
        terminal_dist = pd.DataFrame({
            "scenario_name": pd.array(["s1"], dtype="string"),
            "ticker": pd.array(["AAPL"], dtype="string"),
            "method": pd.array(["gbm"], dtype="string"),
            "s0": [100.0],
            "terminal_mean": [101.0],
            "terminal_std": [5.0],
            "terminal_skew": [0.1],
            "terminal_kurtosis": [0.2],
        })
        mc_metrics = pd.DataFrame({
            "scenario_name": pd.array(["s1"], dtype="string"),
            "ticker": pd.array(["AAPL"], dtype="string"),
            "method": pd.array(["gbm"], dtype="string"),
            "metric_name": pd.array(["var_95"], dtype="string"),
            "value": [0.05],
        })
        drawdown_dist = pd.DataFrame({
            "scenario_name": pd.array(["s1"], dtype="string"),
            "ticker": pd.array(["AAPL"], dtype="string"),
            "method": pd.array(["gbm"], dtype="string"),
            "mean_max_drawdown": [-0.10],
            "median_max_drawdown": [-0.08],
            "p5_max_drawdown": [-0.20],
            "p1_max_drawdown": [-0.30],
            "prob_drawdown_exceeds_20pct": [0.05],
        })
        return path_summary, terminal_dist, mc_metrics, drawdown_dist

    def _make_summary_dict(self) -> dict:
        return {
            "run_timestamp": "2025-01-01T00:00:00Z",
            "input": {"prices_rows": 100, "returns_rows": 99},
            "scenarios_run": {"s1": {"method": "gbm"}},
            "method_comparison": {},
            "config_used": {},
            "duration_sec": 1.0,
        }

    def test_save_simulations_writes_all_four_parquets(self, tmp_path):
        """All four mc_*.parquet files must be written — FR15–FR18."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=None,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=False,
        )

        for fname in ("mc_paths_summary.parquet", "mc_terminal_distribution.parquet",
                      "mc_metrics.parquet", "mc_drawdown_distribution.parquet"):
            assert (processed / fname).exists(), f"{fname} not written to processed_dir"

    def test_save_simulations_writes_summary_json(self, tmp_path):
        """monte_carlo_summary.json must be written to reports_dir — FR20."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=None,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=False,
        )

        json_path = reports / "monte_carlo_summary.json"
        assert json_path.exists(), "monte_carlo_summary.json was not written"
        with open(json_path) as f:
            data = json.load(f)
        assert "run_timestamp" in data, "run_timestamp missing from summary JSON"

    # §8 Test 16
    def test_save_full_paths_flag_false_no_full_parquet(self, tmp_path):
        """save_full_paths=False must not write mc_paths_full.parquet — §8 Test 16, AC §9."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=None,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=False,
        )

        assert not (exports / "mc_paths_full.parquet").exists(), (
            "mc_paths_full.parquet was written even though save_full_paths=False."
        )

    def test_save_full_paths_flag_true_writes_full_parquet(self, tmp_path):
        """save_full_paths=True must write mc_paths_full.parquet to exports_dir — FR19."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        # Build a minimal full_paths list in the format save_simulations expects
        n_sim, horizon = 10, 5
        paths_arr = np.ones((n_sim, horizon + 1)) * 100.0
        full_paths = [{"scenario_name": "s1", "ticker": "AAPL", "paths": paths_arr}]

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=full_paths,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=True,
        )

        assert (exports / "mc_paths_full.parquet").exists(), (
            "mc_paths_full.parquet was not written even though save_full_paths=True."
        )

    def test_save_simulations_parquet_round_trip(self, tmp_path):
        """Parquets can be read back with same shape and values — FR22, AC §9."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=None,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=False,
        )

        mc_metrics_back = pd.read_parquet(processed / "mc_metrics.parquet")
        assert len(mc_metrics_back) == len(mm), (
            f"mc_metrics.parquet row count changed: {len(mm)} → {len(mc_metrics_back)}"
        )
        assert float(mc_metrics_back["value"].iloc[0]) == pytest.approx(float(mm["value"].iloc[0])), (
            "mc_metrics.parquet value changed after round-trip."
        )

    def test_save_simulations_creates_dirs(self, tmp_path):
        """save_simulations must create output directories if they don't exist — FR15."""
        # Don't pre-create any directory
        processed = tmp_path / "new_processed"
        exports = tmp_path / "new_exports"
        reports = tmp_path / "new_reports"
        ps, td, mm, dd = self._make_minimal_dfs()

        save_simulations(
            path_summaries=ps,
            terminal_distributions=td,
            mc_metrics=mm,
            drawdown_distributions=dd,
            full_paths=None,
            summary_dict=self._make_summary_dict(),
            processed_dir=processed,
            exports_dir=exports,
            reports_dir=reports,
            save_full_paths=False,
        )

        assert processed.exists(), "save_simulations did not create processed_dir"
        assert reports.exists(), "save_simulations did not create reports_dir"

    def test_save_simulations_idempotent_overwrite(self, tmp_path):
        """Running save_simulations twice overwrites rather than appending — FR22."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        ps, td, mm, dd = self._make_minimal_dfs()
        summary = self._make_summary_dict()

        save_simulations(ps, td, mm, dd, None, summary, processed, exports, reports, False)
        save_simulations(ps, td, mm, dd, None, summary, processed, exports, reports, False)

        mc_metrics_back = pd.read_parquet(processed / "mc_metrics.parquet")
        assert len(mc_metrics_back) == len(mm), (
            f"mc_metrics.parquet grew on second write: expected {len(mm)}, got {len(mc_metrics_back)}"
        )


# ── Class: TestSchemaEnforcement ─────────────────────────────────────────────

class TestSchemaEnforcement:
    """Tests that pandera schemas are enforced at module boundaries — FR2, AC §9."""

    def test_load_data_schema_rejects_negative_prices(
        self, tmp_processed_dir, mc_scenarios_csv, mc_returns_parquet
    ):
        """prices_clean.parquet with negative close must fail schema validation — FR2."""
        import pandera

        bad_prices = _make_prices_df(["AAPL"], n=20)
        bad_prices.loc[bad_prices.index[5], "close"] = -10.0
        bad_prices.loc[bad_prices.index[5], "open"] = -10.0
        bad_prices.loc[bad_prices.index[5], "high"] = -10.0
        bad_prices.loc[bad_prices.index[5], "low"] = -10.0
        prices_path = tmp_processed_dir / "prices_clean.parquet"
        bad_prices.to_parquet(prices_path, engine="pyarrow", index=False, compression="snappy")

        with pytest.raises(Exception):
            load_data(tmp_processed_dir, mc_scenarios_csv)

    def test_load_data_schema_accepts_valid_data(
        self, tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, mc_scenarios_csv
    ):
        """Valid prices and returns must pass schema validation without exception — FR2."""
        # Should not raise
        prices, returns, scenarios = load_data(tmp_processed_dir, mc_scenarios_csv)
        assert len(prices) > 0, "prices DataFrame must not be empty after successful load"
        assert len(returns) > 0, "returns DataFrame must not be empty after successful load"


# ── Class: TestConfigCompliance ───────────────────────────────────────────────

class TestConfigCompliance:
    """Tests that outputs respect config thresholds — §6 (no magic numbers)."""

    def test_estimate_gbm_params_respects_trading_days_per_year(self):
        """sigma_annual uses the supplied trading_days_per_year, not a hardcoded 252 — §6."""
        df = _make_returns_df(["AAPL"], n=500)
        params_252 = estimate_gbm_params(df, 252, "historical")
        params_360 = estimate_gbm_params(df, 360, "historical")

        sigma_252 = float(params_252["sigma_annual"].iloc[0])
        sigma_360 = float(params_360["sigma_annual"].iloc[0])

        # sigma_annual_360 / sigma_annual_252 should equal sqrt(360/252)
        ratio = sigma_360 / sigma_252
        expected_ratio = np.sqrt(360 / 252)
        assert ratio == pytest.approx(expected_ratio, rel=1e-9), (
            f"sigma_annual ratio {ratio:.6f} != sqrt(360/252)={expected_ratio:.6f} — "
            "trading_days_per_year not used correctly."
        )

    def test_compute_simulated_var_cvar_uses_supplied_levels(self):
        """VaR keys must exactly match the supplied var_levels — §6."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, 30, 5000, rng)
        result = compute_simulated_var_cvar(paths, 100.0, [0.90, 0.95, 0.99], [0.90])
        assert "var_90" in result, "var_90 key missing when 0.90 is in var_levels"
        assert "var_95" in result, "var_95 key missing when 0.95 is in var_levels"
        assert "var_99" in result, "var_99 key missing when 0.99 is in var_levels"
        assert "cvar_90" in result, "cvar_90 key missing when 0.90 is in cvar_levels"

    @pytest.mark.parametrize("n_sim", [100, 1000, 5000])
    def test_simulate_gbm_respects_n_simulations(self, n_sim):
        """Number of simulated paths equals the requested n_simulations — §6."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=10, n_simulations=n_sim, rng=rng)
        assert paths.shape[0] == n_sim, (
            f"Expected {n_sim} paths, got {paths.shape[0]}"
        )

    @pytest.mark.parametrize("horizon", [5, 30, 90])
    def test_simulate_gbm_respects_horizon(self, horizon):
        """Number of time steps equals horizon + 1 (including day 0) — §6."""
        rng = np.random.default_rng(0)
        paths = simulate_gbm(100.0, 0.001, 0.02, horizon=horizon, n_simulations=100, rng=rng)
        assert paths.shape[1] == horizon + 1, (
            f"Expected {horizon + 1} columns (horizon+1), got {paths.shape[1]}"
        )
