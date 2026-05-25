"""
tests/unit/test_monte_carlo.py — Unit tests for src/monte_carlo.py (Step 6).

Uses synthetic white-noise and AR(1) fixtures from conftest.py.
No network calls; no real data files required.
"""

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    load_data,
    estimate_gbm_params,
    simulate_gbm,
    simulate_bootstrap,
    simulate_block_bootstrap,
    simulate_portfolio_correlated,
    compute_path_summary,
    compute_simulated_var_cvar,
    compute_path_drawdowns,
    compute_terminal_distribution,
    save_simulations,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_returns_df(tickers, n=200, seed=0):
    """White-noise returns DataFrame for multiple tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    rows = []
    for ticker in tickers:
        ret = rng.normal(0.001, 0.015, size=n)
        for d, r in zip(dates, ret):
            rows.append({
                "date": d, "ticker": ticker,
                "simple_return": float(r),
                "log_return": float(np.log1p(r)),
            })
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_gbm_params(tickers, mu=0.001, sigma=0.02):
    """Minimal GBM params DataFrame."""
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


# ── Test 1: Missing scenarios CSV raises FileNotFoundError ────────────────────

def test_load_data_raises_on_missing_scenarios_csv(tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, tmp_path):
    missing_csv = tmp_path / "nonexistent_mc_scenarios.csv"
    with pytest.raises(FileNotFoundError, match="mc_scenarios"):
        load_data(tmp_processed_dir, missing_csv)


# ── Test 2: GBM params match hand calculation ─────────────────────────────────

def test_estimate_gbm_params_matches_hand_calc():
    rng = np.random.default_rng(7)
    n = 2000
    true_mu = 0.002
    true_sigma = 0.018
    log_rets = rng.normal(true_mu, true_sigma, size=n)
    dates = pd.bdate_range("2023-01-01", periods=n)
    df = pd.DataFrame({
        "date": dates,
        "ticker": pd.array(["TEST"] * n, dtype="string"),
        "simple_return": log_rets,
        "log_return": log_rets,
    })

    params = estimate_gbm_params(df, 252, "historical")
    row = params[params["ticker"] == "TEST"].iloc[0]

    # mu_daily should be close to true_mu (within ~3 standard errors)
    se_mu = true_sigma / np.sqrt(n)
    assert abs(row["mu_daily"] - true_mu) < 4 * se_mu
    # sigma_daily should be close to true_sigma
    assert abs(row["sigma_daily"] - true_sigma) < 0.003
    # zero drift when drift_method="zero"
    params_zero = estimate_gbm_params(df, 252, "zero")
    assert params_zero[params_zero["ticker"] == "TEST"].iloc[0]["mu_daily"] == 0.0


# ── Test 3: Zero volatility → deterministic path ──────────────────────────────

def test_gbm_zero_volatility_produces_deterministic_path():
    rng = np.random.default_rng(42)
    mu = 0.001
    paths = simulate_gbm(s0=100.0, mu_daily=mu, sigma_daily=0.0, horizon=10, n_simulations=200, rng=rng)
    # All paths must be identical to each other
    assert np.allclose(paths, paths[0])
    # Path should follow S0 * exp(mu * t)
    expected = 100.0 * np.exp(mu * np.arange(11))
    assert np.allclose(paths[0], expected, rtol=1e-10)


# ── Test 4: GBM terminal mean converges to E[S_T] ────────────────────────────

def test_gbm_mean_converges_to_expected():
    rng = np.random.default_rng(0)
    mu, sigma, s0, horizon, n = 0.001, 0.02, 100.0, 5, 50_000
    paths = simulate_gbm(s0, mu, sigma, horizon, n, rng)
    terminal_mean = np.mean(paths[:, -1])
    # E[S_T] = S0 * exp(mu * T)  (GBM with Itô correction preserves this)
    expected_mean = s0 * np.exp(mu * horizon)
    assert abs(terminal_mean / expected_mean - 1.0) < 0.01  # within 1%


# ── Test 5: Bootstrap samples from historical returns ─────────────────────────

def test_bootstrap_uses_with_replacement():
    historical = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    rng = np.random.default_rng(99)
    paths = simulate_bootstrap(s0=100.0, historical_returns=historical, horizon=20, n_simulations=500, rng=rng)
    assert paths.shape == (500, 21)
    # Column 0 must be s0
    assert np.allclose(paths[:, 0], 100.0)
    # No NaN or negative values
    assert not np.any(np.isnan(paths))
    assert np.all(paths > 0)


# ── Test 6: Block bootstrap preserves serial correlation ─────────────────────

def test_block_bootstrap_preserves_serial_correlation():
    # AR(1) process: r[t] = 0.6 * r[t-1] + noise
    rng_gen = np.random.default_rng(1)
    n = 600
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.6 * ar1[i - 1] + rng_gen.standard_normal() * 0.01

    def mean_lag1_autocorr(paths: np.ndarray, n_paths: int = 100) -> float:
        returns = np.diff(np.log(np.clip(paths[:n_paths], 1e-10, None)), axis=1)
        ac = []
        for r in returns:
            if np.std(r) > 1e-12:
                ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        return float(np.nanmean(ac)) if ac else 0.0

    rng1 = np.random.default_rng(42)
    simple_paths = simulate_bootstrap(100.0, ar1, horizon=80, n_simulations=200, rng=rng1)
    rng2 = np.random.default_rng(42)
    block_paths = simulate_block_bootstrap(100.0, ar1, horizon=80, n_simulations=200, block_size=15, rng=rng2)

    simple_ac = mean_lag1_autocorr(simple_paths)
    block_ac = mean_lag1_autocorr(block_paths)

    # Block bootstrap preserves more autocorrelation than independent resampling
    assert block_ac > simple_ac


# ── Test 7: Identity correlation → effectively independent ───────────────────

def test_cholesky_on_identity_equals_independent():
    tickers = ["A", "B"]
    s0 = {"A": 100.0, "B": 200.0}
    gbm_params = _make_gbm_params(tickers)
    identity_corr = pd.DataFrame(np.eye(2), index=tickers, columns=tickers)
    weights = {"A": 0.5, "B": 0.5}

    rng = np.random.default_rng(0)
    per_ticker, portfolio = simulate_portfolio_correlated(
        s0, gbm_params, identity_corr, weights, 30, 2000, rng
    )

    assert set(per_ticker.keys()) == {"A", "B"}
    assert per_ticker["A"].shape == (2000, 31)
    assert portfolio.shape == (2000, 31)

    # With identity corr, per-ticker terminal returns should be near-uncorrelated
    ret_A = per_ticker["A"][:, -1] / 100.0 - 1
    ret_B = per_ticker["B"][:, -1] / 200.0 - 1
    corr = np.corrcoef(ret_A, ret_B)[0, 1]
    assert abs(corr) < 0.15  # should be near 0 with identity correlation


# ── Test 8: Non-PD correlation matrix is handled without crash ────────────────

def test_cholesky_handles_non_pd_matrix():
    tickers = ["A", "B", "C"]
    s0 = {t: 100.0 for t in tickers}
    gbm_params = _make_gbm_params(tickers)
    weights = {t: 1 / 3 for t in tickers}

    # Singular (non-PD) matrix: rows 0 and 1 are identical
    bad_vals = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    bad_corr = pd.DataFrame(bad_vals, index=tickers, columns=tickers)

    rng = np.random.default_rng(5)
    # Must not raise; should produce valid paths after PD projection
    per_ticker, portfolio = simulate_portfolio_correlated(
        s0, gbm_params, bad_corr, weights, 10, 200, rng
    )
    assert portfolio.shape == (200, 11)
    assert not np.any(np.isnan(portfolio))
    assert np.all(portfolio > 0)


# ── Test 9: var_99 >= var_95 ──────────────────────────────────────────────────

def test_var_99_geq_var_95():
    rng = np.random.default_rng(0)
    paths = simulate_gbm(100.0, 0.001, 0.02, 30, 10_000, rng)
    result = compute_simulated_var_cvar(paths, 100.0, [0.95, 0.99], [0.95])
    assert result["var_99"] >= result["var_95"], (
        f"var_99={result['var_99']:.4f} < var_95={result['var_95']:.4f}"
    )


# ── Test 10: cvar_95 >= var_95 ───────────────────────────────────────────────

def test_cvar_geq_var():
    rng = np.random.default_rng(0)
    paths = simulate_gbm(100.0, 0.001, 0.02, 30, 10_000, rng)
    result = compute_simulated_var_cvar(paths, 100.0, [0.95], [0.95])
    assert result["cvar_95"] >= result["var_95"], (
        f"cvar_95={result['cvar_95']:.4f} < var_95={result['var_95']:.4f}"
    )


# ── Test 11: All max drawdown values ≤ 0 ─────────────────────────────────────

def test_path_drawdowns_all_non_positive():
    rng = np.random.default_rng(0)
    paths = simulate_gbm(100.0, 0.001, 0.02, 30, 2000, rng)
    result = compute_path_drawdowns(paths)
    assert result["mean_max_drawdown"] <= 0.0
    assert result["median_max_drawdown"] <= 0.0
    assert result["p5_max_drawdown"] <= 0.0
    assert result["p1_max_drawdown"] <= 0.0
    # p1 must be ≤ p5 (p1 is the worst)
    assert result["p1_max_drawdown"] <= result["p5_max_drawdown"]


# ── Test 12: Percentile bands are monotonic ───────────────────────────────────

def test_percentile_bands_monotonic():
    rng = np.random.default_rng(0)
    paths = simulate_gbm(100.0, 0.001, 0.02, 30, 5_000, rng)
    dates = pd.bdate_range("2025-01-01", periods=31)
    percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]
    summary = compute_path_summary(paths, percentiles, dates)

    pivot = summary.pivot_table(index="day_offset", columns="percentile", values="value")
    for p_lo, p_hi in [(5.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 95.0)]:
        assert (pivot[p_lo] <= pivot[p_hi]).all(), (
            f"Monotonicity violated: P{int(p_lo)} > P{int(p_hi)}"
        )


# ── Test 13: Same seed → identical paths (idempotency) ───────────────────────

def test_idempotent_rerun_with_seed():
    rng1 = np.random.default_rng(42)
    paths1 = simulate_gbm(100.0, 0.001, 0.02, 15, 1_000, rng1)

    rng2 = np.random.default_rng(42)
    paths2 = simulate_gbm(100.0, 0.001, 0.02, 15, 1_000, rng2)

    assert np.array_equal(paths1, paths2), "Paths differ with the same seed — not idempotent."


# ── Test 14: Different scenario names → different random draws ────────────────

def test_per_scenario_seed_produces_different_paths():
    base_seed = 42
    seed_a = (base_seed + hash("scenario_alpha")) % (2 ** 32)
    seed_b = (base_seed + hash("scenario_beta")) % (2 ** 32)

    assert seed_a != seed_b, "Scenario seeds collided — hash function needs adjustment."

    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    paths_a = simulate_gbm(100.0, 0.001, 0.02, 10, 200, rng_a)
    paths_b = simulate_gbm(100.0, 0.001, 0.02, 10, 200, rng_b)

    assert not np.array_equal(paths_a, paths_b), "Different scenario seeds produced identical paths."


# ── Test 15: Unknown method in scenarios CSV raises ValueError ────────────────

def test_unknown_method_raises_value_error(tmp_processed_dir, mc_prices_parquet, mc_returns_parquet, tmp_path):
    bad_csv = tmp_path / "bad_mc_scenarios.csv"
    pd.DataFrame([{
        "scenario_name": "bad_scenario",
        "method": "neural_net_lstm",
        "horizon_days": 30,
        "n_simulations": 100,
    }]).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Unknown method"):
        load_data(tmp_processed_dir, bad_csv)


# ── Test 16: MC_SAVE_FULL_PATHS=False → no mc_paths_full.parquet ─────────────

def test_save_full_paths_flag_respected(tmp_path):
    processed_dir = tmp_path / "processed"
    exports_dir = tmp_path / "exports"
    reports_dir = tmp_path / "reports"
    for d in [processed_dir, exports_dir, reports_dir]:
        d.mkdir()

    mc_metrics = pd.DataFrame({
        "scenario_name": pd.array(["s1"], dtype="string"),
        "ticker": pd.array(["AAPL"], dtype="string"),
        "method": pd.array(["gbm"], dtype="string"),
        "metric_name": pd.array(["var_95"], dtype="string"),
        "value": [0.02],
    })

    save_simulations(
        path_summaries=pd.DataFrame(),
        terminal_distributions=pd.DataFrame(),
        mc_metrics=mc_metrics,
        drawdown_distributions=pd.DataFrame(),
        full_paths=None,
        summary_dict={
            "run_timestamp": "2025-01-01T00:00:00Z",
            "scenarios_run": {},
        },
        processed_dir=processed_dir,
        exports_dir=exports_dir,
        reports_dir=reports_dir,
        save_full_paths=False,
    )

    assert not (exports_dir / "mc_paths_full.parquet").exists(), (
        "mc_paths_full.parquet was written even though save_full_paths=False."
    )
    # The summary JSON should exist
    assert (reports_dir / "monte_carlo_summary.json").exists()
