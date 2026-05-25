"""
tests/unit/test_05-forecasting.py — Unit tests for src/forecasting.py
Uses synthetic time series only; no real data loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.forecasting import (
    check_stationarity,
    compute_forecast_metrics,
    fit_arima,
    fit_naive,
    walk_forward_validate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def white_noise_returns():
    """300 rows of white noise returns for two tickers."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    rows = []
    for ticker in ["AAA", "BBB"]:
        vals = rng.normal(0, 0.01, n)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "simple_return": float(vals[i]),
                    "log_return": float(vals[i]),
                }
            )
    df = pd.DataFrame(rows).astype(
        {"ticker": pd.StringDtype(), "simple_return": float, "log_return": float}
    )
    return df


@pytest.fixture
def ar1_series():
    """AR(1) series with phi=0.5 and 300 points."""
    rng = np.random.default_rng(0)
    n = 300
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.normal(0, 0.01)
    return pd.Series(y)


@pytest.fixture
def trending_series():
    """Linearly upward trending series."""
    return pd.Series(np.arange(200, dtype=float) * 0.1)


@pytest.fixture
def perfect_backtest():
    """Backtest DataFrame where predicted == actual."""
    n = 50
    actual = np.random.default_rng(7).normal(0, 0.01, n)
    df = pd.DataFrame(
        {
            "fold": np.zeros(n, dtype=int),
            "fold_start_date": pd.Timestamp("2023-01-01"),
            "fold_end_date": pd.Timestamp("2023-06-01"),
            "forecast_date": pd.bdate_range("2023-06-02", periods=n, freq="B"),
            "actual": actual,
            "predicted": actual.copy(),
            "lower_ci": actual - 0.02,
            "upper_ci": actual + 0.02,
        }
    )
    return df


@pytest.fixture
def synthetic_backtest():
    """Backtest with known directional accuracy = 0.75."""
    actual =    np.array([ 1.0, -1.0,  1.0,  1.0])
    predicted = np.array([ 0.5, -0.5,  0.5, -0.5])  # 3/4 correct sign
    df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "fold_start_date": pd.Timestamp("2023-01-01"),
            "fold_end_date": pd.Timestamp("2023-06-01"),
            "forecast_date": pd.bdate_range("2023-06-02", periods=4, freq="B"),
            "actual": actual,
            "predicted": predicted,
            "lower_ci": predicted - 0.5,
            "upper_ci": predicted + 0.5,
        }
    )
    return df


# ── Test 1: FileNotFoundError on missing scenarios.csv ───────────────────────


def test_load_data_raises_on_missing_scenarios_csv(tmp_path):
    from src.forecasting import load_data

    processed = tmp_path / "processed"
    processed.mkdir()

    # Create dummy parquet files that pass schema validation
    dates = pd.bdate_range("2023-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["AAA"] * 5, dtype=pd.StringDtype()),
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
            "volume": pd.array([1000] * 5, dtype=pd.Int64Dtype()),
        }
    )
    returns = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["AAA"] * 5, dtype=pd.StringDtype()),
            "simple_return": [0.01] * 5,
            "log_return": [0.01] * 5,
        }
    )
    prices.to_parquet(processed / "prices_clean.parquet", index=False)
    returns.to_parquet(processed / "returns_daily.parquet", index=False)

    with pytest.raises(FileNotFoundError, match="scenarios"):
        load_data(processed, tmp_path / "nonexistent.csv")


# ── Test 2: check_stationarity columns ───────────────────────────────────────


def test_check_stationarity_returns_expected_columns(white_noise_returns):
    result = check_stationarity(white_noise_returns)
    expected = {
        "ticker", "series_type", "adf_statistic", "p_value",
        "is_stationary", "critical_1pct", "critical_5pct",
    }
    assert expected.issubset(set(result.columns))
    assert set(result["series_type"].unique()) == {"returns", "log_prices"}
    assert result["is_stationary"].dtype == bool


# ── Test 3: fit_arima recovers known AR(1) params ────────────────────────────


def test_fit_arima_recovers_known_ar1_params(ar1_series):
    result = fit_arima(ar1_series, max_p=5, max_q=2, max_d=1, seasonal=False, horizon=5, confidence_level=0.95)
    assert result["converged"] is True
    assert len(result["forecast"]) == 5
    assert result["order"] is not None
    # auto_arima should pick p >= 1 for AR(1) data
    p, d, q = result["order"]
    assert p >= 1


# ── Test 4: random walk forecast is constant ─────────────────────────────────


def test_fit_naive_random_walk_forecast_is_constant():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fit_naive(series, method="random_walk", horizon=5, confidence_level=0.95)
    assert result["converged"] is True
    assert np.allclose(result["forecast"], 5.0), "random walk must repeat last value"


# ── Test 5: drift model extrapolates upward trend ────────────────────────────


def test_fit_naive_drift_extrapolates_trend(trending_series):
    result = fit_naive(trending_series, method="drift", horizon=5, confidence_level=0.95)
    assert result["converged"] is True
    fc = result["forecast"]
    # Drift on linearly increasing series must be monotonically increasing
    assert np.all(np.diff(fc) > 0), "drift forecast must increase for upward trend"


# ── Test 6: walk-forward — no train/test leakage ────────────────────────────


def test_walk_forward_no_train_test_leakage(ar1_series):
    dates = pd.Series(pd.bdate_range("2023-01-01", periods=len(ar1_series), freq="B"))

    def _naive_fn(series, **kwargs):
        return fit_naive(series, method="random_walk", horizon=kwargs["horizon"], confidence_level=kwargs["confidence_level"])

    bt = walk_forward_validate(
        series=ar1_series,
        dates=dates,
        model_fn=_naive_fn,
        model_kwargs={"horizon": 10, "confidence_level": 0.95},
        train_initial_days=100,
        step_days=20,
        horizon=10,
        expanding=True,
    )
    assert len(bt) > 0
    for _, row in bt.iterrows():
        assert row["fold_end_date"] < row["forecast_date"], (
            f"fold_end_date {row['fold_end_date']} must be < forecast_date {row['forecast_date']}"
        )


# ── Test 7: walk-forward expanding window grows ──────────────────────────────


def test_walk_forward_expanding_window_grows(ar1_series):
    dates = pd.Series(pd.bdate_range("2023-01-01", periods=len(ar1_series), freq="B"))
    window_sizes = []

    def _recording_fn(series, **kwargs):
        window_sizes.append(len(series))
        return fit_naive(series, method="random_walk", horizon=kwargs["horizon"], confidence_level=kwargs["confidence_level"])

    walk_forward_validate(
        series=ar1_series,
        dates=dates,
        model_fn=_recording_fn,
        model_kwargs={"horizon": 10, "confidence_level": 0.95},
        train_initial_days=100,
        step_days=20,
        horizon=10,
        expanding=True,
    )
    assert len(window_sizes) >= 2
    for i in range(1, len(window_sizes)):
        assert window_sizes[i] >= window_sizes[i - 1], (
            f"Expanding window shrank at fold {i}: {window_sizes[i-1]} → {window_sizes[i]}"
        )


# ── Test 8: coverage rate in [0, 1] ──────────────────────────────────────────


def test_compute_metrics_coverage_in_zero_one_range():
    rng = np.random.default_rng(99)
    n = 100
    actual = rng.normal(0, 1, n)
    predicted = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "fold": np.zeros(n, dtype=int),
            "fold_start_date": pd.Timestamp("2023-01-01"),
            "fold_end_date": pd.Timestamp("2023-06-01"),
            "forecast_date": pd.bdate_range("2023-06-02", periods=n, freq="B"),
            "actual": actual,
            "predicted": predicted,
            "lower_ci": predicted - 2.0,
            "upper_ci": predicted + 2.0,
        }
    )
    metrics = compute_forecast_metrics(df)
    assert 0.0 <= metrics["coverage_rate"] <= 1.0


# ── Test 9: RMSE on perfect forecast is zero ─────────────────────────────────


def test_compute_metrics_rmse_on_perfect_forecast_is_zero(perfect_backtest):
    metrics = compute_forecast_metrics(perfect_backtest)
    assert abs(metrics["rmse"]) < 1e-10, f"Expected RMSE=0, got {metrics['rmse']}"
    assert abs(metrics["mae"]) < 1e-10


# ── Test 10: directional accuracy on synthetic data ──────────────────────────


def test_compute_metrics_directional_accuracy_on_synthetic(synthetic_backtest):
    metrics = compute_forecast_metrics(synthetic_backtest)
    # actual=[1,-1,1,1], predicted=[0.5,-0.5,0.5,-0.5]
    # signs match: T, T, T, F → 3/4 = 0.75
    assert abs(metrics["directional_accuracy"] - 0.75) < 1e-6


# ── Test 11: ARIMA failure falls back to naive ───────────────────────────────


def test_arima_failure_falls_back_to_naive():
    """A constant series causes ARIMA to fail; walk_forward should fall back to naive."""
    constant = pd.Series([5.0] * 200)
    dates = pd.Series(pd.bdate_range("2023-01-01", periods=200, freq="B"))

    def _bad_arima(series, **kwargs):
        return {"converged": False}

    bt = walk_forward_validate(
        series=constant,
        dates=dates,
        model_fn=_bad_arima,
        model_kwargs={"horizon": 5, "confidence_level": 0.95},
        train_initial_days=100,
        step_days=20,
        horizon=5,
        expanding=True,
    )
    # Should have produced forecasts via fallback
    assert len(bt) > 0
    assert not bt["predicted"].isna().any()


# ── Test 12: CI ordering invariant ───────────────────────────────────────────


def test_invariant_lower_ci_leq_forecast_leq_upper_ci():
    series = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300))
    for method in ("random_walk", "drift", "mean"):
        result = fit_naive(series, method=method, horizon=30, confidence_level=0.95)
        assert result["converged"] is True
        fc = result["forecast"]
        lci = result["lower_ci"]
        uci = result["upper_ci"]
        assert np.all(lci <= fc + 1e-10), f"{method}: lower_ci > forecast"
        assert np.all(fc <= uci + 1e-10), f"{method}: forecast > upper_ci"


# ── Test 13: idempotent re-run with same seed ────────────────────────────────


def test_idempotent_rerun_with_seed(ar1_series):
    """Two identical fit_arima calls with the same numpy seed must return identical forecasts."""
    np.random.seed(42)
    r1 = fit_arima(ar1_series, max_p=3, max_q=2, max_d=1, seasonal=False, horizon=5, confidence_level=0.95)
    np.random.seed(42)
    r2 = fit_arima(ar1_series, max_p=3, max_q=2, max_d=1, seasonal=False, horizon=5, confidence_level=0.95)
    if r1["converged"] and r2["converged"]:
        np.testing.assert_array_almost_equal(r1["forecast"], r2["forecast"], decimal=8)


# ── Test 14: unknown model raises ValueError ─────────────────────────────────


def test_unknown_model_raises_value_error(tmp_path):
    from src.forecasting import load_data

    processed = tmp_path / "processed"
    processed.mkdir()
    dates = pd.bdate_range("2023-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["AAA"] * 5, dtype=pd.StringDtype()),
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.5] * 5,
            "volume": pd.array([1000] * 5, dtype=pd.Int64Dtype()),
        }
    )
    returns = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["AAA"] * 5, dtype=pd.StringDtype()),
            "simple_return": [0.01] * 5,
            "log_return": [0.01] * 5,
        }
    )
    prices.to_parquet(processed / "prices_clean.parquet", index=False)
    returns.to_parquet(processed / "returns_daily.parquet", index=False)

    scenarios_csv = tmp_path / "scenarios.csv"
    scenarios_csv.write_text(
        "scenario_name,model,target,horizon_days,confidence_level\n"
        "bad_scenario,lstm_fantasy,returns,30,0.95\n"
    )

    with pytest.raises(ValueError, match="Unknown model"):
        load_data(processed, scenarios_csv)


# ── Test 15: min observations skips short ticker ─────────────────────────────


def test_min_observations_skips_short_ticker():
    from src.forecasting import run_scenario

    n = 50  # fewer than default MIN_OBSERVATIONS_FOR_FORECAST (100)
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["SHORT"] * n, dtype=pd.StringDtype()),
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.5] * n,
            "volume": pd.array([1000] * n, dtype=pd.Int64Dtype()),
        }
    )
    returns = pd.DataFrame(
        {
            "date": dates,
            "ticker": pd.array(["SHORT"] * n, dtype=pd.StringDtype()),
            "simple_return": np.random.default_rng(5).normal(0, 0.01, n).tolist(),
            "log_return": np.random.default_rng(5).normal(0, 0.01, n).tolist(),
        }
    )
    scenario_row = pd.Series(
        {
            "scenario_name": "test_short",
            "model": "naive_random_walk",
            "target": "returns",
            "horizon_days": 10,
            "confidence_level": 0.95,
            "tickers": "all",
        }
    )
    fc_df, m_df = run_scenario(
        scenario_row, prices, returns, {"min_observations": 100}
    )
    # With only 50 rows and min_obs=100, ticker should be skipped
    assert len(fc_df) == 0
    assert len(m_df) == 0
