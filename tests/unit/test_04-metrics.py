"""
tests/unit/test_04-metrics.py

Unit tests for src/metrics.py (Step 4 — Portfolio Metrics).

All test logic is derived exclusively from the spec at
.claude/specs/04-metrics.md.  No implementation details are assumed
beyond public function signatures and the output contracts documented
in the spec.

Hand-computed reference values are provided for CAGR (Test 4) and
VaR (Test 9) so the formulas are verified end-to-end.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_drawdown_series,
    compute_portfolio_metrics,
    compute_return_metrics,
    compute_risk_adjusted,
    compute_risk_metrics,
    compute_rolling_metrics,
    load_data,
    save_metrics,
)


# ---------------------------------------------------------------------------
# Module-level helpers / fixture factories
# ---------------------------------------------------------------------------

from config import METRICS_TRADING_DAYS_PER_YEAR as TRADING_DAYS
from config import RISK_FREE_RATE
VAR_LEVELS = [0.95, 0.99]
CVAR_LEVELS = [0.95]


def _make_prices(
    tickers: list[str],
    close_values: dict[str, list[float]],
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """
    Build a minimal prices_clean DataFrame in long format.
    close_values maps ticker -> list of daily close prices.
    All other OHLC fields are set to consistent synthetic values.
    """
    rows = []
    for ticker in tickers:
        closes = close_values[ticker]
        dates = pd.date_range(start, periods=len(closes), freq="B")
        for d, c in zip(dates, closes):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.98,
                    "close": float(c),
                    "volume": pd.array([1_000_000], dtype=pd.Int64Dtype())[0],
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_returns(
    tickers: list[str],
    return_values: dict[str, list[float]],
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """
    Build a minimal returns_daily DataFrame in long format.
    return_values maps ticker -> list of simple_return values.
    log_return is approximated as log(1 + r).
    """
    rows = []
    for ticker in tickers:
        rets = return_values[ticker]
        dates = pd.date_range(start, periods=len(rets), freq="B")
        for d, r in zip(dates, rets):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "simple_return": float(r),
                    "log_return": float(np.log(1.0 + r)),
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_returns_two_tickers():
    """
    Two tickers (AAPL, MSFT) with 10 non-zero returns each.
    Returns are varied enough to produce non-degenerate VaR/CVaR/Sharpe.
    """
    aapl_rets = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04, -0.015, 0.025, -0.005]
    msft_rets = [0.02, -0.01, 0.015, -0.025, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01]
    return _make_returns(["AAPL", "MSFT"], {"AAPL": aapl_rets, "MSFT": msft_rets})


@pytest.fixture()
def simple_prices_two_tickers():
    """
    Two tickers with 10 trading days of price data (starts at 100).
    Prices move gently so drawdown is non-zero but bounded.
    """
    aapl_closes = [100, 101, 99, 102, 101, 103, 100, 104, 102, 105]
    msft_closes = [200, 202, 198, 204, 202, 206, 200, 208, 204, 210]
    return _make_prices(
        ["AAPL", "MSFT"],
        {"AAPL": aapl_closes, "MSFT": msft_closes},
    )


@pytest.fixture()
def flat_returns_one_ticker():
    """Single ticker with ALL returns == 0 (flat-lined / zero vol)."""
    rets = [0.0] * 20
    return _make_returns(["FLAT"], {"FLAT": rets})


@pytest.fixture()
def flat_prices_one_ticker():
    """Matching prices for the flat-return ticker (constant price = 100)."""
    closes = [100.0] * 20
    return _make_prices(["FLAT"], {"FLAT": closes})


@pytest.fixture()
def monotone_rising_prices():
    """
    Single ticker whose price rises every single day.
    Expected: drawdown is always 0 (never falls below running peak).
    """
    closes = [100.0 + i for i in range(10)]  # 100, 101, ..., 109
    return _make_prices(["RISE"], {"RISE": closes})


@pytest.fixture()
def returns_with_benchmark():
    """
    Three tickers: AAPL, MSFT, BENCH.
    BENCH acts as the benchmark ticker.
    """
    aapl_rets = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04, -0.015, 0.025, -0.005]
    msft_rets = [0.02, -0.01, 0.015, -0.025, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01]
    bench_rets = [0.005, -0.01, 0.02, -0.005, 0.01, -0.015, 0.02, -0.01, 0.015, -0.005]
    return _make_returns(
        ["AAPL", "BENCH", "MSFT"],
        {"AAPL": aapl_rets, "BENCH": bench_rets, "MSFT": msft_rets},
    )


# ---------------------------------------------------------------------------
# Test 1 — load_data raises FileNotFoundError on missing prices
# ---------------------------------------------------------------------------

class TestLoadData:
    """FR1, §6 — load_data error handling."""

    def test_load_data_raises_on_missing_prices(self, tmp_processed_dir):
        """
        FileNotFoundError must be raised (with a hint to run Step 2) when
        prices_clean.parquet is absent.  Returns file is also absent here,
        but the spec says prices is checked first.
        """
        with pytest.raises(FileNotFoundError, match="Step 2"):
            load_data(tmp_processed_dir)

    def test_load_data_raises_on_missing_returns(
        self, tmp_processed_dir, simple_prices_two_tickers
    ):
        """
        FileNotFoundError must be raised (with a hint to run Step 2) when
        returns_daily.parquet is absent but prices_clean.parquet exists.
        """
        _write_parquet(
            simple_prices_two_tickers,
            tmp_processed_dir / "prices_clean.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Step 2"):
            load_data(tmp_processed_dir)

    def test_load_data_raises_on_nan_in_simple_return(
        self, tmp_processed_dir, simple_prices_two_tickers, simple_returns_two_tickers
    ):
        """
        ValueError is raised when returns_daily.parquet contains NaN
        simple_return values (should have been cleaned in Step 2 — §6).
        """
        bad_returns = simple_returns_two_tickers.copy()
        bad_returns.loc[bad_returns.index[0], "simple_return"] = np.nan
        _write_parquet(simple_prices_two_tickers, tmp_processed_dir / "prices_clean.parquet")
        _write_parquet(bad_returns, tmp_processed_dir / "returns_daily.parquet")

        with pytest.raises(ValueError):
            load_data(tmp_processed_dir)

    def test_load_data_returns_tuple_of_two_dataframes(
        self, tmp_processed_dir, simple_prices_two_tickers, simple_returns_two_tickers
    ):
        """
        Happy path: load_data returns (prices, returns) when both files exist
        and data is valid.
        """
        _write_parquet(simple_prices_two_tickers, tmp_processed_dir / "prices_clean.parquet")
        _write_parquet(simple_returns_two_tickers, tmp_processed_dir / "returns_daily.parquet")

        result = load_data(tmp_processed_dir)

        assert isinstance(result, tuple), "load_data must return a tuple"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        prices_out, returns_out = result
        assert isinstance(prices_out, pd.DataFrame), "First element must be a DataFrame"
        assert isinstance(returns_out, pd.DataFrame), "Second element must be a DataFrame"


# ---------------------------------------------------------------------------
# Test 2 — CAGR hand-computed (FR3, §13)
# ---------------------------------------------------------------------------

class TestComputeReturnMetrics:
    """FR3 — per-ticker return metrics."""

    def test_compute_return_metrics_cagr_matches_hand_calc(self):
        """
        Hand-computed CAGR for a constant +1 % daily return series of 5 days:
          total_return = (1.01)^5 - 1 = 0.05101005...
          cagr         = (1 + total_return)^(252/5) - 1

        Spec formula (§13): (1 + total_return)^(trading_days_per_year / n_days) - 1
        """
        r = 0.01
        n_days = 5
        tdy = 252
        rets_list = [r] * n_days
        returns_df = _make_returns(["TEST"], {"TEST": rets_list})
        prices_df = _make_prices(
            ["TEST"],
            {"TEST": [100.0 * (1.0 + r) ** i for i in range(n_days)]},
        )

        result = compute_return_metrics(returns_df, prices_df, tdy)

        total_ret_row = result.loc[result["metric_name"] == "total_return"]
        cagr_row = result.loc[result["metric_name"] == "cagr"]

        expected_total = (1.0 + r) ** n_days - 1.0
        expected_cagr = (1.0 + expected_total) ** (tdy / n_days) - 1.0

        actual_total = float(total_ret_row["value"].iloc[0])
        actual_cagr = float(cagr_row["value"].iloc[0])

        assert actual_total == pytest.approx(expected_total, rel=1e-9), (
            f"total_return mismatch: expected {expected_total:.8f}, got {actual_total:.8f}"
        )
        assert actual_cagr == pytest.approx(expected_cagr, rel=1e-9), (
            f"CAGR mismatch: expected {expected_cagr:.8f}, got {actual_cagr:.8f}"
        )

    def test_compute_return_metrics_output_columns(self, simple_returns_two_tickers, simple_prices_two_tickers):
        """
        Output DataFrame must contain columns [ticker, metric_name, value, category]
        and category must be 'return' for all rows (FR3, §4.1).
        """
        result = compute_return_metrics(
            simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS
        )

        expected_cols = {"ticker", "metric_name", "value", "category"}
        assert expected_cols.issubset(result.columns), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )
        assert (result["category"] == "return").all(), (
            "All rows from compute_return_metrics must have category='return'"
        )

    def test_compute_return_metrics_expected_metric_names(
        self, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Per spec FR3, each ticker must produce exactly these metric_names:
        total_return, cagr, mean_daily_return, mean_monthly_return,
        mean_annual_return, best_day_return, worst_day_return, pct_positive_days.
        """
        expected_names = {
            "total_return",
            "cagr",
            "mean_daily_return",
            "mean_monthly_return",
            "mean_annual_return",
            "best_day_return",
            "worst_day_return",
            "pct_positive_days",
        }
        result = compute_return_metrics(
            simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS
        )
        for ticker in ["AAPL", "MSFT"]:
            ticker_names = set(result.loc[result["ticker"] == ticker, "metric_name"])
            missing = expected_names - ticker_names
            assert not missing, (
                f"Ticker {ticker} is missing metric_names: {missing}"
            )

    def test_compute_return_metrics_does_not_mutate_input(
        self, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Spec §5: input DataFrames must not be modified (defensive copy at entry).
        """
        returns_copy = simple_returns_two_tickers.copy(deep=True)
        prices_copy = simple_prices_two_tickers.copy(deep=True)

        compute_return_metrics(simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS)

        pd.testing.assert_frame_equal(
            simple_returns_two_tickers.reset_index(drop=True),
            returns_copy.reset_index(drop=True),
            check_dtype=True,
            obj="returns DataFrame was mutated by compute_return_metrics",
        )
        pd.testing.assert_frame_equal(
            simple_prices_two_tickers.reset_index(drop=True),
            prices_copy.reset_index(drop=True),
            check_dtype=True,
            obj="prices DataFrame was mutated by compute_return_metrics",
        )


# ---------------------------------------------------------------------------
# Tests 3, 4, 9 — Risk metrics: VaR positive, CVaR ≥ VaR, hand-computed VaR
# ---------------------------------------------------------------------------

class TestComputeRiskMetrics:
    """FR4 — per-ticker risk metrics."""

    def test_compute_risk_metrics_var_is_positive(self, simple_returns_two_tickers):
        """
        Spec §5: VaR must be returned as a positive loss value.
        var_95 and var_99 must both be > 0 for a return series with losses.
        """
        result = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )

        for ticker in ["AAPL", "MSFT"]:
            for metric in ["var_95", "var_99"]:
                val = float(
                    result.loc[
                        (result["ticker"] == ticker) & (result["metric_name"] == metric),
                        "value",
                    ].iloc[0]
                )
                assert val >= 0, (
                    f"{ticker} {metric} must be a positive loss value, got {val}"
                )

    def test_compute_risk_metrics_cvar_geq_var(self, simple_returns_two_tickers):
        """
        Spec §7 invariant: CVaR_95 ≥ VaR_95 in magnitude for every ticker.
        CVaR is the expected shortfall beyond the VaR threshold, so it cannot
        be smaller than VaR.
        """
        result = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )

        for ticker in ["AAPL", "MSFT"]:
            var95 = float(
                result.loc[
                    (result["ticker"] == ticker) & (result["metric_name"] == "var_95"),
                    "value",
                ].iloc[0]
            )
            cvar95 = float(
                result.loc[
                    (result["ticker"] == ticker) & (result["metric_name"] == "cvar_95"),
                    "value",
                ].iloc[0]
            )
            assert cvar95 >= var95 - 1e-10, (
                f"{ticker}: CVaR_95 ({cvar95:.6f}) must be >= VaR_95 ({var95:.6f})"
            )

    def test_compute_risk_metrics_var95_leq_var99(self, simple_returns_two_tickers):
        """
        Spec §7 invariant: VaR_95 ≤ VaR_99 in magnitude.
        A 99th-percentile loss is at least as large as a 95th-percentile loss.
        """
        result = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )

        for ticker in ["AAPL", "MSFT"]:
            var95 = float(
                result.loc[
                    (result["ticker"] == ticker) & (result["metric_name"] == "var_95"),
                    "value",
                ].iloc[0]
            )
            var99 = float(
                result.loc[
                    (result["ticker"] == ticker) & (result["metric_name"] == "var_99"),
                    "value",
                ].iloc[0]
            )
            assert var95 <= var99 + 1e-10, (
                f"{ticker}: VaR_95 ({var95:.6f}) must be <= VaR_99 ({var99:.6f})"
            )

    def test_compute_risk_metrics_var_hand_computed(self):
        """
        Hand-computed VaR_95 verification.

        Spec formula (§13): np.quantile(returns, 1 - confidence_level)
        then take absolute value.

        For returns = [-0.05, -0.03, 0.0, 0.02, 0.04]:
          np.quantile(returns, 0.05) = value at 5th percentile
          expected_var_95 = abs(np.quantile(returns, 0.05))
        """
        raw_returns = [-0.05, -0.03, 0.0, 0.02, 0.04]
        expected_var_95 = float(abs(np.quantile(raw_returns, 1.0 - 0.95)))

        returns_df = _make_returns(["HAND"], {"HAND": raw_returns})
        result = compute_risk_metrics(returns_df, TRADING_DAYS, [0.95], [0.95])

        actual_var_95 = float(
            result.loc[
                (result["ticker"] == "HAND") & (result["metric_name"] == "var_95"),
                "value",
            ].iloc[0]
        )
        assert actual_var_95 == pytest.approx(expected_var_95, rel=1e-9), (
            f"VaR_95 mismatch: expected {expected_var_95:.8f}, got {actual_var_95:.8f}"
        )

    def test_compute_risk_metrics_max_drawdown_is_negative(self, simple_returns_two_tickers):
        """
        Spec §5 / §7: max_drawdown must be a negative value (or zero) for every ticker.
        """
        result = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )

        for ticker in ["AAPL", "MSFT"]:
            mdd = float(
                result.loc[
                    (result["ticker"] == ticker) & (result["metric_name"] == "max_drawdown"),
                    "value",
                ].iloc[0]
            )
            assert mdd <= 0.0, (
                f"{ticker} max_drawdown must be <= 0, got {mdd}"
            )

    def test_compute_risk_metrics_output_category_is_risk(self, simple_returns_two_tickers):
        """All rows returned by compute_risk_metrics must have category='risk' (§4.1)."""
        result = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        assert (result["category"] == "risk").all(), (
            "All rows from compute_risk_metrics must have category='risk'"
        )

    def test_compute_risk_metrics_does_not_mutate_input(self, simple_returns_two_tickers):
        """Spec §5: defensive copy — input must not be mutated."""
        original = simple_returns_two_tickers.copy(deep=True)
        compute_risk_metrics(simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS)
        pd.testing.assert_frame_equal(
            simple_returns_two_tickers.reset_index(drop=True),
            original.reset_index(drop=True),
            check_dtype=True,
            obj="returns DataFrame was mutated by compute_risk_metrics",
        )


# ---------------------------------------------------------------------------
# Tests 5, 6 — Drawdown series: running peak monotonic, pct ≤ 0
# ---------------------------------------------------------------------------

class TestComputeDrawdownSeries:
    """FR10 — per-ticker drawdown time series."""

    def test_compute_drawdown_running_peak_monotonic(self, simple_prices_two_tickers):
        """
        Spec §4.4: running_peak is the cumulative maximum and must be
        non-decreasing over time for each ticker.
        """
        result = compute_drawdown_series(simple_prices_two_tickers)

        for ticker in ["AAPL", "MSFT"]:
            sub = result[result["ticker"] == ticker].sort_values("date")
            peaks = sub["running_peak"].values
            assert (np.diff(peaks) >= -1e-12).all(), (
                f"{ticker}: running_peak is not monotonically non-decreasing: {peaks}"
            )

    def test_compute_drawdown_all_negative_or_zero(self, simple_prices_two_tickers):
        """
        Spec §4.4 / §7: drawdown_pct must always be ≤ 0.
        A price at or above its running peak gives drawdown_pct = 0;
        below the peak gives a negative value.
        """
        result = compute_drawdown_series(simple_prices_two_tickers)

        violations = result[result["drawdown_pct"] > 1e-12]
        assert violations.empty, (
            f"drawdown_pct contains positive values:\n{violations}"
        )

    def test_compute_drawdown_monotone_rising_series_zero_drawdown(
        self, monotone_rising_prices
    ):
        """
        A price series that rises every day should have drawdown_pct == 0
        on every row (price always equals its running peak).
        """
        result = compute_drawdown_series(monotone_rising_prices)
        sub = result[result["ticker"] == "RISE"]

        assert (sub["drawdown_pct"].abs() < 1e-12).all(), (
            "Monotone rising series must have drawdown_pct == 0 everywhere, "
            f"got: {sub['drawdown_pct'].values}"
        )

    def test_compute_drawdown_output_columns(self, simple_prices_two_tickers):
        """
        Output must contain exactly the columns specified in §4.4:
        [date, ticker, close, running_peak, drawdown_pct].
        """
        result = compute_drawdown_series(simple_prices_two_tickers)
        expected_cols = {"date", "ticker", "close", "running_peak", "drawdown_pct"}
        assert expected_cols.issubset(result.columns), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

    def test_compute_drawdown_does_not_mutate_input(self, simple_prices_two_tickers):
        """Spec §5: input prices DataFrame must not be modified."""
        original = simple_prices_two_tickers.copy(deep=True)
        compute_drawdown_series(simple_prices_two_tickers)
        pd.testing.assert_frame_equal(
            simple_prices_two_tickers.reset_index(drop=True),
            original.reset_index(drop=True),
            check_dtype=True,
            obj="prices DataFrame was mutated by compute_drawdown_series",
        )


# ---------------------------------------------------------------------------
# Test 7 — Sharpe is NaN for zero-vol ticker, not a crash
# ---------------------------------------------------------------------------

class TestComputeRiskAdjusted:
    """FR5 — risk-adjusted metrics (Sharpe, Sortino, Calmar)."""

    def test_sharpe_zero_vol_returns_nan(
        self, flat_returns_one_ticker, flat_prices_one_ticker
    ):
        """
        Spec §6: A flat-lined ticker (all returns zero, vol = 0) must produce
        Sharpe = NaN, not raise an exception or produce ±inf.
        """
        return_metrics = compute_return_metrics(
            flat_returns_one_ticker, flat_prices_one_ticker, TRADING_DAYS
        )
        risk_metrics = compute_risk_metrics(
            flat_returns_one_ticker, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        result = compute_risk_adjusted(
            return_metrics, risk_metrics, RISK_FREE_RATE, TRADING_DAYS
        )

        sharpe_row = result.loc[
            (result["ticker"] == "FLAT") & (result["metric_name"] == "sharpe")
        ]
        assert not sharpe_row.empty, "Sharpe row must exist even for flat-lined ticker"
        sharpe_val = sharpe_row["value"].iloc[0]
        assert math.isnan(float(sharpe_val)), (
            f"Sharpe must be NaN for zero-vol ticker, got {sharpe_val}"
        )

    def test_risk_adjusted_output_category(self, simple_returns_two_tickers, simple_prices_two_tickers):
        """All rows must have category='risk_adjusted' (§4.1)."""
        return_metrics = compute_return_metrics(
            simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS
        )
        risk_metrics = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        result = compute_risk_adjusted(
            return_metrics, risk_metrics, RISK_FREE_RATE, TRADING_DAYS
        )

        assert (result["category"] == "risk_adjusted").all(), (
            "All rows from compute_risk_adjusted must have category='risk_adjusted'"
        )

    def test_risk_adjusted_metric_names_present(
        self, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Per FR5 and §3.2, each ticker must have metric_names:
        sharpe, sortino, calmar.
        """
        return_metrics = compute_return_metrics(
            simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS
        )
        risk_metrics = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        result = compute_risk_adjusted(
            return_metrics, risk_metrics, RISK_FREE_RATE, TRADING_DAYS
        )

        expected_names = {"sharpe", "sortino", "calmar"}
        for ticker in ["AAPL", "MSFT"]:
            names = set(result.loc[result["ticker"] == ticker, "metric_name"])
            missing = expected_names - names
            assert not missing, (
                f"{ticker} missing risk_adjusted metric_names: {missing}"
            )


# ---------------------------------------------------------------------------
# Tests 8, 9, 10 — Portfolio metrics: weights validation, equal default,
# diversification ratio
# ---------------------------------------------------------------------------

class TestComputePortfolioMetrics:
    """FR6, FR7, FR8 — portfolio-level metrics."""

    def test_portfolio_weights_must_sum_to_one(self, simple_returns_two_tickers):
        """
        Spec §5 / §7: ValueError must be raised when custom weights don't
        sum to 1.0 (within tolerance 1e-6).
        """
        bad_weights = {"AAPL": 0.6, "MSFT": 0.6}  # sums to 1.2
        with pytest.raises(ValueError, match="1"):
            compute_portfolio_metrics(
                simple_returns_two_tickers,
                weights=bad_weights,
                benchmark_ticker=None,
                exclude_benchmark=False,
                risk_free_rate=RISK_FREE_RATE,
                trading_days_per_year=TRADING_DAYS,
                var_levels=VAR_LEVELS,
                cvar_levels=CVAR_LEVELS,
            )

    def test_portfolio_weights_missing_ticker_raises(self, simple_returns_two_tickers):
        """
        Spec §6: ValueError raised when PORTFOLIO_WEIGHTS references tickers
        not present in the dataset.
        """
        bad_weights = {"AAPL": 0.5, "FAKE": 0.5}
        with pytest.raises(ValueError):
            compute_portfolio_metrics(
                simple_returns_two_tickers,
                weights=bad_weights,
                benchmark_ticker=None,
                exclude_benchmark=False,
                risk_free_rate=RISK_FREE_RATE,
                trading_days_per_year=TRADING_DAYS,
                var_levels=VAR_LEVELS,
                cvar_levels=CVAR_LEVELS,
            )

    def test_portfolio_equal_weights_default(self, simple_returns_two_tickers):
        """
        Spec FR6: when weights=None, equal weights must be applied across
        all non-benchmark tickers.  For 2 tickers with no benchmark, each
        should get weight 0.5.  The weights_strategy column must be 'equal'.
        """
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        assert result.shape[0] == 1, (
            f"Portfolio metrics must be a single-row DataFrame, got {result.shape[0]} rows"
        )
        assert str(result["weights_strategy"].iloc[0]) == "equal", (
            f"weights_strategy must be 'equal' when weights=None, "
            f"got {result['weights_strategy'].iloc[0]!r}"
        )

    def test_portfolio_custom_weights_strategy_label(self, simple_returns_two_tickers):
        """
        Spec §4.2: weights_strategy column must be 'custom' when explicit
        weights are provided.
        """
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights={"AAPL": 0.5, "MSFT": 0.5},
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        assert str(result["weights_strategy"].iloc[0]) == "custom", (
            "weights_strategy must be 'custom' when explicit weights are provided"
        )

    def test_diversification_ratio_single_ticker(self):
        """
        Spec §6: when only one ticker is in the portfolio, diversification_ratio
        must equal 1.0 (no diversification benefit possible — §6 states ratio=1.0).
        """
        single_rets = _make_returns(
            ["ONLY"],
            {"ONLY": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.015]},
        )
        result = compute_portfolio_metrics(
            single_rets,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        dr = float(result["diversification_ratio"].iloc[0])
        assert dr == pytest.approx(1.0, abs=1e-9), (
            f"Single-ticker portfolio diversification_ratio must be 1.0, got {dr}"
        )

    def test_diversification_ratio_multiple_tickers_geq_one(
        self, simple_returns_two_tickers
    ):
        """
        Spec FR8 / §10 Test 10: diversification_ratio ≥ 1.0 for a portfolio
        of uncorrelated or positively correlated assets.
        """
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        dr = float(result["diversification_ratio"].iloc[0])
        assert dr >= 1.0 - 1e-9, (
            f"diversification_ratio must be >= 1.0, got {dr}"
        )

    def test_beta_skipped_when_benchmark_none(self, simple_returns_two_tickers):
        """
        Spec §7 / §6: when benchmark_ticker=None, beta_vs_benchmark must be
        NaN and the function must not raise any exception.
        """
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        assert "beta_vs_benchmark" in result.columns, (
            "beta_vs_benchmark column must be present even when benchmark=None"
        )
        beta_val = result["beta_vs_benchmark"].iloc[0]
        assert math.isnan(float(beta_val)), (
            f"beta_vs_benchmark must be NaN when benchmark_ticker=None, got {beta_val}"
        )

    def test_portfolio_max_drawdown_is_negative_or_zero(
        self, simple_returns_two_tickers
    ):
        """Spec §7: portfolio max_drawdown must be <= 0."""
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        mdd = float(result["max_drawdown"].iloc[0])
        assert mdd <= 0.0, (
            f"Portfolio max_drawdown must be <= 0, got {mdd}"
        )

    def test_portfolio_var95_leq_var99(self, simple_returns_two_tickers):
        """Spec §7: portfolio VaR_95 ≤ VaR_99 in magnitude."""
        result = compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=[0.95, 0.99],
            cvar_levels=CVAR_LEVELS,
        )

        var95 = float(result["var_95"].iloc[0])
        var99 = float(result["var_99"].iloc[0])
        assert var95 <= var99 + 1e-10, (
            f"Portfolio VaR_95 ({var95:.6f}) must be <= VaR_99 ({var99:.6f})"
        )

    def test_benchmark_excluded_from_portfolio_when_flag_set(
        self, returns_with_benchmark
    ):
        """
        Spec FR6 / config.EXCLUDE_BENCHMARK_FROM_PORTFOLIO:
        when exclude_benchmark=True, the benchmark ticker must not
        participate in the equal-weight portfolio construction.
        With 3 tickers (AAPL, BENCH, MSFT) and exclude=True, only
        AAPL and MSFT are used → each gets weight 0.5 → weights_strategy='equal'.
        """
        result = compute_portfolio_metrics(
            returns_with_benchmark,
            weights=None,
            benchmark_ticker="BENCH",
            exclude_benchmark=True,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        assert result.shape[0] == 1, "Portfolio metrics must be single-row"
        assert str(result["weights_strategy"].iloc[0]) == "equal", (
            "weights_strategy must be 'equal' for equal-weight portfolio"
        )

    def test_portfolio_does_not_mutate_input(self, simple_returns_two_tickers):
        """Spec §5: input returns DataFrame must not be modified."""
        original = simple_returns_two_tickers.copy(deep=True)
        compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )
        pd.testing.assert_frame_equal(
            simple_returns_two_tickers.reset_index(drop=True),
            original.reset_index(drop=True),
            check_dtype=True,
            obj="returns DataFrame was mutated by compute_portfolio_metrics",
        )


# ---------------------------------------------------------------------------
# Test 11 — Rolling metrics: initial NaN rows dropped, window respected
# ---------------------------------------------------------------------------

class TestComputeRollingMetrics:
    """FR9 — rolling metrics."""

    def test_rolling_sharpe_window_drops_initial_nans(self):
        """
        Spec §3.2 / FR9 / §10 Test 11:
        The first (window - 1) rows do not have a full window of data and
        therefore produce NaN.  The function must DROP these NaN rows so
        the output contains only valid values.

        With 10 daily returns and a sharpe_window of 5:
          valid rolling rows = 10 - 5 + 1 = 6
        """
        n_rows = 10
        sharpe_window = 5
        rets_list = [0.01 * ((-1) ** i) for i in range(n_rows)]
        returns_df = _make_returns(["ROLL"], {"ROLL": rets_list})

        result = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=sharpe_window,
            beta_window=5,
            corr_window=5,
        )

        sharpe_rows = result[
            (result["ticker"] == "ROLL")
            & (result["metric_name"] == f"rolling_sharpe_{sharpe_window}")
        ]
        expected_rows = n_rows - sharpe_window + 1
        assert len(sharpe_rows) == expected_rows, (
            f"Expected {expected_rows} rolling Sharpe rows (window={sharpe_window}, "
            f"n={n_rows}), got {len(sharpe_rows)}"
        )
        assert sharpe_rows["value"].notna().all(), (
            "Rolling Sharpe output must contain no NaN values after initial window drop"
        )

    def test_rolling_metrics_skipped_when_fewer_rows_than_window(self):
        """
        Spec §6: when a ticker has fewer rows than ROLLING_SHARPE_WINDOW,
        the module must skip rolling metrics for that ticker (log warning,
        no exception).
        """
        n_rows = 5
        sharpe_window = 90  # far larger than available rows
        rets_list = [0.01, -0.01, 0.02, -0.02, 0.01]
        returns_df = _make_returns(["SHORT"], {"SHORT": rets_list})

        # Must not raise; output may be empty for this ticker
        result = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=sharpe_window,
            beta_window=60,
            corr_window=60,
        )

        ticker_rows = result[result["ticker"] == "SHORT"] if not result.empty else pd.DataFrame()
        assert ticker_rows.empty, (
            f"Ticker with {n_rows} rows < window {sharpe_window} should produce "
            f"no rolling output, got {len(ticker_rows)} rows"
        )

    def test_rolling_metrics_no_benchmark_excludes_beta_corr(self):
        """
        When benchmark_ticker=None, rolling_beta and rolling_corr columns
        must not appear in the output (FR7 / FR9).
        """
        rets_list = [0.01 * ((-1) ** i) for i in range(100)]
        returns_df = _make_returns(["ROLL"], {"ROLL": rets_list})

        result = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=20,
            beta_window=20,
            corr_window=20,
        )

        beta_rows = result[result["metric_name"].str.contains("rolling_beta", na=False)]
        corr_rows = result[result["metric_name"].str.contains("rolling_corr", na=False)]
        assert beta_rows.empty, (
            "No rolling_beta rows expected when benchmark_ticker=None"
        )
        assert corr_rows.empty, (
            "No rolling_corr rows expected when benchmark_ticker=None"
        )

    def test_rolling_metrics_output_columns(self):
        """
        Output must contain columns [date, ticker, metric_name, value] per §4.3.
        """
        rets_list = [0.01 * ((-1) ** i) for i in range(100)]
        returns_df = _make_returns(["ROLL"], {"ROLL": rets_list})

        result = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=20,
            beta_window=20,
            corr_window=20,
        )

        expected_cols = {"date", "ticker", "metric_name", "value"}
        assert expected_cols.issubset(result.columns), (
            f"Missing columns in rolling metrics: {expected_cols - set(result.columns)}"
        )

    def test_rolling_metrics_does_not_mutate_input(self):
        """Spec §5: input returns DataFrame must not be modified."""
        rets_list = [0.01, -0.01] * 50
        returns_df = _make_returns(["ROLL"], {"ROLL": rets_list})
        original = returns_df.copy(deep=True)

        compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=20,
            beta_window=20,
            corr_window=20,
        )

        pd.testing.assert_frame_equal(
            returns_df.reset_index(drop=True),
            original.reset_index(drop=True),
            check_dtype=True,
            obj="returns DataFrame was mutated by compute_rolling_metrics",
        )


# ---------------------------------------------------------------------------
# Test 12 — Idempotency: running save_metrics twice produces identical Parquet
# ---------------------------------------------------------------------------

class TestIdempotency:
    """FR17 / §7: identical inputs → byte-identical Parquet output."""

    def _build_all_frames(self, returns_df, prices_df):
        """Helper: compute all intermediate frames needed for save_metrics."""
        return_metrics = compute_return_metrics(returns_df, prices_df, TRADING_DAYS)
        risk_metrics = compute_risk_metrics(
            returns_df, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        drawdowns = compute_drawdown_series(prices_df)
        risk_adj = compute_risk_adjusted(
            return_metrics, risk_metrics, RISK_FREE_RATE, TRADING_DAYS
        )
        per_ticker = pd.concat([return_metrics, risk_metrics, risk_adj], ignore_index=True)
        portfolio = compute_portfolio_metrics(
            returns_df,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )
        rolling = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=5,
            beta_window=5,
            corr_window=5,
        )
        return per_ticker, portfolio, rolling, drawdowns

    def test_idempotent_rerun(self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers):
        """
        Spec FR17 / §7: running save_metrics twice with identical inputs must
        produce byte-for-byte identical Parquet files.
        """
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        processed.mkdir()
        reports.mkdir()

        per_ticker, portfolio, rolling, drawdowns = self._build_all_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary_dict = {"run_timestamp": "fixed", "test": True}

        # First run
        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary_dict, processed, reports,
        )
        first_bytes = {
            fname: (processed / fname).read_bytes()
            for fname in [
                "metrics_per_ticker.parquet",
                "portfolio_metrics.parquet",
                "rolling_metrics.parquet",
                "drawdown_series.parquet",
            ]
        }

        # Second run — must overwrite with identical content
        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary_dict, processed, reports,
        )
        second_bytes = {
            fname: (processed / fname).read_bytes()
            for fname in first_bytes
        }

        for fname in first_bytes:
            assert first_bytes[fname] == second_bytes[fname], (
                f"Parquet file {fname} is not byte-identical after re-run "
                f"(first={len(first_bytes[fname])} bytes, "
                f"second={len(second_bytes[fname])} bytes)"
            )


# ---------------------------------------------------------------------------
# Test 13 — No modifications to input DataFrames
# ---------------------------------------------------------------------------
# (Already covered individually inside each compute_* test class above.
#  This combined test verifies the entire compute pipeline preserves inputs.)

class TestNoInputMutation:
    """Spec §5: every public compute function must make a defensive copy."""

    def test_full_pipeline_no_mutation(
        self, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Run all four compute functions in pipeline order and confirm that
        the original returns and prices DataFrames are unchanged after each step.
        """
        returns_snapshot = simple_returns_two_tickers.copy(deep=True)
        prices_snapshot = simple_prices_two_tickers.copy(deep=True)

        ret_metrics = compute_return_metrics(
            simple_returns_two_tickers, simple_prices_two_tickers, TRADING_DAYS
        )
        risk_metrics_df = compute_risk_metrics(
            simple_returns_two_tickers, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        compute_drawdown_series(simple_prices_two_tickers)
        compute_risk_adjusted(ret_metrics, risk_metrics_df, RISK_FREE_RATE, TRADING_DAYS)
        compute_portfolio_metrics(
            simple_returns_two_tickers,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )

        pd.testing.assert_frame_equal(
            simple_returns_two_tickers.reset_index(drop=True),
            returns_snapshot.reset_index(drop=True),
            check_dtype=True,
            obj="returns DataFrame was mutated during full pipeline run",
        )
        pd.testing.assert_frame_equal(
            simple_prices_two_tickers.reset_index(drop=True),
            prices_snapshot.reset_index(drop=True),
            check_dtype=True,
            obj="prices DataFrame was mutated during full pipeline run",
        )


# ---------------------------------------------------------------------------
# Test 14 — save_metrics side-effect verification (FR11–FR15)
# ---------------------------------------------------------------------------

class TestSaveMetrics:
    """FR11–FR15: all output files created with correct content."""

    def _build_frames(self, returns_df, prices_df):
        return_metrics = compute_return_metrics(returns_df, prices_df, TRADING_DAYS)
        risk_metrics = compute_risk_metrics(
            returns_df, TRADING_DAYS, VAR_LEVELS, CVAR_LEVELS
        )
        drawdowns = compute_drawdown_series(prices_df)
        risk_adj = compute_risk_adjusted(
            return_metrics, risk_metrics, RISK_FREE_RATE, TRADING_DAYS
        )
        per_ticker = pd.concat([return_metrics, risk_metrics, risk_adj], ignore_index=True)
        portfolio = compute_portfolio_metrics(
            returns_df,
            weights=None,
            benchmark_ticker=None,
            exclude_benchmark=False,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            var_levels=VAR_LEVELS,
            cvar_levels=CVAR_LEVELS,
        )
        rolling = compute_rolling_metrics(
            returns_df,
            benchmark_ticker=None,
            risk_free_rate=RISK_FREE_RATE,
            trading_days_per_year=TRADING_DAYS,
            sharpe_window=5,
            beta_window=5,
            corr_window=5,
        )
        return per_ticker, portfolio, rolling, drawdowns

    def test_save_metrics_creates_all_output_files(
        self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        FR11–FR15: After save_metrics completes, all five output paths
        must exist on disk.
        """
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        processed.mkdir()
        reports.mkdir()

        per_ticker, portfolio, rolling, drawdowns = self._build_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary = {"run_timestamp": "test", "tickers": 2}

        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary, processed, reports,
        )

        expected_files = [
            processed / "metrics_per_ticker.parquet",
            processed / "portfolio_metrics.parquet",
            processed / "rolling_metrics.parquet",
            processed / "drawdown_series.parquet",
            reports / "metrics_summary.json",
        ]
        for path in expected_files:
            assert path.exists(), f"Expected output file was not created: {path}"

    def test_save_metrics_creates_directories_if_missing(
        self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Spec §3.2 (save_metrics docstring): directories must be created if
        they do not already exist.
        """
        processed = tmp_path / "nested" / "processed"
        reports = tmp_path / "nested" / "reports"
        # Do NOT create directories — save_metrics must create them

        per_ticker, portfolio, rolling, drawdowns = self._build_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary = {"test": True}

        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary, processed, reports,
        )

        assert (processed / "metrics_per_ticker.parquet").exists(), (
            "save_metrics must create processed_dir if it does not exist"
        )
        assert (reports / "metrics_summary.json").exists(), (
            "save_metrics must create reports_dir if it does not exist"
        )

    def test_save_metrics_json_is_valid_json(
        self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        FR15: metrics_summary.json must be valid JSON (parseable by json.load).
        """
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        processed.mkdir()
        reports.mkdir()

        per_ticker, portfolio, rolling, drawdowns = self._build_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary = {"run_timestamp": "2026-05-25T00:00:00Z", "tickers": 2, "value": 1.5}

        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary, processed, reports,
        )

        json_path = reports / "metrics_summary.json"
        with open(json_path, encoding="utf-8") as fh:
            parsed = json.load(fh)

        assert isinstance(parsed, dict), (
            f"metrics_summary.json must parse to a dict, got {type(parsed)}"
        )

    def test_save_metrics_parquet_readable_roundtrip(
        self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Write metrics then read them back: confirm the DataFrames round-trip
        correctly (no data loss, correct row counts).
        """
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        processed.mkdir()
        reports.mkdir()

        per_ticker, portfolio, rolling, drawdowns = self._build_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary = {}

        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary, processed, reports,
        )

        pt_read = pd.read_parquet(processed / "metrics_per_ticker.parquet")
        port_read = pd.read_parquet(processed / "portfolio_metrics.parquet")
        dd_read = pd.read_parquet(processed / "drawdown_series.parquet")

        assert pt_read.shape[0] > 0, "metrics_per_ticker.parquet must not be empty"
        assert port_read.shape[0] == 1, (
            f"portfolio_metrics.parquet must have exactly 1 row, got {port_read.shape[0]}"
        )
        assert dd_read.shape[0] > 0, "drawdown_series.parquet must not be empty"

    def test_save_metrics_overwrites_existing_files(
        self, tmp_path, simple_returns_two_tickers, simple_prices_two_tickers
    ):
        """
        Spec §6: existing files from a prior run must be overwritten without
        prompting.  Write a sentinel file at the parquet path first, then
        confirm save_metrics replaces it.
        """
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        processed.mkdir()
        reports.mkdir()

        # Write a sentinel so we can prove overwriting works
        sentinel = b"SENTINEL_CONTENT_SHOULD_BE_REPLACED"
        (processed / "metrics_per_ticker.parquet").write_bytes(sentinel)

        per_ticker, portfolio, rolling, drawdowns = self._build_frames(
            simple_returns_two_tickers, simple_prices_two_tickers
        )
        summary = {}

        save_metrics(
            per_ticker, portfolio, rolling, drawdowns,
            summary, processed, reports,
        )

        content = (processed / "metrics_per_ticker.parquet").read_bytes()
        assert content != sentinel, (
            "save_metrics must overwrite existing metrics_per_ticker.parquet"
        )


# ---------------------------------------------------------------------------
# Parametrized edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence_level,quantile_arg", [
    (0.95, 0.05),
    (0.99, 0.01),
])
def test_var_formula_parametrized(confidence_level, quantile_arg):
    """
    Spec §13: VaR formula is np.quantile(returns, 1 - confidence_level),
    returned as positive loss.  Parametrized for both 95 % and 99 % levels.
    """
    raw_returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    expected_var = float(abs(np.quantile(raw_returns, quantile_arg)))

    returns_df = _make_returns(["PTEST"], {"PTEST": raw_returns})
    result = compute_risk_metrics(
        returns_df, TRADING_DAYS, [confidence_level], [confidence_level]
    )

    metric_name = f"var_{int(confidence_level * 100)}"
    actual_var = float(
        result.loc[
            (result["ticker"] == "PTEST") & (result["metric_name"] == metric_name),
            "value",
        ].iloc[0]
    )
    assert actual_var == pytest.approx(expected_var, rel=1e-9), (
        f"VaR_{int(confidence_level*100)} mismatch: "
        f"expected {expected_var:.8f}, got {actual_var:.8f}"
    )


@pytest.mark.parametrize("n_tickers", [1, 2, 3])
def test_compute_return_metrics_one_row_per_metric_per_ticker(n_tickers):
    """
    Data-integrity test: no ticker should produce duplicate (ticker, metric_name)
    pairs in the return metrics output.
    """
    tickers = [f"T{i}" for i in range(n_tickers)]
    rets = {t: [0.01, -0.01, 0.02, -0.02, 0.01] for t in tickers}
    closes = {t: [100.0, 101.0, 99.0, 101.5, 100.5] for t in tickers}

    returns_df = _make_returns(tickers, rets)
    prices_df = _make_prices(tickers, closes)

    result = compute_return_metrics(returns_df, prices_df, TRADING_DAYS)

    duplicates = result.groupby(["ticker", "metric_name"]).size()
    dupes = duplicates[duplicates > 1]
    assert dupes.empty, (
        f"Duplicate (ticker, metric_name) pairs found in return metrics:\n{dupes}"
    )
