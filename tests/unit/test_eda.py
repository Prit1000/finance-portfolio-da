"""
tests/unit/test_eda.py
======================
Pytest test suite for src/eda.py — Step 3 of the Finance Portfolio Analysis Pipeline.

All test logic is derived from the spec at .claude/specs/03-eda.md.
No implementation details assumed beyond public function signatures and
data contracts defined in the spec. No real files from data/ are read.
Synthetic DataFrames and tmp_path are used throughout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    detect_outliers,
    load_processed_data,
    plot_correlations,
    plot_price_trends,
    plot_return_distributions,
    plot_volatility,
    save_eda_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_prices(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 60,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Long-format OHLCV DataFrame satisfying prices_clean_schema."""
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            close = 100.0 + i * 0.1
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": pd.array([1_000_000], dtype="Int64")[0],
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("string")
    return df


def _make_returns(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 59,
    start: str = "2024-01-03",
    return_val: float = 0.001,
) -> pd.DataFrame:
    """Long-format returns DataFrame satisfying returns_schema."""
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for d in dates:
            sr = return_val
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "simple_return": sr,
                    "log_return": float(np.log(1.0 + sr)),
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("string")
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_load_processed_data_raises_on_missing_file
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadProcessedData:
    """FR1/FR2 — load and validate parquet inputs; raise on missing files."""

    def test_raises_file_not_found_for_missing_prices_parquet(self, tmp_path):
        """FileNotFoundError with 'Step 2' hint when prices_clean.parquet absent."""
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        with pytest.raises(FileNotFoundError, match="(?i)step.?2|run.*cleaning|prices_clean"):
            load_processed_data(processed, raw)

    def test_raises_file_not_found_for_missing_returns_parquet(self, tmp_path):
        """FileNotFoundError with 'Step 2' hint when returns_daily.parquet absent."""
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        prices = _make_prices(tickers=("AAPL",), n_days=5)
        _write_parquet(prices, processed / "prices_clean.parquet")
        # returns_daily.parquet deliberately not written

        with pytest.raises(FileNotFoundError, match="(?i)step.?2|run.*cleaning|returns_daily"):
            load_processed_data(processed, raw)

    def test_returns_empty_metadata_when_metadata_json_missing(self, tmp_path):
        """§6 — missing metadata.json logs a warning; pipeline must not raise."""
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        _, _, meta = load_processed_data(processed, raw)

        assert meta == {}, (
            "load_processed_data must return empty dict when metadata.json is absent"
        )

    def test_returns_correct_shapes(self, tmp_path):
        """FR1 — loaded DataFrames have the expected row counts."""
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        p, r, _ = load_processed_data(processed, raw)

        assert len(p) == len(prices)
        assert len(r) == len(returns)


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_plot_price_trends_creates_expected_files
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotPriceTrends:
    """FR3 — one price PNG per ticker saved at the correct path."""

    def test_creates_one_png_per_ticker(self, tmp_path):
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20, 50], out, dpi=72)

        tickers = prices["ticker"].unique().tolist()
        assert len(saved) == len(tickers), (
            f"Expected {len(tickers)} price PNGs, got {len(saved)}"
        )

    def test_files_exist_at_correct_paths(self, tmp_path):
        prices = _make_prices(tickers=("AAPL",), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20], out, dpi=72)

        for p in saved:
            assert p.exists(), f"Expected PNG at {p}"
            assert p.suffix == ".png"

    def test_output_directory_created_automatically(self, tmp_path):
        prices = _make_prices(tickers=("AAPL",), n_days=60)
        out = tmp_path / "deep" / "nested" / "plots"

        plot_price_trends(prices, [20], out, dpi=72)

        assert (out / "01_price_trends").exists()

    def test_returns_list_of_paths(self, tmp_path):
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        result = plot_price_trends(prices, [20, 50], out, dpi=72)

        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_plot_return_distributions_returns_stats_df
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotReturnDistributions:
    """FR5/FR7 — stats DataFrame has all expected columns; one row per ticker."""

    _EXPECTED_COLS = {
        "ticker", "mean", "std", "skew", "kurtosis", "min", "max", "jarque_bera_pvalue",
    }

    def test_stats_df_has_all_required_columns(self, tmp_path):
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=50)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert self._EXPECTED_COLS.issubset(set(stats_df.columns)), (
            f"Missing columns: {self._EXPECTED_COLS - set(stats_df.columns)}"
        )

    def test_stats_df_has_one_row_per_ticker(self, tmp_path):
        tickers = ("AAPL", "MSFT", "GOOGL")
        returns = _make_returns(tickers=tickers, n_days=50)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert len(stats_df) == len(tickers), (
            f"Expected {len(tickers)} rows in stats_df, got {len(stats_df)}"
        )

    def test_stats_df_ticker_values_match_input(self, tmp_path):
        tickers = ("AAPL", "MSFT")
        returns = _make_returns(tickers=tickers, n_days=50)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert set(stats_df["ticker"].tolist()) == set(tickers)

    def test_boxplot_file_created(self, tmp_path):
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=50)
        out = tmp_path / "plots"

        saved, _ = plot_return_distributions(returns, out, dpi=72)

        boxplot_files = [p for p in saved if "boxplot" in p.name]
        assert len(boxplot_files) == 1, (
            "Expected exactly one all_tickers_boxplot.png"
        )

    def test_returns_tuple_of_paths_and_dataframe(self, tmp_path):
        returns = _make_returns(tickers=("AAPL",), n_days=50)
        out = tmp_path / "plots"

        result = plot_return_distributions(returns, out, dpi=72)

        assert isinstance(result, tuple) and len(result) == 2
        paths, df = result
        assert isinstance(paths, list)
        assert isinstance(df, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_plot_volatility_skips_tickers_below_window
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotVolatility:
    """FR8/§6 — tickers with insufficient data are skipped gracefully."""

    def test_skips_ticker_with_fewer_rows_than_window(self, tmp_path):
        """Ticker with 5 rows must be skipped when window=30."""
        short_returns = _make_returns(tickers=("TINY",), n_days=5)
        long_returns = _make_returns(tickers=("AAPL",), n_days=60)
        returns = pd.concat([short_returns, long_returns], ignore_index=True)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72)

        # TINY should have no rolling vol chart; AAPL should
        tiny_plots = [p for p in saved if "TINY" in p.name]
        aapl_plots = [p for p in saved if "AAPL" in p.name and "rolling_vol" in p.name]

        assert len(tiny_plots) == 0, "TINY has fewer rows than window — must be skipped"
        assert len(aapl_plots) == 1, "AAPL has sufficient data — rolling vol must be generated"

    def test_no_crash_when_all_tickers_below_window(self, tmp_path):
        """All tickers below window: returns empty monthly_vol DataFrame, no crash."""
        returns = _make_returns(tickers=("AAPL",), n_days=5)
        out = tmp_path / "plots"

        saved, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        assert isinstance(saved, list)
        assert isinstance(monthly_vol, pd.DataFrame)
        assert monthly_vol.empty

    def test_returns_monthly_vol_pivot_for_sufficient_tickers(self, tmp_path):
        """monthly_vol_df has months as index and tickers as columns."""
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        assert not monthly_vol.empty
        assert "AAPL" in monthly_vol.columns
        assert "MSFT" in monthly_vol.columns

    def test_heatmap_file_created(self, tmp_path):
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        heatmap_files = [p for p in saved if "monthly_vol_heatmap" in p.name]
        assert len(heatmap_files) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_plot_correlations_top_pairs_sorted_by_abs_value
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotCorrelations:
    """FR10/FR11 — top pairs returned in descending |correlation| order."""

    def _make_correlated_returns(self) -> pd.DataFrame:
        """
        Three tickers: AAPL and MSFT highly correlated (0.95),
        XOM is uncorrelated with both.
        """
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        base = np.random.normal(0.001, 0.01, n)
        rows = []
        for t, noise_scale in [("AAPL", 0.001), ("MSFT", 0.001), ("XOM", 0.02)]:
            noise = np.random.normal(0, noise_scale, n)
            if t == "XOM":
                sr = np.random.normal(0.001, 0.015, n)
            else:
                sr = base + noise
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "simple_return": float(sr[i]),
                        "log_return": float(np.log(1 + sr[i])) if sr[i] > -1 else 0.0,
                    }
                )
        df = pd.DataFrame(rows)
        df["ticker"] = df["ticker"].astype("string")
        return df

    def test_top_pairs_sorted_descending_by_abs_correlation(self, tmp_path):
        """FR11 — top_pairs list must be sorted by |correlation| descending."""
        returns = self._make_correlated_returns()
        out = tmp_path / "plots"

        _, _, top_pairs = plot_correlations(returns, {}, top_n_pairs=3, out_dir=out, dpi=72)

        abs_corrs = [abs(c) for _, _, c in top_pairs]
        assert abs_corrs == sorted(abs_corrs, reverse=True), (
            "top_pairs must be sorted by |correlation| descending"
        )

    def test_top_n_pairs_count_respected(self, tmp_path):
        """FR11 — exactly top_n scatter plots generated when enough pairs exist."""
        returns = self._make_correlated_returns()
        out = tmp_path / "plots"
        top_n = 2

        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=top_n, out_dir=out, dpi=72
        )

        assert len(top_pairs) == top_n, (
            f"Expected {top_n} top pairs, got {len(top_pairs)}"
        )

    def test_correlation_matrix_is_symmetric(self, tmp_path):
        """FR10 — corr_matrix_df must be symmetric."""
        returns = self._make_correlated_returns()
        out = tmp_path / "plots"

        _, corr_matrix, _ = plot_correlations(returns, {}, top_n_pairs=2, out_dir=out, dpi=72)

        np.testing.assert_allclose(
            corr_matrix.values,
            corr_matrix.values.T,
            atol=1e-10,
            err_msg="Correlation matrix must be symmetric",
        )

    def test_diagonal_is_one(self, tmp_path):
        """FR10 — diagonal of correlation matrix must be 1.0."""
        returns = self._make_correlated_returns()
        out = tmp_path / "plots"

        _, corr_matrix, _ = plot_correlations(returns, {}, top_n_pairs=2, out_dir=out, dpi=72)

        np.testing.assert_allclose(
            np.diag(corr_matrix.values),
            np.ones(len(corr_matrix)),
            atol=1e-10,
            err_msg="Diagonal of correlation matrix must be 1.0",
        )

    def test_skips_analysis_for_single_ticker(self, tmp_path):
        """§6 — single ticker: correlation analysis skipped, no crash."""
        returns = _make_returns(tickers=("AAPL",), n_days=50)
        out = tmp_path / "plots"

        paths, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=3, out_dir=out, dpi=72
        )

        assert paths == []
        assert corr_matrix.empty
        assert top_pairs == []

    def test_top_n_exceeding_pairs_uses_all_available(self, tmp_path):
        """§6 — EDA_TOP_N_CORRELATIONS > num_pairs: use all available pairs."""
        # Use random (varied) returns so that correlation is well-defined
        returns = self._make_correlated_returns()
        # Keep only AAPL and MSFT → 1 unique pair
        returns = returns[returns["ticker"].isin(["AAPL", "MSFT"])].copy()
        out = tmp_path / "plots"

        # 2 tickers → 1 unique pair; request 5 → should return all 1 pair
        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=5, out_dir=out, dpi=72
        )

        assert len(top_pairs) == 1, (
            "With only 2 tickers there is 1 unique pair; all should be returned"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. test_detect_outliers_returns_top_n_per_ticker
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectOutliers:
    """FR13/FR14 — exactly top_n moves per ticker; cross-reference with cleaning report."""

    def _make_varied_returns(self, ticker: str = "AAPL", n: int = 50) -> pd.DataFrame:
        np.random.seed(0)
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        sr = np.random.normal(0.001, 0.015, n)
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "simple_return": sr,
                "log_return": np.log(1 + sr),
            }
        )
        df["ticker"] = df["ticker"].astype("string")
        return df

    def test_returns_exactly_top_n_per_ticker(self):
        """Exactly top_n moves returned when ticker has enough rows."""
        returns = self._make_varied_returns("AAPL", n=50)
        top_n = 5

        report = detect_outliers(returns, {}, top_n=top_n)

        aapl_moves = report["top_moves_per_ticker"].get("AAPL", [])
        assert len(aapl_moves) == top_n, (
            f"Expected {top_n} moves for AAPL, got {len(aapl_moves)}"
        )

    def test_top_moves_sorted_by_descending_absolute_return(self):
        """Top moves should be the largest |return| values."""
        returns = self._make_varied_returns("AAPL", n=50)
        top_n = 5

        report = detect_outliers(returns, {}, top_n=top_n)

        moves = report["top_moves_per_ticker"]["AAPL"]
        abs_returns = [abs(m["simple_return"]) for m in moves]
        assert abs_returns == sorted(abs_returns, reverse=True), (
            "Top moves must be ordered by |simple_return| descending"
        )

    def test_each_move_has_required_keys(self):
        """Each move dict must have date, simple_return, and log_return."""
        returns = self._make_varied_returns("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)
        for move in report["top_moves_per_ticker"]["AAPL"]:
            assert "date" in move
            assert "simple_return" in move
            assert "log_return" in move

    def test_eda_vs_cleaning_match_keys_present(self):
        """eda_vs_cleaning_match must have eda_count, cleaning_count, overlap for each ticker."""
        returns = self._make_varied_returns("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)

        match = report["eda_vs_cleaning_match"]["AAPL"]
        assert "eda_count" in match
        assert "cleaning_count" in match
        assert "overlap" in match

    def test_cleaning_report_count_reflected(self):
        """cleaning_count in eda_vs_cleaning_match mirrors cleaning_report value."""
        returns = self._make_varied_returns("AAPL", n=20)
        cleaning_report = {"actions": {"outliers_flagged": {"AAPL": 7}}}

        report = detect_outliers(returns, cleaning_report, top_n=3)

        assert report["eda_vs_cleaning_match"]["AAPL"]["cleaning_count"] == 7

    def test_return_struct_has_all_required_keys(self):
        """Return dict has all four required top-level keys."""
        returns = self._make_varied_returns("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)

        assert "top_moves_per_ticker" in report
        assert "eda_vs_cleaning_match" in report
        assert "zero_volume_days" in report
        assert "zero_price_change_days" in report

    def test_handles_fewer_rows_than_top_n(self):
        """When ticker has fewer rows than top_n, returns all available rows."""
        returns = self._make_varied_returns("AAPL", n=3)
        report = detect_outliers(returns, {}, top_n=10)

        aapl_moves = report["top_moves_per_ticker"]["AAPL"]
        assert len(aapl_moves) == 3, (
            "When ticker has 3 rows and top_n=10, all 3 should be returned"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. test_save_eda_summary_writes_valid_json
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveEdaSummary:
    """FR16 — JSON file written, parseable, contains required keys."""

    def _minimal_args(self, tmp_path: Path) -> dict:
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=59)
        out = tmp_path / "reports"

        _, dist_stats = plot_return_distributions(returns, tmp_path / "plots", dpi=72)
        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=tmp_path / "plots", dpi=72
        )
        _, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=tmp_path / "plots", dpi=72
        )
        outlier_report = detect_outliers(returns, {}, top_n=5)

        return dict(
            distribution_stats=dist_stats,
            monthly_vol=monthly_vol,
            corr_matrix=corr_matrix,
            top_pairs=top_pairs,
            outlier_report=outlier_report,
            saved_plot_paths=[],
            out_dir=out,
        )

    def test_json_file_exists_after_call(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        path = save_eda_summary(**kwargs)

        assert path.exists(), "eda_summary.json must exist after save_eda_summary"
        assert path.suffix == ".json"

    def test_json_is_parseable(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        path = save_eda_summary(**kwargs)

        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert isinstance(parsed, dict), "eda_summary.json must deserialise to a dict"

    def test_json_contains_required_keys(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))

        required = {
            "run_timestamp",
            "distribution_stats",
            "monthly_volatility",
            "correlations",
            "outliers",
            "plots_generated",
        }
        assert required.issubset(set(parsed.keys())), (
            f"Missing required keys: {required - set(parsed.keys())}"
        )

    def test_correlations_section_has_matrix_and_top_pairs(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert "matrix" in parsed["correlations"]
        assert "top_pairs" in parsed["correlations"]

    def test_creates_out_dir_when_missing(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        non_existent = tmp_path / "brand_new" / "reports"
        kwargs["out_dir"] = non_existent

        save_eda_summary(**kwargs)

        assert non_existent.exists()

    def test_overwrites_existing_file(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        save_eda_summary(**kwargs)
        # Second call must not raise
        path = save_eda_summary(**kwargs)
        assert path.exists()

    def test_json_has_indent_formatting(self, tmp_path):
        """indent=2 means lines start with two spaces."""
        kwargs = self._minimal_args(tmp_path)
        path = save_eda_summary(**kwargs)

        text = path.read_text(encoding="utf-8")
        assert "\n  " in text, "eda_summary.json must be formatted with indent=2"

    def test_returns_path_object(self, tmp_path):
        kwargs = self._minimal_args(tmp_path)
        result = save_eda_summary(**kwargs)
        assert isinstance(result, Path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. test_idempotent_rerun
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotentRerun:
    """FR18 — running EDA twice on identical input produces identical JSON analysis content."""

    def test_analysis_content_identical_on_second_run(self, tmp_path):
        """
        Running save_eda_summary twice on the same DataFrames should produce
        identical analysis values (distribution_stats, correlations, outliers).
        The run_timestamp is fixed so byte equality can be checked.
        """
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=59)

        _, dist_stats = plot_return_distributions(returns, tmp_path / "plots1", dpi=72)
        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=tmp_path / "plots1", dpi=72
        )
        _, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=tmp_path / "plots1", dpi=72
        )
        outlier_report = detect_outliers(returns, {}, top_n=5)

        fixed_ts = "2026-01-01T00:00:00Z"
        common_kwargs = dict(
            distribution_stats=dist_stats,
            monthly_vol=monthly_vol,
            corr_matrix=corr_matrix,
            top_pairs=top_pairs,
            outlier_report=outlier_report,
            saved_plot_paths=[],
            run_timestamp=fixed_ts,
        )

        path1 = save_eda_summary(**common_kwargs, out_dir=tmp_path / "reports1")
        path2 = save_eda_summary(**common_kwargs, out_dir=tmp_path / "reports2")

        parsed1 = json.loads(path1.read_text(encoding="utf-8"))
        parsed2 = json.loads(path2.read_text(encoding="utf-8"))

        assert parsed1["distribution_stats"] == parsed2["distribution_stats"]
        assert parsed1["correlations"] == parsed2["correlations"]
        assert parsed1["outliers"]["top_moves_per_ticker"] == (
            parsed2["outliers"]["top_moves_per_ticker"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. test_handles_missing_metadata_gracefully
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingMetadata:
    """§6 — missing metadata.json → sector analysis skipped; pipeline continues."""

    def test_plot_correlations_works_without_metadata(self, tmp_path):
        """plot_correlations with empty metadata dict must not crash."""
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=50)
        out = tmp_path / "plots"

        paths, corr_matrix, top_pairs = plot_correlations(
            returns, metadata={}, top_n_pairs=2, out_dir=out, dpi=72
        )

        assert not corr_matrix.empty, "Correlation matrix must still be computed"
        # No sector heatmap expected (no metadata)
        sector_plots = [p for p in paths if "sector" in p.name]
        assert len(sector_plots) == 0, "No sector heatmap should be generated without metadata"

    def test_correlation_matrix_still_generated_without_metadata(self, tmp_path):
        """Correlation matrix heatmap generated even with no metadata."""
        returns = _make_returns(tickers=("AAPL", "MSFT", "GOOGL"), n_days=50)
        out = tmp_path / "plots"

        paths, _, _ = plot_correlations(
            returns, metadata={}, top_n_pairs=1, out_dir=out, dpi=72
        )

        corr_matrix_plots = [p for p in paths if "correlation_matrix" in p.name]
        assert len(corr_matrix_plots) == 1

    def test_load_processed_data_missing_metadata_returns_empty_dict(self, tmp_path):
        """load_processed_data returns {} for metadata when metadata.json absent."""
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()
        # raw/metadata.json deliberately absent

        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        _, _, metadata = load_processed_data(processed, raw)
        assert metadata == {}


# ─────────────────────────────────────────────────────────────────────────────
# 10. test_no_modifications_to_input_dataframes
# ─────────────────────────────────────────────────────────────────────────────

class TestNoInputModification:
    """Spec constraint — input prices/returns DataFrames must be unchanged after run."""

    def _hash(self, df: pd.DataFrame) -> int:
        return int(pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).sum())

    def test_plot_price_trends_does_not_modify_prices(self, tmp_path):
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        h_before = self._hash(prices)

        plot_price_trends(prices, [20, 50], tmp_path / "plots", dpi=72)

        assert self._hash(prices) == h_before, "plot_price_trends must not modify prices"

    def test_plot_return_distributions_does_not_modify_returns(self, tmp_path):
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=50)
        h_before = self._hash(returns)

        plot_return_distributions(returns, tmp_path / "plots", dpi=72)

        assert self._hash(returns) == h_before, (
            "plot_return_distributions must not modify returns"
        )

    def test_plot_volatility_does_not_modify_returns(self, tmp_path):
        returns = _make_returns(tickers=("AAPL",), n_days=60)
        h_before = self._hash(returns)

        plot_volatility(returns, window=30, trading_days_per_year=252,
                        out_dir=tmp_path / "plots", dpi=72)

        assert self._hash(returns) == h_before, (
            "plot_volatility must not modify returns"
        )

    def test_plot_correlations_does_not_modify_returns(self, tmp_path):
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=50)
        h_before = self._hash(returns)

        plot_correlations(returns, {}, top_n_pairs=2, out_dir=tmp_path / "plots", dpi=72)

        assert self._hash(returns) == h_before, (
            "plot_correlations must not modify returns"
        )

    def test_detect_outliers_does_not_modify_returns(self):
        returns = _make_returns(tickers=("AAPL",), n_days=20)
        h_before = self._hash(returns)

        detect_outliers(returns, {}, top_n=5)

        assert self._hash(returns) == h_before, (
            "detect_outliers must not modify returns"
        )
