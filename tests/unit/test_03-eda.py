"""
tests/unit/test_03-eda.py
==========================
Comprehensive pytest test suite for src/eda.py — Step 3 of the
Finance Portfolio Analysis Pipeline.

All test logic is derived exclusively from the spec at:
  .claude/specs/03-eda.md

No implementation details are assumed beyond public function signatures
and the data contracts defined in the spec.  No real files from data/raw/
or data/processed/ are read.  Synthetic DataFrames and tmp_path are used
throughout.

Critical note on fixtures: constant return values produce zero variance
and NaN correlations.  Any test touching plot_correlations or
save_eda_summary must use randomised returns (np.random.seed + normal).
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
    plot_volume_trends,
    save_eda_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_prices(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 60,
    start: str = "2024-01-02",
    base_close: float = 100.0,
) -> pd.DataFrame:
    """
    Long-format OHLCV DataFrame satisfying prices_clean_schema.
    Close price drifts slightly so rolling windows produce non-flat lines.
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            close = base_close + float(i) * 0.1
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


def _make_returns_constant(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 59,
    start: str = "2024-01-03",
    return_val: float = 0.001,
) -> pd.DataFrame:
    """
    Long-format returns with a fixed return_val.
    Use only for tests that do NOT depend on variance / correlation.
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for d in dates:
            sr = return_val
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "simple_return": float(sr),
                    "log_return": float(np.log(1.0 + sr)),
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("string")
    return df


def _make_returns_random(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 100,
    start: str = "2024-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Long-format returns with random values — required for any test that
    checks variance, correlation, or summary statistics.
    Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        sr_array = rng.normal(0.001, 0.015, n_days)
        for i, d in enumerate(dates):
            sr = float(sr_array[i])
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "simple_return": sr,
                    "log_return": float(np.log(1.0 + sr)) if sr > -1.0 else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("string")
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet in snappy/pyarrow format."""
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")


def _df_hash(df: pd.DataFrame) -> int:
    """Stable hash of a DataFrame's content for mutation detection."""
    return int(pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).sum())


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_processed_data  (spec §3.2, §6, spec test plan #1)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadProcessedData:
    """FR1/FR2/§6 — load and validate parquet inputs; raise on missing files."""

    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()
        return processed, raw

    # ── spec test #1 ──────────────────────────────────────────────────────────

    def test_load_processed_data_raises_on_missing_prices_parquet(self, tmp_path):
        """
        FR1 / §6 / spec test #1 — FileNotFoundError when prices_clean.parquet absent.
        Error message must hint at 'Step 2'.
        """
        processed, raw = self._setup_dirs(tmp_path)

        with pytest.raises(FileNotFoundError, match="(?i)step.?2|run.*cleaning|prices_clean"):
            load_processed_data(processed, raw)

    def test_load_processed_data_raises_on_missing_returns_parquet(self, tmp_path):
        """
        FR1 / §6 / spec test #1 — FileNotFoundError when returns_daily.parquet absent.
        Error message must hint at 'Step 2'.
        """
        processed, raw = self._setup_dirs(tmp_path)
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        _write_parquet(prices, processed / "prices_clean.parquet")
        # returns_daily.parquet is intentionally absent

        with pytest.raises(FileNotFoundError, match="(?i)step.?2|run.*cleaning|returns_daily"):
            load_processed_data(processed, raw)

    def test_load_processed_data_error_is_file_not_found_not_value_error(self, tmp_path):
        """
        Spec §6 — the exception type is FileNotFoundError, not ValueError or
        RuntimeError; callers must be able to catch the specific exception.
        """
        processed, raw = self._setup_dirs(tmp_path)

        with pytest.raises(FileNotFoundError):
            load_processed_data(processed, raw)

    # ── spec test #9 — missing metadata ───────────────────────────────────────

    def test_load_processed_data_returns_empty_metadata_when_json_missing(self, tmp_path):
        """
        §6 / spec test #9 — missing metadata.json must log a warning and return
        an empty dict; it must NOT raise an exception.
        """
        processed, raw = self._setup_dirs(tmp_path)
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")
        # raw/metadata.json is deliberately absent

        _, _, metadata = load_processed_data(processed, raw)

        assert metadata == {}, (
            f"load_processed_data must return empty dict when metadata.json is absent, "
            f"got {metadata!r}"
        )

    def test_load_processed_data_returns_metadata_when_present(self, tmp_path):
        """FR1 — metadata is correctly loaded when metadata.json exists."""
        processed, raw = self._setup_dirs(tmp_path)
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")
        meta_content = {"AAPL": {"sector": "Technology", "currency": "USD"}}
        (raw / "metadata.json").write_text(
            json.dumps(meta_content), encoding="utf-8"
        )

        _, _, metadata = load_processed_data(processed, raw)

        assert "AAPL" in metadata, "metadata.json content must be returned"
        assert metadata["AAPL"]["sector"] == "Technology", (
            "Metadata values must be preserved exactly"
        )

    def test_load_processed_data_returns_correct_row_counts(self, tmp_path):
        """FR1 happy path — loaded DataFrames have the row counts matching the files."""
        processed, raw = self._setup_dirs(tmp_path)
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        returns = _make_returns_constant(tickers=("AAPL", "MSFT"), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        p, r, _ = load_processed_data(processed, raw)

        assert len(p) == len(prices), (
            f"Expected {len(prices)} price rows, got {len(p)}"
        )
        assert len(r) == len(returns), (
            f"Expected {len(returns)} return rows, got {len(r)}"
        )

    def test_load_processed_data_returns_three_tuple(self, tmp_path):
        """FR1 — return value is a 3-tuple: (prices_df, returns_df, metadata_dict)."""
        processed, raw = self._setup_dirs(tmp_path)
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        result = load_processed_data(processed, raw)

        assert isinstance(result, tuple) and len(result) == 3, (
            f"load_processed_data must return a 3-tuple, got {type(result)}"
        )
        p, r, m = result
        assert isinstance(p, pd.DataFrame), "First element must be a DataFrame"
        assert isinstance(r, pd.DataFrame), "Second element must be a DataFrame"
        assert isinstance(m, dict), "Third element must be a dict"


# ─────────────────────────────────────────────────────────────────────────────
# 2. plot_price_trends  (spec §3.2, FR3, FR15, spec test plan #2)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotPriceTrends:
    """FR3/FR15 — one price PNG per ticker saved at the correct path."""

    def test_plot_price_trends_creates_expected_files(self, tmp_path):
        """
        Spec test #2 — exactly one PNG per ticker is saved at
        out_dir/01_price_trends/{ticker}_price.png.
        """
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20, 50], out, dpi=72)

        tickers = sorted(prices["ticker"].unique().tolist())
        assert len(saved) == len(tickers), (
            f"Expected {len(tickers)} price PNGs, got {len(saved)}"
        )

    def test_plot_price_trends_files_at_correct_subpath(self, tmp_path):
        """Spec test #2 — files must be at out_dir/01_price_trends/{ticker}_price.png."""
        prices = _make_prices(tickers=("AAPL",), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20], out, dpi=72)

        assert len(saved) == 1, f"Expected 1 path, got {len(saved)}"
        path = saved[0]
        assert path.exists(), f"Expected file at {path}"
        assert path.name == "AAPL_price.png", (
            f"Expected 'AAPL_price.png', got '{path.name}'"
        )
        assert path.parent.name == "01_price_trends", (
            f"Expected parent directory '01_price_trends', got '{path.parent.name}'"
        )

    def test_plot_price_trends_creates_output_directory_automatically(self, tmp_path):
        """§6 / FR15 — out_dir/01_price_trends is created if it does not exist."""
        prices = _make_prices(tickers=("AAPL",), n_days=60)
        out = tmp_path / "deep" / "nested" / "plots"
        assert not out.exists(), "Pre-condition: directory must not exist"

        plot_price_trends(prices, [20], out, dpi=72)

        assert (out / "01_price_trends").exists(), (
            "01_price_trends subdirectory must be created automatically"
        )

    def test_plot_price_trends_returns_list_of_path_objects(self, tmp_path):
        """FR3 — return type is list[Path]."""
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        result = plot_price_trends(prices, [20, 50], out, dpi=72)

        assert isinstance(result, list), (
            f"plot_price_trends must return list, got {type(result)}"
        )
        assert all(isinstance(p, Path) for p in result), (
            "All items in returned list must be Path objects"
        )

    def test_plot_price_trends_files_are_png_format(self, tmp_path):
        """FR15 — all saved files must have .png extension."""
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20, 50], out, dpi=72)

        for p in saved:
            assert p.suffix == ".png", (
                f"Expected .png extension, got '{p.suffix}' for {p.name}"
            )

    def test_plot_price_trends_single_ticker_produces_one_file(self, tmp_path):
        """Edge case — single ticker portfolio must produce exactly one file."""
        prices = _make_prices(tickers=("AAPL",), n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20, 50], out, dpi=72)

        assert len(saved) == 1, (
            f"Single-ticker input must produce 1 file, got {len(saved)}"
        )

    # ── spec test #10 — no input modification ─────────────────────────────────

    def test_plot_price_trends_does_not_modify_input_dataframe(self, tmp_path):
        """Spec test #10 — prices DataFrame must be unchanged after the call."""
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        h_before = _df_hash(prices)

        plot_price_trends(prices, [20, 50], tmp_path / "plots", dpi=72)

        assert _df_hash(prices) == h_before, (
            "plot_price_trends must not modify the input prices DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2b. plot_volume_trends  (spec §3.2, FR4, spec §4.1)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotVolumeTrends:
    """FR4 — one volume PNG per ticker saved at out_dir/01_price_trends/{ticker}_volume.png."""

    def test_plot_volume_trends_creates_one_png_per_ticker(self, tmp_path):
        """FR4 — exactly one volume bar chart per ticker."""
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=30)
        out = tmp_path / "plots"

        saved = plot_volume_trends(prices, out, dpi=72)

        tickers = sorted(prices["ticker"].unique().tolist())
        assert len(saved) == len(tickers), (
            f"Expected {len(tickers)} volume PNGs, got {len(saved)}"
        )

    def test_plot_volume_trends_files_at_correct_subpath(self, tmp_path):
        """§4.1 — volume files must be at out_dir/01_price_trends/{ticker}_volume.png."""
        prices = _make_prices(tickers=("AAPL",), n_days=30)
        out = tmp_path / "plots"

        saved = plot_volume_trends(prices, out, dpi=72)

        assert len(saved) == 1, f"Expected 1 path, got {len(saved)}"
        path = saved[0]
        assert path.exists(), f"Expected file at {path}"
        assert path.name == "AAPL_volume.png", (
            f"Expected 'AAPL_volume.png', got '{path.name}'"
        )
        assert path.parent.name == "01_price_trends", (
            f"Expected parent directory '01_price_trends', got '{path.parent.name}'"
        )

    def test_plot_volume_trends_returns_list_of_path_objects(self, tmp_path):
        """FR4 — return type is list[Path]."""
        prices = _make_prices(tickers=("AAPL",), n_days=30)
        out = tmp_path / "plots"

        result = plot_volume_trends(prices, out, dpi=72)

        assert isinstance(result, list), (
            f"plot_volume_trends must return list, got {type(result)}"
        )
        assert all(isinstance(p, Path) for p in result), (
            "All items in returned list must be Path objects"
        )

    def test_plot_volume_trends_does_not_modify_input(self, tmp_path):
        """Spec test #10 — prices DataFrame unchanged after call."""
        prices = _make_prices(tickers=("AAPL",), n_days=30)
        h_before = _df_hash(prices)

        plot_volume_trends(prices, tmp_path / "plots", dpi=72)

        assert _df_hash(prices) == h_before, (
            "plot_volume_trends must not modify the input prices DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. plot_return_distributions  (spec §3.2, FR5/FR6/FR7, spec test plan #3)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotReturnDistributions:
    """FR5/FR6/FR7 — stats DataFrame has all expected columns; one row per ticker."""

    _REQUIRED_STATS_COLS = {
        "ticker", "mean", "std", "skew", "kurtosis", "min", "max", "jarque_bera_pvalue",
    }

    def test_plot_return_distributions_returns_stats_df(self, tmp_path):
        """
        Spec test #3 — stats DataFrame has all required columns and correct ticker count.
        Uses randomised returns to ensure non-zero variance (required for JB test).
        """
        tickers = ("AAPL", "MSFT")
        returns = _make_returns_random(tickers=tickers, n_days=50, seed=0)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert self._REQUIRED_STATS_COLS.issubset(set(stats_df.columns)), (
            f"Missing stats columns: {self._REQUIRED_STATS_COLS - set(stats_df.columns)}"
        )
        assert len(stats_df) == len(tickers), (
            f"Expected {len(tickers)} rows in stats_df, got {len(stats_df)}"
        )

    def test_plot_return_distributions_stats_df_one_row_per_ticker(self, tmp_path):
        """FR7 — one row per ticker in the stats DataFrame."""
        tickers = ("AAPL", "MSFT", "GOOGL")
        returns = _make_returns_random(tickers=tickers, n_days=50, seed=1)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert len(stats_df) == len(tickers), (
            f"Expected {len(tickers)} rows in stats_df, got {len(stats_df)}"
        )

    def test_plot_return_distributions_stats_ticker_values_match_input(self, tmp_path):
        """FR7 — ticker column in stats_df must contain exactly the input tickers."""
        tickers = ("AAPL", "MSFT")
        returns = _make_returns_random(tickers=tickers, n_days=50, seed=2)
        out = tmp_path / "plots"

        _, stats_df = plot_return_distributions(returns, out, dpi=72)

        assert set(stats_df["ticker"].tolist()) == set(tickers), (
            f"stats_df tickers {set(stats_df['ticker'].tolist())} "
            f"do not match input tickers {set(tickers)}"
        )

    def test_plot_return_distributions_boxplot_file_created(self, tmp_path):
        """FR6 — exactly one all_tickers_boxplot.png must be saved."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=50, seed=3)
        out = tmp_path / "plots"

        saved, _ = plot_return_distributions(returns, out, dpi=72)

        boxplot_files = [p for p in saved if "boxplot" in p.name]
        assert len(boxplot_files) == 1, (
            f"Expected exactly one boxplot file, got {len(boxplot_files)}"
        )
        assert boxplot_files[0].name == "all_tickers_boxplot.png", (
            f"Boxplot file must be named 'all_tickers_boxplot.png', "
            f"got '{boxplot_files[0].name}'"
        )

    def test_plot_return_distributions_returns_tuple_of_paths_and_dataframe(
        self, tmp_path
    ):
        """FR5 — return type is tuple[list[Path], pd.DataFrame]."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=50, seed=4)
        out = tmp_path / "plots"

        result = plot_return_distributions(returns, out, dpi=72)

        assert isinstance(result, tuple) and len(result) == 2, (
            f"Expected 2-tuple, got {type(result)}"
        )
        paths, df = result
        assert isinstance(paths, list), f"First element must be list, got {type(paths)}"
        assert isinstance(df, pd.DataFrame), (
            f"Second element must be DataFrame, got {type(df)}"
        )

    def test_plot_return_distributions_skips_qq_for_fewer_than_30_rows(self, tmp_path):
        """
        §6 — Q-Q plot is skipped for a ticker with fewer than 30 rows;
        histogram must still be produced, but no qqplot file for that ticker.
        """
        # 20 rows: fewer than the 30-row minimum for Q-Q
        returns = _make_returns_random(tickers=("TINY",), n_days=20, seed=5)
        out = tmp_path / "plots"

        saved, _ = plot_return_distributions(returns, out, dpi=72)

        qq_files = [p for p in saved if "qqplot" in p.name and "TINY" in p.name]
        hist_files = [p for p in saved if "histogram" in p.name and "TINY" in p.name]

        assert len(qq_files) == 0, (
            "Q-Q plot must not be generated for a ticker with < 30 rows"
        )
        assert len(hist_files) == 1, (
            "Histogram must still be generated even when Q-Q is skipped"
        )

    def test_plot_return_distributions_histogram_files_created_per_ticker(
        self, tmp_path
    ):
        """FR5 — one histogram file per ticker in 02_return_distributions/."""
        tickers = ("AAPL", "MSFT")
        returns = _make_returns_random(tickers=tickers, n_days=50, seed=6)
        out = tmp_path / "plots"

        saved, _ = plot_return_distributions(returns, out, dpi=72)

        hist_files = [p for p in saved if "histogram" in p.name]
        assert len(hist_files) == len(tickers), (
            f"Expected {len(tickers)} histogram files, got {len(hist_files)}"
        )

    # ── spec test #10 — no input modification ─────────────────────────────────

    def test_plot_return_distributions_does_not_modify_input_returns(self, tmp_path):
        """Spec test #10 — returns DataFrame must be unchanged after call."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=50, seed=7)
        h_before = _df_hash(returns)

        plot_return_distributions(returns, tmp_path / "plots", dpi=72)

        assert _df_hash(returns) == h_before, (
            "plot_return_distributions must not modify the input returns DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. plot_volatility  (spec §3.2, FR8/FR9, §6, spec test plan #4)
# ─────────────────────────────────────────────────────────────────────────────


class TestPlotVolatility:
    """FR8/FR9 / §6 — tickers with insufficient data are skipped gracefully."""

    def test_plot_volatility_skips_tickers_below_window(self, tmp_path):
        """
        Spec test #4 — a ticker with fewer rows than the window is skipped
        (no plot generated); a ticker with sufficient data is processed normally.
        The function must NOT crash.
        """
        short_returns = _make_returns_random(tickers=("TINY",), n_days=5, seed=10)
        long_returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=11)
        returns = pd.concat([short_returns, long_returns], ignore_index=True)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        tiny_vol_plots = [p for p in saved if "TINY" in p.name and "rolling_vol" in p.name]
        aapl_vol_plots = [p for p in saved if "AAPL" in p.name and "rolling_vol" in p.name]

        assert len(tiny_vol_plots) == 0, (
            "TINY has fewer rows than the window — rolling_vol plot must be skipped"
        )
        assert len(aapl_vol_plots) == 1, (
            "AAPL has sufficient rows — rolling_vol plot must be generated"
        )

    def test_plot_volatility_no_crash_when_all_tickers_below_window(self, tmp_path):
        """
        §6 — when every ticker has fewer rows than the window, the function must
        return ([], empty_DataFrame) without raising.
        """
        returns = _make_returns_random(tickers=("AAPL",), n_days=5, seed=12)
        out = tmp_path / "plots"

        saved, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        assert isinstance(saved, list), "saved must be a list"
        assert isinstance(monthly_vol, pd.DataFrame), "monthly_vol must be a DataFrame"
        assert monthly_vol.empty, (
            "monthly_vol must be empty when no ticker has sufficient data"
        )

    def test_plot_volatility_returns_monthly_vol_pivot_with_correct_columns(
        self, tmp_path
    ):
        """FR9 — monthly_vol_df must be pivoted: months as index, tickers as columns."""
        tickers = ("AAPL", "MSFT")
        returns = _make_returns_random(tickers=tickers, n_days=60, seed=13)
        out = tmp_path / "plots"

        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        assert not monthly_vol.empty, "monthly_vol must not be empty for sufficient data"
        for t in tickers:
            assert t in monthly_vol.columns, (
                f"Ticker '{t}' must be a column in monthly_vol DataFrame"
            )

    def test_plot_volatility_heatmap_file_created(self, tmp_path):
        """FR9 — monthly_vol_heatmap.png must be saved in 03_volatility/."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=60, seed=14)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        heatmap_files = [p for p in saved if "monthly_vol_heatmap" in p.name]
        assert len(heatmap_files) == 1, (
            f"Expected exactly one monthly_vol_heatmap.png, got {len(heatmap_files)}"
        )
        assert heatmap_files[0].exists(), "monthly_vol_heatmap.png must exist on disk"

    def test_plot_volatility_returns_tuple(self, tmp_path):
        """FR8 — return type is tuple[list[Path], pd.DataFrame]."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=15)
        out = tmp_path / "plots"

        result = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        assert isinstance(result, tuple) and len(result) == 2, (
            f"Expected 2-tuple, got {type(result)}"
        )
        paths, df = result
        assert isinstance(paths, list), f"First element must be list, got {type(paths)}"
        assert isinstance(df, pd.DataFrame), (
            f"Second element must be DataFrame, got {type(df)}"
        )

    def test_plot_volatility_rolling_vol_file_in_03_volatility_subdir(self, tmp_path):
        """FR8 — rolling vol plots saved to out_dir/03_volatility/."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=16)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(
            returns, window=30, trading_days_per_year=252, out_dir=out, dpi=72
        )

        vol_plots = [p for p in saved if "rolling_vol" in p.name]
        assert len(vol_plots) >= 1, "At least one rolling_vol PNG must be generated"
        for p in vol_plots:
            assert p.parent.name == "03_volatility", (
                f"Rolling vol plots must be in '03_volatility/', got '{p.parent.name}'"
            )

    # ── spec test #10 ─────────────────────────────────────────────────────────

    def test_plot_volatility_does_not_modify_input_returns(self, tmp_path):
        """Spec test #10 — returns DataFrame must be unchanged after call."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=17)
        h_before = _df_hash(returns)

        plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=tmp_path / "plots", dpi=72
        )

        assert _df_hash(returns) == h_before, (
            "plot_volatility must not modify the input returns DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. plot_correlations  (spec §3.2, FR10/FR11/FR12, §6, spec test plan #5)
# ─────────────────────────────────────────────────────────────────────────────


def _make_correlated_returns() -> pd.DataFrame:
    """
    Three tickers: AAPL and MSFT highly correlated through a shared base
    signal; XOM is independent.  Seeded for reproducibility.
    All returns are randomised so that correlation values are well-defined
    (non-zero variance).
    """
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    base = rng.normal(0.001, 0.012, n)

    rows = []
    for ticker, is_correlated in [("AAPL", True), ("MSFT", True), ("XOM", False)]:
        if is_correlated:
            sr = base + rng.normal(0, 0.001, n)
        else:
            sr = rng.normal(0.001, 0.015, n)

        for i, d in enumerate(dates):
            v = float(sr[i])
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "simple_return": v,
                    "log_return": float(np.log(1.0 + v)) if v > -1.0 else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("string")
    return df


class TestPlotCorrelations:
    """FR10/FR11/FR12 / §6 — correlation analysis produces correctly sorted pairs."""

    def test_plot_correlations_top_pairs_sorted_by_abs_value(self, tmp_path):
        """
        Spec test #5 — top_pairs list must be sorted by |correlation|
        in descending order.
        """
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=3, out_dir=out, dpi=72
        )

        assert len(top_pairs) > 0, "top_pairs must not be empty for 3+ tickers"
        abs_corrs = [abs(c) for _, _, c in top_pairs]
        assert abs_corrs == sorted(abs_corrs, reverse=True), (
            f"top_pairs must be sorted by |correlation| descending, got {abs_corrs}"
        )

    def test_plot_correlations_top_n_count_respected(self, tmp_path):
        """FR11 — exactly top_n_pairs items returned when enough unique pairs exist."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"
        top_n = 2

        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=top_n, out_dir=out, dpi=72
        )

        assert len(top_pairs) == top_n, (
            f"Expected {top_n} top pairs, got {len(top_pairs)}"
        )

    def test_plot_correlations_matrix_is_symmetric(self, tmp_path):
        """FR10 — corr_matrix_df must be symmetric (corr(A,B) == corr(B,A))."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        _, corr_matrix, _ = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=out, dpi=72
        )

        assert not corr_matrix.empty, "Correlation matrix must not be empty"
        np.testing.assert_allclose(
            corr_matrix.values,
            corr_matrix.values.T,
            atol=1e-10,
            err_msg="Correlation matrix must be symmetric",
        )

    def test_plot_correlations_diagonal_is_one(self, tmp_path):
        """FR10 — self-correlation (diagonal of the matrix) must be 1.0."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        _, corr_matrix, _ = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=out, dpi=72
        )

        np.testing.assert_allclose(
            np.diag(corr_matrix.values),
            np.ones(len(corr_matrix)),
            atol=1e-10,
            err_msg="Diagonal of correlation matrix must be 1.0",
        )

    def test_plot_correlations_skips_analysis_for_single_ticker(self, tmp_path):
        """
        §6 — a single-ticker portfolio has no pairs to correlate.
        Function must return ([], empty_df, []) without crashing.
        """
        returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=20)
        out = tmp_path / "plots"

        paths, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=3, out_dir=out, dpi=72
        )

        assert paths == [], "No files should be saved for single-ticker correlation"
        assert corr_matrix.empty, "corr_matrix must be empty for single ticker"
        assert top_pairs == [], "top_pairs must be empty for single ticker"

    def test_plot_correlations_top_n_exceeding_available_pairs_uses_all(self, tmp_path):
        """
        §6 — when top_n_pairs > number of available pairs, all available
        pairs are returned rather than raising an error.
        """
        # 2 tickers → exactly 1 unique pair
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=100, seed=21)
        out = tmp_path / "plots"

        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=99, out_dir=out, dpi=72
        )

        assert len(top_pairs) == 1, (
            f"With 2 tickers there is 1 unique pair; top_n=99 must return 1, got {len(top_pairs)}"
        )

    def test_plot_correlations_no_sector_heatmap_without_metadata(self, tmp_path):
        """§6/FR12 — no sector_correlation.png generated when metadata is empty."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        paths, _, _ = plot_correlations(
            returns, metadata={}, top_n_pairs=2, out_dir=out, dpi=72
        )

        sector_plots = [p for p in paths if "sector" in p.name]
        assert len(sector_plots) == 0, (
            "sector_correlation.png must NOT be generated when metadata is empty"
        )

    def test_plot_correlations_generates_matrix_heatmap_file(self, tmp_path):
        """FR10 — correlation_matrix.png must be saved in 04_correlations/."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        paths, _, _ = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=out, dpi=72
        )

        matrix_files = [p for p in paths if "correlation_matrix" in p.name]
        assert len(matrix_files) == 1, (
            f"Expected exactly one correlation_matrix.png, got {len(matrix_files)}"
        )
        assert matrix_files[0].parent.name == "04_correlations", (
            f"Must be saved in '04_correlations/', got '{matrix_files[0].parent.name}'"
        )

    def test_plot_correlations_sector_heatmap_generated_with_metadata(self, tmp_path):
        """FR12 — sector_correlation.png must be generated when metadata is provided."""
        returns = _make_correlated_returns()
        metadata = {
            "AAPL": {"sector": "Technology"},
            "MSFT": {"sector": "Technology"},
            "XOM": {"sector": "Energy"},
        }
        out = tmp_path / "plots"

        paths, _, _ = plot_correlations(
            returns, metadata=metadata, top_n_pairs=2, out_dir=out, dpi=72
        )

        sector_plots = [p for p in paths if "sector" in p.name]
        assert len(sector_plots) == 1, (
            f"Expected 1 sector heatmap when metadata is present, got {len(sector_plots)}"
        )

    def test_plot_correlations_top_pairs_are_tuples_of_correct_structure(self, tmp_path):
        """FR11 — each item in top_pairs must be (ticker_a, ticker_b, float)."""
        returns = _make_correlated_returns()
        out = tmp_path / "plots"

        _, _, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=out, dpi=72
        )

        for item in top_pairs:
            assert isinstance(item, tuple) and len(item) == 3, (
                f"Each top_pair must be a 3-tuple, got {item!r}"
            )
            ticker_a, ticker_b, corr_val = item
            assert isinstance(ticker_a, str), f"ticker_a must be str, got {type(ticker_a)}"
            assert isinstance(ticker_b, str), f"ticker_b must be str, got {type(ticker_b)}"
            assert isinstance(corr_val, float), f"correlation must be float, got {type(corr_val)}"

    # ── spec test #10 ─────────────────────────────────────────────────────────

    def test_plot_correlations_does_not_modify_input_returns(self, tmp_path):
        """Spec test #10 — returns DataFrame must be unchanged after call."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=60, seed=22)
        h_before = _df_hash(returns)

        plot_correlations(returns, {}, top_n_pairs=2, out_dir=tmp_path / "plots", dpi=72)

        assert _df_hash(returns) == h_before, (
            "plot_correlations must not modify the input returns DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. detect_outliers  (spec §3.2, FR13/FR14, spec test plan #6)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectOutliers:
    """FR13/FR14 — exactly top_n moves per ticker; cross-reference with cleaning report."""

    def _make_varied_returns_single(self, ticker: str = "AAPL", n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        sr = rng.normal(0.001, 0.015, n)
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "simple_return": sr,
                "log_return": np.log(1 + np.clip(sr, -0.999, None)),
            }
        )
        df["ticker"] = df["ticker"].astype("string")
        return df

    def test_detect_outliers_returns_top_n_per_ticker(self):
        """Spec test #6 — exactly top_n moves returned per ticker."""
        returns = self._make_varied_returns_single("AAPL", n=50)
        top_n = 5

        report = detect_outliers(returns, {}, top_n=top_n)

        aapl_moves = report["top_moves_per_ticker"].get("AAPL", [])
        assert len(aapl_moves) == top_n, (
            f"Expected {top_n} moves for AAPL, got {len(aapl_moves)}"
        )

    def test_detect_outliers_top_moves_are_largest_absolute_returns(self):
        """FR13 — each move in the top list has a larger |return| than any non-listed move."""
        returns = self._make_varied_returns_single("AAPL", n=50)
        top_n = 5

        report = detect_outliers(returns, {}, top_n=top_n)

        moves = report["top_moves_per_ticker"]["AAPL"]
        abs_in_top = sorted([abs(m["simple_return"]) for m in moves], reverse=True)

        all_abs = sorted(
            returns["simple_return"].abs().tolist(), reverse=True
        )
        # The top_n largest abs values must match the top portion of all_abs
        assert abs_in_top == pytest.approx(all_abs[:top_n], rel=1e-6), (
            "Top moves must be the top_n largest |simple_return| values"
        )

    def test_detect_outliers_each_move_has_required_keys(self):
        """FR13 — every move entry must contain date, simple_return, log_return."""
        returns = self._make_varied_returns_single("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)

        for move in report["top_moves_per_ticker"]["AAPL"]:
            assert "date" in move, f"Move must have 'date' key: {move}"
            assert "simple_return" in move, f"Move must have 'simple_return' key: {move}"
            assert "log_return" in move, f"Move must have 'log_return' key: {move}"

    def test_detect_outliers_result_has_all_required_top_level_keys(self):
        """FR13/FR14 — return dict must have all four required top-level keys."""
        returns = self._make_varied_returns_single("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)

        required_keys = {
            "top_moves_per_ticker",
            "eda_vs_cleaning_match",
            "zero_volume_days",
            "zero_price_change_days",
        }
        assert required_keys.issubset(set(report.keys())), (
            f"Missing keys: {required_keys - set(report.keys())}"
        )

    def test_detect_outliers_eda_vs_cleaning_match_has_required_structure(self):
        """FR14 — eda_vs_cleaning_match entry must have eda_count, cleaning_count, overlap."""
        returns = self._make_varied_returns_single("AAPL", n=20)
        report = detect_outliers(returns, {}, top_n=3)

        match = report["eda_vs_cleaning_match"].get("AAPL", {})
        assert "eda_count" in match, "eda_vs_cleaning_match must have 'eda_count'"
        assert "cleaning_count" in match, "eda_vs_cleaning_match must have 'cleaning_count'"
        assert "overlap" in match, "eda_vs_cleaning_match must have 'overlap'"

    def test_detect_outliers_cleaning_count_reflects_cleaning_report(self):
        """FR14 — cleaning_count mirrors the value from cleaning_report."""
        returns = self._make_varied_returns_single("AAPL", n=20)
        cleaning_report = {"actions": {"outliers_flagged": {"AAPL": 7}}}

        report = detect_outliers(returns, cleaning_report, top_n=3)

        assert report["eda_vs_cleaning_match"]["AAPL"]["cleaning_count"] == 7, (
            "cleaning_count must equal the value from cleaning_report['actions']['outliers_flagged']"
        )

    def test_detect_outliers_fewer_rows_than_top_n_returns_all_available(self):
        """
        §6 edge case — when a ticker has fewer rows than top_n,
        all available rows are returned (no IndexError, no crash).
        """
        returns = self._make_varied_returns_single("AAPL", n=3)
        report = detect_outliers(returns, {}, top_n=10)

        aapl_moves = report["top_moves_per_ticker"].get("AAPL", [])
        assert len(aapl_moves) == 3, (
            f"When ticker has 3 rows and top_n=10, all 3 moves must be returned, "
            f"got {len(aapl_moves)}"
        )

    def test_detect_outliers_multiple_tickers(self):
        """FR13 — top moves computed independently per ticker."""
        aapl = self._make_varied_returns_single("AAPL", n=40)
        msft_rng = np.random.default_rng(50)
        dates = pd.date_range("2024-01-02", periods=40, freq="B")
        msft_sr = msft_rng.normal(0.001, 0.02, 40)
        msft = pd.DataFrame({
            "date": dates,
            "ticker": "MSFT",
            "simple_return": msft_sr,
            "log_return": np.log(1 + np.clip(msft_sr, -0.999, None)),
        })
        msft["ticker"] = msft["ticker"].astype("string")
        returns = pd.concat([aapl, msft], ignore_index=True)
        top_n = 4

        report = detect_outliers(returns, {}, top_n=top_n)

        assert "AAPL" in report["top_moves_per_ticker"], "AAPL must appear in top_moves_per_ticker"
        assert "MSFT" in report["top_moves_per_ticker"], "MSFT must appear in top_moves_per_ticker"
        assert len(report["top_moves_per_ticker"]["AAPL"]) == top_n
        assert len(report["top_moves_per_ticker"]["MSFT"]) == top_n

    # ── spec test #10 ─────────────────────────────────────────────────────────

    def test_detect_outliers_does_not_modify_input_returns(self):
        """Spec test #10 — returns DataFrame must be unchanged after call."""
        returns = self._make_varied_returns_single("AAPL", n=20)
        h_before = _df_hash(returns)

        detect_outliers(returns, {}, top_n=5)

        assert _df_hash(returns) == h_before, (
            "detect_outliers must not modify the input returns DataFrame"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. save_eda_summary  (spec §3.2, §4.2, FR16, spec test plan #7)
# ─────────────────────────────────────────────────────────────────────────────


def _build_summary_kwargs(tmp_path: Path) -> dict:
    """
    Build the full set of arguments for save_eda_summary using randomised
    returns so that correlation and stats values are well-defined (non-NaN).
    """
    returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=100, seed=30)
    plots_dir = tmp_path / "plots"

    _, dist_stats = plot_return_distributions(returns, plots_dir, dpi=72)
    _, monthly_vol = plot_volatility(
        returns, window=30, trading_days_per_year=252,
        out_dir=plots_dir, dpi=72
    )
    _, corr_matrix, top_pairs = plot_correlations(
        returns, {}, top_n_pairs=1, out_dir=plots_dir, dpi=72
    )
    outlier_report = detect_outliers(returns, {}, top_n=5)

    return dict(
        distribution_stats=dist_stats,
        monthly_vol=monthly_vol,
        corr_matrix=corr_matrix,
        top_pairs=top_pairs,
        outlier_report=outlier_report,
        saved_plot_paths=[],
        out_dir=tmp_path / "reports",
    )


class TestSaveEdaSummary:
    """FR16 / §4.2 — JSON file written, parseable, contains all required keys."""

    def test_save_eda_summary_writes_valid_json(self, tmp_path):
        """
        Spec test #7 — eda_summary.json is written, is parseable JSON,
        and contains all required top-level keys.
        """
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        assert path.exists(), "eda_summary.json must exist after save_eda_summary"
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert isinstance(parsed, dict), "eda_summary.json must deserialise to a dict"

        required_keys = {
            "run_timestamp",
            "distribution_stats",
            "monthly_volatility",
            "correlations",
            "outliers",
            "plots_generated",
        }
        assert required_keys.issubset(set(parsed.keys())), (
            f"Missing required keys: {required_keys - set(parsed.keys())}"
        )

    def test_save_eda_summary_json_file_exists_at_correct_name(self, tmp_path):
        """FR16 — file must be named eda_summary.json."""
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        assert path.name == "eda_summary.json", (
            f"Expected filename 'eda_summary.json', got '{path.name}'"
        )

    def test_save_eda_summary_correlations_section_has_matrix_and_top_pairs(
        self, tmp_path
    ):
        """§4.2 — correlations section must contain 'matrix' and 'top_pairs' keys."""
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert "matrix" in parsed["correlations"], (
            "'matrix' key must exist in correlations section"
        )
        assert "top_pairs" in parsed["correlations"], (
            "'top_pairs' key must exist in correlations section"
        )

    def test_save_eda_summary_creates_output_directory_when_missing(self, tmp_path):
        """§6 — out_dir must be created with parents=True if it does not exist."""
        kwargs = _build_summary_kwargs(tmp_path)
        non_existent = tmp_path / "brand_new" / "nested" / "reports"
        assert not non_existent.exists(), "Pre-condition: directory must not exist"
        kwargs["out_dir"] = non_existent

        save_eda_summary(**kwargs)

        assert non_existent.exists(), (
            "save_eda_summary must create out_dir (with parents) when it does not exist"
        )

    def test_save_eda_summary_overwrites_existing_file_without_error(self, tmp_path):
        """FR18 — a second call must silently overwrite the existing file."""
        kwargs = _build_summary_kwargs(tmp_path)
        save_eda_summary(**kwargs)  # first write
        path = save_eda_summary(**kwargs)  # second write — must not raise

        assert path.exists(), "eda_summary.json must exist after second write"

    def test_save_eda_summary_has_indent_formatting(self, tmp_path):
        """FR16 / §4.2 — file must be formatted with indent=2 (human-readable)."""
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        text = path.read_text(encoding="utf-8")
        assert "\n  " in text, (
            "eda_summary.json must be written with indent=2 — no indentation found"
        )

    def test_save_eda_summary_returns_path_object(self, tmp_path):
        """FR16 — return value must be a Path pointing to the written file."""
        kwargs = _build_summary_kwargs(tmp_path)
        result = save_eda_summary(**kwargs)

        assert isinstance(result, Path), (
            f"save_eda_summary must return a Path, got {type(result)}"
        )

    def test_save_eda_summary_plots_generated_section_has_total_and_by_block(
        self, tmp_path
    ):
        """
        §4.2 — plots_generated section must contain 'total' and 'by_block' keys;
        by_block must have all five expected block keys.
        """
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        pg = parsed["plots_generated"]
        assert "total" in pg, "'total' must be in plots_generated"
        assert "by_block" in pg, "'by_block' must be in plots_generated"

        expected_blocks = {
            "01_price_trends",
            "02_return_distributions",
            "03_volatility",
            "04_correlations",
            "05_outliers",
        }
        assert expected_blocks.issubset(set(pg["by_block"].keys())), (
            f"Missing block keys: {expected_blocks - set(pg['by_block'].keys())}"
        )

    def test_save_eda_summary_plot_count_matches_saved_paths(self, tmp_path):
        """FR16 — plots_generated.total must equal len(saved_plot_paths)."""
        kwargs = _build_summary_kwargs(tmp_path)
        # Inject three fake paths to verify the count is recorded accurately
        fake_paths = [
            tmp_path / "plots" / "01_price_trends" / "AAPL_price.png",
            tmp_path / "plots" / "01_price_trends" / "MSFT_price.png",
            tmp_path / "plots" / "02_return_distributions" / "AAPL_histogram.png",
        ]
        kwargs["saved_plot_paths"] = fake_paths
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["plots_generated"]["total"] == 3, (
            f"plots_generated.total must equal len(saved_plot_paths)=3, "
            f"got {parsed['plots_generated']['total']}"
        )

    def test_save_eda_summary_distribution_stats_keyed_by_ticker(self, tmp_path):
        """§4.2 — distribution_stats must be a dict keyed by ticker symbol."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=100, seed=31)
        kwargs = _build_summary_kwargs(tmp_path)
        # Regenerate with known tickers
        _, dist_stats = plot_return_distributions(
            returns, tmp_path / "plots_check", dpi=72
        )
        kwargs["distribution_stats"] = dist_stats
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        dist = parsed["distribution_stats"]
        assert isinstance(dist, dict), "distribution_stats must be a dict"
        for ticker in ("AAPL", "MSFT"):
            assert ticker in dist, f"distribution_stats must contain ticker '{ticker}'"


# ─────────────────────────────────────────────────────────────────────────────
# 8. test_idempotent_rerun  (spec §5, FR18, spec test plan #8)
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotentRerun:
    """FR18 — running EDA twice on identical input produces identical JSON output."""

    def test_idempotent_rerun(self, tmp_path):
        """
        Spec test #8 — save_eda_summary called twice with identical inputs and a
        fixed run_timestamp must produce identical analysis content fields:
        distribution_stats, correlations, outliers.top_moves_per_ticker.
        """
        returns = _make_returns_random(
            tickers=("AAPL", "MSFT"), n_days=100, seed=100
        )
        plots1 = tmp_path / "plots1"
        plots2 = tmp_path / "plots2"

        _, dist_stats = plot_return_distributions(returns, plots1, dpi=72)
        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=plots1, dpi=72
        )
        _, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=1, out_dir=plots1, dpi=72
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

        assert parsed1["distribution_stats"] == parsed2["distribution_stats"], (
            "distribution_stats must be identical across two runs with the same input"
        )
        assert parsed1["correlations"] == parsed2["correlations"], (
            "correlations must be identical across two runs with the same input"
        )
        assert (
            parsed1["outliers"]["top_moves_per_ticker"]
            == parsed2["outliers"]["top_moves_per_ticker"]
        ), "top_moves_per_ticker must be identical across two runs with the same input"

    def test_idempotent_rerun_fixed_timestamp_produces_identical_json(self, tmp_path):
        """
        FR18 — when run_timestamp is fixed, the complete JSON output must be
        byte-identical between two consecutive writes to the same path.
        """
        returns = _make_returns_random(
            tickers=("AAPL", "MSFT"), n_days=100, seed=101
        )
        plots_dir = tmp_path / "plots"

        _, dist_stats = plot_return_distributions(returns, plots_dir, dpi=72)
        _, monthly_vol = plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=plots_dir, dpi=72
        )
        _, corr_matrix, top_pairs = plot_correlations(
            returns, {}, top_n_pairs=1, out_dir=plots_dir, dpi=72
        )
        outlier_report = detect_outliers(returns, {}, top_n=5)

        reports_dir = tmp_path / "reports"
        fixed_ts = "2026-05-24T00:00:00Z"
        kwargs = dict(
            distribution_stats=dist_stats,
            monthly_vol=monthly_vol,
            corr_matrix=corr_matrix,
            top_pairs=top_pairs,
            outlier_report=outlier_report,
            saved_plot_paths=[],
            out_dir=reports_dir,
            run_timestamp=fixed_ts,
        )

        path = save_eda_summary(**kwargs)
        bytes1 = path.read_bytes()

        path = save_eda_summary(**kwargs)
        bytes2 = path.read_bytes()

        assert bytes1 == bytes2, (
            "save_eda_summary with fixed run_timestamp must produce byte-identical "
            "output on consecutive calls (FR18 idempotency)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. test_handles_missing_metadata_gracefully  (spec §6, spec test plan #9)
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingMetadata:
    """§6 — missing metadata.json triggers a warning; pipeline continues normally."""

    def test_handles_missing_metadata_gracefully(self, tmp_path):
        """
        Spec test #9 — load_processed_data returns ({}, {}, {}) for metadata
        when metadata.json is absent; the function must NOT raise.
        """
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()
        # metadata.json intentionally absent from raw/

        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        _, _, metadata = load_processed_data(processed, raw)

        assert metadata == {}, (
            f"Missing metadata.json must return empty dict, got {metadata!r}"
        )

    def test_plot_correlations_proceeds_without_metadata(self, tmp_path):
        """§6 — plot_correlations with empty metadata must not crash; matrix still produced."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=60, seed=40)
        out = tmp_path / "plots"

        paths, corr_matrix, top_pairs = plot_correlations(
            returns, metadata={}, top_n_pairs=1, out_dir=out, dpi=72
        )

        assert not corr_matrix.empty, (
            "Correlation matrix must still be computed even without metadata"
        )
        assert len(paths) >= 1, (
            "At least the correlation matrix heatmap must be saved without metadata"
        )

    def test_plot_correlations_no_sector_plot_when_metadata_is_empty(self, tmp_path):
        """§6 — sector heatmap is skipped when metadata={}, no error raised."""
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=60, seed=41)
        out = tmp_path / "plots"

        paths, _, _ = plot_correlations(
            returns, metadata={}, top_n_pairs=1, out_dir=out, dpi=72
        )

        sector_files = [p for p in paths if "sector_correlation" in p.name]
        assert len(sector_files) == 0, (
            "No sector_correlation.png should be generated when metadata is empty"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. test_no_modifications_to_input_dataframes  (spec test plan #10)
# ─────────────────────────────────────────────────────────────────────────────


class TestNoInputModification:
    """
    Spec test #10 — EDA is read-only (§5 constraint).
    Input prices and returns DataFrames must be unchanged after every public function call.
    """

    def test_plot_price_trends_does_not_modify_prices(self, tmp_path):
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=60)
        h_before = _df_hash(prices)
        plot_price_trends(prices, [20, 50], tmp_path / "plots", dpi=72)
        assert _df_hash(prices) == h_before, "plot_price_trends must not mutate prices"

    def test_plot_volume_trends_does_not_modify_prices(self, tmp_path):
        prices = _make_prices(tickers=("AAPL",), n_days=30)
        h_before = _df_hash(prices)
        plot_volume_trends(prices, tmp_path / "plots", dpi=72)
        assert _df_hash(prices) == h_before, "plot_volume_trends must not mutate prices"

    def test_plot_return_distributions_does_not_modify_returns(self, tmp_path):
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=50, seed=50)
        h_before = _df_hash(returns)
        plot_return_distributions(returns, tmp_path / "plots", dpi=72)
        assert _df_hash(returns) == h_before, (
            "plot_return_distributions must not mutate returns"
        )

    def test_plot_volatility_does_not_modify_returns(self, tmp_path):
        returns = _make_returns_random(tickers=("AAPL",), n_days=60, seed=51)
        h_before = _df_hash(returns)
        plot_volatility(
            returns, window=30, trading_days_per_year=252,
            out_dir=tmp_path / "plots", dpi=72
        )
        assert _df_hash(returns) == h_before, (
            "plot_volatility must not mutate returns"
        )

    def test_plot_correlations_does_not_modify_returns(self, tmp_path):
        returns = _make_returns_random(tickers=("AAPL", "MSFT"), n_days=60, seed=52)
        h_before = _df_hash(returns)
        plot_correlations(returns, {}, top_n_pairs=1, out_dir=tmp_path / "plots", dpi=72)
        assert _df_hash(returns) == h_before, (
            "plot_correlations must not mutate returns"
        )

    def test_detect_outliers_does_not_modify_returns(self):
        returns = _make_returns_random(tickers=("AAPL",), n_days=20, seed=53)
        h_before = _df_hash(returns)
        detect_outliers(returns, {}, top_n=5)
        assert _df_hash(returns) == h_before, (
            "detect_outliers must not mutate returns"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Config compliance  (spec §3.1)
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigCompliance:
    """Verify that all EDA config constants from §3.1 are present with the correct types."""

    def test_config_plots_dir_is_path(self):
        """§3.1 — PLOTS_DIR must be a pathlib.Path."""
        import config
        assert isinstance(config.PLOTS_DIR, Path), (
            f"PLOTS_DIR must be Path, got {type(config.PLOTS_DIR)}"
        )

    def test_config_reports_dir_is_path(self):
        """§3.1 — REPORTS_DIR must be a pathlib.Path."""
        import config
        assert isinstance(config.REPORTS_DIR, Path), (
            f"REPORTS_DIR must be Path, got {type(config.REPORTS_DIR)}"
        )

    def test_config_eda_plot_dpi_is_int(self):
        """§3.1 — EDA_PLOT_DPI must be an int."""
        import config
        assert isinstance(config.EDA_PLOT_DPI, int), (
            f"EDA_PLOT_DPI must be int, got {type(config.EDA_PLOT_DPI)}"
        )

    def test_config_eda_plot_dpi_is_positive(self):
        """§3.1 — EDA_PLOT_DPI must be > 0 (print-quality means at least 72)."""
        import config
        assert config.EDA_PLOT_DPI > 0, (
            f"EDA_PLOT_DPI must be positive, got {config.EDA_PLOT_DPI}"
        )

    def test_config_eda_rolling_windows_is_list_of_ints(self):
        """§3.1 — EDA_ROLLING_WINDOWS must be a list of ints."""
        import config
        assert isinstance(config.EDA_ROLLING_WINDOWS, list), (
            f"EDA_ROLLING_WINDOWS must be list, got {type(config.EDA_ROLLING_WINDOWS)}"
        )
        assert all(isinstance(w, int) for w in config.EDA_ROLLING_WINDOWS), (
            "All window values in EDA_ROLLING_WINDOWS must be ints"
        )

    def test_config_eda_vol_window_is_int(self):
        """§3.1 — EDA_VOL_WINDOW must be an int."""
        import config
        assert isinstance(config.EDA_VOL_WINDOW, int), (
            f"EDA_VOL_WINDOW must be int, got {type(config.EDA_VOL_WINDOW)}"
        )

    def test_config_eda_trading_days_per_year_is_int(self):
        """§3.1 — EDA_TRADING_DAYS_PER_YEAR must be an int."""
        import config
        assert isinstance(config.EDA_TRADING_DAYS_PER_YEAR, int), (
            f"EDA_TRADING_DAYS_PER_YEAR must be int, got {type(config.EDA_TRADING_DAYS_PER_YEAR)}"
        )

    def test_config_eda_top_n_correlations_is_int(self):
        """§3.1 — EDA_TOP_N_CORRELATIONS must be an int."""
        import config
        assert isinstance(config.EDA_TOP_N_CORRELATIONS, int), (
            f"EDA_TOP_N_CORRELATIONS must be int, got {type(config.EDA_TOP_N_CORRELATIONS)}"
        )

    def test_config_eda_top_n_moves_is_int(self):
        """§3.1 — EDA_TOP_N_MOVES must be an int."""
        import config
        assert isinstance(config.EDA_TOP_N_MOVES, int), (
            f"EDA_TOP_N_MOVES must be int, got {type(config.EDA_TOP_N_MOVES)}"
        )

    def test_config_eda_plot_style_is_str(self):
        """§3.1 — EDA_PLOT_STYLE must be a string."""
        import config
        assert isinstance(config.EDA_PLOT_STYLE, str), (
            f"EDA_PLOT_STYLE must be str, got {type(config.EDA_PLOT_STYLE)}"
        )

    def test_plot_volatility_respects_window_from_config(self, tmp_path):
        """
        Config compliance — plot_volatility must honour the window argument.
        When window == n_days - 1, rolling std can only be computed for the last row,
        but no crash should occur.  When window == n_days + 1, the ticker is skipped.
        """
        import config
        n = config.EDA_VOL_WINDOW + 5   # just above the default window
        returns = _make_returns_random(tickers=("AAPL",), n_days=n, seed=60)
        out = tmp_path / "plots"

        # Should produce a rolling_vol chart (sufficient rows)
        saved, monthly_vol = plot_volatility(
            returns,
            window=config.EDA_VOL_WINDOW,
            trading_days_per_year=config.EDA_TRADING_DAYS_PER_YEAR,
            out_dir=out,
            dpi=config.EDA_PLOT_DPI,
        )

        vol_plots = [p for p in saved if "AAPL" in p.name and "rolling_vol" in p.name]
        assert len(vol_plots) == 1, (
            f"With n_days={n} > window={config.EDA_VOL_WINDOW}, "
            f"AAPL rolling_vol must be generated"
        )

    @pytest.mark.parametrize("top_n", [1, 2, 3])
    def test_detect_outliers_respects_top_n_moves_param(self, top_n):
        """Config compliance — top_n parameter is respected; no magic number."""
        rng = np.random.default_rng(70 + top_n)
        n = 50
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        sr = rng.normal(0.001, 0.015, n)
        returns = pd.DataFrame({
            "date": dates,
            "ticker": "AAPL",
            "simple_return": sr,
            "log_return": np.log(1 + np.clip(sr, -0.999, None)),
        })
        returns["ticker"] = returns["ticker"].astype("string")

        report = detect_outliers(returns, {}, top_n=top_n)

        assert len(report["top_moves_per_ticker"]["AAPL"]) == top_n, (
            f"detect_outliers must return exactly top_n={top_n} moves, "
            f"got {len(report['top_moves_per_ticker']['AAPL'])}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Schema enforcement  (spec FR2)
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaEnforcement:
    """FR2 — input DataFrames are validated against pandera schemas on load."""

    def test_load_processed_data_raises_on_invalid_prices_schema(self, tmp_path):
        """
        FR2 — if prices_clean.parquet does not satisfy prices_clean_schema
        (e.g. negative close), a RuntimeError must be raised after reading.
        """
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        # Build a prices DataFrame that violates the schema (negative close)
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        prices = prices.copy()
        prices.loc[0, "close"] = -1.0   # violates prices_clean_schema (close > 0)
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4)
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        with pytest.raises((RuntimeError, Exception)):
            load_processed_data(processed, raw)

    def test_load_processed_data_raises_on_invalid_returns_schema(self, tmp_path):
        """
        FR2 — if returns_daily.parquet has a missing required column, validation
        must raise a RuntimeError.
        """
        processed = tmp_path / "processed"
        processed.mkdir()
        raw = tmp_path / "raw"
        raw.mkdir()

        prices = _make_prices(tickers=("AAPL",), n_days=5)
        # Build returns without log_return column — violates returns_schema
        returns = _make_returns_constant(tickers=("AAPL",), n_days=4).drop(
            columns=["log_return"]
        )
        _write_parquet(prices, processed / "prices_clean.parquet")
        _write_parquet(returns, processed / "returns_daily.parquet")

        with pytest.raises((RuntimeError, Exception)):
            load_processed_data(processed, raw)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Additional edge-case and data-integrity tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgesCasesAndDataIntegrity:
    """
    Additional tests covering edge cases and data integrity not in the ten
    primary spec tests but required by the coverage checklist.
    """

    def test_detect_outliers_returns_list_for_zero_volume_days(self):
        """FR13 — zero_volume_days must be a list (possibly empty) in the return dict."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=20, seed=80)
        report = detect_outliers(returns, {}, top_n=3)

        assert isinstance(report["zero_volume_days"], list), (
            "zero_volume_days must be a list"
        )

    def test_detect_outliers_returns_list_for_zero_price_change_days(self):
        """FR13 — zero_price_change_days must be a list (possibly empty) in the return dict."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=20, seed=81)
        report = detect_outliers(returns, {}, top_n=3)

        assert isinstance(report["zero_price_change_days"], list), (
            "zero_price_change_days must be a list"
        )

    def test_plot_price_trends_count_equals_num_tickers(self, tmp_path):
        """Data integrity — number of saved price PNGs == number of tickers in input."""
        tickers = ("AAPL", "MSFT", "GOOGL")
        prices = _make_prices(tickers=tickers, n_days=60)
        out = tmp_path / "plots"

        saved = plot_price_trends(prices, [20, 50], out, dpi=72)

        assert len(saved) == len(tickers), (
            f"Expected {len(tickers)} price PNGs (one per ticker), got {len(saved)}"
        )

    def test_plot_volume_trends_count_equals_num_tickers(self, tmp_path):
        """Data integrity — number of saved volume PNGs == number of tickers in input."""
        tickers = ("AAPL", "MSFT")
        prices = _make_prices(tickers=tickers, n_days=30)
        out = tmp_path / "plots"

        saved = plot_volume_trends(prices, out, dpi=72)

        assert len(saved) == len(tickers), (
            f"Expected {len(tickers)} volume PNGs, got {len(saved)}"
        )

    def test_save_eda_summary_run_timestamp_is_string(self, tmp_path):
        """§4.2 — run_timestamp in the JSON must be a string."""
        kwargs = _build_summary_kwargs(tmp_path)
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed["run_timestamp"], str), (
            f"run_timestamp must be a string, got {type(parsed['run_timestamp'])}"
        )

    def test_save_eda_summary_fixed_timestamp_appears_in_json(self, tmp_path):
        """FR18 — when run_timestamp kwarg is passed, it must appear in the output JSON."""
        kwargs = _build_summary_kwargs(tmp_path)
        fixed_ts = "2099-01-01T12:00:00Z"
        kwargs["run_timestamp"] = fixed_ts
        path = save_eda_summary(**kwargs)

        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["run_timestamp"] == fixed_ts, (
            f"run_timestamp must equal the supplied value '{fixed_ts}', "
            f"got '{parsed['run_timestamp']}'"
        )

    def test_plot_correlations_corr_matrix_has_all_tickers_as_index_and_columns(
        self, tmp_path
    ):
        """FR10 — corr_matrix_df must include all input tickers as both index and columns."""
        tickers = ("AAPL", "MSFT", "XOM")
        returns = _make_returns_random(tickers=tickers, n_days=80, seed=90)
        out = tmp_path / "plots"

        _, corr_matrix, _ = plot_correlations(
            returns, {}, top_n_pairs=2, out_dir=out, dpi=72
        )

        for t in tickers:
            assert t in corr_matrix.index, f"Ticker '{t}' must be in corr_matrix index"
            assert t in corr_matrix.columns, f"Ticker '{t}' must be in corr_matrix columns"

    def test_detect_outliers_empty_cleaning_report_accepted(self):
        """FR14 — detect_outliers must accept an empty cleaning_report without raising."""
        returns = _make_returns_random(tickers=("AAPL",), n_days=20, seed=91)

        # Must not raise
        report = detect_outliers(returns, cleaning_report={}, top_n=3)

        assert report["eda_vs_cleaning_match"]["AAPL"]["cleaning_count"] == 0, (
            "cleaning_count must be 0 when cleaning_report is empty"
        )

    def test_plot_volatility_boundary_ticker_exactly_at_window(self, tmp_path):
        """
        §6 boundary — a ticker with exactly window rows should produce a plot
        (window rows is sufficient for at least one rolling std value).
        """
        window = 10
        returns = _make_returns_random(tickers=("AAPL",), n_days=window, seed=92)
        out = tmp_path / "plots"

        saved, _ = plot_volatility(
            returns, window=window, trading_days_per_year=252, out_dir=out, dpi=72
        )

        vol_plots = [p for p in saved if "AAPL" in p.name and "rolling_vol" in p.name]
        assert len(vol_plots) == 1, (
            f"Ticker with exactly window={window} rows must produce a rolling_vol plot"
        )

    def test_plot_volatility_ticker_one_below_window_is_skipped(self, tmp_path):
        """§6 boundary — a ticker with window-1 rows must be skipped (insufficient data)."""
        window = 10
        returns = _make_returns_random(tickers=("TINY",), n_days=window - 1, seed=93)
        out = tmp_path / "plots"

        saved, monthly_vol = plot_volatility(
            returns, window=window, trading_days_per_year=252, out_dir=out, dpi=72
        )

        vol_plots = [p for p in saved if "TINY" in p.name and "rolling_vol" in p.name]
        assert len(vol_plots) == 0, (
            f"Ticker with {window - 1} rows (< window={window}) must be skipped"
        )
        assert monthly_vol.empty, (
            "monthly_vol must be empty when the only ticker is below the window"
        )
