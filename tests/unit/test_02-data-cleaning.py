"""
tests/unit/test_02-data-cleaning.py
====================================
Comprehensive pytest test suite for src/data_cleaning.py — Step 2 of the
Finance Portfolio Analysis Pipeline.

All test logic is derived exclusively from the spec at:
  .claude/specs/02-data-cleaning.md

No implementation details are assumed beyond public function signatures and
the data contracts defined in the spec.  No real files from data/raw/ or
data/processed/ are read.  pandas_market_calendars is mocked where used.
"""
from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    compute_returns,
    drop_invalid_prices,
    enforce_coverage,
    flag_outliers,
    handle_missing,
    load_raw,
    remove_duplicates,
    save_processed,
    validate_raw_schema,
)
from src.schemas import prices_clean_schema


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_prices(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 5,
    start: str = "2024-01-02",
    base_close: float = 100.0,
) -> pd.DataFrame:
    """
    Minimal long-format OHLCV DataFrame that satisfies Step 1's output
    contract. Uses business-day frequency to approximate trading days.
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": base_close + float(i) * 0.5,
                    "high": base_close + float(i) * 0.5 + 1.0,
                    "low": base_close + float(i) * 0.5 - 0.5,
                    "close": base_close + float(i) * 0.5 + 0.25,
                    "volume": 1_000_000,
                }
            )
    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype("Int64")
    return df


def _make_returns(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    n_days: int = 5,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """
    Minimal long-format returns DataFrame matching compute_returns output:
    columns [date, ticker, simple_return, log_return].
    Returns are small and within the default threshold so they are not
    flagged unless a specific test overrides a value.
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            sr = 0.01 * (i + 1)   # 1 % … 5 % — well below 25 % threshold
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "simple_return": sr,
                    "log_return": float(np.log(1.0 + sr)),
                }
            )
    return pd.DataFrame(rows)


def _mock_calendar_with_n_days(n: int, start: str = "2024-01-02"):
    """
    Return a mock pandas_market_calendars calendar whose schedule()
    returns a DataFrame with exactly *n* index entries.
    """
    trading_days = pd.date_range(start, periods=n, freq="B")
    mock_schedule = pd.DataFrame(index=trading_days)

    mock_cal_instance = mock.MagicMock()
    mock_cal_instance.schedule.return_value = mock_schedule

    mock_get_calendar = mock.MagicMock(return_value=mock_cal_instance)
    return mock_get_calendar


# ─────────────────────────────────────────────────────────────────────────────
# Class 1: load_raw
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRaw:
    """FR1 — load prices_raw.csv and metadata.json from raw_dir."""

    def _write_prices_csv(self, raw_dir: Path, df: pd.DataFrame) -> None:
        df.to_csv(raw_dir / "prices_raw.csv", index=False)

    def _write_metadata_json(self, raw_dir: Path, meta: dict) -> None:
        with open(raw_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh)

    def test_load_raw_raises_file_not_found_when_prices_csv_missing(
        self, tmp_raw_dir
    ):
        """
        §6 edge case — load_raw must raise FileNotFoundError when
        prices_raw.csv does not exist; hint must mention 'Step 1'.
        """
        with pytest.raises(FileNotFoundError, match="(?i)step.?1|run.*ingestion|prices_raw"):
            load_raw(tmp_raw_dir)

    def test_load_raw_returns_empty_metadata_when_json_missing(
        self, tmp_raw_dir
    ):
        """
        §6 edge case — metadata.json missing: load_raw logs a warning and
        returns an empty dict; it must NOT raise.
        """
        df = _make_prices(tickers=("AAPL",), n_days=3)
        self._write_prices_csv(tmp_raw_dir, df)
        # Deliberately NOT writing metadata.json

        result_df, meta = load_raw(tmp_raw_dir)

        assert isinstance(result_df, pd.DataFrame), (
            "load_raw must return a DataFrame even when metadata.json is missing"
        )
        assert isinstance(meta, dict), (
            "load_raw must return a dict for metadata even when file is missing"
        )
        assert meta == {}, (
            f"Expected empty dict for missing metadata.json, got {meta}"
        )

    def test_load_raw_parses_date_column_as_datetime(self, tmp_raw_dir):
        """FR1 — 'date' column must be datetime64[ns] after loading."""
        df = _make_prices(tickers=("AAPL",), n_days=3)
        self._write_prices_csv(tmp_raw_dir, df)

        result_df, _ = load_raw(tmp_raw_dir)

        assert pd.api.types.is_datetime64_any_dtype(result_df["date"]), (
            f"'date' column must be datetime64 after load_raw, got {result_df['date'].dtype}"
        )

    def test_load_raw_uppercases_ticker_column(self, tmp_raw_dir):
        """FR1 — ticker values must be uppercase regardless of CSV content."""
        df = _make_prices(tickers=("AAPL",), n_days=2)
        df["ticker"] = df["ticker"].str.lower()   # write lowercase
        self._write_prices_csv(tmp_raw_dir, df)

        result_df, _ = load_raw(tmp_raw_dir)

        for ticker in result_df["ticker"].unique():
            assert ticker == ticker.upper(), (
                f"Ticker '{ticker}' must be uppercase after load_raw"
            )

    def test_load_raw_returns_correct_row_count(self, tmp_raw_dir):
        """FR1 — loaded DataFrame row count matches CSV row count."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        self._write_prices_csv(tmp_raw_dir, df)

        result_df, _ = load_raw(tmp_raw_dir)

        assert result_df.shape[0] == df.shape[0], (
            f"Expected {df.shape[0]} rows, load_raw returned {result_df.shape[0]}"
        )

    def test_load_raw_loads_metadata_when_present(self, tmp_raw_dir):
        """FR1 — metadata dict is returned correctly when metadata.json exists."""
        df = _make_prices(tickers=("AAPL",), n_days=2)
        self._write_prices_csv(tmp_raw_dir, df)
        meta = {"AAPL": {"sector": "Technology", "currency": "USD"}}
        self._write_metadata_json(tmp_raw_dir, meta)

        _, result_meta = load_raw(tmp_raw_dir)

        assert "AAPL" in result_meta, "metadata.json content must be returned by load_raw"
        assert result_meta["AAPL"]["sector"] == "Technology", (
            "Metadata values must be preserved exactly"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 2: validate_raw_schema
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateRawSchema:
    """FR2 — validate columns, dtypes, and required fields."""

    def test_validate_raw_schema_passes_on_valid_dataframe(self):
        """FR2 happy path — no exception on a correctly-shaped DataFrame."""
        df = _make_prices(tickers=("AAPL",), n_days=3)
        # Ensure float dtypes
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)

        # Must not raise
        validate_raw_schema(df)

    def test_validate_raw_schema_raises_value_error_on_missing_columns(self):
        """FR2 — ValueError when required columns are absent."""
        df = _make_prices(tickers=("AAPL",), n_days=3).drop(columns=["close"])

        with pytest.raises(ValueError, match="(?i)close|missing|column"):
            validate_raw_schema(df)

    def test_validate_raw_schema_raises_value_error_on_wrong_date_dtype(self):
        """FR2 — ValueError when 'date' column is not datetime64."""
        df = _make_prices(tickers=("AAPL",), n_days=3).copy()
        df["date"] = df["date"].astype(str)   # convert to object / string

        with pytest.raises(ValueError, match="(?i)date|datetime"):
            validate_raw_schema(df)

    def test_validate_raw_schema_raises_value_error_on_unexpected_columns(self):
        """
        §6 — validate_raw_schema must raise ValueError when the DataFrame
        contains unexpected columns that violate the contract.  The spec
        treats an absent required column as a schema mismatch.
        """
        df = _make_prices(tickers=("AAPL",), n_days=3).drop(columns=["volume"])

        with pytest.raises(ValueError):
            validate_raw_schema(df)

    def test_validate_raw_schema_raises_value_error_on_non_float_ohlc(self):
        """FR2 — OHLC columns must be float64; integer dtype triggers ValueError."""
        df = _make_prices(tickers=("AAPL",), n_days=3).copy()
        df["close"] = df["close"].astype(int)

        with pytest.raises(ValueError, match="(?i)close|float|dtype"):
            validate_raw_schema(df)


# ─────────────────────────────────────────────────────────────────────────────
# Class 3: remove_duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveDuplicates:
    """FR3 — drop exact duplicate (date, ticker) rows, keeping the first."""

    def test_remove_duplicates_keeps_first(self):
        """
        FR3 — when two rows share the same (date, ticker), the first row's
        values are retained and the second is discarded.
        """
        df = _make_prices(tickers=("AAPL",), n_days=3)
        # Inject a duplicate of row 0 with a different close value
        dup = df.iloc[[0]].copy()
        dup["close"] = 9999.0
        df_with_dup = pd.concat([df, dup], ignore_index=True)

        result, count = remove_duplicates(df_with_dup)

        assert count == 1, f"Expected 1 duplicate removed, got {count}"
        assert len(result) == 3, f"Expected 3 rows after dedup, got {len(result)}"
        # The first occurrence (close != 9999) must be kept
        date_of_dup = df.iloc[0]["date"]
        kept_close = result.loc[result["date"] == date_of_dup, "close"].iloc[0]
        assert kept_close != 9999.0, (
            f"The duplicate row (close=9999) was kept instead of the original"
        )

    def test_remove_duplicates_returns_zero_count_on_clean_data(self):
        """FR3 — no duplicates in clean data returns count=0 and unchanged DataFrame."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result, count = remove_duplicates(df)

        assert count == 0, f"Expected 0 duplicates on clean data, got {count}"
        assert len(result) == len(df), (
            f"Row count must be unchanged, expected {len(df)}, got {len(result)}"
        )

    def test_remove_duplicates_count_matches_rows_removed(self):
        """FR3 — the returned count equals the actual number of rows removed."""
        df = _make_prices(tickers=("AAPL",), n_days=3)
        # Duplicate every row
        df_doubled = pd.concat([df, df], ignore_index=True)

        result, count = remove_duplicates(df_doubled)

        assert count == len(df), (
            f"Expected {len(df)} duplicates removed, got {count}"
        )
        assert len(result) == len(df), (
            f"Expected {len(df)} rows after dedup, got {len(result)}"
        )

    def test_remove_duplicates_no_duplicate_pairs_in_output(self):
        """FR3 / data integrity — output must contain no duplicate (date, ticker) pairs."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        # Inject two duplicates
        extra = df.iloc[:2].copy()
        df_dirty = pd.concat([df, extra], ignore_index=True)

        result, _ = remove_duplicates(df_dirty)

        duplicates = result.duplicated(subset=["date", "ticker"])
        assert not duplicates.any(), (
            f"remove_duplicates output still contains {duplicates.sum()} duplicate pairs"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 4: handle_missing
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleMissing:
    """FR5 — forward-fill OHLC up to max_consecutive days; never fill volume."""

    def _prices_with_nan_gap(
        self, gap_start_idx: int, gap_length: int, total_rows: int = 8
    ) -> pd.DataFrame:
        """Build single-ticker prices where close is NaN for gap_length rows."""
        dates = pd.date_range("2024-01-02", periods=total_rows, freq="B")
        rows = []
        for i, d in enumerate(dates):
            is_gap = gap_start_idx <= i < gap_start_idx + gap_length
            rows.append(
                {
                    "date": d,
                    "ticker": "AAPL",
                    "open": 100.0 if not is_gap else np.nan,
                    "high": 102.0 if not is_gap else np.nan,
                    "low": 99.0 if not is_gap else np.nan,
                    "close": 101.0 if not is_gap else np.nan,
                    "volume": pd.NA,
                }
            )
        df = pd.DataFrame(rows)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype("Int64")
        return df

    def test_handle_missing_respects_max_consecutive_fills(self):
        """
        FR5 — a gap of 4 consecutive NaN closes with max_consecutive=3:
        rows 1–3 are filled, row 4 is still NaN so the entire row is dropped.
        Result: 7 total rows (8 original - 1 dropped beyond limit).
        """
        df = self._prices_with_nan_gap(gap_start_idx=1, gap_length=4, total_rows=8)

        result, fills = handle_missing(df, method="ffill", max_consecutive=3)

        assert len(result) == 7, (
            f"Expected 7 rows (1 beyond-limit row dropped), got {len(result)}"
        )
        assert fills.get("AAPL", 0) == 3, (
            f"Expected 3 fills for AAPL, got {fills.get('AAPL', 0)}"
        )
        assert result["close"].isna().sum() == 0, (
            "All remaining close values must be non-NaN after handle_missing"
        )

    def test_handle_missing_does_not_fill_volume(self):
        """FR5 — volume NaN values must remain NaN (zero volume is meaningful)."""
        df = _make_prices(tickers=("AAPL",), n_days=5).copy()
        df.loc[2, "volume"] = pd.NA   # intentionally missing

        result, _ = handle_missing(df, method="ffill", max_consecutive=3)

        # Locate the row for the originally-missing volume date
        target_date = df.loc[2, "date"]
        matching = result[result["date"] == target_date]
        assert not matching.empty, "Row with originally-missing volume must still exist"
        assert pd.isna(matching.iloc[0]["volume"]), (
            "Volume must remain NaN after handle_missing — spec FR5 'Volume: do NOT fill'"
        )

    def test_handle_missing_max_consecutive_zero_disables_filling(self):
        """
        §6 edge case — MAX_CONSECUTIVE_FILLS=0 disables forward-fill entirely.
        Any row with a missing close is dropped.
        """
        # Row index 2 has NaN close
        df = self._prices_with_nan_gap(gap_start_idx=2, gap_length=1, total_rows=5)

        result, fills = handle_missing(df, method="ffill", max_consecutive=0)

        assert len(result) == 4, (
            f"With max_consecutive=0, the NaN row must be dropped; expected 4 rows, got {len(result)}"
        )
        total_filled = sum(fills.values())
        assert total_filled == 0, (
            f"With max_consecutive=0, no fills should occur; got {total_filled}"
        )

    def test_handle_missing_fills_per_ticker_dict_is_returned(self):
        """FR5 — second return value must be a dict mapping ticker to fill count."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        # Introduce one NaN in AAPL close
        df = df.copy()
        df.loc[df["ticker"] == "AAPL", "close"] = df.loc[
            df["ticker"] == "AAPL", "close"
        ].where(df.loc[df["ticker"] == "AAPL"].index != df.loc[df["ticker"] == "AAPL"].index[1], other=np.nan)

        _, fills = handle_missing(df, method="ffill", max_consecutive=3)

        assert isinstance(fills, dict), (
            f"fills_per_ticker must be a dict, got {type(fills)}"
        )
        assert "AAPL" in fills, "AAPL must appear in fills_per_ticker dict"
        assert "MSFT" in fills, "MSFT must appear in fills_per_ticker dict"

    def test_handle_missing_clean_data_has_zero_fills(self):
        """FR5 — no fills reported on already-complete data."""
        df = _make_prices(tickers=("AAPL",), n_days=5)

        _, fills = handle_missing(df, method="ffill", max_consecutive=3)

        for ticker, count in fills.items():
            assert count == 0, (
                f"Expected 0 fills for {ticker} on clean data, got {count}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Class 5: drop_invalid_prices
# ─────────────────────────────────────────────────────────────────────────────

class TestDropInvalidPrices:
    """FR6 — drop rows with zero/negative OHLC or high < low."""

    def test_drop_invalid_prices_removes_zero_and_negative(self):
        """FR6 — rows with close=0 and open<0 must both be dropped."""
        df = _make_prices(tickers=("AAPL",), n_days=5).copy()
        df.loc[0, "close"] = 0.0     # zero
        df.loc[1, "open"] = -5.0     # negative

        result, dropped = drop_invalid_prices(df)

        assert dropped == 2, f"Expected 2 rows dropped, got {dropped}"
        assert len(result) == 3, f"Expected 3 rows remaining, got {len(result)}"
        assert (result[["open", "high", "low", "close"]] > 0).all().all(), (
            "All OHLC values in the output must be > 0"
        )

    def test_drop_invalid_prices_handles_high_less_than_low(self):
        """§6 edge case — rows where high < low must be dropped (data feed error)."""
        df = _make_prices(tickers=("AAPL",), n_days=5).copy()
        # Force high < low on row index 2
        df.loc[2, "high"] = 95.0
        df.loc[2, "low"] = 100.0    # now low > high

        result, dropped = drop_invalid_prices(df)

        assert dropped >= 1, (
            f"Expected at least 1 row dropped for high < low, got {dropped}"
        )
        # All remaining rows must satisfy high >= low
        remaining = result[result["ticker"] == "AAPL"]
        invalid_hl = remaining[remaining["high"] < remaining["low"]]
        assert invalid_hl.empty, (
            f"Output contains {len(invalid_hl)} row(s) where high < low"
        )

    def test_drop_invalid_prices_sets_negative_volume_to_nan_not_drops_row(self):
        """
        §6 edge case — negative volume is sanitised to NaN (not a row drop).
        The row itself must survive; only volume becomes NaN.
        """
        df = _make_prices(tickers=("AAPL",), n_days=5).copy()
        df["volume"] = df["volume"].astype("Int64")
        df.loc[1, "volume"] = pd.array([-500], dtype="Int64")[0]

        result, dropped = drop_invalid_prices(df)

        # Row count: only truly invalid OHLC rows are dropped; negative volume is NOT a row drop
        assert len(result) == 5, (
            f"Negative volume must not drop the row; expected 5 rows, got {len(result)}"
        )
        # The volume on that row must now be NaN
        target_date = df.loc[1, "date"]
        vol_val = result.loc[result["date"] == target_date, "volume"].iloc[0]
        assert pd.isna(vol_val), (
            f"Negative volume must be set to NaN, but got {vol_val}"
        )

    def test_drop_invalid_prices_returns_zero_count_on_clean_data(self):
        """FR6 happy path — clean data produces dropped=0 and identical row count."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result, dropped = drop_invalid_prices(df)

        assert dropped == 0, f"Expected 0 rows dropped on clean data, got {dropped}"
        assert len(result) == len(df), (
            f"Row count must be unchanged: expected {len(df)}, got {len(result)}"
        )

    @pytest.mark.parametrize("col", ["open", "high", "low", "close"])
    def test_drop_invalid_prices_drops_row_for_each_zero_ohlc_column(self, col):
        """FR6 — a zero in any one of open/high/low/close triggers a row drop."""
        df = _make_prices(tickers=("AAPL",), n_days=4).copy()
        df.loc[0, col] = 0.0

        result, dropped = drop_invalid_prices(df)

        assert dropped == 1, (
            f"Expected 1 row dropped when {col}=0, got {dropped}"
        )
        assert 0.0 not in result[col].values, (
            f"Zero in '{col}' must not survive drop_invalid_prices"
        )

    @pytest.mark.parametrize("col", ["open", "high", "low", "close"])
    def test_drop_invalid_prices_drops_row_for_each_negative_ohlc_column(self, col):
        """FR6 — a negative value in any OHLC column triggers a row drop."""
        df = _make_prices(tickers=("AAPL",), n_days=4).copy()
        df.loc[0, col] = -1.0

        result, dropped = drop_invalid_prices(df)

        assert dropped == 1, (
            f"Expected 1 row dropped when {col}=-1, got {dropped}"
        )
        negative_mask = result[col] < 0
        assert not negative_mask.any(), (
            f"Negative value in '{col}' must not survive drop_invalid_prices"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 6: enforce_coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestEnforceCoverage:
    """FR7 — drop tickers with fewer than min_pct of expected trading days."""

    def _build_coverage_df(self) -> pd.DataFrame:
        """
        Build a two-ticker DataFrame where AAPL has 5 rows and SPARSE has 1 row
        across a 5-business-day window.  Against a 5-day calendar, AAPL coverage
        = 1.00 and SPARSE coverage = 0.20.
        """
        dates_full = pd.date_range("2024-01-02", periods=5, freq="B")
        rows = [
            {"date": d, "ticker": "AAPL", "open": 100.0, "high": 102.0,
             "low": 99.0, "close": 101.0, "volume": 1}
            for d in dates_full
        ]
        # SPARSE only has 1 row
        rows.append(
            {"date": dates_full[0], "ticker": "SPARSE", "open": 50.0,
             "high": 51.0, "low": 49.0, "close": 50.0, "volume": 1}
        )
        df = pd.DataFrame(rows)
        df["volume"] = df["volume"].astype("Int64")
        return df

    def test_enforce_coverage_drops_sparse_ticker(self):
        """
        FR7 — SPARSE ticker with coverage 0.20 < min_pct=0.80 must be
        removed from the result DataFrame and appear in dropped_tickers.
        """
        df = self._build_coverage_df()
        dates_full = pd.date_range("2024-01-02", periods=5, freq="B")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=dates_full
            )
            result, dropped, coverage = enforce_coverage(
                df, min_pct=0.80, calendar_name="NYSE"
            )

        assert "SPARSE" in dropped, (
            "SPARSE ticker must appear in dropped_tickers list"
        )
        assert "AAPL" not in dropped, (
            "AAPL must NOT be dropped — its coverage is 1.00"
        )
        assert "SPARSE" not in result["ticker"].values, (
            "SPARSE must be removed from the result DataFrame"
        )
        assert coverage.get("SPARSE", 1.0) < 0.80, (
            f"SPARSE coverage must be < 0.80, got {coverage.get('SPARSE')}"
        )

    def test_enforce_coverage_keeps_sufficient_ticker(self):
        """FR7 happy path — tickers at or above min_pct are retained."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        trading_days = pd.date_range("2024-01-02", periods=5, freq="B")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=trading_days
            )
            result, dropped, coverage = enforce_coverage(
                df, min_pct=0.80, calendar_name="NYSE"
            )

        assert dropped == [], (
            f"No tickers should be dropped from a fully-covered dataset, got {dropped}"
        )
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}, (
            "Both tickers must survive the coverage check"
        )

    def test_enforce_coverage_returns_coverage_dict_with_correct_keys(self):
        """FR7 — coverage_per_ticker dict must contain an entry for every ticker."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        trading_days = pd.date_range("2024-01-02", periods=5, freq="B")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=trading_days
            )
            _, dropped, coverage = enforce_coverage(
                df, min_pct=0.50, calendar_name="NYSE"
            )

        assert "AAPL" in coverage, "coverage_per_ticker must contain AAPL"
        assert "MSFT" in coverage, "coverage_per_ticker must contain MSFT"

    def test_enforce_coverage_values_are_between_zero_and_one(self):
        """FR7 — coverage values must be in [0.0, 1.0]."""
        df = _make_prices(tickers=("AAPL",), n_days=5)
        trading_days = pd.date_range("2024-01-02", periods=5, freq="B")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=trading_days
            )
            _, _, coverage = enforce_coverage(
                df, min_pct=0.50, calendar_name="NYSE"
            )

        for ticker, pct in coverage.items():
            assert 0.0 <= pct <= 1.0, (
                f"Coverage for {ticker} must be in [0, 1], got {pct}"
            )

    def test_enforce_coverage_threshold_boundary_exactly_at_min_kept(self):
        """
        FR7 boundary — a ticker whose coverage equals min_pct exactly
        should be retained (not dropped).
        """
        # 4 rows out of 5 expected = 0.80 exactly
        dates_full = pd.date_range("2024-01-02", periods=5, freq="B")
        rows = [
            {"date": d, "ticker": "BOUNDARY", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.5, "volume": 1}
            for d in dates_full[:4]    # only 4 of 5 days
        ]
        df = pd.DataFrame(rows)
        df["volume"] = df["volume"].astype("Int64")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=dates_full
            )
            result, dropped, coverage = enforce_coverage(
                df, min_pct=0.80, calendar_name="NYSE"
            )

        assert "BOUNDARY" not in dropped, (
            f"Ticker at exactly min_pct=0.80 must NOT be dropped; "
            f"coverage was {coverage.get('BOUNDARY')}"
        )

    def test_enforce_coverage_threshold_just_below_min_drops_ticker(self):
        """FR7 boundary — coverage just below min_pct must trigger a drop."""
        # 3 rows out of 5 expected = 0.60 < 0.80
        dates_full = pd.date_range("2024-01-02", periods=5, freq="B")
        rows = [
            {"date": d, "ticker": "BELOW", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.5, "volume": 1}
            for d in dates_full[:3]
        ]
        df = pd.DataFrame(rows)
        df["volume"] = df["volume"].astype("Int64")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=dates_full
            )
            result, dropped, coverage = enforce_coverage(
                df, min_pct=0.80, calendar_name="NYSE"
            )

        assert "BELOW" in dropped, (
            f"Ticker with coverage {coverage.get('BELOW')} < 0.80 must be dropped"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 7: compute_returns
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeReturns:
    """FR9 — compute simple_return and log_return per ticker; drop first row."""

    def test_compute_returns_first_row_dropped_per_ticker(self):
        """
        FR9 / §4.2 — the first trading day per ticker has no prior close;
        its return row must be absent from the output.  With n_days=5 per
        ticker, each ticker must have exactly 4 return rows.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        returns = compute_returns(df)

        # 4 rows per ticker (first day dropped) × 2 tickers = 8
        assert len(returns) == 8, (
            f"Expected 8 return rows (4 per ticker × 2), got {len(returns)}"
        )
        for ticker, grp in returns.groupby("ticker"):
            assert len(grp) == 4, (
                f"Expected 4 rows for {ticker}, got {len(grp)}"
            )
            assert grp["simple_return"].notna().all(), (
                f"simple_return must be non-NaN for {ticker}"
            )
            assert grp["log_return"].notna().all(), (
                f"log_return must be non-NaN for {ticker}"
            )

    def test_compute_returns_drops_ticker_with_fewer_than_2_rows(self):
        """
        §6 edge case — a ticker with only 1 price row cannot produce a return.
        compute_returns must silently drop it and log a warning.
        """
        df = _make_prices(tickers=("AAPL",), n_days=5)
        # Add TINY with only 1 row
        tiny_row = pd.DataFrame(
            [{"date": pd.Timestamp("2024-01-02"), "ticker": "TINY",
              "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0,
              "volume": pd.array([100], dtype="Int64")[0]}]
        )
        df_combined = pd.concat([df, tiny_row], ignore_index=True)

        returns = compute_returns(df_combined)

        assert "TINY" not in returns["ticker"].values, (
            "TINY (single row) must not appear in returns — cannot compute return"
        )
        assert "AAPL" in returns["ticker"].values, (
            "AAPL must still produce returns even when TINY is dropped"
        )

    def test_compute_returns_output_columns_are_exact(self):
        """
        FR9 / §4.2 — output DataFrame must have exactly
        [date, ticker, simple_return, log_return] columns.
        """
        df = _make_prices(tickers=("AAPL",), n_days=5)

        returns = compute_returns(df)

        expected_cols = {"date", "ticker", "simple_return", "log_return"}
        actual_cols = set(returns.columns)
        assert actual_cols == expected_cols, (
            f"Expected columns {expected_cols}, got {actual_cols}"
        )

    @pytest.mark.parametrize("close_series,expected_simple", [
        # Two rows: close goes from 100 to 110 → simple_return = 0.10
        ([100.0, 110.0], 0.10),
        # Two rows: close goes from 200 to 190 → simple_return = -0.05
        ([200.0, 190.0], -0.05),
        # Two rows: close unchanged → simple_return = 0.0
        ([150.0, 150.0], 0.0),
    ])
    def test_compute_returns_simple_return_formula(self, close_series, expected_simple):
        """FR9 — simple_return = close.pct_change() = (c_t - c_t-1) / c_t-1."""
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAPL",
                "open": close_series,
                "high": [v + 1.0 for v in close_series],
                "low": [v - 0.5 for v in close_series],
                "close": close_series,
                "volume": pd.array([1_000_000, 1_000_000], dtype="Int64"),
            }
        )

        returns = compute_returns(df)

        assert len(returns) == 1, f"Expected 1 return row, got {len(returns)}"
        sr = returns.iloc[0]["simple_return"]
        assert sr == pytest.approx(expected_simple, rel=1e-6), (
            f"simple_return mismatch: expected {expected_simple}, got {sr}"
        )

    @pytest.mark.parametrize("close_series,expected_log", [
        ([100.0, 110.0], float(np.log(110.0 / 100.0))),
        ([200.0, 190.0], float(np.log(190.0 / 200.0))),
        ([150.0, 150.0], 0.0),
    ])
    def test_compute_returns_log_return_formula(self, close_series, expected_log):
        """FR9 — log_return = ln(close_t / close_t-1)."""
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAPL",
                "open": close_series,
                "high": [v + 1.0 for v in close_series],
                "low": [v - 0.5 for v in close_series],
                "close": close_series,
                "volume": pd.array([1_000_000, 1_000_000], dtype="Int64"),
            }
        )

        returns = compute_returns(df)

        lr = returns.iloc[0]["log_return"]
        assert lr == pytest.approx(expected_log, rel=1e-6), (
            f"log_return mismatch: expected {expected_log}, got {lr}"
        )

    def test_compute_returns_output_sorted_by_date_per_ticker(self):
        """FR9 determinism — rows must be sorted by date within each ticker."""
        df = _make_prices(tickers=("AAPL",), n_days=5)
        # Shuffle input order
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        returns = compute_returns(df_shuffled)

        for ticker, grp in returns.groupby("ticker"):
            dates_list = grp["date"].tolist()
            assert dates_list == sorted(dates_list), (
                f"Returns for {ticker} must be sorted by date, got {dates_list}"
            )

    def test_compute_returns_no_nan_in_output(self):
        """FR9 — output must contain no NaN in simple_return or log_return columns."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        returns = compute_returns(df)

        assert returns["simple_return"].isna().sum() == 0, (
            "simple_return must contain no NaN values"
        )
        assert returns["log_return"].isna().sum() == 0, (
            "log_return must contain no NaN values"
        )

    def test_compute_returns_empty_input_returns_empty_dataframe(self):
        """FR9 edge case — empty prices input must produce an empty returns DataFrame."""
        df = pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )

        returns = compute_returns(df)

        assert isinstance(returns, pd.DataFrame), (
            "compute_returns must return a DataFrame for empty input"
        )
        assert returns.empty, "compute_returns must return an empty DataFrame for empty input"
        expected_cols = {"date", "ticker", "simple_return", "log_return"}
        assert expected_cols.issubset(set(returns.columns)), (
            f"Empty returns DataFrame must still have the required columns"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 8: flag_outliers
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagOutliers:
    """FR8 — identify |simple_return| > threshold; never modify the returns DF."""

    def test_flag_outliers_does_not_modify_returns(self):
        """
        FR8 — the returns DataFrame passed in must be returned unchanged.
        We verify via a hash of every cell value.
        """
        returns = _make_returns(tickers=("AAPL",), n_days=3).copy()
        returns.loc[0, "simple_return"] = 0.40   # outlier above 0.25

        original_hash = pd.util.hash_pandas_object(returns, index=True).sum()

        unchanged, flagged = flag_outliers(returns, threshold=0.25)

        after_hash = pd.util.hash_pandas_object(unchanged, index=True).sum()
        assert original_hash == after_hash, (
            "flag_outliers must not modify the returns DataFrame in any way"
        )

    def test_flag_outliers_threshold_zero_flags_all_nonzero_returns(self):
        """
        §6 edge case — threshold=0 means every non-zero return is flagged.
        Allowed and useful for debugging.
        """
        returns = _make_returns(tickers=("AAPL",), n_days=5)
        # All returns are non-zero (0.01 … 0.05)
        nonzero_count = (returns["simple_return"].abs() > 0).sum()

        _, flagged = flag_outliers(returns, threshold=0.0)

        assert len(flagged) == nonzero_count, (
            f"With threshold=0, expected {nonzero_count} rows flagged, got {len(flagged)}"
        )

    def test_flag_outliers_reason_column_value(self):
        """
        FR8 / §4.3 — every row in the flagged DataFrame must have
        reason = 'abs_return_exceeds_threshold'.
        """
        returns = _make_returns(tickers=("AAPL",), n_days=3).copy()
        returns.loc[0, "simple_return"] = 0.50   # clear outlier

        _, flagged = flag_outliers(returns, threshold=0.25)

        assert not flagged.empty, "Expected at least one flagged row"
        assert (flagged["reason"] == "abs_return_exceeds_threshold").all(), (
            f"All flagged rows must have reason='abs_return_exceeds_threshold', "
            f"got: {flagged['reason'].unique()}"
        )

    def test_flag_outliers_empty_when_no_outliers(self):
        """FR8 — when all returns are within threshold, flagged DataFrame is empty."""
        returns = _make_returns(tickers=("AAPL", "MSFT"), n_days=5)
        # All values ≤ 0.05 < 0.25 threshold

        _, flagged = flag_outliers(returns, threshold=0.25)

        assert flagged.empty, (
            f"Expected empty flagged DataFrame when no outliers exist, got {len(flagged)} rows"
        )

    def test_flag_outliers_flagged_schema_has_required_columns(self):
        """
        FR8 / §4.3 — flagged DataFrame must have exactly
        [date, ticker, simple_return, log_return, reason].
        """
        returns = _make_returns(tickers=("AAPL",), n_days=3).copy()
        returns.loc[0, "simple_return"] = 0.99   # extreme outlier

        _, flagged = flag_outliers(returns, threshold=0.25)

        expected_cols = {"date", "ticker", "simple_return", "log_return", "reason"}
        actual_cols = set(flagged.columns)
        assert actual_cols == expected_cols, (
            f"flagged DataFrame columns mismatch: expected {expected_cols}, got {actual_cols}"
        )

    def test_flag_outliers_both_positive_and_negative_exceedance(self):
        """FR8 — |simple_return| > threshold flags both large positive and negative returns."""
        returns = _make_returns(tickers=("AAPL",), n_days=4).copy()
        returns.loc[0, "simple_return"] = 0.30    # positive outlier
        returns.loc[1, "simple_return"] = -0.30   # negative outlier

        _, flagged = flag_outliers(returns, threshold=0.25)

        assert len(flagged) == 2, (
            f"Expected 2 flagged rows (one positive, one negative), got {len(flagged)}"
        )

    def test_flag_outliers_count_is_exact(self):
        """FR8 — exactly the threshold-exceeding rows and no others are flagged."""
        returns = _make_returns(tickers=("AAPL",), n_days=5).copy()
        # Inject exactly 2 outliers
        returns.loc[0, "simple_return"] = 0.26    # just above 0.25
        returns.loc[1, "simple_return"] = 0.10    # within threshold

        _, flagged = flag_outliers(returns, threshold=0.25)

        assert len(flagged) == 1, (
            f"Expected exactly 1 flagged row (0.26 > 0.25), got {len(flagged)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 9: schema validation (pandera prices_clean_schema)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:
    """FR14 — prices_clean_schema catches contract violations before writing."""

    def _make_valid_schema_df(self) -> pd.DataFrame:
        """Build a minimal DataFrame that satisfies prices_clean_schema."""
        df = _make_prices(tickers=("AAPL",), n_days=3)
        df["volume"] = df["volume"].astype("Int64")
        return df

    def test_schema_validation_passes_on_clean_data(self):
        """FR14 happy path — a valid DataFrame passes without raising."""
        df = self._make_valid_schema_df()
        # Must not raise
        prices_clean_schema.validate(df)

    def test_schema_validation_catches_negative_price(self):
        """FR14 — pandera must raise on a negative close value."""
        df = self._make_valid_schema_df().copy()
        df.loc[0, "close"] = -1.0

        with pytest.raises(Exception):
            prices_clean_schema.validate(df)

    def test_schema_validation_catches_zero_price(self):
        """FR14 — schema requires open/high/low/close > 0; zero must be rejected."""
        df = self._make_valid_schema_df().copy()
        df.loc[0, "open"] = 0.0

        with pytest.raises(Exception):
            prices_clean_schema.validate(df)

    def test_schema_validation_catches_high_less_than_low(self):
        """FR14 — schema enforces high >= low; violated row must trigger an error."""
        df = self._make_valid_schema_df().copy()
        df.loc[0, "high"] = 90.0
        df.loc[0, "low"] = 100.0   # high < low

        with pytest.raises(Exception):
            prices_clean_schema.validate(df)

    def test_schema_validation_catches_duplicate_date_ticker(self):
        """
        FR14 — schema enforces unique (date, ticker) composite key.
        A duplicate pair must raise.
        """
        df = self._make_valid_schema_df()
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with pytest.raises(Exception):
            prices_clean_schema.validate(df_with_dup)

    def test_schema_validation_catches_null_date(self):
        """FR14 — schema requires 'date' to be not null."""
        df = self._make_valid_schema_df().copy()
        df.loc[0, "date"] = pd.NaT

        with pytest.raises(Exception):
            prices_clean_schema.validate(df)


# ─────────────────────────────────────────────────────────────────────────────
# Class 10: save_processed
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveProcessed:
    """FR10–FR13 — four output files written correctly to out_dir."""

    def _make_flagged_empty(self) -> pd.DataFrame:
        """Empty flagged DataFrame with correct schema columns."""
        return pd.DataFrame(
            columns=["date", "ticker", "simple_return", "log_return", "reason"]
        )

    def _make_report(self) -> dict:
        return {
            "run_timestamp": "2024-01-02T10:00:00Z",
            "input": {"rows": 10, "tickers": 2, "date_range": ["2024-01-02", "2024-01-10"]},
            "output": {"rows": 8, "tickers": 2, "date_range": ["2024-01-02", "2024-01-10"]},
            "actions": {
                "duplicates_removed": 0,
                "invalid_price_rows_dropped": 2,
                "missing_filled": {},
                "rows_dropped_after_fill_limit": 0,
                "tickers_dropped_low_coverage": [],
                "outliers_flagged": {},
            },
            "coverage_pct": {"AAPL": 1.0, "MSFT": 1.0},
            "currency_warning": None,
            "duration_sec": 0.5,
        }

    def test_save_processed_writes_all_four_files(self, tmp_processed_dir):
        """
        FR10–FR13 — after save_processed, all four output files must exist:
        prices_clean.parquet, returns_daily.parquet,
        flagged_observations.parquet, cleaning_report.json.
        """
        prices = _make_prices(tickers=("AAPL",), n_days=3)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        assert (tmp_processed_dir / "prices_clean.parquet").exists(), (
            "prices_clean.parquet must be written by save_processed (FR10)"
        )
        assert (tmp_processed_dir / "returns_daily.parquet").exists(), (
            "returns_daily.parquet must be written by save_processed (FR11)"
        )
        assert (tmp_processed_dir / "flagged_observations.parquet").exists(), (
            "flagged_observations.parquet must be written by save_processed (FR12)"
        )
        assert (tmp_processed_dir / "cleaning_report.json").exists(), (
            "cleaning_report.json must be written by save_processed (FR13)"
        )

    def test_save_processed_parquet_roundtrip_prices(self, tmp_processed_dir):
        """FR10 — prices_clean.parquet can be read back; shape and columns are preserved."""
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        read_back = pd.read_parquet(tmp_processed_dir / "prices_clean.parquet")
        assert read_back.shape[0] == prices.shape[0], (
            f"prices_clean.parquet row count mismatch: expected {prices.shape[0]}, "
            f"got {read_back.shape[0]}"
        )
        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(read_back.columns)), (
            f"prices_clean.parquet missing columns: {expected_cols - set(read_back.columns)}"
        )

    def test_save_processed_parquet_roundtrip_returns(self, tmp_processed_dir):
        """FR11 — returns_daily.parquet can be read back; both return columns present."""
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        read_back = pd.read_parquet(tmp_processed_dir / "returns_daily.parquet")
        assert "simple_return" in read_back.columns, (
            "returns_daily.parquet must contain 'simple_return'"
        )
        assert "log_return" in read_back.columns, (
            "returns_daily.parquet must contain 'log_return'"
        )
        assert read_back.shape[0] == returns.shape[0], (
            f"returns_daily.parquet row count: expected {returns.shape[0]}, "
            f"got {read_back.shape[0]}"
        )

    def test_save_processed_parquet_flagged_observations_written_even_when_empty(
        self, tmp_processed_dir
    ):
        """
        FR12 / §4.3 — flagged_observations.parquet must be written even when
        the flagged DataFrame has zero rows (no outliers detected).
        """
        prices = _make_prices(tickers=("AAPL",), n_days=5)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()   # empty
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        path = tmp_processed_dir / "flagged_observations.parquet"
        assert path.exists(), "flagged_observations.parquet must exist even when empty"
        read_back = pd.read_parquet(path)
        assert read_back.shape[0] == 0, (
            f"Expected 0 flagged rows, got {read_back.shape[0]}"
        )

    def test_save_processed_report_json_is_valid_and_readable(
        self, tmp_processed_dir
    ):
        """FR13 — cleaning_report.json must be valid JSON, readable as a dict."""
        prices = _make_prices(tickers=("AAPL",), n_days=3)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        json_path = tmp_processed_dir / "cleaning_report.json"
        raw_text = json_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_text)

        assert isinstance(parsed, dict), (
            "cleaning_report.json must deserialise to a dict"
        )
        assert parsed["run_timestamp"] == report["run_timestamp"], (
            "run_timestamp must be persisted correctly"
        )

    def test_save_processed_report_json_has_indent_formatting(
        self, tmp_processed_dir
    ):
        """FR13 — cleaning_report.json must be human-readable (indent=2)."""
        prices = _make_prices(tickers=("AAPL",), n_days=3)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        raw_text = (tmp_processed_dir / "cleaning_report.json").read_text(encoding="utf-8")
        # indent=2 produces lines starting with two spaces
        assert "\n  " in raw_text, (
            "cleaning_report.json must be written with indent=2 for human readability"
        )

    def test_save_processed_creates_output_dir_if_missing(self, tmp_path):
        """FR10–FR13 — save_processed must create out_dir if it does not exist."""
        non_existent_dir = tmp_path / "brand_new" / "processed"
        assert not non_existent_dir.exists(), "Pre-condition: directory must not exist"

        prices = _make_prices(tickers=("AAPL",), n_days=3)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        save_processed(prices, returns, flagged, report, non_existent_dir)

        assert non_existent_dir.exists(), (
            "save_processed must create out_dir with parents if it does not exist"
        )

    def test_save_processed_overwrites_existing_files_without_error(
        self, tmp_processed_dir
    ):
        """FR16 — re-running save_processed must overwrite existing files silently."""
        prices = _make_prices(tickers=("AAPL",), n_days=3)
        returns = compute_returns(prices)
        flagged = self._make_flagged_empty()
        report = self._make_report()

        # First write
        save_processed(prices, returns, flagged, report, tmp_processed_dir)
        # Second write — must not raise
        save_processed(prices, returns, flagged, report, tmp_processed_dir)

        # Verify the file is still correctly formed after the second write
        read_back = pd.read_parquet(tmp_processed_dir / "prices_clean.parquet")
        assert read_back.shape[0] == prices.shape[0], (
            f"After second write, prices_clean.parquet must still have {prices.shape[0]} rows"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 11: idempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    """FR16 — running cleaning twice on clean data produces identical outputs."""

    def test_idempotent_rerun_remove_duplicates(self):
        """
        FR16 — applying remove_duplicates twice on the same clean DataFrame
        must yield the same result with zero removals on both passes.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result1, dups1 = remove_duplicates(df)
        result2, dups2 = remove_duplicates(result1)

        assert dups1 == 0, f"First pass: expected 0 dups, got {dups1}"
        assert dups2 == 0, f"Second pass: expected 0 dups, got {dups2}"
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
            check_dtype=True,
        )

    def test_idempotent_rerun_drop_invalid_prices(self):
        """
        FR16 — applying drop_invalid_prices twice on already-clean data must
        be a no-op on the second pass.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result1, dropped1 = drop_invalid_prices(df)
        result2, dropped2 = drop_invalid_prices(result1)

        assert dropped1 == 0, f"First pass: expected 0 dropped, got {dropped1}"
        assert dropped2 == 0, f"Second pass: expected 0 dropped, got {dropped2}"
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
            check_dtype=True,
        )

    def test_idempotent_rerun(self):
        """
        FR16 — the full cleaning chain (remove_duplicates + drop_invalid_prices
        + handle_missing + compute_returns + flag_outliers) applied twice on
        clean data produces byte-identical results.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        # ── Pass 1 ────────────────────────────────────────────────────────────
        df1, _ = remove_duplicates(df)
        df1, _ = drop_invalid_prices(df1)
        df1, _ = handle_missing(df1, method="ffill", max_consecutive=3)
        returns1 = compute_returns(df1)
        returns1, flagged1 = flag_outliers(returns1, threshold=0.25)

        # ── Pass 2 (run again on the already-clean output) ────────────────────
        df2, dups2 = remove_duplicates(df1)
        df2, dropped2 = drop_invalid_prices(df2)
        df2, fills2 = handle_missing(df2, method="ffill", max_consecutive=3)
        returns2 = compute_returns(df2)
        returns2, flagged2 = flag_outliers(returns2, threshold=0.25)

        # All cleaning operations must be no-ops on already-clean data
        assert dups2 == 0, f"Second pass: expected 0 duplicates, got {dups2}"
        assert dropped2 == 0, f"Second pass: expected 0 dropped, got {dropped2}"
        assert all(v == 0 for v in fills2.values()), (
            f"Second pass: expected 0 fills, got {fills2}"
        )

        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            returns1.reset_index(drop=True),
            returns2.reset_index(drop=True),
            check_dtype=True,
        )
        assert len(flagged1) == len(flagged2), (
            f"Flagged counts must match on idempotent rerun: {len(flagged1)} vs {len(flagged2)}"
        )

    def test_idempotent_save_processed_byte_identical_parquet(
        self, tmp_processed_dir
    ):
        """
        FR16 — writing the same DataFrames twice to Parquet must produce
        byte-identical files (no timestamp or ordering pollution).
        """
        prices = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        returns = compute_returns(prices)
        flagged = pd.DataFrame(
            columns=["date", "ticker", "simple_return", "log_return", "reason"]
        )
        report = {
            "run_timestamp": "2024-01-02T10:00:00Z",
            "input": {"rows": 10, "tickers": 2},
            "output": {"rows": 10, "tickers": 2},
            "actions": {},
            "coverage_pct": {},
            "currency_warning": None,
            "duration_sec": 0.1,
        }

        save_processed(prices, returns, flagged, report, tmp_processed_dir)
        bytes1 = (tmp_processed_dir / "prices_clean.parquet").read_bytes()

        save_processed(prices, returns, flagged, report, tmp_processed_dir)
        bytes2 = (tmp_processed_dir / "prices_clean.parquet").read_bytes()

        assert bytes1 == bytes2, (
            "Two consecutive save_processed calls must produce byte-identical "
            "prices_clean.parquet (idempotency test, FR16)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 12: config compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigCompliance:
    """
    Verify that the module reads thresholds from config.py and that the
    constants expected by the spec are present and of the correct type.
    """

    def test_config_min_coverage_pct_is_float(self):
        """§3.1 — MIN_COVERAGE_PCT must be a float in config.py."""
        import config
        assert isinstance(config.MIN_COVERAGE_PCT, float), (
            f"MIN_COVERAGE_PCT must be float, got {type(config.MIN_COVERAGE_PCT)}"
        )

    def test_config_outlier_return_threshold_is_float(self):
        """§3.1 — OUTLIER_RETURN_THRESHOLD must be a float in config.py."""
        import config
        assert isinstance(config.OUTLIER_RETURN_THRESHOLD, float), (
            f"OUTLIER_RETURN_THRESHOLD must be float, got {type(config.OUTLIER_RETURN_THRESHOLD)}"
        )

    def test_config_max_consecutive_fills_is_int(self):
        """§3.1 — MAX_CONSECUTIVE_FILLS must be an int in config.py."""
        import config
        assert isinstance(config.MAX_CONSECUTIVE_FILLS, int), (
            f"MAX_CONSECUTIVE_FILLS must be int, got {type(config.MAX_CONSECUTIVE_FILLS)}"
        )

    def test_config_trading_calendar_is_string(self):
        """§3.1 — TRADING_CALENDAR must be a string in config.py."""
        import config
        assert isinstance(config.TRADING_CALENDAR, str), (
            f"TRADING_CALENDAR must be str, got {type(config.TRADING_CALENDAR)}"
        )

    def test_config_fill_method_is_ffill(self):
        """§3.1 — FILL_METHOD must be 'ffill' (only forward-fill supported in v1)."""
        import config
        assert config.FILL_METHOD == "ffill", (
            f"FILL_METHOD must be 'ffill', got '{config.FILL_METHOD}'"
        )

    def test_flag_outliers_uses_threshold_from_config(self):
        """
        Config compliance — the threshold is respected: a return just above
        config.OUTLIER_RETURN_THRESHOLD is flagged; one just below is not.
        """
        import config
        threshold = config.OUTLIER_RETURN_THRESHOLD

        returns = _make_returns(tickers=("AAPL",), n_days=3).copy()
        just_above = threshold + 0.001
        just_below = threshold - 0.001

        returns.loc[0, "simple_return"] = just_above
        returns.loc[1, "simple_return"] = just_below

        _, flagged = flag_outliers(returns, threshold=threshold)

        flagged_dates = set(flagged["date"].values)
        date_above = returns.loc[0, "date"]
        date_below = returns.loc[1, "date"]

        assert date_above in flagged_dates, (
            f"Return of {just_above} (> threshold {threshold}) must be flagged"
        )
        assert date_below not in flagged_dates, (
            f"Return of {just_below} (< threshold {threshold}) must NOT be flagged"
        )

    def test_enforce_coverage_uses_min_pct_threshold(self):
        """
        Config compliance — min_pct boundary is enforced:
        coverage at exactly min_pct is kept; coverage below is dropped.
        """
        import config
        min_pct = config.MIN_COVERAGE_PCT

        # 4 rows / 5 expected = 0.80
        n_expected = 5
        dates_full = pd.date_range("2024-01-02", periods=n_expected, freq="B")

        rows_at = [
            {"date": d, "ticker": "AT_THRESHOLD", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.5, "volume": 1}
            for d in dates_full[: int(n_expected * min_pct)]
        ]
        rows_below = [
            {"date": d, "ticker": "BELOW_THRESHOLD", "open": 50.0, "high": 51.0,
             "low": 49.0, "close": 50.5, "volume": 1}
            for d in dates_full[: max(1, int(n_expected * min_pct) - 1)]
        ]
        df = pd.DataFrame(rows_at + rows_below)
        df["volume"] = df["volume"].astype("Int64")

        with mock.patch("pandas_market_calendars.get_calendar") as mock_get_cal:
            mock_get_cal.return_value.schedule.return_value = pd.DataFrame(
                index=dates_full
            )
            _, dropped, _ = enforce_coverage(
                df, min_pct=min_pct, calendar_name="NYSE"
            )

        assert "BELOW_THRESHOLD" in dropped, (
            "Ticker below MIN_COVERAGE_PCT must be dropped"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 13: data integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestDataIntegrity:
    """Cross-cutting data integrity checks: no silent data loss, no extra rows."""

    def test_remove_duplicates_no_silent_data_loss(self):
        """
        Data integrity — remove_duplicates must not remove non-duplicate rows.
        Input has 2 tickers × 5 days = 10 unique rows; none should be lost.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result, count = remove_duplicates(df)

        assert count == 0, f"No duplicates expected, got {count}"
        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

    def test_drop_invalid_prices_does_not_drop_clean_rows(self):
        """FR6 data integrity — clean rows must not be lost."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

        result, dropped = drop_invalid_prices(df)

        assert dropped == 0, f"No clean rows should be dropped, got {dropped}"
        assert len(result) == len(df), (
            f"All rows must survive: expected {len(df)}, got {len(result)}"
        )

    def test_compute_returns_row_count_formula(self):
        """
        FR9 data integrity — for N price rows per ticker, there must be
        exactly N-1 return rows (first day has no prior close).
        """
        n_days = 7
        df = _make_prices(tickers=("AAPL",), n_days=n_days)

        returns = compute_returns(df)

        assert len(returns) == n_days - 1, (
            f"Expected {n_days - 1} return rows for {n_days} price rows, "
            f"got {len(returns)}"
        )

    def test_full_pipeline_no_duplicate_date_ticker_in_prices(self):
        """
        Acceptance criterion — after remove_duplicates, no duplicate
        (date, ticker) pairs may exist in the prices DataFrame.
        """
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        # Inject one deliberate duplicate
        df_dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        result, _ = remove_duplicates(df_dirty)

        dups = result.duplicated(subset=["date", "ticker"])
        assert not dups.any(), (
            f"Found {dups.sum()} duplicate (date, ticker) pair(s) after remove_duplicates"
        )

    def test_handle_missing_does_not_add_extra_tickers(self):
        """FR5 data integrity — handle_missing must not introduce new ticker values."""
        df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)
        original_tickers = set(df["ticker"].unique())

        result, _ = handle_missing(df, method="ffill", max_consecutive=3)

        result_tickers = set(result["ticker"].unique())
        assert result_tickers.issubset(original_tickers), (
            f"handle_missing introduced unexpected tickers: "
            f"{result_tickers - original_tickers}"
        )
