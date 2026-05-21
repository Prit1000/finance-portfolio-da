"""
tests/unit/test_01-data-integestion.py

Pytest test suite for src/data_ingestion.py — Step 1 of the pipeline.

Tests are derived exclusively from the spec at .claude/specs/01-data-integestion.md.
No implementation logic is assumed beyond public function signatures and the
data contracts defined in the spec.

Network calls are always mocked via unittest.mock.patch — yfinance is never
invoked for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — build realistic yfinance mock return values
# ---------------------------------------------------------------------------

_TRADING_DATES = pd.date_range("2024-01-02", periods=5, freq="B")
_TICKERS = ["AAPL", "MSFT"]


def _make_multiindex_download(tickers: list[str], dates=_TRADING_DATES) -> pd.DataFrame:
    """
    Construct a wide-format MultiIndex DataFrame that mimics what
    yfinance >=0.2 returns for a multi-ticker batch download.
    Column level 0 = ticker, level 1 = field name (title-cased).
    """
    arrays = []
    for ticker in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            arrays.append((ticker, field))

    columns = pd.MultiIndex.from_tuples(arrays)
    data = {}
    for ticker in tickers:
        base = 100.0 if ticker == "AAPL" else 200.0
        for field in ["Open", "High", "Low", "Close"]:
            data[(ticker, field)] = [base + i * 0.5 for i in range(len(dates))]
        data[(ticker, "Volume")] = [1_000_000 + i * 1000 for i in range(len(dates))]

    df = pd.DataFrame(data, index=dates, columns=columns)
    df.index.name = "Date"
    return df


def _make_single_ticker_download(ticker: str = "AAPL", dates=_TRADING_DATES) -> pd.DataFrame:
    """Flat DataFrame returned by yfinance when only one ticker is requested."""
    base = 100.0
    df = pd.DataFrame(
        {
            "Open": [base + i * 0.5 for i in range(len(dates))],
            "High": [base + i * 0.5 + 1.0 for i in range(len(dates))],
            "Low": [base + i * 0.5 - 0.5 for i in range(len(dates))],
            "Close": [base + i * 0.5 + 0.25 for i in range(len(dates))],
            "Volume": [1_000_000 + i * 1000 for i in range(len(dates))],
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_ticker_info(ticker: str = "AAPL") -> dict:
    """Realistic .info dict returned by yf.Ticker(t).info."""
    return {
        "shortName": f"{ticker} Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3_400_000_000_000,
        "currency": "USD",
        "beta": 1.25,
        "trailingPE": 32.5,
        "fiftyTwoWeekHigh": 250.0,
        "fiftyTwoWeekLow": 160.0,
        "someOtherKey": "should_be_ignored",
    }


# ---------------------------------------------------------------------------
# Class 1: fetch_prices — happy path
# ---------------------------------------------------------------------------


class TestFetchPricesHappyPath:
    """FR1, FR3 — Batch fetch returns correct long-format DataFrame."""

    @patch("src.data_ingestion.yf.download")
    def test_returns_long_format_dataframe(self, mock_download):
        """fetch_prices returns a DataFrame in long format (one row per date/ticker)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        assert isinstance(result, pd.DataFrame), "fetch_prices must return a DataFrame"

    @patch("src.data_ingestion.yf.download")
    def test_output_has_required_columns(self, mock_download):
        """fetch_prices output contains exactly the columns specified in §3.2."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_date_column_dtype_is_datetime(self, mock_download):
        """The 'date' column must be datetime64[ns] per the spec schema (§3.2)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        assert pd.api.types.is_datetime64_any_dtype(result["date"]), (
            f"'date' column must be datetime64, got {result['date'].dtype}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_ticker_column_contains_all_requested_tickers(self, mock_download):
        """All requested tickers that returned data appear in the 'ticker' column."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        for ticker in _TICKERS:
            assert ticker in result["ticker"].values, (
                f"Ticker '{ticker}' missing from result"
            )

    @patch("src.data_ingestion.yf.download")
    def test_row_count_matches_dates_times_tickers(self, mock_download):
        """Row count equals number_of_tickers * number_of_trading_days (5 days x 2 tickers = 10)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        expected_rows = len(_TICKERS) * len(_TRADING_DATES)
        assert result.shape[0] == expected_rows, (
            f"Expected {expected_rows} rows, got {result.shape[0]}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_no_duplicate_date_ticker_pairs(self, mock_download):
        """There must be no duplicate (date, ticker) combinations in the output."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        duplicates = result.duplicated(subset=["date", "ticker"])
        assert not duplicates.any(), (
            f"Found {duplicates.sum()} duplicate (date, ticker) pairs"
        )

    @patch("src.data_ingestion.yf.download")
    def test_ticker_values_are_uppercase(self, mock_download):
        """All ticker values in the output must be uppercase strings."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        for ticker in result["ticker"].unique():
            assert ticker == ticker.upper(), (
                f"Ticker '{ticker}' is not uppercase"
            )

    @patch("src.data_ingestion.yf.download")
    def test_ohlc_columns_are_float(self, mock_download):
        """open, high, low, close columns must be float64 per the spec schema (§3.2)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        for col in ("open", "high", "low", "close"):
            assert pd.api.types.is_float_dtype(result[col]), (
                f"Column '{col}' must be float64, got {result[col].dtype}"
            )

    @patch("src.data_ingestion.yf.download")
    def test_volume_column_is_integer_type(self, mock_download):
        """volume column must be an integer-compatible type per the spec schema (§3.2)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        assert pd.api.types.is_integer_dtype(result["volume"]) or isinstance(
            result["volume"].dtype, pd.Int64Dtype
        ), f"'volume' column must be integer type, got {result['volume'].dtype}"

    @patch("src.data_ingestion.yf.download")
    def test_single_ticker_returns_long_format(self, mock_download):
        """A single-ticker download also produces correct long-format output."""
        mock_download.return_value = _make_single_ticker_download("AAPL")

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["AAPL"], "2024-01-02", "2024-01-08")

        assert "ticker" in result.columns, "Single-ticker result must have 'ticker' column"
        assert (result["ticker"] == "AAPL").all(), (
            "All rows in single-ticker result must have ticker='AAPL'"
        )
        assert result.shape[0] == len(_TRADING_DATES), (
            f"Expected {len(_TRADING_DATES)} rows for single ticker, got {result.shape[0]}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_yfinance_called_with_auto_adjust_true(self, mock_download):
        """FR3 — yf.download must be called with auto_adjust=True."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is True, (
            "yf.download must be called with auto_adjust=True (spec FR3)"
        )

    @patch("src.data_ingestion.yf.download")
    def test_yfinance_called_as_single_batch(self, mock_download):
        """FR1 — prices are fetched in one batch call, not one call per ticker."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        assert mock_download.call_count == 1, (
            f"yf.download should be called once (batch), got {mock_download.call_count} calls"
        )


# ---------------------------------------------------------------------------
# Class 2: fetch_prices — edge cases and error handling
# ---------------------------------------------------------------------------


class TestFetchPricesEdgeCases:
    """FR8 — bad tickers are skipped; spec §5 edge cases."""

    @patch("src.data_ingestion.yf.download")
    def test_empty_download_returns_empty_dataframe_with_schema(self, mock_download):
        """When yf.download returns empty, fetch_prices returns empty DataFrame with correct columns."""
        mock_download.return_value = pd.DataFrame()

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["FAKE123"], "2024-01-02", "2024-01-08")

        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(result.columns)), (
            f"Empty result must still have schema columns, missing: {expected_cols - set(result.columns)}"
        )
        assert result.shape[0] == 0, (
            f"Empty download must produce 0 rows, got {result.shape[0]}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_fake_ticker_excluded_from_output(self, mock_download):
        """FR8 — a bad ticker that yfinance returns no data for is excluded silently."""
        # Return a MultiIndex df with only AAPL data; FAKE123 column is all NaN
        dates = _TRADING_DATES
        real_df = _make_multiindex_download(["AAPL"], dates=dates)

        # Add a FAKE123 column group with all-NaN close
        fake_cols = pd.MultiIndex.from_tuples(
            [("FAKE123", f) for f in ["Open", "High", "Low", "Close", "Volume"]]
        )
        fake_data = pd.DataFrame(
            float("nan"),
            index=dates,
            columns=fake_cols,
        )
        combined = pd.concat([real_df, fake_data], axis=1)
        mock_download.return_value = combined

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["AAPL", "FAKE123"], "2024-01-02", "2024-01-08")

        assert "FAKE123" not in result["ticker"].values, (
            "FAKE123 (all-NaN data) must not appear in result ticker column"
        )
        assert "AAPL" in result["ticker"].values, (
            "AAPL with valid data must still appear in result"
        )

    @patch("src.data_ingestion.yf.download")
    def test_all_tickers_fail_returns_empty_dataframe(self, mock_download):
        """When all tickers return no valid data, result is an empty DataFrame with schema columns."""
        mock_download.return_value = pd.DataFrame()

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["FAKE1", "FAKE2"], "2024-01-02", "2024-01-08")

        assert result.shape[0] == 0, (
            f"All-fail scenario must produce 0 rows, got {result.shape[0]}"
        )

    @patch("src.data_ingestion.yf.download")
    def test_partial_ticker_failure_keeps_valid_tickers(self, mock_download):
        """When one ticker fails, valid tickers are still returned (no data loss)."""
        mock_download.return_value = _make_multiindex_download(["AAPL"])

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["AAPL", "FAKE123"], "2024-01-02", "2024-01-08")

        assert "AAPL" in result["ticker"].values, (
            "AAPL data must survive even when another ticker fails"
        )
        assert result.shape[0] > 0, "Result must not be empty when at least one ticker succeeds"

    @patch("src.data_ingestion.yf.download")
    def test_deterministic_output_same_input(self, mock_download):
        """Same mock input produces byte-identical output on repeated calls (determinism)."""
        mock_download.return_value = _make_multiindex_download(_TICKERS)

        from src.data_ingestion import fetch_prices

        result_a = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")
        result_b = fetch_prices(_TICKERS, "2024-01-02", "2024-01-08")

        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
            check_dtype=True,
        )


# ---------------------------------------------------------------------------
# Class 3: fetch_metadata — happy path
# ---------------------------------------------------------------------------


class TestFetchMetadataHappyPath:
    """FR2, FR7 — metadata dict has correct keys and values."""

    _EXPECTED_KEYS = {
        "shortName",
        "sector",
        "industry",
        "marketCap",
        "currency",
        "beta",
        "trailingPE",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    }

    def _make_mock_ticker(self, info: dict) -> MagicMock:
        mock_ticker = MagicMock()
        mock_ticker.info = info
        return mock_ticker

    @patch("src.data_ingestion.yf.Ticker")
    def test_returns_dict_keyed_by_ticker(self, mock_ticker_cls):
        """fetch_metadata returns a dict with one entry per requested ticker."""
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(_make_ticker_info(t))

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL", "MSFT"])

        assert isinstance(result, dict), "fetch_metadata must return a dict"
        assert "AAPL" in result, "AAPL must be a key in the metadata dict"
        assert "MSFT" in result, "MSFT must be a key in the metadata dict"
        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"

    @patch("src.data_ingestion.yf.Ticker")
    def test_each_entry_has_all_required_keys(self, mock_ticker_cls):
        """Each ticker entry in metadata must contain all 9 spec-defined keys (§3.2)."""
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(_make_ticker_info(t))

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert self._EXPECTED_KEYS == set(result["AAPL"].keys()), (
            f"Expected keys {self._EXPECTED_KEYS}, got {set(result['AAPL'].keys())}"
        )

    @patch("src.data_ingestion.yf.Ticker")
    def test_no_extra_keys_beyond_spec(self, mock_ticker_cls):
        """Extra keys returned by yfinance (e.g. 'someOtherKey') are not stored."""
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(_make_ticker_info(t))

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        unexpected = set(result["AAPL"].keys()) - self._EXPECTED_KEYS
        assert not unexpected, (
            f"Unexpected extra keys in metadata: {unexpected}"
        )

    @patch("src.data_ingestion.yf.Ticker")
    def test_values_match_yfinance_info(self, mock_ticker_cls):
        """Values in the metadata entry must come directly from yfinance .info."""
        info = _make_ticker_info("AAPL")
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(info)

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert result["AAPL"]["sector"] == "Technology", (
            f"sector mismatch: expected 'Technology', got {result['AAPL']['sector']}"
        )
        assert result["AAPL"]["currency"] == "USD", (
            f"currency mismatch: expected 'USD', got {result['AAPL']['currency']}"
        )
        assert pytest.approx(result["AAPL"]["beta"], rel=1e-6) == 1.25, (
            "beta value must match yfinance .info"
        )

    @patch("src.data_ingestion.yf.Ticker")
    def test_single_ticker_metadata(self, mock_ticker_cls):
        """fetch_metadata works correctly for a single-item ticker list."""
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(_make_ticker_info(t))

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert len(result) == 1, f"Expected 1 entry, got {len(result)}"
        assert "AAPL" in result, "AAPL must be in the result"


# ---------------------------------------------------------------------------
# Class 4: fetch_metadata — missing / empty info edge cases
# ---------------------------------------------------------------------------


class TestFetchMetadataEdgeCases:
    """Spec §5 — graceful handling of empty .info dicts and missing keys."""

    _EXPECTED_KEYS = {
        "shortName", "sector", "industry", "marketCap", "currency",
        "beta", "trailingPE", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    }

    def _make_mock_ticker(self, info: dict) -> MagicMock:
        mock_ticker = MagicMock()
        mock_ticker.info = info
        return mock_ticker

    @patch("src.data_ingestion.yf.Ticker")
    def test_empty_info_dict_stores_all_none(self, mock_ticker_cls):
        """
        Spec §5: if yf.Ticker(t).info returns an empty dict,
        all 9 metadata fields must be stored as None (do not raise).
        """
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker({})

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert "AAPL" in result, "Ticker with empty info must still appear in result"
        for key in self._EXPECTED_KEYS:
            assert result["AAPL"][key] is None, (
                f"Key '{key}' must be None when .info is empty, got {result['AAPL'][key]}"
            )

    @patch("src.data_ingestion.yf.Ticker")
    def test_partial_info_missing_keys_become_none(self, mock_ticker_cls):
        """Missing keys in .info are stored as None, not omitted or raised."""
        partial_info = {"sector": "Technology", "currency": "USD"}
        mock_ticker_cls.side_effect = lambda t: self._make_mock_ticker(partial_info)

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert result["AAPL"]["sector"] == "Technology", "Present key must keep its value"
        assert result["AAPL"]["beta"] is None, (
            "Missing key 'beta' must be None, not raise"
        )
        assert result["AAPL"]["marketCap"] is None, (
            "Missing key 'marketCap' must be None"
        )

    @patch("src.data_ingestion._fetch_single_metadata")
    def test_network_failure_after_retries_stores_all_none(self, mock_retry_fn):
        """
        Spec §5 / FR8: if .info fetch raises after all retries,
        metadata entry must be all-None — do not crash.
        """
        mock_retry_fn.side_effect = Exception("Connection timeout")

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL"])

        assert "AAPL" in result, "Failed ticker must still appear in result"
        for key in (
            "shortName", "sector", "industry", "marketCap", "currency",
            "beta", "trailingPE", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ):
            assert result["AAPL"][key] is None, (
                f"Key '{key}' must be None after network failure, got {result['AAPL'][key]}"
            )

    @patch("src.data_ingestion._fetch_single_metadata")
    def test_failure_on_one_ticker_does_not_stop_others(self, mock_retry_fn):
        """FR8 — one ticker's metadata failure must not prevent other tickers from succeeding."""
        def side_effect(ticker):
            if ticker == "FAKE123":
                raise Exception("Ticker not found")
            return _make_ticker_info(ticker)

        mock_retry_fn.side_effect = side_effect

        from src.data_ingestion import fetch_metadata

        result = fetch_metadata(["AAPL", "FAKE123"])

        assert "AAPL" in result, "AAPL must succeed even when FAKE123 fails"
        assert "FAKE123" in result, "Failed ticker must still appear in result (with None values)"
        assert result["AAPL"]["sector"] is not None, (
            "AAPL sector must have a value — should not be contaminated by FAKE123 failure"
        )


# ---------------------------------------------------------------------------
# Class 5: save_raw
# ---------------------------------------------------------------------------


class TestSaveRaw:
    """FR6, FR7, FR10 — file persistence, format, overwrite behaviour."""

    def _make_prices(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        rows = []
        for ticker in ["AAPL", "MSFT"]:
            for d in dates:
                rows.append({
                    "date": d, "ticker": ticker,
                    "open": 100.0, "high": 102.0,
                    "low": 99.0, "close": 101.0,
                    "volume": pd.array([1_000_000], dtype="Int64")[0],
                })
        return pd.DataFrame(rows)

    def _make_metadata(self) -> dict:
        return {
            "AAPL": {
                "shortName": "Apple Inc.", "sector": "Technology",
                "industry": "Consumer Electronics", "marketCap": 3_400_000_000_000,
                "currency": "USD", "beta": 1.25, "trailingPE": 32.5,
                "fiftyTwoWeekHigh": 250.0, "fiftyTwoWeekLow": 160.0,
            }
        }

    def test_prices_csv_created(self, tmp_raw_dir):
        """FR6 — prices_raw.csv must exist after save_raw completes."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        assert (tmp_raw_dir / "prices_raw.csv").exists(), (
            "prices_raw.csv must be created by save_raw"
        )

    def test_metadata_json_created(self, tmp_raw_dir):
        """FR7 — metadata.json must exist after save_raw completes."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        assert (tmp_raw_dir / "metadata.json").exists(), (
            "metadata.json must be created by save_raw"
        )

    def test_prices_csv_has_no_index_column(self, tmp_raw_dir):
        """FR6 — prices_raw.csv must not have an unnamed index column."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        df = pd.read_csv(tmp_raw_dir / "prices_raw.csv")
        assert "Unnamed: 0" not in df.columns, (
            "prices_raw.csv must be written with index=False (no unnamed index column)"
        )

    def test_prices_csv_roundtrip_preserves_schema(self, tmp_raw_dir):
        """Reading prices_raw.csv back produces a DataFrame with the same columns."""
        from src.data_ingestion import save_raw

        prices = self._make_prices()
        save_raw(prices, self._make_metadata(), tmp_raw_dir)

        read_back = pd.read_csv(tmp_raw_dir / "prices_raw.csv")
        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(read_back.columns)), (
            f"Read-back CSV missing columns: {expected_cols - set(read_back.columns)}"
        )

    def test_prices_csv_row_count_preserved(self, tmp_raw_dir):
        """Row count in prices_raw.csv must equal the number of rows passed to save_raw."""
        from src.data_ingestion import save_raw

        prices = self._make_prices()
        save_raw(prices, self._make_metadata(), tmp_raw_dir)

        read_back = pd.read_csv(tmp_raw_dir / "prices_raw.csv")
        assert read_back.shape[0] == prices.shape[0], (
            f"Expected {prices.shape[0]} rows in CSV, got {read_back.shape[0]}"
        )

    def test_metadata_json_is_valid_json(self, tmp_raw_dir):
        """metadata.json must be valid, parseable JSON."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        with open(tmp_raw_dir / "metadata.json", "r", encoding="utf-8") as fh:
            parsed = json.load(fh)

        assert isinstance(parsed, dict), "metadata.json must deserialise to a dict"

    def test_metadata_json_keyed_by_ticker(self, tmp_raw_dir):
        """metadata.json top-level keys must be ticker symbols."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        with open(tmp_raw_dir / "metadata.json", "r", encoding="utf-8") as fh:
            parsed = json.load(fh)

        assert "AAPL" in parsed, "AAPL must be a top-level key in metadata.json"

    def test_metadata_json_indent_formatting(self, tmp_raw_dir):
        """FR7 — metadata.json must be human-readable (indent=2)."""
        from src.data_ingestion import save_raw

        save_raw(self._make_prices(), self._make_metadata(), tmp_raw_dir)

        raw_text = (tmp_raw_dir / "metadata.json").read_text(encoding="utf-8")
        # indent=2 produces lines starting with "  "
        assert "\n  " in raw_text, (
            "metadata.json must be written with indent=2 (human-readable)"
        )

    def test_creates_raw_dir_if_not_exists(self, tmp_path):
        """FR6/FR7 spec §5 — save_raw must create raw_dir if it does not exist."""
        from src.data_ingestion import save_raw

        nonexistent_dir = tmp_path / "brand_new_dir" / "raw"
        assert not nonexistent_dir.exists(), "Pre-condition: directory must not exist"

        save_raw(self._make_prices(), self._make_metadata(), nonexistent_dir)

        assert nonexistent_dir.exists(), (
            "save_raw must create the raw directory if it does not exist"
        )

    def test_overwrite_existing_csv_without_error(self, tmp_raw_dir):
        """FR10 — re-running save_raw overwrites prices_raw.csv without raising."""
        from src.data_ingestion import save_raw

        prices = self._make_prices()
        metadata = self._make_metadata()

        # First write
        save_raw(prices, metadata, tmp_raw_dir)
        # Second write — must not raise
        save_raw(prices, metadata, tmp_raw_dir)

        read_back = pd.read_csv(tmp_raw_dir / "prices_raw.csv")
        assert read_back.shape[0] == prices.shape[0], (
            "Overwritten CSV must still have the correct row count"
        )

    def test_idempotent_two_writes_produce_identical_files(self, tmp_raw_dir):
        """Idempotency — writing twice must produce byte-identical CSV output."""
        from src.data_ingestion import save_raw

        prices = self._make_prices()
        metadata = self._make_metadata()

        save_raw(prices, metadata, tmp_raw_dir)
        content_first = (tmp_raw_dir / "prices_raw.csv").read_bytes()

        save_raw(prices, metadata, tmp_raw_dir)
        content_second = (tmp_raw_dir / "prices_raw.csv").read_bytes()

        assert content_first == content_second, (
            "Two consecutive save_raw calls must produce byte-identical prices_raw.csv"
        )


# ---------------------------------------------------------------------------
# Class 6: run_ingestion — orchestration and summary dict
# ---------------------------------------------------------------------------


class TestRunIngestion:
    """FR9 — summary dict shape; FR8 — bad tickers don't crash; spec §5 edge cases."""

    def _patch_all(
        self,
        *,
        tickers=None,
        start="2024-01-02",
        end="2024-01-08",
        download_return=None,
        ticker_info=None,
    ):
        """Return a context-manager stack that patches config + yfinance together."""
        if tickers is None:
            tickers = _TICKERS
        if download_return is None:
            download_return = _make_multiindex_download(tickers)
        if ticker_info is None:
            ticker_info = {t: _make_ticker_info(t) for t in tickers}

        def make_ticker(t):
            m = MagicMock()
            m.info = ticker_info.get(t, {})
            return m

        patches = [
            patch("config.TICKERS", tickers),
            patch("config.DATE_START", start),
            patch("config.DATE_END", end),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", Path("data/raw_test_REPLACE")),  # overridden per test
            patch("src.data_ingestion.yf.download", return_value=download_return),
            patch("src.data_ingestion.yf.Ticker", side_effect=make_ticker),
        ]
        return patches

    def test_returns_dict_with_required_keys(self, tmp_raw_dir):
        """FR9 — run_ingestion must return a dict with the 5 required keys."""
        required_keys = {
            "tickers_requested",
            "tickers_succeeded",
            "tickers_failed",
            "rows_fetched",
            "duration_sec",
        }

        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        assert required_keys == set(summary.keys()), (
            f"Summary dict keys mismatch. Expected {required_keys}, got {set(summary.keys())}"
        )

    def test_tickers_requested_matches_config(self, tmp_raw_dir):
        """FR9 — tickers_requested must equal len(config.TICKERS)."""
        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        assert summary["tickers_requested"] == len(_TICKERS), (
            f"tickers_requested must be {len(_TICKERS)}, got {summary['tickers_requested']}"
        )

    def test_tickers_failed_is_list(self, tmp_raw_dir):
        """FR9 — tickers_failed must be a list (iterable of failed ticker symbols)."""
        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        assert isinstance(summary["tickers_failed"], list), (
            f"tickers_failed must be a list, got {type(summary['tickers_failed'])}"
        )

    def test_rows_fetched_equals_csv_row_count(self, tmp_raw_dir):
        """FR9 — rows_fetched in summary must match the actual rows in prices_raw.csv."""
        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        csv_path = tmp_raw_dir / "prices_raw.csv"
        assert csv_path.exists(), "prices_raw.csv must be written by run_ingestion"
        written_df = pd.read_csv(csv_path)
        assert summary["rows_fetched"] == written_df.shape[0], (
            f"rows_fetched ({summary['rows_fetched']}) must equal CSV rows ({written_df.shape[0]})"
        )

    def test_duration_sec_is_non_negative_float(self, tmp_raw_dir):
        """FR9 — duration_sec must be a non-negative number."""
        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        assert isinstance(summary["duration_sec"], (int, float)), (
            "duration_sec must be numeric"
        )
        assert summary["duration_sec"] >= 0, (
            f"duration_sec must be non-negative, got {summary['duration_sec']}"
        )

    def test_fake_ticker_in_config_does_not_crash(self, tmp_raw_dir):
        """
        Spec §6 acceptance criterion — introducing FAKE123 into TICKERS must not
        crash the pipeline; FAKE123 must appear in tickers_failed.
        """
        tickers_with_fake = ["AAPL", "FAKE123"]
        # Only AAPL returns real data
        download_result = _make_multiindex_download(["AAPL"])

        with (
            patch("config.TICKERS", tickers_with_fake),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=download_result),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()  # must not raise

        assert "FAKE123" in summary["tickers_failed"], (
            "FAKE123 must appear in tickers_failed"
        )
        assert summary["tickers_succeeded"] >= 1, (
            "AAPL must be counted in tickers_succeeded"
        )

    def test_all_tickers_fail_no_csv_written(self, tmp_raw_dir):
        """Spec §5 — when ALL tickers fail, prices_raw.csv must NOT be written."""
        with (
            patch("config.TICKERS", ["FAKE1", "FAKE2"]),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=pd.DataFrame()),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info={})),
        ):
            from src.data_ingestion import run_ingestion

            summary = run_ingestion()

        assert not (tmp_raw_dir / "prices_raw.csv").exists(), (
            "prices_raw.csv must NOT be written when all tickers fail"
        )
        assert summary["tickers_succeeded"] == 0, (
            f"tickers_succeeded must be 0 when all tickers fail, got {summary['tickers_succeeded']}"
        )

    def test_idempotent_rerun_produces_identical_csv(self, tmp_raw_dir):
        """
        Spec §6 / FR10 — re-running ingestion must overwrite and produce
        identical output (idempotency).
        """
        common_kwargs = dict(
            tickers=_TICKERS,
            start="2024-01-02",
            end="2024-01-08",
            raw_dir=tmp_raw_dir,
        )

        ctx_patches = (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        )

        from src.data_ingestion import run_ingestion

        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            run_ingestion()
            bytes_first = (tmp_raw_dir / "prices_raw.csv").read_bytes()

        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download", return_value=_make_multiindex_download(_TICKERS)),
            patch("src.data_ingestion.yf.Ticker", side_effect=lambda t: MagicMock(info=_make_ticker_info(t))),
        ):
            run_ingestion()
            bytes_second = (tmp_raw_dir / "prices_raw.csv").read_bytes()

        assert bytes_first == bytes_second, (
            "Two consecutive run_ingestion calls must produce byte-identical prices_raw.csv"
        )


# ---------------------------------------------------------------------------
# Class 7: Input validation (ValueError conditions)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Spec §5 — early ValueError for invalid config combinations."""

    def test_empty_tickers_list_raises_value_error(self, tmp_raw_dir):
        """Spec §5 — empty TICKERS must raise ValueError before any network call."""
        with (
            patch("config.TICKERS", []),
            patch("config.DATE_START", "2024-01-02"),
            patch("config.DATE_END", "2024-01-08"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download") as mock_dl,
        ):
            from src.data_ingestion import run_ingestion

            with pytest.raises(ValueError, match="(?i)empty|tickers|nothing"):
                run_ingestion()

            mock_dl.assert_not_called()

    def test_start_after_end_raises_value_error(self, tmp_raw_dir):
        """Spec §5 — DATE_START > DATE_END must raise ValueError before fetching."""
        with (
            patch("config.TICKERS", _TICKERS),
            patch("config.DATE_START", "2024-12-31"),
            patch("config.DATE_END", "2024-01-01"),
            patch("config.FETCH_INTERVAL", "1d"),
            patch("config.RAW_DATA_DIR", tmp_raw_dir),
            patch("src.data_ingestion.yf.download") as mock_dl,
        ):
            from src.data_ingestion import run_ingestion

            with pytest.raises(ValueError, match="(?i)start|date|end|before"):
                run_ingestion()

            mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Class 8: network isolation guarantee
# ---------------------------------------------------------------------------


class TestNetworkIsolation:
    """Guarantee that yfinance is the only network dependency in this module."""

    def test_yfinance_import_absent_from_other_modules(self):
        """
        Architectural constraint — no module other than data_ingestion.py
        may import yfinance.  We verify by inspecting the import of the
        modules that exist at this stage (src/data_ingestion only in Step 1).
        """
        import src.data_ingestion as mod
        import inspect

        source = inspect.getsource(mod)
        # yfinance import is expected in data_ingestion
        assert "import yfinance" in source or "import yf" in source or "yfinance" in source, (
            "data_ingestion.py itself must import yfinance"
        )

    def test_fetch_prices_does_not_call_yfinance_when_mocked(self):
        """
        Ensure the mock is effective: fetch_prices must not bypass the mock
        and hit the real yfinance.download.
        """
        with patch("src.data_ingestion.yf.download", return_value=pd.DataFrame()) as mock_dl:
            from src.data_ingestion import fetch_prices

            fetch_prices(["AAPL"], "2024-01-02", "2024-01-08")

            assert mock_dl.called, (
                "yf.download mock was not invoked — the patch may not be applied correctly"
            )


# ---------------------------------------------------------------------------
# Class 9: parametrised tests — multiple ticker counts and date ranges
# ---------------------------------------------------------------------------


class TestParametrised:
    """Data-driven coverage of different ticker counts and date ranges."""

    @pytest.mark.parametrize("tickers", [
        ["AAPL"],
        ["AAPL", "MSFT"],
        ["AAPL", "MSFT", "GOOGL"],
    ])
    @patch("src.data_ingestion.yf.download")
    def test_correct_ticker_count_in_output(self, mock_download, tickers):
        """fetch_prices returns exactly the tickers that have valid data (parametrised)."""
        mock_download.return_value = _make_multiindex_download(tickers)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(tickers, "2024-01-02", "2024-01-08")

        returned_tickers = set(result["ticker"].unique())
        expected_tickers = set(tickers)
        assert returned_tickers == expected_tickers, (
            f"Expected tickers {expected_tickers}, got {returned_tickers}"
        )

    @pytest.mark.parametrize("n_days", [1, 5, 10])
    @patch("src.data_ingestion.yf.download")
    def test_row_count_scales_with_trading_days(self, mock_download, n_days):
        """Row count scales linearly with the number of trading days returned."""
        dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
        mock_download.return_value = _make_multiindex_download(["AAPL"], dates=dates)

        from src.data_ingestion import fetch_prices

        result = fetch_prices(["AAPL"], "2024-01-02", "2024-06-30")

        assert result.shape[0] == n_days, (
            f"Expected {n_days} rows for {n_days} trading days, got {result.shape[0]}"
        )

    @pytest.mark.parametrize("ticker,expected_sector", [
        ("AAPL", "Technology"),
        ("MSFT", "Technology"),
    ])
    def test_metadata_sector_by_ticker(self, ticker, expected_sector):
        """fetch_metadata correctly maps sector per ticker (parametrised)."""
        info = _make_ticker_info(ticker)

        with patch("src.data_ingestion.yf.Ticker") as mock_cls:
            mock_cls.return_value = MagicMock(info=info)

            from src.data_ingestion import fetch_metadata

            result = fetch_metadata([ticker])

        assert result[ticker]["sector"] == expected_sector, (
            f"Expected sector '{expected_sector}' for {ticker}, got {result[ticker]['sector']}"
        )
