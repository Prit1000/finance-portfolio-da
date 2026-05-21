"""
tests/unit/test_data_cleaning.py — Unit tests for src/data_cleaning.py

Uses small synthetic DataFrames (no real data from data/raw/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    compute_returns,
    drop_invalid_prices,
    enforce_coverage,
    flag_outliers,
    handle_missing,
    remove_duplicates,
)
from src.schemas import prices_clean_schema


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_prices(tickers=("AAPL", "MSFT"), n_days=5, start="2024-01-02") -> pd.DataFrame:
    """Minimal long-format OHLCV DataFrame for a set of tickers."""
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for d in dates:
            rows.append(
                {"date": d, "ticker": t, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1_000_000}
            )
    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype("Int64")
    return df


def _make_returns(tickers=("AAPL", "MSFT"), n_days=5, start="2024-01-02") -> pd.DataFrame:
    """Minimal returns DataFrame for flag_outliers tests."""
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "simple_return": 0.01 * (i + 1),
                    "log_return": np.log(1 + 0.01 * (i + 1)),
                }
            )
    return pd.DataFrame(rows)


# ── Test 1: remove_duplicates keeps first occurrence ─────────────────────────


def test_remove_duplicates_keeps_first():
    df = _make_prices(tickers=("AAPL",), n_days=3)
    # Inject a duplicate of the first row with different close
    dup = df.iloc[[0]].copy()
    dup["close"] = 999.0
    df = pd.concat([df, dup], ignore_index=True)

    result, count = remove_duplicates(df)

    assert count == 1
    assert len(result) == 3
    # First occurrence (close=101) kept, not the duplicate (close=999)
    assert result.loc[result["date"] == df.iloc[0]["date"], "close"].iloc[0] == 101.0


# ── Test 2: drop_invalid_prices removes zero and negative rows ───────────────


def test_drop_invalid_prices_removes_zero_and_negative():
    df = _make_prices(tickers=("AAPL",), n_days=5)
    df = df.copy()
    df.loc[0, "close"] = 0.0   # zero
    df.loc[1, "open"] = -5.0   # negative

    result, dropped = drop_invalid_prices(df)

    assert dropped == 2
    assert len(result) == 3
    assert (result[["open", "high", "low", "close"]] > 0).all().all()


# ── Test 3: handle_missing respects max_consecutive_fills ────────────────────


def test_handle_missing_respects_max_consecutive_fills():
    """A gap of 4 consecutive NaN closes with max_consecutive=3: first 3 filled, 4th dropped."""
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    rows = []
    for i, d in enumerate(dates):
        close = 100.0 if i == 0 else (None if 1 <= i <= 4 else 101.0)
        rows.append({"date": d, "ticker": "AAPL", "open": 100.0, "high": 102.0, "low": 99.0, "close": close, "volume": pd.NA})
    df = pd.DataFrame(rows)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype("Int64")

    result, fills = handle_missing(df, method="ffill", max_consecutive=3)

    # 4-day gap: 3 filled + 1 dropped → 6 rows remain (original 7 - 1 dropped)
    assert len(result) == 6
    assert fills["AAPL"] == 3
    assert result["close"].isna().sum() == 0


# ── Test 4: handle_missing does NOT fill volume ───────────────────────────────


def test_handle_missing_does_not_fill_volume():
    df = _make_prices(tickers=("AAPL",), n_days=5)
    df = df.copy()
    df.loc[2, "volume"] = pd.NA  # intentional missing volume

    result, _ = handle_missing(df, method="ffill", max_consecutive=3)

    assert pd.isna(result.loc[result.index[2], "volume"])


# ── Test 5: enforce_coverage drops sparse ticker ─────────────────────────────


def test_enforce_coverage_drops_sparse_ticker():
    """SPARSE ticker has only 1 of 5 expected days → coverage=0.20 < 0.80 → dropped."""
    # Build a calendar-like DataFrame manually; mock the market calendar
    dates_aapl = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = [{"date": d, "ticker": "AAPL", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1} for d in dates_aapl]
    # SPARSE only has 1 row for the same 5-day period
    rows.append({"date": dates_aapl[0], "ticker": "SPARSE", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 1})
    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype("Int64")

    # Monkeypatch pandas_market_calendars to return 5 schedule rows
    import unittest.mock as mock

    mock_schedule = pd.DataFrame(index=dates_aapl)

    with mock.patch("pandas_market_calendars.get_calendar") as mock_cal:
        mock_cal_instance = mock.MagicMock()
        mock_cal_instance.schedule.return_value = mock_schedule
        mock_cal.return_value = mock_cal_instance

        result, dropped, coverage = enforce_coverage(df, min_pct=0.80, calendar_name="NYSE")

    assert "SPARSE" in dropped
    assert "AAPL" not in dropped
    assert coverage["SPARSE"] < 0.80
    assert "SPARSE" not in result["ticker"].values


# ── Test 6: compute_returns first row per ticker is excluded ─────────────────


def test_compute_returns_first_row_dropped():
    df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

    returns = compute_returns(df)

    # Each ticker has 5 rows of prices → 4 return rows (first day dropped per ticker)
    assert len(returns) == 8  # 4 per ticker × 2 tickers
    for ticker, grp in returns.groupby("ticker"):
        assert grp["simple_return"].notna().all()
        assert grp["log_return"].notna().all()


# ── Test 7: flag_outliers does NOT modify returns ────────────────────────────


def test_flag_outliers_does_not_modify_returns():
    returns = _make_returns(tickers=("AAPL",), n_days=3)
    # Inject one large return
    returns = returns.copy()
    returns.loc[0, "simple_return"] = 0.40  # > 0.25 threshold

    original_hash = pd.util.hash_pandas_object(returns).sum()

    unchanged, flagged = flag_outliers(returns, threshold=0.25)

    assert pd.util.hash_pandas_object(unchanged).sum() == original_hash
    assert len(flagged) == 1
    assert flagged.iloc[0]["reason"] == "abs_return_exceeds_threshold"
    assert "simple_return" in flagged.columns


# ── Test 8: schema validation catches negative price ─────────────────────────


def test_schema_validation_catches_negative_price():
    df = _make_prices(tickers=("AAPL",), n_days=3)
    df["volume"] = df["volume"].astype("Int64")
    df = df.copy()
    df.loc[0, "close"] = -1.0  # invalid

    with pytest.raises(Exception):
        prices_clean_schema.validate(df)


# ── Test 9: idempotent re-run on already-clean data ──────────────────────────


def test_idempotent_rerun():
    """Applying remove_duplicates and drop_invalid_prices twice yields the same result."""
    df = _make_prices(tickers=("AAPL", "MSFT"), n_days=5)

    result1, dups1 = remove_duplicates(df)
    result1, dropped1 = drop_invalid_prices(result1)

    result2, dups2 = remove_duplicates(result1)
    result2, dropped2 = drop_invalid_prices(result2)

    assert dups1 == 0
    assert dups2 == 0
    assert dropped1 == 0
    assert dropped2 == 0
    pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))
