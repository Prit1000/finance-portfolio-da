"""
tests/conftest.py — Shared fixtures for the Finance Portfolio Analysis Pipeline test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_prices_raw():
    """
    Minimal long-format OHLCV DataFrame matching Step 1's output contract.
    2 tickers, 5 trading days.
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for d in dates:
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_raw_dir(tmp_path):
    """Isolated raw data directory per test."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Isolated processed data directory per test."""
    d = tmp_path / "processed"
    d.mkdir()
    return d
