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


# ── Monte Carlo fixtures (Step 6) ─────────────────────────────────────────────

@pytest.fixture
def mc_prices_parquet(tmp_processed_dir):
    """252 trading days, 2 tickers (AAPL, MSFT) written to tmp_processed_dir."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=252)
    rows = []
    for ticker in ["AAPL", "MSFT"]:
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
    path = tmp_processed_dir / "prices_clean.parquet"
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path


@pytest.fixture
def mc_returns_parquet(tmp_processed_dir, mc_prices_parquet):
    """Compute returns from the mc_prices_parquet fixture."""
    prices = pd.read_parquet(mc_prices_parquet)
    rows = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        close = grp["close"].values
        for i in range(1, len(close)):
            sr = (close[i] / close[i - 1]) - 1.0
            lr = float(np.log(close[i] / close[i - 1]))
            rows.append({
                "date": grp["date"].iloc[i],
                "ticker": ticker,
                "simple_return": sr,
                "log_return": lr,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    path = tmp_processed_dir / "returns_daily.parquet"
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path


@pytest.fixture
def mc_scenarios_csv(tmp_path):
    """Minimal mc_scenarios.csv with one GBM scenario."""
    rows = [{
        "scenario_name": "test_gbm",
        "method": "gbm",
        "horizon_days": 10,
        "n_simulations": 500,
        "block_size": 5,
        "drift_method": "historical",
        "tickers": "all",
        "simulate_portfolio": True,
        "notes": "test scenario",
    }]
    path = tmp_path / "mc_scenarios.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
