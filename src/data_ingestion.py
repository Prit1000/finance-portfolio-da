"""
src/data_ingestion.py — Step 1 of the Finance Portfolio Analysis Pipeline.

Fetches historical OHLCV price data and fundamental metadata for a
multi-asset portfolio from Yahoo Finance and persists results to data/raw/.
This is the ONLY module allowed to make network calls.
"""

from __future__ import annotations

import json
import time
from datetime import date as _date
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

# ── Constants ─────────────────────────────────────────────────────────────────
_METADATA_KEYS = (
    "shortName",
    "sector",
    "industry",
    "marketCap",
    "currency",
    "beta",
    "trailingPE",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
)


# ── Public Functions ──────────────────────────────────────────────────────────


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV for all tickers in a single yf.download() call.

    Returns a long-format DataFrame with columns:
        date, ticker, open, high, low, close, volume

    Note: volume uses pandas nullable Int64 (not int64) to handle NaN on
    trading days with no reported volume without losing other rows.
    """
    logger.info(f"Fetching prices for {len(tickers)} tickers [{start} to {end}]")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        logger.warning("yf.download() returned an empty DataFrame")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    # Determine MultiIndex ticker level (yfinance >=0.2 uses ticker as level 0)
    ticker_level: int = 0
    if isinstance(raw.columns, pd.MultiIndex):
        level0_vals = raw.columns.get_level_values(0).unique().tolist()
        if not any(t in level0_vals for t in tickers):
            ticker_level = 1  # fall back if tickers are in level 1

    # Convert wide → long
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(ticker, axis=1, level=ticker_level).copy()
            else:
                # Single ticker — raw IS the ticker frame
                df = raw.copy()

            # Normalise field names to lowercase
            df.columns = df.columns.str.lower()

            if df.empty or "close" not in df.columns or df["close"].isna().all():
                logger.warning(f"{ticker}: no price data returned -- skipping")
                continue

            # Reset index; handle both "Date" and "Datetime" index names
            df = df.reset_index()
            date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
            if date_col is None:
                logger.warning(f"{ticker}: date column not found -- skipping")
                continue
            df = df.rename(columns={date_col: "date"})

            df["ticker"] = ticker
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            df["volume"] = df["volume"].astype("Int64")
            frames.append(df)
            logger.info(f"{ticker}: {len(df)} rows fetched")
        except KeyError:
            logger.warning(f"{ticker}: not found in downloaded data -- skipping")

    if not frames:
        logger.error("No valid price data for any ticker")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    prices = pd.concat(frames, ignore_index=True)

    # Warn if multiple currencies detected (metadata concern, but detectable here
    # only if metadata is available — deferred to run_ingestion)
    logger.info(f"Total price rows fetched: {len(prices)}")
    return prices


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_single_metadata(ticker: str) -> dict:
    """Fetch .info for one ticker with exponential-backoff retry on network errors."""
    return yf.Ticker(ticker).info


def fetch_metadata(tickers: list[str]) -> dict[str, dict]:
    """
    Loop over tickers, call yf.Ticker(t).info, extract a fixed set of keys.
    Missing keys → None (do not raise).

    Returns a dict keyed by ticker symbol.
    """
    logger.info(f"Fetching metadata for {len(tickers)} tickers")
    metadata: dict[str, dict] = {}

    for ticker in tickers:
        logger.info(f"{ticker}: fetching metadata …")
        try:
            info = _fetch_single_metadata(ticker)
            if not info:
                logger.warning(f"{ticker}: .info returned empty dict — storing nulls")
                info = {}
            entry = {key: info.get(key) for key in _METADATA_KEYS}
            metadata[ticker] = entry
            logger.info(f"{ticker}: metadata OK — sector={entry.get('sector')}, currency={entry.get('currency')}")
        except Exception as exc:
            logger.error(f"{ticker}: metadata fetch failed after {config.MAX_RETRIES} retries — {exc}")
            metadata[ticker] = {key: None for key in _METADATA_KEYS}

    return metadata


def save_raw(prices: pd.DataFrame, metadata: dict, raw_dir: Path) -> None:
    """
    Persist prices → prices_raw.csv (no index column)
    Persist metadata → metadata.json (indent=2, ensure_ascii=False)
    Creates raw_dir if it does not exist.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    prices_path = raw_dir / "prices_raw.csv"
    prices.to_csv(prices_path, index=False)
    logger.info(f"Saved prices → {prices_path}  ({len(prices)} rows)")

    metadata_path = raw_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata → {metadata_path}  ({len(metadata)} tickers)")


def run_ingestion() -> dict:
    """
    Orchestrator — called by main.py as Step 1 of the pipeline.

    Returns a summary dict:
        tickers_requested, tickers_succeeded, tickers_failed,
        rows_fetched, duration_sec
    """
    t0 = time.perf_counter()

    tickers: list[str] = config.TICKERS
    start: str = config.DATE_START
    end: str = config.DATE_END
    interval: str = config.FETCH_INTERVAL
    raw_dir: Path = config.RAW_DATA_DIR

    # ── Input validation ──────────────────────────────────────────────────────
    if not tickers:
        raise ValueError("config.TICKERS is empty — nothing to fetch")

    if _date.fromisoformat(start) > _date.fromisoformat(end):
        raise ValueError(f"DATE_START ({start}) must be before DATE_END ({end})")

    logger.info(f"run_ingestion: tickers={tickers}, range=[{start}, {end}], interval={interval}")

    # ── Fetch prices (batch) ──────────────────────────────────────────────────
    prices = fetch_prices(tickers, start, end, interval)

    tickers_with_data = set(prices["ticker"].unique()) if not prices.empty else set()
    tickers_failed = [t for t in tickers if t not in tickers_with_data]

    if tickers_failed:
        logger.warning(f"Tickers with no price data: {tickers_failed}")

    # ── Fetch metadata (per-ticker with retry) ────────────────────────────────
    successful_tickers = [t for t in tickers if t in tickers_with_data]
    metadata = fetch_metadata(successful_tickers)

    # ── Currency warning ──────────────────────────────────────────────────────
    currencies = {v.get("currency") for v in metadata.values() if v.get("currency")}
    if len(currencies) > 1:
        logger.warning(f"Mixed currencies detected in portfolio: {currencies}. "
                       "Currency normalization is a downstream concern.")

    # ── Persist ───────────────────────────────────────────────────────────────
    if prices.empty:
        logger.critical("All tickers failed — not writing empty CSV")
    else:
        save_raw(prices, metadata, raw_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    duration = round(time.perf_counter() - t0, 2)
    summary = {
        "tickers_requested": len(tickers),
        "tickers_succeeded": len(successful_tickers),
        "tickers_failed": tickers_failed,
        "rows_fetched": len(prices),
        "duration_sec": duration,
    }

    logger.info(f"Ingestion complete in {duration}s — {summary}")
    return summary


# ── Smoke test (run directly: python -m src.data_ingestion) ──────────────────
# if __name__ == "__main__":
#     import sys
#     from loguru import logger as _log
#     _log.add(sys.stderr, level="DEBUG")
#     result = run_ingestion()
#     assert result["tickers_requested"] > 0
#     assert result["rows_fetched"] > 0
#     print("Smoke test PASSED")
