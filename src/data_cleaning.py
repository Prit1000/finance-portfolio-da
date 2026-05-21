"""
src/data_cleaning.py — Step 2 of the Finance Portfolio Analysis Pipeline.

Reads raw OHLCV data produced by data_ingestion.py, applies a deterministic
set of cleaning rules, and persists analysis-ready datasets to data/processed/.
This module must NOT import yfinance or make any network calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger

import config
from src.schemas import prices_clean_schema

_PRICE_COLS = ["open", "high", "low", "close"]
_SUPPORTED_FILL_METHODS = {"ffill"}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")


# ── Public Functions ──────────────────────────────────────────────────────────


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read prices_raw.csv and metadata.json. Parse `date` to datetime,
    uppercase tickers, set explicit dtypes. Raise FileNotFoundError if
    prices_raw.csv is missing; warn and return {} if metadata.json is missing.
    """
    prices_path = raw_dir / "prices_raw.csv"
    metadata_path = raw_dir / "metadata.json"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found. Run Step 1 first (data_ingestion.run_ingestion())."
        )

    df = pd.read_csv(
        prices_path,
        dtype={"ticker": str, "open": float, "high": float, "low": float, "close": float},
    )
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper().astype("string")
    if "volume" in df.columns:
        df["volume"] = pd.array(df["volume"], dtype="Int64")

    if not metadata_path.exists():
        logger.warning(f"{metadata_path} not found — continuing without metadata")
        metadata: dict = {}
    else:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    logger.info(
        f"Loaded {len(df)} rows for {df['ticker'].nunique()} tickers "
        f"[{df['date'].min().date()} to {df['date'].max().date()}]"
    )
    return df, metadata


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Assert columns, dtypes, and required fields match Step 1's output contract.
    Raise ValueError with a descriptive message on mismatch.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw schema validation failed — missing columns: {sorted(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError(f"Column 'date' must be datetime64, got {df['date'].dtype}")

    for col in _PRICE_COLS:
        if not pd.api.types.is_float_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be float64, got {df[col].dtype}")

    logger.info("Raw schema validation passed")


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate (date, ticker) rows. Keep first.
    Returns (deduped_df, num_duplicates_removed).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed} duplicate (date, ticker) rows")
    return df, removed


def reindex_to_calendar(df: pd.DataFrame, calendar_name: str) -> pd.DataFrame:
    """
    For each ticker, reindex to the full set of expected trading days within
    [df.date.min(), df.date.max()] using pandas_market_calendars.
    Missing trading days become NaN rows (to be handled by handle_missing).
    """
    start = df["date"].min()
    end = df["date"].max()

    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=start, end_date=end)

    if schedule.empty:
        raise ValueError(
            f"Calendar '{calendar_name}' returned no trading days for {start.date()} "
            f"to {end.date()}. Check date range or calendar name."
        )

    # schedule.index is a tz-naive DatetimeIndex of trading days
    trading_days = schedule.index.normalize()
    if trading_days.tz is not None:
        trading_days = trading_days.tz_convert(None)

    pieces = []
    for ticker, grp in df.groupby("ticker", sort=False):
        reindexed = (
            grp.set_index("date")
            .reindex(trading_days)
            .rename_axis("date")
            .reset_index()
        )
        reindexed["ticker"] = ticker
        pieces.append(reindexed)

    result = pd.concat(pieces, ignore_index=True)
    logger.info(
        f"Reindexed {df['ticker'].nunique()} tickers to {len(trading_days)} trading days "
        f"via '{calendar_name}' calendar"
    )
    return result


def handle_missing(
    df: pd.DataFrame,
    method: str,
    max_consecutive: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Per ticker:
      - Forward-fill close, open, high, low up to max_consecutive days.
      - Beyond max_consecutive, drop the rows.
      - Volume: do NOT fill (zero volume is meaningful — halt day).
    Returns (filled_df, fills_per_ticker).
    """
    if method not in _SUPPORTED_FILL_METHODS:
        raise ValueError(
            f"Unsupported fill method '{method}'. Supported: {_SUPPORTED_FILL_METHODS}"
        )

    fills_per_ticker: dict[str, int] = {}
    pieces = []

    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()

        was_nan = grp["close"].isna()

        if max_consecutive > 0:
            for col in _PRICE_COLS:
                if method == "ffill":
                    grp[col] = grp[col].ffill(limit=max_consecutive)
        # max_consecutive == 0 means no filling at all — fall through to dropna

        still_nan = grp["close"].isna()
        fills_per_ticker[ticker] = int((was_nan & ~still_nan).sum())

        # Drop rows where any price column is still NaN
        grp = grp.dropna(subset=_PRICE_COLS)
        pieces.append(grp)

    result = pd.concat(pieces, ignore_index=True)
    total_filled = sum(fills_per_ticker.values())
    if total_filled:
        logger.info(f"Forward-filled {total_filled} rows across tickers: {fills_per_ticker}")
    return result, fills_per_ticker


def drop_invalid_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where any of open/high/low/close <= 0.
    Also drops rows where high < low (data feed error).
    Returns (filtered_df, num_rows_dropped).
    """
    before = len(df)

    # Negative or zero prices
    nonpos_mask = (df[_PRICE_COLS] <= 0).any(axis=1)
    if nonpos_mask.any():
        logger.warning(f"Dropping {nonpos_mask.sum()} rows with zero or negative OHLC values")

    # high < low (data feed error)
    bad_hl_mask = df["high"] < df["low"]
    if bad_hl_mask.any():
        logger.warning(f"Dropping {bad_hl_mask.sum()} rows where high < low")

    # Negative volume → set to NaN (not a row drop, just sanitise)
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).fillna(False)
        if neg_vol.any():
            logger.warning(f"Setting {neg_vol.sum()} negative volume values to NaN")
            df = df.copy()
            df.loc[neg_vol, "volume"] = pd.NA

    drop_mask = nonpos_mask | bad_hl_mask
    df = df[~drop_mask].reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped


def enforce_coverage(
    df: pd.DataFrame,
    min_pct: float,
    calendar_name: str,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """
    For each ticker, compute coverage = actual_days / expected_trading_days.
    Drop tickers below min_pct.
    Returns (filtered_df, dropped_tickers, coverage_per_ticker).
    """
    start = df["date"].min()
    end = df["date"].max()
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=start, end_date=end)
    expected_days = len(schedule)

    coverage_per_ticker: dict[str, float] = {}
    dropped_tickers: list[str] = []

    for ticker, grp in df.groupby("ticker"):
        coverage = len(grp) / expected_days if expected_days > 0 else 0.0
        coverage_per_ticker[ticker] = round(coverage, 4)
        if coverage < min_pct:
            dropped_tickers.append(ticker)
            logger.warning(
                f"Dropping ticker '{ticker}': coverage {coverage:.1%} < threshold {min_pct:.1%}"
            )

    if dropped_tickers:
        df = df[~df["ticker"].isin(dropped_tickers)].reset_index(drop=True)

    logger.info(
        f"Coverage check: {len(dropped_tickers)} ticker(s) dropped, "
        f"{df['ticker'].nunique()} surviving"
    )
    return df, dropped_tickers, coverage_per_ticker


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ticker, sorted by date:
      simple_return = close.pct_change()
      log_return = np.log(close / close.shift(1))
    First row per ticker will have NaN returns — drop these.
    Returns long-format DataFrame: [date, ticker, simple_return, log_return].
    """
    pieces = []
    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        if len(grp) < 2:
            logger.warning(
                f"Ticker '{ticker}' has fewer than 2 rows — cannot compute returns; dropping"
            )
            continue
        # Zero-close rows are excluded upstream by drop_invalid_prices
        grp["simple_return"] = grp["close"].pct_change()
        grp["log_return"] = np.log(grp["close"] / grp["close"].shift(1))
        grp = grp.dropna(subset=["simple_return"])
        pieces.append(grp[["date", "ticker", "simple_return", "log_return"]])

    if not pieces:
        return pd.DataFrame(columns=["date", "ticker", "simple_return", "log_return"])

    result = pd.concat(pieces, ignore_index=True)
    logger.info(f"Computed returns: {len(result)} rows across {result['ticker'].nunique()} tickers")
    return result


def flag_outliers(
    returns: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify rows where |simple_return| > threshold.
    Returns (returns_unchanged, flagged_df).
    Flagged DataFrame schema: [date, ticker, simple_return, log_return, reason].
    Does NOT modify the returns DataFrame — flagging only.
    """
    mask = returns["simple_return"].abs() > threshold
    flagged = returns[mask].copy()
    flagged["reason"] = "abs_return_exceeds_threshold"
    flagged = flagged[["date", "ticker", "simple_return", "log_return", "reason"]]

    if not flagged.empty:
        by_ticker = flagged.groupby("ticker").size().to_dict()
        logger.info(f"Flagged {len(flagged)} outlier return(s): {by_ticker}")

    return returns, flagged.reset_index(drop=True)


def save_processed(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    flagged: pd.DataFrame,
    report: dict,
    out_dir: Path,
) -> None:
    """
    Write prices_clean.parquet, returns_daily.parquet,
    flagged_observations.parquet, and cleaning_report.json.
    Creates out_dir if missing; overwrites existing files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(prices, out_dir / "prices_clean.parquet")
    _write_parquet(returns, out_dir / "returns_daily.parquet")
    _write_parquet(flagged, out_dir / "flagged_observations.parquet")

    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info(f"Saved cleaned outputs to {out_dir}")


def run_cleaning() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_raw → validate_raw_schema → remove_duplicates → reindex_to_calendar
      → handle_missing → drop_invalid_prices → enforce_coverage
      → compute_returns → flag_outliers → schema_validate → save_processed
    Returns the cleaning_report dict (also persisted to JSON).
    """
    t0 = datetime.now(timezone.utc)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / f"cleaning_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    log_id = logger.add(log_path, rotation="1 day", level="INFO", enqueue=True)

    try:
        # ── Load ────────────────────────────────────────────────────────────
        df, metadata = load_raw(config.RAW_DATA_DIR)

        input_rows = len(df)
        input_tickers = df["ticker"].nunique()
        input_date_range = [
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        ]

        # ── Validate ─────────────────────────────────────────────────────────
        validate_raw_schema(df)

        # ── Deduplicate ───────────────────────────────────────────────────────
        df, dups_removed = remove_duplicates(df)

        # ── Reindex to trading calendar ───────────────────────────────────────
        df = reindex_to_calendar(df, config.TRADING_CALENDAR)

        # ── Handle missing prices ─────────────────────────────────────────────
        rows_after_reindex = len(df)
        df, fills_per_ticker = handle_missing(df, config.FILL_METHOD, config.MAX_CONSECUTIVE_FILLS)
        rows_dropped_after_fill_limit = rows_after_reindex - len(df)

        # ── Drop invalid prices ───────────────────────────────────────────────
        df, invalid_dropped = drop_invalid_prices(df)

        # ── Enforce coverage ──────────────────────────────────────────────────
        df, dropped_tickers, coverage_per_ticker = enforce_coverage(
            df, config.MIN_COVERAGE_PCT, config.TRADING_CALENDAR
        )

        if df.empty:
            raise RuntimeError(
                "All tickers failed coverage check — pipeline cannot continue with empty data."
            )

        # Drop tickers with fewer than 2 rows (cannot compute returns)
        tiny = [t for t, g in df.groupby("ticker") if len(g) < 2]
        if tiny:
            logger.warning(f"Dropping tickers with < 2 rows: {tiny}")
            df = df[~df["ticker"].isin(tiny)].reset_index(drop=True)

        if df.empty:
            raise RuntimeError("No tickers remain after cleaning — pipeline cannot continue.")

        # ── Currency warning ──────────────────────────────────────────────────
        currency_warning = None
        if metadata:
            surviving = set(df["ticker"].unique())
            currencies = {
                t: v.get("currency")
                for t, v in metadata.items()
                if t in surviving and v.get("currency") is not None
            }
            unique_currencies = set(currencies.values())
            if len(unique_currencies) > 1:
                currency_warning = (
                    f"Multiple currencies detected among surviving tickers: "
                    f"{sorted(unique_currencies)}"
                )
                logger.warning(currency_warning)

        # ── Ensure correct dtypes before schema validation ────────────────────
        df["ticker"] = df["ticker"].astype("string")
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype("Int64")

        # ── Schema validation ─────────────────────────────────────────────────
        try:
            prices_clean_schema.validate(df)
        except Exception as exc:
            raise RuntimeError(f"Final prices schema validation failed: {exc}") from exc

        logger.info("Cleaned prices schema validation passed")

        # ── Compute returns ───────────────────────────────────────────────────
        returns = compute_returns(df)

        # ── Flag outliers ─────────────────────────────────────────────────────
        returns, flagged = flag_outliers(returns, config.OUTLIER_RETURN_THRESHOLD)

        # ── Build report ──────────────────────────────────────────────────────
        outliers_per_ticker: dict[str, int] = {}
        if not flagged.empty:
            outliers_per_ticker = flagged.groupby("ticker").size().to_dict()

        duration = (datetime.now(timezone.utc) - t0).total_seconds()
        report: dict = {
            "run_timestamp": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {
                "rows": input_rows,
                "tickers": input_tickers,
                "date_range": input_date_range,
            },
            "output": {
                "rows": len(df),
                "tickers": df["ticker"].nunique(),
                "date_range": [
                    df["date"].min().strftime("%Y-%m-%d"),
                    df["date"].max().strftime("%Y-%m-%d"),
                ],
            },
            "actions": {
                "duplicates_removed": dups_removed,
                "invalid_price_rows_dropped": invalid_dropped,
                "missing_filled": {k: v for k, v in fills_per_ticker.items() if v > 0},
                "rows_dropped_after_fill_limit": rows_dropped_after_fill_limit,
                "tickers_dropped_low_coverage": dropped_tickers,
                "outliers_flagged": outliers_per_ticker,
            },
            "coverage_pct": coverage_per_ticker,
            "currency_warning": currency_warning,
            "duration_sec": round(duration, 2),
        }

        # ── Save ──────────────────────────────────────────────────────────────
        save_processed(df, returns, flagged, report, config.PROCESSED_DATA_DIR)

        logger.info(
            f"Cleaning complete: {len(df)} rows, {df['ticker'].nunique()} tickers, "
            f"{len(flagged)} outlier(s) flagged"
        )
        return report

    finally:
        logger.remove(log_id)
