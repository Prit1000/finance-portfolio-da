"""
src/metrics.py — Portfolio Metrics Module (Step 4/7).

Reads cleaned prices and returns; computes per-ticker, portfolio-level, and
rolling metrics. No network calls; reads data/processed/ only.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

import config
from pandera.errors import SchemaError as _PanderaSchemaError
from src.schemas import (
    drawdown_schema,
    metrics_per_ticker_schema,
    prices_clean_schema,
    returns_schema,
    rolling_metrics_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable values to JSON-safe types."""
    if obj is None or obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _dd_stats_from_array(r: np.ndarray) -> tuple[float, int, bool]:
    """
    Compute (max_drawdown, max_drawdown_duration_days, recovered) from a
    sorted simple-return array by reconstructing a normalized price series.

    Prepends 1.0 so that an immediate negative first return is captured.
    Duration counts trading-day steps from peak to recovery (or end of data).

    NOTE: This reconstructs a normalized series starting at 1.0 from the first
    return. Results may differ slightly from compute_drawdown_series() if the
    returns series starts one day later than the prices series (e.g. the first
    row was dropped during cleaning). compute_drawdown_series() uses actual close
    prices and is the authoritative source for the drawdown time series; this
    function exists only to compute scalar summary stats efficiently.
    """
    prices = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    running_peak = np.maximum.accumulate(prices)
    dd = prices / running_peak - 1.0

    max_dd = float(np.min(dd))

    max_dur = 0
    in_dd = False
    start_i = 0
    recovered = True
    for i, d in enumerate(dd):
        if d < 0.0 and not in_dd:
            in_dd = True
            start_i = i
        elif d >= 0.0 and in_dd:
            max_dur = max(max_dur, i - start_i)
            in_dd = False
    if in_dd:
        max_dur = max(max_dur, len(dd) - 1 - start_i)
        recovered = False

    return max_dd, max_dur, recovered


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read prices_clean.parquet and returns_daily.parquet from processed_dir.
    Validate against pandera schemas before returning.
    Raises FileNotFoundError (with hint) if either file is missing.
    Returns (prices, returns).
    """
    prices_path = processed_dir / "prices_clean.parquet"
    returns_path = processed_dir / "returns_daily.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found. Run Step 2 (data_cleaning) first."
        )
    if not returns_path.exists():
        raise FileNotFoundError(
            f"{returns_path} not found. Run Step 2 (data_cleaning) first."
        )

    logger.info(f"Loading {prices_path}")
    prices = pd.read_parquet(prices_path)
    logger.info(f"Loading {returns_path}")
    returns = pd.read_parquet(returns_path)

    try:
        prices_clean_schema.validate(prices)
    except _PanderaSchemaError as exc:
        raise ValueError(
            f"prices_clean.parquet failed schema validation — rerun Step 2. Detail: {exc}"
        ) from exc
    try:
        returns_schema.validate(returns)
    except _PanderaSchemaError as exc:
        raise ValueError(
            f"returns_daily.parquet failed schema validation — rerun Step 2. Detail: {exc}"
        ) from exc

    return prices, returns


def compute_return_metrics(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - total_return: prod(1 + r) - 1
      - cagr: (1 + total_return)^(trading_days_per_year / n_days) - 1
      - mean_daily_return, mean_monthly_return (~21 td/mo), mean_annual_return
      - best_day_return, worst_day_return
      - pct_positive_days

    Returns long-format DataFrame: [ticker, metric_name, value, category].
    """
    returns = returns.copy()
    # prices is accepted for API stability (spec §3.2) but not used in v1;
    # reserved for dollar-weighted return calculations in a future revision.
    _ = prices
    rows: list[dict] = []

    for ticker, grp in returns.groupby("ticker", sort=True):
        r = grp.sort_values("date")["simple_return"].values
        n_days = len(r)

        total_return = float(np.prod(1.0 + r) - 1.0)
        cagr = float((1.0 + total_return) ** (trading_days_per_year / n_days) - 1.0)
        mean_daily = float(np.mean(r))
        mean_monthly = mean_daily * 21
        mean_annual = mean_daily * trading_days_per_year
        best_day = float(np.max(r))
        worst_day = float(np.min(r))
        pct_positive = float(np.mean(r > 0))

        for name, val in [
            ("total_return", total_return),
            ("cagr", cagr),
            ("mean_daily_return", mean_daily),
            ("mean_monthly_return", mean_monthly),
            ("mean_annual_return", mean_annual),
            ("best_day_return", best_day),
            ("worst_day_return", worst_day),
            ("pct_positive_days", pct_positive),
        ]:
            rows.append({"ticker": ticker, "metric_name": name, "value": val, "category": "return"})

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    df["category"] = df["category"].astype(pd.StringDtype())
    return df


def compute_risk_metrics(
    returns: pd.DataFrame,
    trading_days_per_year: int,
    var_levels: list[float],
    cvar_levels: list[float],
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - daily_vol: std(r, ddof=1)
      - annual_vol: daily_vol * sqrt(trading_days_per_year)
      - downside_deviation: std(r[r<0], ddof=1) * sqrt(trading_days_per_year)
      - var_{p}: historical quantile at (1-p), returned as positive loss
      - cvar_{p}: mean of returns <= var threshold, returned as positive
      - max_drawdown: min of running drawdown series (negative value)
      - max_drawdown_duration_days: longest peak-to-recovery span (trading days)

    VaR/CVaR use the historical (empirical) method.
    Prices reconstructed from returns for drawdown (normalized series).
    Returns long-format DataFrame: [ticker, metric_name, value, category].
    """
    returns = returns.copy()
    rows: list[dict] = []

    for ticker, grp in returns.groupby("ticker", sort=True):
        r = grp.sort_values("date")["simple_return"].values
        neg_r = r[r < 0]

        daily_vol = float(np.std(r, ddof=1))
        annual_vol = daily_vol * math.sqrt(trading_days_per_year)
        downside_dev = (
            float(np.std(neg_r, ddof=1)) * math.sqrt(trading_days_per_year)
            if len(neg_r) > 1
            else np.nan
        )

        if daily_vol == 0.0:
            logger.warning(f"Ticker {ticker}: zero volatility (flat-lined returns); Sharpe/Sortino will be NaN")

        ticker_metrics: dict[str, float] = {
            "daily_vol": daily_vol,
            "annual_vol": annual_vol,
            "downside_deviation": downside_dev,
        }

        for level in var_levels:
            key = f"var_{int(level * 100)}"
            ticker_metrics[key] = float(abs(np.quantile(r, 1.0 - level)))

        for level in cvar_levels:
            key = f"cvar_{int(level * 100)}"
            threshold = np.quantile(r, 1.0 - level)
            tail = r[r <= threshold]
            ticker_metrics[key] = float(abs(np.mean(tail))) if len(tail) > 0 else np.nan

        max_dd, max_dur, _ = _dd_stats_from_array(r)
        ticker_metrics["max_drawdown"] = max_dd
        ticker_metrics["max_drawdown_duration_days"] = float(max_dur)

        for name, val in ticker_metrics.items():
            rows.append({"ticker": ticker, "metric_name": name, "value": val, "category": "risk"})

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    df["category"] = df["category"].astype(pd.StringDtype())
    return df


def compute_drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Per ticker, compute drawdown_pct = (close / running_max_close) - 1.
    Running max is cumulative and does NOT reset on partial recovery.

    Returns long-format DataFrame: [date, ticker, close, running_peak, drawdown_pct].
    """
    prices = prices.copy()
    parts: list[pd.DataFrame] = []

    for ticker, grp in prices.groupby("ticker", sort=True):
        grp = grp.sort_values("date")[["date", "close"]].copy()
        grp["running_peak"] = grp["close"].cummax()
        grp["drawdown_pct"] = grp["close"] / grp["running_peak"] - 1.0
        grp["ticker"] = ticker
        parts.append(grp[["date", "ticker", "close", "running_peak", "drawdown_pct"]])

    df = pd.concat(parts, ignore_index=True)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    return df


def compute_risk_adjusted(
    return_metrics: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    risk_free_rate: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """
    Per ticker, compute:
      - sharpe:  (mean_annual_return - risk_free_rate) / annual_vol
      - sortino: (mean_annual_return - risk_free_rate) / downside_deviation
      - calmar:  cagr / abs(max_drawdown)

    risk_free_rate is annual. Both numerator and denominator are annualized.
    Division-by-zero → NaN (logged as warning).
    Returns long-format DataFrame: [ticker, metric_name, value, category].
    """
    ret_pivot = return_metrics.pivot(index="ticker", columns="metric_name", values="value")
    risk_pivot = risk_metrics.pivot(index="ticker", columns="metric_name", values="value")

    tickers = sorted(ret_pivot.index.tolist())
    rows: list[dict] = []

    for ticker in tickers:
        annual_return = float(ret_pivot.at[ticker, "mean_annual_return"])
        annual_vol = float(risk_pivot.at[ticker, "annual_vol"])
        downside_dev = float(risk_pivot.at[ticker, "downside_deviation"])
        cagr = float(ret_pivot.at[ticker, "cagr"])
        max_dd = float(risk_pivot.at[ticker, "max_drawdown"])

        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0.0 else np.nan
        sortino = (
            (annual_return - risk_free_rate) / downside_dev
            if not np.isnan(downside_dev) and downside_dev != 0.0
            else np.nan
        )
        calmar = cagr / abs(max_dd) if not np.isnan(max_dd) and max_dd != 0.0 else np.nan

        if not np.isnan(sharpe) and not math.isfinite(sharpe):
            logger.warning(f"Ticker {ticker}: Sharpe is not finite ({sharpe}); setting to NaN")
            sharpe = np.nan

        for name, val in [("sharpe", sharpe), ("sortino", sortino), ("calmar", calmar)]:
            rows.append({"ticker": ticker, "metric_name": name, "value": val, "category": "risk_adjusted"})

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    df["category"] = df["category"].astype(pd.StringDtype())
    return df


def compute_portfolio_metrics(
    returns: pd.DataFrame,
    weights: dict[str, float] | None,
    benchmark_ticker: str | None,
    exclude_benchmark: bool,
    risk_free_rate: float,
    trading_days_per_year: int,
    var_levels: list[float],
    cvar_levels: list[float],
) -> pd.DataFrame:
    """
    Construct portfolio return series as weighted sum of ticker returns.
    Equal weights when weights is None; benchmark excluded if exclude_benchmark=True.

    Computes the same metrics as per-ticker, plus:
      - beta_vs_benchmark: cov(port, bench) / var(bench)  [NaN if no benchmark]
      - diversification_ratio: sum(w_i * vol_i) / portfolio_vol

    Returns single-row wide DataFrame.
    """
    returns = returns.copy()

    ret_wide = returns.pivot(index="date", columns="ticker", values="simple_return")
    ret_wide.columns.name = None
    all_tickers = list(ret_wide.columns)

    portfolio_tickers = [
        t for t in all_tickers
        if not (exclude_benchmark and t == benchmark_ticker)
    ]

    if len(portfolio_tickers) == 0:
        raise ValueError("No portfolio tickers remain after excluding benchmark.")

    if weights is None:
        n = len(portfolio_tickers)
        weights = {t: 1.0 / n for t in portfolio_tickers}
        weights_strategy = "equal"
    else:
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"PORTFOLIO_WEIGHTS must sum to 1.0, got {sum(weights.values()):.8f}"
            )
        extra = set(weights.keys()) - set(portfolio_tickers)
        missing = set(portfolio_tickers) - set(weights.keys())
        if extra:
            raise ValueError(f"PORTFOLIO_WEIGHTS contains unknown tickers: {sorted(extra)}")
        if missing:
            raise ValueError(f"PORTFOLIO_WEIGHTS missing tickers: {sorted(missing)}")
        weights_strategy = "custom"

    port_ret: pd.Series = sum(ret_wide[t] * w for t, w in weights.items())
    port_ret = port_ret.dropna()
    r = port_ret.values
    n_days = len(r)

    total_return = float(np.prod(1.0 + r) - 1.0)
    cagr = float((1.0 + total_return) ** (trading_days_per_year / n_days) - 1.0)
    mean_annual = float(np.mean(r)) * trading_days_per_year
    daily_vol = float(np.std(r, ddof=1))
    annual_vol = daily_vol * math.sqrt(trading_days_per_year)

    sharpe = (mean_annual - risk_free_rate) / annual_vol if annual_vol != 0.0 else np.nan
    neg_r = r[r < 0]
    downside_dev = (
        float(np.std(neg_r, ddof=1)) * math.sqrt(trading_days_per_year)
        if len(neg_r) > 1
        else np.nan
    )
    sortino = (
        (mean_annual - risk_free_rate) / downside_dev
        if not np.isnan(downside_dev) and downside_dev != 0.0
        else np.nan
    )

    max_dd, _, _ = _dd_stats_from_array(r)
    calmar = cagr / abs(max_dd) if max_dd != 0.0 else np.nan

    result: dict[str, Any] = {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
    }

    for level in var_levels:
        key = f"var_{int(level * 100)}"
        result[key] = float(abs(np.quantile(r, 1.0 - level)))

    for level in cvar_levels:
        key = f"cvar_{int(level * 100)}"
        threshold = np.quantile(r, 1.0 - level)
        tail = r[r <= threshold]
        result[key] = float(abs(np.mean(tail))) if len(tail) > 0 else np.nan

    # Beta vs benchmark
    beta = np.nan
    if benchmark_ticker and benchmark_ticker in ret_wide.columns:
        bench = ret_wide[benchmark_ticker].reindex(port_ret.index).dropna()
        aligned_port = port_ret.reindex(bench.index).dropna()
        bench = bench.reindex(aligned_port.index)
        bench_var = float(np.var(bench.values, ddof=1))
        if len(bench) > 1 and bench_var != 0.0:
            beta = float(np.cov(aligned_port.values, bench.values)[0, 1] / bench_var)
        else:
            logger.warning(
                f"Benchmark {benchmark_ticker} has constant returns; portfolio beta set to NaN"
            )
    elif benchmark_ticker:
        logger.warning(
            f"Benchmark {benchmark_ticker} not found in returns data; portfolio beta set to NaN"
        )
    result["beta_vs_benchmark"] = beta

    # Diversification ratio
    ticker_daily_vols = {
        t: float(np.std(ret_wide[t].dropna().values, ddof=1)) for t in portfolio_tickers
    }
    weighted_avg_vol = sum(weights[t] * ticker_daily_vols[t] for t in portfolio_tickers)
    result["diversification_ratio"] = weighted_avg_vol / daily_vol if daily_vol != 0.0 else np.nan

    result["weights_strategy"] = weights_strategy
    result["weights_used"] = json.dumps({t: round(w, 8) for t, w in weights.items()})

    return pd.DataFrame([result])


def compute_rolling_metrics(
    returns: pd.DataFrame,
    benchmark_ticker: str | None,
    risk_free_rate: float,
    trading_days_per_year: int,
    sharpe_window: int,
    beta_window: int,
    corr_window: int,
) -> pd.DataFrame:
    """
    Per ticker, compute rolling time series of:
      - rolling_sharpe_{sharpe_window}: (roll_mean - rf_daily) / roll_std * sqrt(252)
      - rolling_volatility_{sharpe_window}: roll_std * sqrt(252)
      - rolling_beta_{beta_window}: roll_cov(asset, bench) / roll_var(bench)
      - rolling_corr_{corr_window}: rolling Pearson correlation with benchmark

    NaN rows from initial window are dropped.
    Returns long-format DataFrame: [date, ticker, metric_name, value].
    """
    returns = returns.copy()
    rf_daily = risk_free_rate / trading_days_per_year
    sqrt_tdy = math.sqrt(trading_days_per_year)

    ret_wide = returns.pivot(index="date", columns="ticker", values="simple_return")
    ret_wide.columns.name = None

    rows: list[dict] = []

    for ticker in sorted(ret_wide.columns):
        r = ret_wide[ticker].dropna().sort_index()

        if len(r) < sharpe_window:
            logger.warning(
                f"Ticker {ticker}: {len(r)} rows < ROLLING_SHARPE_WINDOW ({sharpe_window}); "
                "skipping rolling metrics for this ticker"
            )
            continue

        roll_mean = r.rolling(sharpe_window).mean()
        roll_std = r.rolling(sharpe_window).std()
        roll_sharpe = ((roll_mean - rf_daily) / roll_std * sqrt_tdy).dropna()
        roll_vol = (roll_std * sqrt_tdy).dropna()

        for date, val in roll_sharpe.items():
            rows.append({
                "date": date, "ticker": ticker,
                "metric_name": f"rolling_sharpe_{sharpe_window}", "value": float(val),
            })
        for date, val in roll_vol.items():
            rows.append({
                "date": date, "ticker": ticker,
                "metric_name": f"rolling_volatility_{sharpe_window}", "value": float(val),
            })

        if benchmark_ticker and benchmark_ticker in ret_wide.columns:
            bench = ret_wide[benchmark_ticker].dropna().sort_index()
            aligned = pd.concat([r, bench], axis=1, join="inner")
            aligned.columns = ["asset", "bench"]

            if len(aligned) >= beta_window:
                roll_cov = aligned["asset"].rolling(beta_window).cov(aligned["bench"])
                roll_var = aligned["bench"].rolling(beta_window).var()
                roll_beta = (roll_cov / roll_var).dropna()
                for date, val in roll_beta.items():
                    rows.append({
                        "date": date, "ticker": ticker,
                        "metric_name": f"rolling_beta_{beta_window}", "value": float(val),
                    })

            if len(aligned) >= corr_window:
                roll_corr = aligned["asset"].rolling(corr_window).corr(aligned["bench"]).dropna()
                for date, val in roll_corr.items():
                    rows.append({
                        "date": date, "ticker": ticker,
                        "metric_name": f"rolling_corr_{corr_window}", "value": float(val),
                    })

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "metric_name", "value"])

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    return df


def save_metrics(
    per_ticker: pd.DataFrame,
    portfolio: pd.DataFrame,
    rolling: pd.DataFrame,
    drawdowns: pd.DataFrame,
    summary_dict: dict,
    processed_dir: Path,
    reports_dir: Path,
) -> None:
    """
    Write:
      - data/processed/metrics_per_ticker.parquet
      - data/processed/portfolio_metrics.parquet
      - data/processed/rolling_metrics.parquet
      - data/processed/drawdown_series.parquet
      - outputs/reports/metrics_summary.json (indent=2)

    Creates directories if missing. Overwrites without prompting.
    DataFrames are sorted before writing to ensure idempotent byte output.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"engine": "pyarrow", "index": False, "compression": "snappy"}

    per_ticker_sorted = per_ticker.sort_values(["ticker", "category", "metric_name"]).reset_index(drop=True)
    metrics_per_ticker_schema.validate(per_ticker_sorted)
    per_ticker_sorted.to_parquet(processed_dir / "metrics_per_ticker.parquet", **kwargs)
    logger.info(f"Saved metrics_per_ticker.parquet ({len(per_ticker_sorted)} rows)")

    portfolio.to_parquet(processed_dir / "portfolio_metrics.parquet", **kwargs)
    logger.info("Saved portfolio_metrics.parquet (1 row)")

    if not rolling.empty:
        rolling_sorted = rolling.sort_values(["ticker", "date", "metric_name"]).reset_index(drop=True)
        rolling_metrics_schema.validate(rolling_sorted)
    else:
        rolling_sorted = rolling
    rolling_sorted.to_parquet(processed_dir / "rolling_metrics.parquet", **kwargs)
    logger.info(f"Saved rolling_metrics.parquet ({len(rolling_sorted)} rows)")

    dd_sorted = drawdowns.sort_values(["ticker", "date"]).reset_index(drop=True)
    drawdown_schema.validate(dd_sorted)
    dd_sorted.to_parquet(processed_dir / "drawdown_series.parquet", **kwargs)
    logger.info(f"Saved drawdown_series.parquet ({len(dd_sorted)} rows)")

    with (reports_dir / "metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(summary_dict), fh, indent=2)
    logger.info("Saved metrics_summary.json")


def _compute_per_ticker_beta(
    returns: pd.DataFrame,
    benchmark: str | None,
    all_tickers: list[str],
) -> dict[str, float | None]:
    """Return {ticker: beta | None} for every ticker vs benchmark (full history)."""
    if not benchmark:
        return {t: None for t in all_tickers}
    ret_wide = returns.pivot(index="date", columns="ticker", values="simple_return")
    ret_wide.columns.name = None
    betas: dict[str, float | None] = {}
    for ticker in all_tickers:
        if benchmark in ret_wide.columns and ticker != benchmark:
            r_a = ret_wide[ticker].dropna()
            r_b = ret_wide[benchmark].reindex(r_a.index).dropna()
            r_a = r_a.reindex(r_b.index)
            bench_var = float(np.var(r_b.values, ddof=1))
            if len(r_b) > 1 and bench_var != 0.0:
                betas[ticker] = float(np.cov(r_a.values, r_b.values)[0, 1] / bench_var)
            else:
                betas[ticker] = None
        else:
            betas[ticker] = None
    return betas


def run_metrics() -> dict:
    """
    Orchestrator for Step 4. Called by main.py.

    Pipeline:
      load_data → validate schemas
      → compute_return_metrics → compute_risk_metrics → compute_drawdown_series
      → compute_risk_adjusted → compute_portfolio_metrics → compute_rolling_metrics
      → save_metrics

    Returns:
      {
        "tickers_analyzed": int,
        "portfolio_sharpe": float,
        "portfolio_max_drawdown": float,
        "benchmark_used": str | None,
        "weights_strategy": "equal" | "custom",
        "outputs_written": list[str],
        "duration_sec": float,
      }
    """
    t0 = time.time()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_id = logger.add(
        config.LOG_DIR / f"metrics_{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO",
        rotation="1 day",
    )

    try:
        # ── Config guards ──────────────────────────────────────────────────
        if config.RISK_FREE_RATE < 0:
            logger.info(f"RISK_FREE_RATE is negative ({config.RISK_FREE_RATE}); allowed for negative-rate currencies")
        if config.RISK_FREE_RATE > 0.5:
            logger.warning(f"RISK_FREE_RATE ({config.RISK_FREE_RATE}) > 0.5 — likely a config error")

        for level in config.VAR_CONFIDENCE_LEVELS + config.CVAR_CONFIDENCE_LEVELS:
            if level <= 0 or level >= 1:
                raise ValueError(f"Confidence level must be in (0, 1), got {level}")

        # ── Load & validate ────────────────────────────────────────────────
        logger.info("Loading cleaned prices and returns")
        prices, returns = load_data(config.PROCESSED_DATA_DIR)

        all_tickers: list[str] = sorted(returns["ticker"].unique().tolist())
        benchmark = config.BENCHMARK_TICKER

        if benchmark and benchmark not in all_tickers:
            logger.warning(
                f"BENCHMARK_TICKER={benchmark!r} not found in data; "
                "beta-related metrics will be NaN"
            )

        date_range = [
            str(returns["date"].min().date()),
            str(returns["date"].max().date()),
        ]
        # If tickers ever include non-public identifiers, redact before logging.
        logger.info(f"Tickers: {all_tickers} | Benchmark: {benchmark} | Date range: {date_range}")

        # ── Return metrics ─────────────────────────────────────────────────
        logger.info("Computing return metrics")
        return_metrics = compute_return_metrics(returns, prices, config.METRICS_TRADING_DAYS_PER_YEAR)

        # ── Risk metrics ───────────────────────────────────────────────────
        logger.info("Computing risk metrics")
        risk_metrics = compute_risk_metrics(
            returns,
            config.METRICS_TRADING_DAYS_PER_YEAR,
            config.VAR_CONFIDENCE_LEVELS,
            config.CVAR_CONFIDENCE_LEVELS,
        )

        # ── Drawdown series ────────────────────────────────────────────────
        logger.info("Computing drawdown series")
        drawdown_series = compute_drawdown_series(prices)

        # ── Risk-adjusted metrics ──────────────────────────────────────────
        logger.info("Computing risk-adjusted metrics (Sharpe, Sortino, Calmar)")
        risk_adjusted = compute_risk_adjusted(
            return_metrics, risk_metrics, config.RISK_FREE_RATE, config.METRICS_TRADING_DAYS_PER_YEAR
        )

        per_ticker = pd.concat([return_metrics, risk_metrics, risk_adjusted], ignore_index=True)

        # ── Invariant checks ───────────────────────────────────────────────
        pt_pivot = per_ticker.pivot_table(index="ticker", columns="metric_name", values="value", aggfunc="first")
        if "var_95" in pt_pivot.columns and "var_99" in pt_pivot.columns:
            bad = pt_pivot.index[pt_pivot["var_95"] > pt_pivot["var_99"]].tolist()
            if bad:
                logger.warning(f"VaR_95 > VaR_99 for tickers: {bad}")
        if "cvar_95" in pt_pivot.columns and "var_95" in pt_pivot.columns:
            bad = pt_pivot.index[pt_pivot["cvar_95"] < pt_pivot["var_95"]].tolist()
            if bad:
                logger.warning(f"CVaR_95 < VaR_95 for tickers: {bad}")
        if "max_drawdown" in pt_pivot.columns:
            bad = pt_pivot.index[pt_pivot["max_drawdown"] > 0].tolist()
            if bad:
                logger.warning(f"max_drawdown > 0 (unexpected) for tickers: {bad}")

        # ── Portfolio metrics ──────────────────────────────────────────────
        logger.info("Computing portfolio metrics")
        portfolio_df = compute_portfolio_metrics(
            returns,
            config.PORTFOLIO_WEIGHTS,
            config.BENCHMARK_TICKER,
            config.EXCLUDE_BENCHMARK_FROM_PORTFOLIO,
            config.RISK_FREE_RATE,
            config.METRICS_TRADING_DAYS_PER_YEAR,
            config.VAR_CONFIDENCE_LEVELS,
            config.CVAR_CONFIDENCE_LEVELS,
        )

        port_sharpe = float(portfolio_df["sharpe"].iloc[0])
        if not math.isfinite(port_sharpe):
            logger.warning(f"Portfolio Sharpe is not finite: {port_sharpe}")
        elif port_sharpe < -1 or port_sharpe > 3:
            logger.warning(f"Portfolio Sharpe ({port_sharpe:.3f}) outside typical range [-1, 3]")

        # ── Rolling metrics ────────────────────────────────────────────────
        logger.info("Computing rolling metrics")
        rolling_df = compute_rolling_metrics(
            returns,
            config.BENCHMARK_TICKER,
            config.RISK_FREE_RATE,
            config.METRICS_TRADING_DAYS_PER_YEAR,
            config.ROLLING_SHARPE_WINDOW,
            config.ROLLING_BETA_WINDOW,
            config.ROLLING_CORR_WINDOW,
        )

        # ── Build summary JSON ─────────────────────────────────────────────
        # Per-ticker beta and portfolio weights come from the computed results,
        # not re-derived from config, so the summary stays in sync with the data.
        ticker_betas = _compute_per_ticker_beta(returns, benchmark, all_tickers)
        used_weights = json.loads(portfolio_df["weights_used"].iloc[0])

        # Per-ticker drawdown recovery flag
        ticker_recovered: dict[str, bool] = {}
        for ticker, grp in returns.groupby("ticker"):
            r_arr = grp.sort_values("date")["simple_return"].values
            _, _, recovered = _dd_stats_from_array(r_arr)
            ticker_recovered[ticker] = recovered

        per_ticker_summary: dict[str, Any] = {}
        for ticker in all_tickers:
            t_vals = per_ticker[per_ticker["ticker"] == ticker].set_index("metric_name")["value"].to_dict()
            t_entry = {k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in t_vals.items()}
            t_entry["beta_vs_benchmark"] = ticker_betas.get(ticker)
            t_entry["recovered"] = ticker_recovered.get(ticker, True)
            per_ticker_summary[ticker] = t_entry

        port_row = portfolio_df.iloc[0].to_dict()
        weights_strategy = str(port_row.get("weights_strategy", "equal"))

        summary: dict[str, Any] = {
            "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {
                "prices_rows": len(prices),
                "returns_rows": len(returns),
                "tickers": len(all_tickers),
                "benchmark": config.BENCHMARK_TICKER,
                "date_range": date_range,
            },
            "per_ticker": per_ticker_summary,
            "portfolio": {
                "weights_strategy": weights_strategy,
                "weights": used_weights,
                "total_return": port_row.get("total_return"),
                "cagr": port_row.get("cagr"),
                "annual_vol": port_row.get("annual_vol"),
                "sharpe": port_row.get("sharpe"),
                "sortino": port_row.get("sortino"),
                "calmar": port_row.get("calmar"),
                "max_drawdown": port_row.get("max_drawdown"),
                "var_95": port_row.get("var_95"),
                "cvar_95": port_row.get("cvar_95"),
                "beta_vs_benchmark": port_row.get("beta_vs_benchmark"),
                "diversification_ratio": port_row.get("diversification_ratio"),
            },
            "config_used": {
                "risk_free_rate": config.RISK_FREE_RATE,
                "benchmark_ticker": config.BENCHMARK_TICKER,
                "trading_days_per_year": config.METRICS_TRADING_DAYS_PER_YEAR,
                "var_confidence_levels": config.VAR_CONFIDENCE_LEVELS,
                "rolling_sharpe_window": config.ROLLING_SHARPE_WINDOW,
                "rolling_beta_window": config.ROLLING_BETA_WINDOW,
            },
            "duration_sec": 0.0,  # updated after save
        }

        # ── Save outputs ───────────────────────────────────────────────────
        outputs_written = [
            str(config.PROCESSED_DATA_DIR / "metrics_per_ticker.parquet"),
            str(config.PROCESSED_DATA_DIR / "portfolio_metrics.parquet"),
            str(config.PROCESSED_DATA_DIR / "rolling_metrics.parquet"),
            str(config.PROCESSED_DATA_DIR / "drawdown_series.parquet"),
            str(config.REPORTS_DIR / "metrics_summary.json"),
        ]
        duration = round(time.time() - t0, 2)
        summary["duration_sec"] = duration

        logger.info("Saving all metric outputs")
        save_metrics(
            per_ticker, portfolio_df, rolling_df, drawdown_series,
            summary, config.PROCESSED_DATA_DIR, config.REPORTS_DIR,
        )

        port_max_dd = float(portfolio_df["max_drawdown"].iloc[0])
        logger.info(
            f"Step 4 complete | tickers={len(all_tickers)} | "
            f"portfolio_sharpe={port_sharpe:.3f} | "
            f"portfolio_max_drawdown={port_max_dd:.3f} | "
            f"duration={duration}s"
        )

        return {
            "tickers_analyzed": len(all_tickers),
            "portfolio_sharpe": port_sharpe,
            "portfolio_max_drawdown": port_max_dd,
            "benchmark_used": config.BENCHMARK_TICKER,
            "weights_strategy": weights_strategy,
            "outputs_written": outputs_written,
            "duration_sec": duration,
        }

    finally:
        logger.remove(log_id)
