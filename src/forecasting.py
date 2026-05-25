"""
src/forecasting.py — Time-series forecasting module (Step 5).

Reads cleaned price/return data, fits ARIMA / Prophet / naive baselines per
ticker-scenario, runs walk-forward backtesting, and persists outputs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller

import config
from src.schemas import (
    forecasts_schema,
    forecast_metrics_schema,
    stationarity_schema,
    prices_clean_schema,
    returns_schema,
)

# Silence cmdstanpy verbose output if Prophet is available
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

_VALID_MODELS = {
    "arima_returns",
    "arima_log_prices",
    "prophet",
    "naive_random_walk",
    "naive_drift",
    "naive_mean",
}

_SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


# ── Public functions ──────────────────────────────────────────────────────────


def load_data(
    processed_dir: Path,
    scenarios_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read prices_clean.parquet, returns_daily.parquet, and scenarios.csv.
    Validate pandera schemas. Validate scenario CSV column presence and model names.
    Raise FileNotFoundError if any file is missing.
    Returns (prices, returns, scenarios).
    """
    prices_path = processed_dir / "prices_clean.parquet"
    returns_path = processed_dir / "returns_daily.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(f"{prices_path} not found. Run Step 2 first.")
    if not returns_path.exists():
        raise FileNotFoundError(f"{returns_path} not found. Run Step 2 first.")
    if not scenarios_csv.exists():
        raise FileNotFoundError(
            f"{scenarios_csv} not found. Create a scenarios CSV with columns: "
            "scenario_name, model, target, horizon_days, confidence_level."
        )

    prices = pd.read_parquet(prices_path)
    returns = pd.read_parquet(returns_path)
    scenarios = pd.read_csv(scenarios_csv)

    prices_clean_schema.validate(prices)
    returns_schema.validate(returns)

    required_cols = {"scenario_name", "model", "target", "horizon_days", "confidence_level"}
    missing = required_cols - set(scenarios.columns)
    if missing:
        raise ValueError(f"scenarios.csv is missing columns: {missing}")
    if len(scenarios) == 0:
        raise ValueError("scenarios.csv has no rows — nothing to forecast.")

    bad_names = scenarios[~scenarios["scenario_name"].str.match(_SCENARIO_NAME_RE)]
    if len(bad_names):
        raise ValueError(
            f"Invalid scenario_name(s) — must match [A-Za-z0-9_-]{{1,64}}: "
            f"{bad_names['scenario_name'].tolist()}"
        )

    unknown = set(scenarios["model"].unique()) - _VALID_MODELS
    if unknown:
        bad_rows = scenarios[scenarios["model"].isin(unknown)][["scenario_name", "model"]]
        raise ValueError(f"Unknown model(s) in scenarios.csv:\n{bad_rows.to_string()}")

    logger.info(
        f"Loaded {len(prices)} price rows, {len(returns)} return rows, "
        f"{len(scenarios)} scenario(s)."
    )
    return prices, returns, scenarios


def check_stationarity(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test per ticker on simple_return and log_prices proxy.
    Returns DataFrame: [ticker, series_type, adf_statistic, p_value, is_stationary,
    critical_1pct, critical_5pct].
    is_stationary = True if p_value < 0.05.
    """
    rows = []
    # Vectorisation not applicable: adfuller returns non-tabular output per series
    for ticker, grp in returns.groupby("ticker"):
        for series_type, series in [
            ("returns", grp["simple_return"].dropna()),
            ("log_prices", grp["log_return"].cumsum().dropna()),
        ]:
            row: dict = {
                "ticker": ticker,
                "series_type": series_type,
                "adf_statistic": None,
                "p_value": None,
                "is_stationary": False,
                "critical_1pct": None,
                "critical_5pct": None,
            }
            if len(series) < 20:
                logger.warning(f"[{ticker}] Too few observations for ADF test on {series_type}.")
                rows.append(row)
                continue
            try:
                result = adfuller(series.values, autolag="AIC")
                adf_stat, p_val = result[0], result[1]
                crits = result[4]
                if np.isnan(adf_stat) or np.isnan(p_val):
                    logger.warning(
                        f"[{ticker}] ADF returned NaN for {series_type} (constant series?)."
                    )
                    rows.append(row)
                    continue
                row.update(
                    {
                        "adf_statistic": float(adf_stat),
                        "p_value": float(p_val),
                        "is_stationary": bool(p_val < 0.05),
                        "critical_1pct": float(crits["1%"]),
                        "critical_5pct": float(crits["5%"]),
                    }
                )
            except Exception as exc:
                logger.warning(f"[{ticker}] ADF failed for {series_type}: {exc}")
            rows.append(row)

    df = pd.DataFrame(rows).astype(
        {
            "ticker": pd.StringDtype(),
            "series_type": pd.StringDtype(),
            "is_stationary": bool,
        }
    )
    return df


def fit_arima(
    series: pd.Series,
    max_p: int,
    max_q: int,
    max_d: int,
    seasonal: bool,
    horizon: int,
    confidence_level: float,
    train_dates: pd.Series | None = None,  # accepted for interface consistency; not used by ARIMA
) -> dict:
    """
    Fit ARIMA via pmdarima.auto_arima with AIC criterion.
    Forecast `horizon` steps ahead with confidence intervals.
    Returns dict with keys: order, aic, forecast, lower_ci, upper_ci, converged.
    On convergence failure, returns converged=False.
    """
    try:
        import pmdarima as pm
    except ImportError as exc:
        raise ImportError("pmdarima is required: pip install pmdarima") from exc

    try:
        model = pm.auto_arima(
            series.values,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            seasonal=seasonal,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
        if model is None:
            return {"converged": False}

        alpha = 1.0 - confidence_level
        fc, conf = model.predict(n_periods=horizon, return_conf_int=True, alpha=alpha)
        return {
            "order": model.order,
            "aic": float(model.aic()),
            "forecast": np.asarray(fc, dtype=float),
            "lower_ci": np.asarray(conf[:, 0], dtype=float),
            "upper_ci": np.asarray(conf[:, 1], dtype=float),
            "converged": True,
        }
    except Exception as exc:
        logger.error(f"ARIMA fit failed: {exc}")
        return {"converged": False}


def fit_prophet(
    series: pd.Series,
    dates: pd.Series,
    yearly: bool,
    weekly: bool,
    daily: bool,
    horizon: int,
    confidence_level: float,
) -> dict:
    """
    Fit Prophet model. Expects series and dates aligned by position.
    Returns dict with keys: forecast, lower_ci, upper_ci, trend,
    yearly_component, weekly_component, converged.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("prophet is required: pip install prophet") from exc

    try:
        df_prophet = pd.DataFrame({"ds": pd.to_datetime(dates.values), "y": series.values})
        df_prophet = df_prophet.dropna()

        m = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            interval_width=confidence_level,
        )
        m.fit(df_prophet)

        last_date = df_prophet["ds"].max()
        future = m.make_future_dataframe(periods=horizon, freq="B")
        future = future[future["ds"] > last_date]

        forecast_df = m.predict(future)

        yearly_comp = (
            forecast_df["yearly"].values if "yearly" in forecast_df.columns else None
        )
        weekly_comp = (
            forecast_df["weekly"].values if "weekly" in forecast_df.columns else None
        )

        return {
            "forecast": forecast_df["yhat"].values.astype(float),
            "lower_ci": forecast_df["yhat_lower"].values.astype(float),
            "upper_ci": forecast_df["yhat_upper"].values.astype(float),
            "trend": forecast_df["trend"].values.astype(float),
            "yearly_component": yearly_comp,
            "weekly_component": weekly_comp,
            "converged": True,
        }
    except Exception as exc:
        logger.error(f"Prophet fit failed: {exc}")
        return {"converged": False}


def fit_naive(
    series: pd.Series,
    method: str,
    horizon: int,
    confidence_level: float,
    train_dates: pd.Series | None = None,  # accepted for interface consistency; not used by naive models
) -> dict:
    """
    Fit naive baseline. method in {"random_walk", "drift", "mean"}.
      - random_walk: forecast = last observed value (constant)
      - drift: forecast = last + mean_change * step
      - mean: forecast = historical mean (constant)

    CIs computed from historical residual std × sqrt(step) for random_walk/drift,
    or historical std for mean.

    Returns same dict structure as fit_arima for consistency.
    """
    vals = series.dropna().values
    if len(vals) < 2:
        return {"converged": False}

    alpha = 1.0 - confidence_level
    z = _z_score(1.0 - alpha / 2.0)
    steps = np.arange(1, horizon + 1, dtype=float)

    if method == "random_walk":
        last = vals[-1]
        resid_std = np.std(np.diff(vals), ddof=1)
        fc = np.full(horizon, last)
        half_width = z * resid_std * np.sqrt(steps)
    elif method == "drift":
        last = vals[-1]
        mean_change = np.mean(np.diff(vals))
        resid_std = np.std(np.diff(vals) - mean_change, ddof=1)
        fc = last + mean_change * steps
        half_width = z * resid_std * np.sqrt(steps)
    elif method == "mean":
        mu = np.mean(vals)
        std = np.std(vals, ddof=1)
        fc = np.full(horizon, mu)
        half_width = np.full(horizon, z * std)
    else:
        raise ValueError(
            f"Unknown naive method: {method!r}. Must be random_walk, drift, or mean."
        )

    return {
        "order": None,
        "aic": None,
        "forecast": fc,
        "lower_ci": fc - half_width,
        "upper_ci": fc + half_width,
        "converged": True,
    }


def walk_forward_validate(
    series: pd.Series,
    dates: pd.Series,
    model_fn: Callable,
    model_kwargs: dict,
    train_initial_days: int,
    step_days: int,
    horizon: int,
    expanding: bool,
) -> pd.DataFrame:
    """
    Walk-forward backtest:
      - Start with train_initial_days of training data
      - Forecast horizon days ahead
      - Roll forward by step_days, refit, repeat until end of series
      - If expanding=True, training window grows; else fixed size

    Returns long-format DataFrame:
      [fold, fold_start_date, fold_end_date, forecast_date,
       actual, predicted, lower_ci, upper_ci]
    """
    vals = series.values
    dts = pd.to_datetime(dates.values)
    n = len(vals)

    if n < train_initial_days + horizon:
        logger.warning(
            f"Series length {n} < train_initial_days({train_initial_days}) + horizon({horizon}). "
            "Skipping walk-forward."
        )
        return pd.DataFrame(
            columns=[
                "fold", "fold_start_date", "fold_end_date", "forecast_date",
                "actual", "predicted", "lower_ci", "upper_ci",
            ]
        )

    rows = []
    fold = 0
    train_start = 0
    train_end = train_initial_days

    while train_end + horizon <= n:
        train_series = pd.Series(vals[train_start:train_end])
        train_dates_slice = pd.Series(dts[train_start:train_end])  # actual dates for Prophet
        test_vals = vals[train_end: train_end + horizon]
        test_dates = dts[train_end: train_end + horizon]

        result = model_fn(train_series, train_dates=train_dates_slice, **model_kwargs)
        if not result.get("converged", False):
            result = fit_naive(
                train_series,
                method="random_walk",
                horizon=horizon,
                confidence_level=model_kwargs["confidence_level"],
            )

        fc = result["forecast"]
        lci = result["lower_ci"]
        uci = result["upper_ci"]

        # Align lengths (Prophet may return fewer points than horizon)
        actual_horizon = min(len(fc), len(test_vals))
        for i in range(actual_horizon):
            rows.append(
                {
                    "fold": fold,
                    "fold_start_date": dts[train_start],
                    "fold_end_date": dts[train_end - 1],
                    "forecast_date": test_dates[i],
                    "actual": float(test_vals[i]),
                    "predicted": float(fc[i]),
                    "lower_ci": float(lci[i]),
                    "upper_ci": float(uci[i]),
                }
            )

        fold += 1
        train_end += step_days
        if not expanding:
            train_start += step_days

    return pd.DataFrame(rows)


def compute_forecast_metrics(backtest_results: pd.DataFrame) -> dict:
    """
    Compute aggregate metrics across all walk-forward folds.
      - rmse: sqrt(mean((actual - predicted)^2))
      - mae: mean(|actual - predicted|)
      - mape: mean(|actual - predicted| / |actual|) * 100, skipping rows where actual≈0
      - directional_accuracy: % of times sign(predicted) == sign(actual) (for returns)
      - coverage_rate: % of actuals inside [lower_ci, upper_ci]
      - mean_interval_width: mean(upper_ci - lower_ci)
      - n_folds: int
      - n_predictions: int
    """
    if len(backtest_results) == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "mape": np.nan,
            "directional_accuracy": np.nan, "coverage_rate": np.nan,
            "mean_interval_width": np.nan, "n_folds": 0, "n_predictions": 0,
        }

    actual = backtest_results["actual"].values
    predicted = backtest_results["predicted"].values
    lower = backtest_results["lower_ci"].values
    upper = backtest_results["upper_ci"].values

    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    eps = 1e-8
    mask = np.abs(actual) > eps
    mape = float(np.mean(np.abs(residuals[mask] / actual[mask])) * 100) if mask.any() else np.nan

    dir_acc = float(np.mean(np.sign(predicted) == np.sign(actual)))
    coverage = float(np.mean((lower <= actual) & (actual <= upper)))
    interval_width = float(np.mean(upper - lower))

    if coverage < config.FORECAST_COVERAGE_WARN_BELOW:
        logger.warning(
            f"Coverage rate {coverage:.2%} < {config.FORECAST_COVERAGE_WARN_BELOW:.0%} "
            "— model underestimates uncertainty."
        )
    elif coverage > config.FORECAST_COVERAGE_WARN_ABOVE:
        logger.info(
            f"Coverage rate {coverage:.2%} > {config.FORECAST_COVERAGE_WARN_ABOVE:.0%} "
            "— CI may be too wide."
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "coverage_rate": coverage,
        "mean_interval_width": interval_width,
        "n_folds": int(backtest_results["fold"].nunique()),
        "n_predictions": len(backtest_results),
    }


def run_scenario(
    scenario_row: pd.Series,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config_overrides: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute one scenario row end-to-end.
    Returns (forecasts_df, metrics_df) both tagged with scenario_name.
    forecasts_df schema: [scenario_name, ticker, forecast_date, forecast,
                          lower_ci, upper_ci, model_type, target, confidence_level]
    metrics_df schema: [scenario_name, ticker, model_type, metric_name, value]
    """
    scenario_name = scenario_row["scenario_name"]
    model = scenario_row["model"]
    target = scenario_row["target"]
    horizon = int(scenario_row["horizon_days"])
    confidence_level = float(scenario_row["confidence_level"])

    train_initial = config_overrides.get("train_initial_days", config.TRAIN_INITIAL_DAYS)
    step_days = config_overrides.get("walk_forward_step_days", config.WALK_FORWARD_STEP_DAYS)
    expanding = config_overrides.get("walk_forward_expanding", config.WALK_FORWARD_EXPANDING)
    min_obs = config_overrides.get("min_observations", config.MIN_OBSERVATIONS_FOR_FORECAST)

    tickers_filter = _parse_tickers(scenario_row.get("tickers", "all"), prices)

    all_forecasts: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for ticker in tickers_filter:
        logger.info(
            f"[{scenario_name}][{ticker}] Fitting model={model} "
            f"target={target} horizon={horizon}"
        )

        series, dates = _get_series(ticker, target, prices, returns)
        if series is None:
            logger.warning(f"[{scenario_name}][{ticker}] No data; skipping.")
            continue

        # Compute clean series once; align dates to non-NaN positions
        clean_series = series.dropna().reset_index(drop=True)
        clean_dates = dates.iloc[series.dropna().index].reset_index(drop=True)

        if len(clean_series) < min_obs:
            logger.warning(
                f"[{scenario_name}][{ticker}] Only {len(clean_series)} obs < "
                f"MIN_OBSERVATIONS_FOR_FORECAST({min_obs}); skipping."
            )
            continue
        if horizon > len(clean_series):
            logger.error(
                f"[{scenario_name}][{ticker}] horizon({horizon}) > series "
                f"length({len(clean_series)}); skipping."
            )
            continue

        model_fn, model_kwargs = _build_model_fn(model, confidence_level, horizon)

        backtest = walk_forward_validate(
            series=clean_series,
            dates=clean_dates,
            model_fn=model_fn,
            model_kwargs=model_kwargs,
            train_initial_days=train_initial,
            step_days=step_days,
            horizon=horizon,
            expanding=expanding,
        )

        metrics_dict = compute_forecast_metrics(backtest)

        # Final out-of-sample forecast on full clean series
        final_result = model_fn(clean_series, train_dates=clean_dates, **model_kwargs)
        if not final_result.get("converged", False):
            logger.warning(
                f"[{scenario_name}][{ticker}] Final fit failed; using random walk fallback."
            )
            final_result = fit_naive(
                clean_series,
                method="random_walk",
                horizon=horizon,
                confidence_level=confidence_level,
            )

        last_date = pd.to_datetime(clean_dates.iloc[-1])
        forecast_dates = pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
        fc_len = min(len(final_result["forecast"]), horizon)
        forecast_dates = forecast_dates[:fc_len]

        fc_df = _build_forecast_df(
            scenario_name, ticker, model, target, confidence_level,
            forecast_dates, final_result, fc_len,
        )
        all_forecasts.append(fc_df)

        metric_rows = [
            {
                "scenario_name": scenario_name,
                "ticker": ticker,
                "model_type": model,
                "metric_name": metric_name,
                "value": (
                    float(value)
                    if value is not None and not (isinstance(value, float) and np.isnan(value))
                    else np.nan
                ),
            }
            for metric_name, value in metrics_dict.items()
        ]
        m_df = pd.DataFrame(metric_rows).astype(
            {
                "scenario_name": pd.StringDtype(),
                "ticker": pd.StringDtype(),
                "model_type": pd.StringDtype(),
                "metric_name": pd.StringDtype(),
            }
        )
        all_metrics.append(m_df)

    forecasts_out = (
        pd.concat(all_forecasts, ignore_index=True) if all_forecasts else _empty_forecasts_df()
    )
    metrics_out = (
        pd.concat(all_metrics, ignore_index=True) if all_metrics else _empty_metrics_df()
    )
    return forecasts_out, metrics_out


def save_forecasts(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    stationarity: pd.DataFrame,
    summary_dict: dict,
    processed_dir: Path,
    reports_dir: Path,
) -> None:
    """
    Validate output contracts then write:
      - data/processed/forecasts.parquet
      - data/processed/forecast_metrics.parquet
      - data/processed/stationarity_tests.parquet
      - outputs/reports/forecasting_summary.json (indent=2)
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if len(forecasts):
        forecasts_schema.validate(forecasts)
    if len(metrics):
        forecast_metrics_schema.validate(metrics)
    if len(stationarity):
        stationarity_schema.validate(stationarity)

    _write_parquet(forecasts, processed_dir / "forecasts.parquet")
    _write_parquet(metrics, processed_dir / "forecast_metrics.parquet")
    _write_parquet(stationarity, processed_dir / "stationarity_tests.parquet")

    summary_path = reports_dir / "forecasting_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    logger.info(f"Saved forecasting outputs to {processed_dir} and {reports_dir}")


def run_forecasting() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_data → check_stationarity
      → for each scenario: run_scenario → walk_forward_validate → compute_forecast_metrics
      → aggregate → save_forecasts

    Returns summary dict with scenarios_run, tickers_forecasted, scenarios_failed,
    best_model_per_ticker, total_forecasts, duration_sec.
    """
    np.random.seed(config.RANDOM_SEED)

    start_time = time.time()

    log_path = config.LOG_DIR / f"forecasting_{datetime.now().strftime('%Y-%m-%d')}.log"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_id = logger.add(log_path, level="INFO", rotation="1 day")

    try:
        prices, returns, scenarios = load_data(config.PROCESSED_DATA_DIR, config.SCENARIOS_CSV)

        stationarity = check_stationarity(returns)
        logger.info(
            f"Stationarity: "
            f"{stationarity[stationarity.series_type == 'returns'].is_stationary.sum()} "
            f"/ {len(stationarity[stationarity.series_type == 'returns'])} "
            "tickers stationary on returns."
        )

        all_forecasts: list[pd.DataFrame] = []
        all_metrics: list[pd.DataFrame] = []
        scenarios_run: dict = {}
        scenarios_failed: list[str] = []

        config_overrides: dict = {}

        for scenario_dict in scenarios.to_dict("records"):
            row = pd.Series(scenario_dict)
            scenario_name = row["scenario_name"]
            logger.info(f"Running scenario: {scenario_name}")
            try:
                fc_df, m_df = run_scenario(row, prices, returns, config_overrides)
                succeeded = fc_df["ticker"].unique().tolist() if len(fc_df) else []
                tickers_filter = _parse_tickers(row.get("tickers", "all"), prices)
                failed_tickers = [t for t in tickers_filter if t not in succeeded]

                avg_rmse: float | None = None
                avg_coverage: float | None = None
                if len(m_df):
                    rmse_rows = m_df[m_df["metric_name"] == "rmse"]["value"]
                    cov_rows = m_df[m_df["metric_name"] == "coverage_rate"]["value"]
                    avg_rmse = float(rmse_rows.mean()) if len(rmse_rows) else None
                    avg_coverage = float(cov_rows.mean()) if len(cov_rows) else None

                scenarios_run[scenario_name] = {
                    "tickers_succeeded": len(succeeded),
                    "tickers_failed": failed_tickers,
                    "avg_rmse": avg_rmse,
                    "avg_coverage_rate": avg_coverage,
                }

                all_forecasts.append(fc_df)
                all_metrics.append(m_df)
            except Exception as exc:
                logger.error(f"Scenario {scenario_name!r} failed entirely: {exc}")
                scenarios_failed.append(scenario_name)

        forecasts_combined = (
            pd.concat(all_forecasts, ignore_index=True)
            if all_forecasts
            else _empty_forecasts_df()
        )
        metrics_combined = (
            pd.concat(all_metrics, ignore_index=True)
            if all_metrics
            else _empty_metrics_df()
        )

        best_model_per_ticker = _find_best_models(metrics_combined)
        tickers_forecasted = (
            int(forecasts_combined["ticker"].nunique()) if len(forecasts_combined) else 0
        )

        stat_ret = stationarity[stationarity["series_type"] == "returns"]
        stat_lp = stationarity[stationarity["series_type"] == "log_prices"]
        date_range = _date_range_str(prices)

        summary = {
            "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {
                "prices_rows": len(prices),
                "returns_rows": len(returns),
                "tickers": int(prices["ticker"].nunique()),
                "scenarios_loaded": len(scenarios),
                "date_range": date_range,
            },
            "stationarity": {
                "returns_stationary_count": int(stat_ret["is_stationary"].sum()),
                "log_prices_stationary_count": int(stat_lp["is_stationary"].sum()),
            },
            "scenarios_run": scenarios_run,
            "best_model_per_ticker": best_model_per_ticker,
            "config_used": {
                "train_initial_days": config.TRAIN_INITIAL_DAYS,
                "walk_forward_step_days": config.WALK_FORWARD_STEP_DAYS,
                "expanding_window": config.WALK_FORWARD_EXPANDING,
                "confidence_level": config.FORECAST_CONFIDENCE_LEVEL,
                # random_seed intentionally included for reproducibility documentation
                # Do NOT mirror API keys here if they are added to config.py in the future
                "random_seed": config.RANDOM_SEED,
            },
            "duration_sec": round(time.time() - start_time, 2),
        }

        save_forecasts(
            forecasts_combined,
            metrics_combined,
            stationarity,
            summary,
            config.PROCESSED_DATA_DIR,
            config.REPORTS_DIR,
        )

        result = {
            "scenarios_run": len(scenarios_run),
            "tickers_forecasted": tickers_forecasted,
            "scenarios_failed": scenarios_failed,
            "best_model_per_ticker": {k: v["model"] for k, v in best_model_per_ticker.items()},
            "total_forecasts": len(forecasts_combined),
            "duration_sec": round(time.time() - start_time, 2),
        }
        logger.info(f"Forecasting complete: {result}")
        return result

    finally:
        logger.remove(log_id)


# ── Private helpers ───────────────────────────────────────────────────────────


def _z_score(p: float) -> float:
    """Approximate normal quantile via scipy if available, else hardcode common values."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        if abs(p - 0.975) < 1e-6:
            return 1.96
        if abs(p - 0.995) < 1e-6:
            return 2.576
        logger.warning(
            f"scipy unavailable; _z_score using fallback 1.96 for p={p:.4f}. "
            "Install scipy for non-standard confidence levels."
        )
        return 1.96


def _get_series(
    ticker: str,
    target: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (series, dates) for the ticker and target type."""
    if target == "returns":
        grp = returns[returns["ticker"] == ticker].sort_values("date")
        if grp.empty:
            return None, None
        return (
            grp["simple_return"].reset_index(drop=True),
            grp["date"].reset_index(drop=True),
        )
    elif target == "prices":
        grp = prices[prices["ticker"] == ticker].sort_values("date")
        if grp.empty:
            return None, None
        log_prices = np.log(grp["close"].values)
        return pd.Series(log_prices), grp["date"].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown target: {target!r}. Must be 'returns' or 'prices'.")


def _parse_tickers(tickers_val: object, prices: pd.DataFrame) -> list[str]:
    """Parse the tickers column from scenarios.csv."""
    all_tickers = sorted(prices["ticker"].unique().tolist())
    if (
        pd.isna(tickers_val)
        or str(tickers_val).strip().lower() == "all"
        or str(tickers_val).strip() == ""
    ):
        return all_tickers
    requested = [t.strip().upper() for t in str(tickers_val).split(",")]
    valid, invalid = [], []
    for t in requested:
        if t in all_tickers:
            valid.append(t)
        else:
            invalid.append(t)
    if invalid:
        logger.warning(f"Tickers not found in data; skipping: {invalid}")
    return valid


def _build_model_fn(
    model: str, confidence_level: float, horizon: int
) -> tuple[Callable, dict]:
    """Return (callable, kwargs) for a model name string."""
    if model in ("arima_returns", "arima_log_prices"):
        return fit_arima, {
            "max_p": config.ARIMA_MAX_P,
            "max_q": config.ARIMA_MAX_Q,
            "max_d": config.ARIMA_MAX_D,
            "seasonal": config.ARIMA_SEASONAL,
            "horizon": horizon,
            "confidence_level": confidence_level,
        }
    elif model == "prophet":
        def _prophet_fn(series: pd.Series, train_dates: pd.Series | None = None, **kwargs) -> dict:
            if train_dates is None:
                logger.warning(
                    "Prophet received no train_dates; synthesising from today "
                    "— walk-forward results may be incorrect."
                )
                train_dates = pd.Series(
                    pd.bdate_range(end=pd.Timestamp.today(), periods=len(series), freq="B")
                )
            return fit_prophet(
                series,
                train_dates,
                yearly=kwargs.get("yearly", config.PROPHET_YEARLY_SEASONALITY),
                weekly=kwargs.get("weekly", config.PROPHET_WEEKLY_SEASONALITY),
                daily=kwargs.get("daily", config.PROPHET_DAILY_SEASONALITY),
                horizon=kwargs.get("horizon", horizon),
                confidence_level=kwargs.get("confidence_level", confidence_level),
            )
        return _prophet_fn, {
            "yearly": config.PROPHET_YEARLY_SEASONALITY,
            "weekly": config.PROPHET_WEEKLY_SEASONALITY,
            "daily": config.PROPHET_DAILY_SEASONALITY,
            "horizon": horizon,
            "confidence_level": confidence_level,
        }
    elif model == "naive_random_walk":
        return fit_naive, {
            "method": "random_walk",
            "horizon": horizon,
            "confidence_level": confidence_level,
        }
    elif model == "naive_drift":
        return fit_naive, {
            "method": "drift",
            "horizon": horizon,
            "confidence_level": confidence_level,
        }
    elif model == "naive_mean":
        return fit_naive, {
            "method": "mean",
            "horizon": horizon,
            "confidence_level": confidence_level,
        }
    else:
        raise ValueError(f"Unknown model: {model!r}")


def _build_forecast_df(
    scenario_name: str,
    ticker: str,
    model: str,
    target: str,
    confidence_level: float,
    forecast_dates: pd.DatetimeIndex,
    final_result: dict,
    fc_len: int,
) -> pd.DataFrame:
    """Build the forecast output DataFrame for one ticker-scenario."""
    return pd.DataFrame(
        {
            "scenario_name": pd.array([scenario_name] * fc_len, dtype=pd.StringDtype()),
            "ticker": pd.array([ticker] * fc_len, dtype=pd.StringDtype()),
            "model_type": pd.array([model] * fc_len, dtype=pd.StringDtype()),
            "target": pd.array([target] * fc_len, dtype=pd.StringDtype()),
            "forecast_date": forecast_dates,
            "forecast": final_result["forecast"][:fc_len].astype(float),
            "lower_ci": final_result["lower_ci"][:fc_len].astype(float),
            "upper_ci": final_result["upper_ci"][:fc_len].astype(float),
            "confidence_level": float(confidence_level),
        }
    )


def _find_best_models(metrics: pd.DataFrame) -> dict:
    """Return best model per ticker by lowest RMSE."""
    if len(metrics) == 0:
        return {}
    rmse_df = metrics[metrics["metric_name"] == "rmse"].copy()
    if len(rmse_df) == 0:
        return {}

    rw_rmse = (
        rmse_df[rmse_df["model_type"] == "naive_random_walk"]
        .set_index("ticker")["value"]
        .to_dict()
    )

    result = {}
    for ticker, grp in rmse_df.groupby("ticker"):
        best_row = grp.loc[grp["value"].idxmin()]
        best_model = best_row["model_type"]
        best_rmse = float(best_row["value"])
        rw = rw_rmse.get(ticker)
        beats_rw = None if (rw is None or best_model == "naive_random_walk") else bool(best_rmse < rw)
        result[ticker] = {"model": best_model, "rmse": best_rmse, "beats_random_walk": beats_rw}
    return result


def _date_range_str(prices: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(prices["date"])
    return [str(dates.min().date()), str(dates.max().date())]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    logger.info(f"Wrote {len(df)} rows → {path}")


def _empty_forecasts_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario_name", "ticker", "model_type", "target",
            "forecast_date", "forecast", "lower_ci", "upper_ci", "confidence_level",
        ]
    )


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["scenario_name", "ticker", "model_type", "metric_name", "value"]
    )
