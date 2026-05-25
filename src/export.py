"""
export.py — Step 7: Export pipeline outputs to CSV, Excel, and pipeline_summary.json.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

import config


def run_export() -> dict:
    """Consolidate all pipeline outputs into CSV snapshots, Excel workbook, and a master JSON."""
    t0 = time.perf_counter()
    log_path = config.LOG_DIR / f"export_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_id = logger.add(log_path, level="DEBUG")

    csv_files_written = 0
    excel_written = False
    pipeline_summary_written = False
    outputs_written: list[str] = []

    try:
        config.EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if config.EXPORT_CSV:
            csv_count, csv_paths = _write_csv_exports()
            csv_files_written = csv_count
            outputs_written.extend(csv_paths)

        if config.EXPORT_EXCEL:
            excel_path = _write_excel_report()
            if excel_path is not None:
                excel_written = True
                outputs_written.append(str(excel_path))

        # Capture duration once; pass the same value into the summary JSON so both are consistent.
        duration_sec = round(time.perf_counter() - t0, 3)
        summary_path = _write_pipeline_summary(csv_files_written, excel_written, duration_sec)
        if summary_path is not None:
            pipeline_summary_written = True
            outputs_written.append(str(summary_path))

        logger.info(
            f"Export complete — {csv_files_written} CSVs written, "
            f"excel={excel_written}, summary={pipeline_summary_written}, "
            f"duration={duration_sec}s"
        )
        return {
            "csv_files_written": csv_files_written,
            "excel_written": excel_written,
            "pipeline_summary_written": pipeline_summary_written,
            "outputs_written": outputs_written,
            "duration_sec": duration_sec,
        }
    finally:
        logger.remove(log_id)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _json_default(obj):
    """Narrow JSON serialisation fallback — raises on unrecognised types instead of silently coercing."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _load_parquet(name: str) -> pd.DataFrame | None:
    path = config.PROCESSED_DATA_DIR / name
    if not path.exists():
        logger.warning(f"Missing upstream file: {path} — skipping")
        return None
    return pd.read_parquet(path)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning(f"Missing upstream file: {path} — skipping")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── CSV exports ────────────────────────────────────────────────────────────────


def _write_csv_exports() -> tuple[int, list[str]]:
    csv_map = [
        ("prices_clean.csv", "prices_clean.parquet"),
        ("returns_daily.csv", "returns_daily.parquet"),
        ("metrics_per_ticker.csv", "metrics_per_ticker.parquet"),
        ("portfolio_metrics.csv", "portfolio_metrics.parquet"),
        ("forecasts.csv", "forecasts.parquet"),
        ("mc_metrics.csv", "mc_metrics.parquet"),
    ]
    written = 0
    paths: list[str] = []
    for csv_name, parquet_name in csv_map:
        df = _load_parquet(parquet_name)
        if df is None:
            continue
        out = config.EXPORTS_DATA_DIR / csv_name
        try:
            df.to_csv(out, index=False)
        except Exception as exc:
            logger.warning(f"Failed to write {out} ({exc}) — skipping")
            continue
        logger.debug(f"Wrote {out} ({len(df):,} rows)")
        written += 1
        paths.append(str(out))
    return written, paths


# ── Excel export ───────────────────────────────────────────────────────────────

_EXCEL_DATA_PARQUETS = [
    "portfolio_metrics.parquet",
    "metrics_per_ticker.parquet",
    "drawdown_series.parquet",
    "forecasts.parquet",
    "mc_metrics.parquet",
]


def _write_excel_report() -> Path | None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        logger.warning("openpyxl not installed — skipping Excel export")
        return None

    if not any((config.PROCESSED_DATA_DIR / p).exists() for p in _EXCEL_DATA_PARQUETS):
        logger.warning("No upstream data available for Excel export — skipping")
        return None

    out = config.EXPORTS_DATA_DIR / "portfolio_report.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        _sheet_as_is(writer, "Portfolio Metrics", "portfolio_metrics.parquet")
        _sheet_pivoted(
            writer,
            "Per-Ticker Metrics",
            "metrics_per_ticker.parquet",
            index="ticker",
            columns="metric_name",
            values="value",
        )
        _sheet_drawdown_summary(writer)
        _sheet_as_is(writer, "Forecasts", "forecasts.parquet")
        _sheet_pivoted(
            writer,
            "MC Risk",
            "mc_metrics.parquet",
            index=["scenario_name", "ticker", "method"],
            columns="metric_name",
            values="value",
        )
        _sheet_config(writer)

    logger.info(f"Wrote Excel workbook: {out}")
    return out


def _sheet_as_is(writer: pd.ExcelWriter, sheet: str, parquet_name: str) -> None:
    df = _load_parquet(parquet_name)
    if df is None:
        return
    df.to_excel(writer, sheet_name=sheet, index=False)
    logger.debug(f"Excel sheet '{sheet}': {len(df):,} rows")


def _sheet_pivoted(
    writer: pd.ExcelWriter,
    sheet: str,
    parquet_name: str,
    index: str | list[str],
    columns: str,
    values: str,
) -> None:
    df = _load_parquet(parquet_name)
    if df is None:
        return
    try:
        pivoted = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
        pivoted.columns.name = None
        pivoted = pivoted.reset_index()
        pivoted.to_excel(writer, sheet_name=sheet, index=False)
        logger.debug(f"Excel sheet '{sheet}': pivoted {len(pivoted):,} rows")
    except Exception as exc:
        logger.warning(f"Pivot failed for sheet '{sheet}' ({exc}) — writing long format")
        df.to_excel(writer, sheet_name=sheet, index=False)


def _sheet_drawdown_summary(writer: pd.ExcelWriter) -> None:
    df = _load_parquet("drawdown_series.parquet")
    if df is None:
        return
    try:
        summary = (
            df.groupby("ticker")["drawdown_pct"]
            .agg(max_drawdown="min", mean_drawdown="mean")
            .reset_index()
        )
        summary.to_excel(writer, sheet_name="Drawdown Summary", index=False)
        logger.debug(f"Excel sheet 'Drawdown Summary': {len(summary):,} rows")
    except Exception as exc:
        logger.warning(f"Drawdown summary failed ({exc}) — writing long format")
        df.to_excel(writer, sheet_name="Drawdown Summary", index=False)


def _sheet_config(writer: pd.ExcelWriter) -> None:
    rows = [
        ("tickers", ", ".join(config.TICKERS)),
        ("date_start", config.DATE_START),
        ("date_end", config.DATE_END),
        ("risk_free_rate", config.RISK_FREE_RATE),
        ("benchmark_ticker", str(config.BENCHMARK_TICKER)),
        ("portfolio_weights", str(config.PORTFOLIO_WEIGHTS)),
        ("export_csv", config.EXPORT_CSV),
        ("export_excel", config.EXPORT_EXCEL),
    ]
    pd.DataFrame(rows, columns=["parameter", "value"]).to_excel(
        writer, sheet_name="Config", index=False
    )
    logger.debug("Excel sheet 'Config': written")


# ── pipeline_summary.json ──────────────────────────────────────────────────────


def _write_pipeline_summary(
    csv_files_written: int, excel_written: bool, duration_sec: float
) -> Path | None:
    step_files = {
        "step_2_cleaning": config.PROCESSED_DATA_DIR / "cleaning_report.json",
        "step_3_eda": config.REPORTS_DIR / "eda_summary.json",
        "step_4_metrics": config.REPORTS_DIR / "metrics_summary.json",
        "step_5_forecasting": config.REPORTS_DIR / "forecasting_summary.json",
        "step_6_monte_carlo": config.REPORTS_DIR / "monte_carlo_summary.json",
    }

    steps_data: dict = {}
    for step_key, json_path in step_files.items():
        data = _load_json(json_path)
        if data is not None:
            steps_data[step_key] = data

    summary = {
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": config.PIPELINE_VERSION,
        "date_range": [config.DATE_START, config.DATE_END],
        "tickers": config.TICKERS,
        "steps": steps_data,
        "config_snapshot": {
            "tickers": config.TICKERS,
            "date_start": config.DATE_START,
            "date_end": config.DATE_END,
            "risk_free_rate": config.RISK_FREE_RATE,
            "benchmark_ticker": config.BENCHMARK_TICKER,
            "portfolio_weights": config.PORTFOLIO_WEIGHTS,
        },
        "export": {
            "csv_files_written": csv_files_written,
            "excel_written": excel_written,
            "duration_sec": duration_sec,
        },
    }

    out = config.REPORTS_DIR / "pipeline_summary.json"
    try:
        with out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        logger.info(f"Wrote pipeline summary: {out}")
        return out
    except Exception as exc:
        logger.warning(f"Failed to write pipeline summary ({exc}) — skipping")
        return None
