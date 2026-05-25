"""
tests/unit/test_07-export.py — Unit tests for src/export.py (Step 7).

All test logic is derived exclusively from the spec at
.claude/specs/07-export.md. No implementation details are assumed
beyond the public API ``run_export() -> dict`` and the output
contracts documented in the spec.

Coverage:
  - Happy path: valid upstream files produce correct outputs
  - Return dict: all 5 required keys with correct types
  - CSV exports: 6 files written, contents round-trip, paths in outputs_written
  - Excel export: file created, all 6 required sheet names present, Config sheet structure
  - pipeline_summary.json: top-level keys, steps keys, config_snapshot, export sub-object
  - Missing upstream files: warn + skip, never raise
  - Config knobs: EXPORT_CSV=False, EXPORT_EXCEL=False
  - openpyxl absent: graceful skip, excel_written=False
  - Idempotency: second run overwrites, no row duplication
  - Duration: positive float
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import config


# ---------------------------------------------------------------------------
# Module-level fixture helpers
# ---------------------------------------------------------------------------

def _make_prices_df() -> pd.DataFrame:
    """Minimal prices_clean long-format DataFrame (2 tickers, 5 days)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1_000_000,
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["volume"] = df["volume"].astype(pd.Int64Dtype())
    return df


def _make_returns_df() -> pd.DataFrame:
    """Minimal returns long-format DataFrame."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "simple_return": 0.01,
                    "log_return": 0.00995,
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    return df


def _make_metrics_per_ticker_df() -> pd.DataFrame:
    """Minimal metrics_per_ticker long-format DataFrame."""
    rows = [
        {"ticker": "AAPL", "metric_name": "total_return", "value": 0.15, "category": "return"},
        {"ticker": "MSFT", "metric_name": "total_return", "value": 0.12, "category": "return"},
        {"ticker": "AAPL", "metric_name": "volatility", "value": 0.20, "category": "risk"},
        {"ticker": "MSFT", "metric_name": "volatility", "value": 0.18, "category": "risk"},
    ]
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    df["category"] = df["category"].astype(pd.StringDtype())
    return df


def _make_portfolio_metrics_df() -> pd.DataFrame:
    """Minimal portfolio_metrics single-row wide DataFrame."""
    return pd.DataFrame(
        [{"sharpe_ratio": 1.2, "sortino_ratio": 1.5, "calmar_ratio": 0.8,
          "var_95": 0.03, "cvar_95": 0.04, "max_drawdown": -0.15}]
    )


def _make_forecasts_df() -> pd.DataFrame:
    """Minimal forecasts long-format DataFrame."""
    rows = [
        {
            "scenario_name": "base",
            "ticker": "AAPL",
            "model_type": "naive_random_walk",
            "target": "returns",
            "forecast_date": pd.Timestamp("2024-02-01"),
            "forecast": 0.005,
            "lower_ci": -0.01,
            "upper_ci": 0.02,
            "confidence_level": 0.95,
        }
    ]
    df = pd.DataFrame(rows)
    df["scenario_name"] = df["scenario_name"].astype(pd.StringDtype())
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["model_type"] = df["model_type"].astype(pd.StringDtype())
    df["target"] = df["target"].astype(pd.StringDtype())
    return df


def _make_mc_metrics_df() -> pd.DataFrame:
    """Minimal mc_metrics long-format DataFrame."""
    rows = [
        {
            "scenario_name": "test_gbm",
            "ticker": "AAPL",
            "method": "gbm",
            "metric_name": "var_95",
            "value": 0.08,
        },
        {
            "scenario_name": "test_gbm",
            "ticker": "MSFT",
            "method": "gbm",
            "metric_name": "var_95",
            "value": 0.07,
        },
    ]
    df = pd.DataFrame(rows)
    for col in ("scenario_name", "ticker", "method", "metric_name"):
        df[col] = df[col].astype(pd.StringDtype())
    return df


def _make_rolling_metrics_df() -> pd.DataFrame:
    """Minimal rolling_metrics long-format DataFrame (Excel-only)."""
    rows = [
        {"date": pd.Timestamp("2024-01-10"), "ticker": "AAPL",
         "metric_name": "rolling_sharpe_90", "value": 1.1},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    df["metric_name"] = df["metric_name"].astype(pd.StringDtype())
    return df


def _make_drawdown_series_df() -> pd.DataFrame:
    """Minimal drawdown_series DataFrame (Excel-only)."""
    rows = [
        {"date": pd.Timestamp("2024-01-02"), "ticker": "AAPL",
         "close": 101.0, "running_peak": 101.0, "drawdown_pct": 0.0},
        {"date": pd.Timestamp("2024-01-03"), "ticker": "AAPL",
         "close": 99.0, "running_peak": 101.0, "drawdown_pct": -0.0198},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(pd.StringDtype())
    return df


def _make_forecast_metrics_df() -> pd.DataFrame:
    """Minimal forecast_metrics DataFrame (Excel-only)."""
    rows = [
        {"scenario_name": "base", "ticker": "AAPL", "model_type": "naive_random_walk",
         "metric_name": "rmse", "value": 0.01},
    ]
    df = pd.DataFrame(rows)
    for col in ("scenario_name", "ticker", "model_type", "metric_name"):
        df[col] = df[col].astype(pd.StringDtype())
    return df


def _make_mc_terminal_df() -> pd.DataFrame:
    """Minimal mc_terminal_distribution DataFrame (Excel-only)."""
    rows = [
        {"scenario_name": "test_gbm", "ticker": "AAPL", "method": "gbm",
         "s0": 100.0, "terminal_mean": 105.0, "terminal_std": 10.0,
         "terminal_skew": 0.1, "terminal_kurtosis": 0.2},
    ]
    df = pd.DataFrame(rows)
    for col in ("scenario_name", "ticker", "method"):
        df[col] = df[col].astype(pd.StringDtype())
    return df


def _make_cleaning_report() -> dict:
    return {"duplicates_removed": 0, "rows_dropped": 2, "gaps_filled": 5, "outliers_flagged": 1}


def _make_eda_summary() -> dict:
    return {"plot_count": 12, "tickers_analysed": ["AAPL", "MSFT"]}


def _make_metrics_summary() -> dict:
    return {"per_ticker": {"AAPL": {"sharpe": 1.2}}, "run_timestamp": "2024-01-01T00:00:00Z"}


def _make_forecasting_summary() -> dict:
    return {"scenarios": ["base"], "run_timestamp": "2024-01-01T00:00:00Z"}


def _make_monte_carlo_summary() -> dict:
    return {"scenarios": ["test_gbm"], "run_timestamp": "2024-01-01T00:00:00Z"}


# ---------------------------------------------------------------------------
# Fixture: full upstream environment in tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def upstream_env(tmp_path, monkeypatch):
    """
    Creates all upstream parquet and JSON files under tmp_path and
    monkeypatches config paths so run_export() reads/writes exclusively
    within tmp_path. Returns a dict of the directories for inspection.
    """
    processed = tmp_path / "processed"
    exports = tmp_path / "exports"
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    for d in (processed, exports, reports, logs):
        d.mkdir(parents=True, exist_ok=True)

    # Write all upstream parquets
    parquet_kwargs = {"engine": "pyarrow", "index": False, "compression": "snappy"}

    _make_prices_df().to_parquet(processed / "prices_clean.parquet", **parquet_kwargs)
    _make_returns_df().to_parquet(processed / "returns_daily.parquet", **parquet_kwargs)
    _make_metrics_per_ticker_df().to_parquet(
        processed / "metrics_per_ticker.parquet", **parquet_kwargs
    )
    _make_portfolio_metrics_df().to_parquet(
        processed / "portfolio_metrics.parquet", **parquet_kwargs
    )
    _make_rolling_metrics_df().to_parquet(
        processed / "rolling_metrics.parquet", **parquet_kwargs
    )
    _make_drawdown_series_df().to_parquet(
        processed / "drawdown_series.parquet", **parquet_kwargs
    )
    _make_forecasts_df().to_parquet(processed / "forecasts.parquet", **parquet_kwargs)
    _make_forecast_metrics_df().to_parquet(
        processed / "forecast_metrics.parquet", **parquet_kwargs
    )
    _make_mc_metrics_df().to_parquet(processed / "mc_metrics.parquet", **parquet_kwargs)
    _make_mc_terminal_df().to_parquet(
        processed / "mc_terminal_distribution.parquet", **parquet_kwargs
    )

    # Write all upstream JSON reports
    (processed / "cleaning_report.json").write_text(
        json.dumps(_make_cleaning_report()), encoding="utf-8"
    )
    (reports / "eda_summary.json").write_text(
        json.dumps(_make_eda_summary()), encoding="utf-8"
    )
    (reports / "metrics_summary.json").write_text(
        json.dumps(_make_metrics_summary()), encoding="utf-8"
    )
    (reports / "forecasting_summary.json").write_text(
        json.dumps(_make_forecasting_summary()), encoding="utf-8"
    )
    (reports / "monte_carlo_summary.json").write_text(
        json.dumps(_make_monte_carlo_summary()), encoding="utf-8"
    )

    # Monkeypatch config paths
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "LOG_DIR", logs)
    monkeypatch.setattr(config, "EXPORT_CSV", True)
    monkeypatch.setattr(config, "EXPORT_EXCEL", True)

    # Force re-import so the module picks up patched config references
    import src.export as export_mod
    importlib.reload(export_mod)

    return {
        "processed": processed,
        "exports": exports,
        "reports": reports,
        "logs": logs,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Helper: run run_export() with the given fixture env loaded
# ---------------------------------------------------------------------------

def _run(monkeypatch) -> dict:
    """Re-import and call run_export after monkeypatching."""
    import src.export as export_mod
    importlib.reload(export_mod)
    return export_mod.run_export()


# ---------------------------------------------------------------------------
# Class: TestReturnDict
# ---------------------------------------------------------------------------

class TestReturnDict:
    """Validate the shape and types of the dict returned by run_export()."""

    def test_run_export_returns_all_five_required_keys(self, upstream_env, monkeypatch):
        """run_export() must return a dict with exactly the 5 spec-defined keys."""
        result = _run(monkeypatch)
        required_keys = {
            "csv_files_written",
            "excel_written",
            "pipeline_summary_written",
            "outputs_written",
            "duration_sec",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Return dict missing keys: {missing}"

    def test_run_export_csv_files_written_is_int(self, upstream_env, monkeypatch):
        """csv_files_written must be an int — spec §2."""
        result = _run(monkeypatch)
        assert isinstance(result["csv_files_written"], int), (
            f"csv_files_written must be int, got {type(result['csv_files_written'])}"
        )

    def test_run_export_excel_written_is_bool(self, upstream_env, monkeypatch):
        """excel_written must be a bool — spec §2."""
        result = _run(monkeypatch)
        assert isinstance(result["excel_written"], bool), (
            f"excel_written must be bool, got {type(result['excel_written'])}"
        )

    def test_run_export_pipeline_summary_written_is_bool(self, upstream_env, monkeypatch):
        """pipeline_summary_written must be a bool — spec §2."""
        result = _run(monkeypatch)
        assert isinstance(result["pipeline_summary_written"], bool), (
            f"pipeline_summary_written must be bool, got "
            f"{type(result['pipeline_summary_written'])}"
        )

    def test_run_export_outputs_written_is_list(self, upstream_env, monkeypatch):
        """outputs_written must be a list — spec §2."""
        result = _run(monkeypatch)
        assert isinstance(result["outputs_written"], list), (
            f"outputs_written must be list, got {type(result['outputs_written'])}"
        )

    def test_run_export_duration_sec_is_positive_float(self, upstream_env, monkeypatch):
        """duration_sec must be a non-negative float — spec §2."""
        result = _run(monkeypatch)
        assert isinstance(result["duration_sec"], float), (
            f"duration_sec must be float, got {type(result['duration_sec'])}"
        )
        assert result["duration_sec"] >= 0.0, (
            f"duration_sec must be >= 0, got {result['duration_sec']}"
        )


# ---------------------------------------------------------------------------
# Class: TestCsvExports
# ---------------------------------------------------------------------------

class TestCsvExports:
    """Validate CSV file creation, contents, and count — spec §4 CSV section."""

    # Expected CSV outputs defined by the spec
    EXPECTED_CSV_NAMES = [
        "prices_clean.csv",
        "returns_daily.csv",
        "metrics_per_ticker.csv",
        "portfolio_metrics.csv",
        "forecasts.csv",
        "mc_metrics.csv",
    ]

    def test_csv_files_written_count_is_six(self, upstream_env, monkeypatch):
        """csv_files_written must equal 6 when all 6 upstream parquets are present — spec §4."""
        result = _run(monkeypatch)
        assert result["csv_files_written"] == 6, (
            f"Expected 6 CSV files written, got {result['csv_files_written']}"
        )

    @pytest.mark.parametrize("csv_name", [
        "prices_clean.csv",
        "returns_daily.csv",
        "metrics_per_ticker.csv",
        "portfolio_metrics.csv",
        "forecasts.csv",
        "mc_metrics.csv",
    ])
    def test_csv_file_exists_for_each_expected_name(
        self, upstream_env, monkeypatch, csv_name
    ):
        """Each expected CSV file must exist in data/exports/ — spec §4."""
        _run(monkeypatch)
        out_path = upstream_env["exports"] / csv_name
        assert out_path.exists(), (
            f"Expected CSV file not found: {out_path}"
        )

    def test_prices_clean_csv_round_trip_row_count(self, upstream_env, monkeypatch):
        """prices_clean.csv must contain the same row count as prices_clean.parquet — spec §4."""
        _run(monkeypatch)
        original = pd.read_parquet(upstream_env["processed"] / "prices_clean.parquet")
        written = pd.read_csv(upstream_env["exports"] / "prices_clean.csv")
        assert len(written) == len(original), (
            f"prices_clean.csv row count {len(written)} != parquet row count {len(original)}"
        )

    def test_returns_daily_csv_round_trip_row_count(self, upstream_env, monkeypatch):
        """returns_daily.csv must contain the same row count as returns_daily.parquet — spec §4."""
        _run(monkeypatch)
        original = pd.read_parquet(upstream_env["processed"] / "returns_daily.parquet")
        written = pd.read_csv(upstream_env["exports"] / "returns_daily.csv")
        assert len(written) == len(original), (
            f"returns_daily.csv row count {len(written)} != parquet row count {len(original)}"
        )

    def test_csv_files_in_outputs_written_list(self, upstream_env, monkeypatch):
        """All written CSV file paths must appear in outputs_written — spec §2."""
        result = _run(monkeypatch)
        outputs = result["outputs_written"]
        for csv_name in self.EXPECTED_CSV_NAMES:
            expected_path = str(upstream_env["exports"] / csv_name)
            assert expected_path in outputs, (
                f"CSV path '{expected_path}' not found in outputs_written: {outputs}"
            )

    def test_csv_export_false_writes_no_csv_files(self, upstream_env, monkeypatch):
        """No CSV files must be written when EXPORT_CSV=False — spec §3 config knob."""
        monkeypatch.setattr(config, "EXPORT_CSV", False)
        _run(monkeypatch)
        for csv_name in self.EXPECTED_CSV_NAMES:
            path = upstream_env["exports"] / csv_name
            assert not path.exists(), (
                f"CSV file '{csv_name}' was written even though EXPORT_CSV=False"
            )

    def test_csv_export_false_returns_zero_count(self, upstream_env, monkeypatch):
        """csv_files_written must be 0 when EXPORT_CSV=False — spec §3 config knob."""
        monkeypatch.setattr(config, "EXPORT_CSV", False)
        result = _run(monkeypatch)
        assert result["csv_files_written"] == 0, (
            f"Expected csv_files_written=0 when EXPORT_CSV=False, "
            f"got {result['csv_files_written']}"
        )

    def test_missing_parquet_skips_that_csv_no_exception(
        self, tmp_path, monkeypatch
    ):
        """A missing upstream parquet must produce a warning+skip, not an exception — spec §4."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        logs = tmp_path / "logs"
        for d in (processed, exports, reports, logs):
            d.mkdir(parents=True, exist_ok=True)

        # Only write ONE parquet — all others missing
        _make_prices_df().to_parquet(
            processed / "prices_clean.parquet",
            engine="pyarrow", index=False, compression="snappy"
        )

        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
        monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
        monkeypatch.setattr(config, "REPORTS_DIR", reports)
        monkeypatch.setattr(config, "LOG_DIR", logs)
        monkeypatch.setattr(config, "EXPORT_CSV", True)
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)

        import src.export as export_mod
        importlib.reload(export_mod)

        # Must not raise
        result = export_mod.run_export()
        assert result["csv_files_written"] == 1, (
            f"Expected 1 CSV written (only prices_clean.parquet present), "
            f"got {result['csv_files_written']}"
        )
        assert (exports / "prices_clean.csv").exists(), (
            "prices_clean.csv must exist (its source parquet was present)"
        )

    def test_all_upstream_parquets_missing_returns_zero_csvs(
        self, tmp_path, monkeypatch
    ):
        """Zero CSVs when no upstream parquets exist — spec missing-file skip."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        logs = tmp_path / "logs"
        for d in (processed, exports, reports, logs):
            d.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
        monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
        monkeypatch.setattr(config, "REPORTS_DIR", reports)
        monkeypatch.setattr(config, "LOG_DIR", logs)
        monkeypatch.setattr(config, "EXPORT_CSV", True)
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)

        import src.export as export_mod
        importlib.reload(export_mod)

        result = export_mod.run_export()
        assert result["csv_files_written"] == 0, (
            f"Expected 0 CSVs when all parquets missing, got {result['csv_files_written']}"
        )

    def test_csv_no_index_column(self, upstream_env, monkeypatch):
        """CSVs must be written without a row index column — spec §4 (index=False)."""
        _run(monkeypatch)
        df = pd.read_csv(upstream_env["exports"] / "prices_clean.csv")
        # If index was written, there would be an 'Unnamed: 0' column
        assert "Unnamed: 0" not in df.columns, (
            "prices_clean.csv has an index column — to_csv(index=False) not used."
        )


# ---------------------------------------------------------------------------
# Class: TestExcelExport
# ---------------------------------------------------------------------------

class TestExcelExport:
    """Validate Excel workbook creation and sheet contents — spec §4 Excel section."""

    EXPECTED_SHEETS = [
        "Portfolio Metrics",
        "Per-Ticker Metrics",
        "Drawdown Summary",
        "Forecasts",
        "MC Risk",
        "Config",
    ]

    def test_excel_written_true_when_openpyxl_present(self, upstream_env, monkeypatch):
        """excel_written must be True after a successful Excel run — spec §4."""
        pytest.importorskip("openpyxl")
        result = _run(monkeypatch)
        assert result["excel_written"] is True, (
            f"excel_written should be True when openpyxl is installed, "
            f"got {result['excel_written']}"
        )

    def test_excel_file_exists_after_run(self, upstream_env, monkeypatch):
        """portfolio_report.xlsx must exist in data/exports/ — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        xlsx_path = upstream_env["exports"] / "portfolio_report.xlsx"
        assert xlsx_path.exists(), (
            f"Expected Excel workbook not found: {xlsx_path}"
        )

    def test_excel_path_in_outputs_written(self, upstream_env, monkeypatch):
        """portfolio_report.xlsx path must appear in outputs_written — spec §2."""
        pytest.importorskip("openpyxl")
        result = _run(monkeypatch)
        xlsx_path = str(upstream_env["exports"] / "portfolio_report.xlsx")
        assert xlsx_path in result["outputs_written"], (
            f"xlsx path '{xlsx_path}' not in outputs_written: {result['outputs_written']}"
        )

    @pytest.mark.parametrize("sheet_name", [
        "Portfolio Metrics",
        "Per-Ticker Metrics",
        "Drawdown Summary",
        "Forecasts",
        "MC Risk",
        "Config",
    ])
    def test_excel_has_required_sheet(self, upstream_env, monkeypatch, sheet_name):
        """Excel workbook must contain every spec-defined sheet — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        xlsx_path = upstream_env["exports"] / "portfolio_report.xlsx"
        xl = pd.ExcelFile(xlsx_path)
        assert sheet_name in xl.sheet_names, (
            f"Required sheet '{sheet_name}' not found in workbook. "
            f"Sheets present: {xl.sheet_names}"
        )

    def test_excel_config_sheet_has_parameter_and_value_columns(
        self, upstream_env, monkeypatch
    ):
        """Config sheet must have exactly 'parameter' and 'value' columns — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        df = pd.read_excel(
            upstream_env["exports"] / "portfolio_report.xlsx",
            sheet_name="Config",
        )
        assert "parameter" in df.columns, (
            f"Config sheet missing 'parameter' column. Columns: {list(df.columns)}"
        )
        assert "value" in df.columns, (
            f"Config sheet missing 'value' column. Columns: {list(df.columns)}"
        )

    def test_excel_config_sheet_contains_key_parameters(
        self, upstream_env, monkeypatch
    ):
        """Config sheet must include rows for tickers, date_start, date_end — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        df = pd.read_excel(
            upstream_env["exports"] / "portfolio_report.xlsx",
            sheet_name="Config",
        )
        params = set(df["parameter"].astype(str).tolist())
        for expected_param in ("tickers", "date_start", "date_end", "risk_free_rate"):
            assert expected_param in params, (
                f"Config sheet missing row for parameter '{expected_param}'. "
                f"Parameters present: {params}"
            )

    def test_excel_portfolio_metrics_sheet_has_data(self, upstream_env, monkeypatch):
        """Portfolio Metrics sheet must have at least one data row — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        df = pd.read_excel(
            upstream_env["exports"] / "portfolio_report.xlsx",
            sheet_name="Portfolio Metrics",
        )
        assert len(df) >= 1, (
            f"'Portfolio Metrics' sheet has no data rows, expected >= 1"
        )

    def test_excel_forecasts_sheet_has_data(self, upstream_env, monkeypatch):
        """Forecasts sheet must have at least one data row — spec §4."""
        pytest.importorskip("openpyxl")
        _run(monkeypatch)
        df = pd.read_excel(
            upstream_env["exports"] / "portfolio_report.xlsx",
            sheet_name="Forecasts",
        )
        assert len(df) >= 1, (
            f"'Forecasts' sheet has no data rows, expected >= 1"
        )

    def test_excel_export_false_no_xlsx_written(self, upstream_env, monkeypatch):
        """portfolio_report.xlsx must NOT be written when EXPORT_EXCEL=False — spec §3."""
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)
        _run(monkeypatch)
        xlsx_path = upstream_env["exports"] / "portfolio_report.xlsx"
        assert not xlsx_path.exists(), (
            "portfolio_report.xlsx was written even though EXPORT_EXCEL=False"
        )

    def test_excel_export_false_returns_false(self, upstream_env, monkeypatch):
        """excel_written must be False when EXPORT_EXCEL=False — spec §3 config knob."""
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)
        result = _run(monkeypatch)
        assert result["excel_written"] is False, (
            f"excel_written should be False when EXPORT_EXCEL=False, "
            f"got {result['excel_written']}"
        )

    def test_excel_missing_parquet_skips_gracefully(self, tmp_path, monkeypatch):
        """Excel export must not raise when upstream parquets are missing — spec §4."""
        pytest.importorskip("openpyxl")
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        logs = tmp_path / "logs"
        for d in (processed, exports, reports, logs):
            d.mkdir(parents=True, exist_ok=True)

        # Only portfolio_metrics present; all others missing
        _make_portfolio_metrics_df().to_parquet(
            processed / "portfolio_metrics.parquet",
            engine="pyarrow", index=False, compression="snappy"
        )

        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
        monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
        monkeypatch.setattr(config, "REPORTS_DIR", reports)
        monkeypatch.setattr(config, "LOG_DIR", logs)
        monkeypatch.setattr(config, "EXPORT_CSV", False)
        monkeypatch.setattr(config, "EXPORT_EXCEL", True)

        import src.export as export_mod
        importlib.reload(export_mod)

        # Must not raise even with many missing parquets
        result = export_mod.run_export()
        assert isinstance(result, dict), "run_export() must return a dict even with missing files"

    def test_excel_openpyxl_absent_graceful_skip(self, upstream_env, monkeypatch):
        """When openpyxl is not importable, Excel export is skipped — spec §4 final note."""
        # Sentinel: setting sys.modules["openpyxl"] = None causes `import openpyxl`
        # to raise ImportError, simulating the package being absent.
        original = sys.modules.get("openpyxl", None)
        sys.modules["openpyxl"] = None  # type: ignore[assignment]

        try:
            import src.export as export_mod
            importlib.reload(export_mod)
            result = export_mod.run_export()
            assert result["excel_written"] is False, (
                "excel_written must be False when openpyxl is unavailable, "
                f"got {result['excel_written']}"
            )
            xlsx_path = upstream_env["exports"] / "portfolio_report.xlsx"
            assert not xlsx_path.exists(), (
                "portfolio_report.xlsx was written even though openpyxl is unavailable"
            )
        finally:
            # Restore original sys.modules state
            if original is None:
                sys.modules.pop("openpyxl", None)
            else:
                sys.modules["openpyxl"] = original
            import src.export as export_mod
            importlib.reload(export_mod)


# ---------------------------------------------------------------------------
# Class: TestPipelineSummaryJson
# ---------------------------------------------------------------------------

class TestPipelineSummaryJson:
    """Validate pipeline_summary.json structure — spec §4 JSON section."""

    def test_pipeline_summary_written_is_true(self, upstream_env, monkeypatch):
        """pipeline_summary_written must be True after a successful run — spec §2."""
        result = _run(monkeypatch)
        assert result["pipeline_summary_written"] is True, (
            "pipeline_summary_written should be True when writing succeeded"
        )

    def test_pipeline_summary_file_exists(self, upstream_env, monkeypatch):
        """pipeline_summary.json must exist at outputs/reports/ — spec §4."""
        _run(monkeypatch)
        summary_path = upstream_env["reports"] / "pipeline_summary.json"
        assert summary_path.exists(), (
            f"pipeline_summary.json not found at {summary_path}"
        )

    def test_pipeline_summary_path_in_outputs_written(self, upstream_env, monkeypatch):
        """pipeline_summary.json path must appear in outputs_written — spec §2."""
        result = _run(monkeypatch)
        summary_path = str(upstream_env["reports"] / "pipeline_summary.json")
        assert summary_path in result["outputs_written"], (
            f"pipeline_summary.json path not in outputs_written: {result['outputs_written']}"
        )

    def test_pipeline_summary_top_level_keys(self, upstream_env, monkeypatch):
        """pipeline_summary.json must contain all spec-defined top-level keys — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = {
            "run_timestamp",
            "pipeline_version",
            "date_range",
            "tickers",
            "steps",
            "config_snapshot",
            "export",
        }
        missing = required_keys - set(data.keys())
        assert not missing, (
            f"pipeline_summary.json missing top-level keys: {missing}"
        )

    def test_pipeline_version_is_1_0(self, upstream_env, monkeypatch):
        """pipeline_version must be '1.0' — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["pipeline_version"] == "1.0", (
            f"pipeline_version must be '1.0', got '{data['pipeline_version']}'"
        )

    def test_pipeline_summary_steps_has_all_five_step_keys(
        self, upstream_env, monkeypatch
    ):
        """steps sub-object must contain all 5 step keys when all JSONs present — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        expected_step_keys = {
            "step_2_cleaning",
            "step_3_eda",
            "step_4_metrics",
            "step_5_forecasting",
            "step_6_monte_carlo",
        }
        missing = expected_step_keys - set(data["steps"].keys())
        assert not missing, (
            f"pipeline_summary.json steps missing keys: {missing}"
        )

    def test_pipeline_summary_step_contents_match_source_json(
        self, upstream_env, monkeypatch
    ):
        """Each step entry must contain the contents from the corresponding source JSON — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)

        # Verify step_2_cleaning content from cleaning_report.json
        assert data["steps"]["step_2_cleaning"]["duplicates_removed"] == 0, (
            "step_2_cleaning content does not match cleaning_report.json"
        )
        assert data["steps"]["step_3_eda"]["plot_count"] == 12, (
            "step_3_eda content does not match eda_summary.json"
        )

    def test_pipeline_summary_config_snapshot_keys(self, upstream_env, monkeypatch):
        """config_snapshot must contain the 6 spec-defined keys — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        expected_keys = {
            "tickers",
            "date_start",
            "date_end",
            "risk_free_rate",
            "benchmark_ticker",
            "portfolio_weights",
        }
        missing = expected_keys - set(data["config_snapshot"].keys())
        assert not missing, (
            f"config_snapshot missing keys: {missing}"
        )

    def test_pipeline_summary_date_range_matches_config(
        self, upstream_env, monkeypatch
    ):
        """date_range must equal [config.DATE_START, config.DATE_END] — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["date_range"] == [config.DATE_START, config.DATE_END], (
            f"date_range {data['date_range']} != "
            f"[{config.DATE_START}, {config.DATE_END}]"
        )

    def test_pipeline_summary_tickers_matches_config(
        self, upstream_env, monkeypatch
    ):
        """tickers must equal config.TICKERS — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["tickers"] == config.TICKERS, (
            f"tickers in summary {data['tickers']} != config.TICKERS {config.TICKERS}"
        )

    def test_pipeline_summary_export_section_keys(self, upstream_env, monkeypatch):
        """export sub-object must contain csv_files_written, excel_written, duration_sec — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        expected_export_keys = {"csv_files_written", "excel_written", "duration_sec"}
        missing = expected_export_keys - set(data["export"].keys())
        assert not missing, (
            f"export sub-object missing keys: {missing}"
        )

    def test_pipeline_summary_export_csv_count_matches_return_value(
        self, upstream_env, monkeypatch
    ):
        """export.csv_files_written in JSON must match the returned csv_files_written — spec §4."""
        result = _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["export"]["csv_files_written"] == result["csv_files_written"], (
            f"JSON export.csv_files_written {data['export']['csv_files_written']} "
            f"!= return value {result['csv_files_written']}"
        )

    def test_pipeline_summary_export_excel_matches_return_value(
        self, upstream_env, monkeypatch
    ):
        """export.excel_written in JSON must match the returned excel_written — spec §4."""
        pytest.importorskip("openpyxl")
        result = _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["export"]["excel_written"] == result["excel_written"], (
            f"JSON export.excel_written {data['export']['excel_written']} "
            f"!= return value {result['excel_written']}"
        )

    def test_pipeline_summary_missing_step_json_skips_gracefully(
        self, tmp_path, monkeypatch
    ):
        """Missing step JSONs must produce a warning+skip, not an exception — spec §4."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        logs = tmp_path / "logs"
        for d in (processed, exports, reports, logs):
            d.mkdir(parents=True, exist_ok=True)

        # Write NO step JSON files
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
        monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
        monkeypatch.setattr(config, "REPORTS_DIR", reports)
        monkeypatch.setattr(config, "LOG_DIR", logs)
        monkeypatch.setattr(config, "EXPORT_CSV", False)
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)

        import src.export as export_mod
        importlib.reload(export_mod)

        # Must not raise
        result = export_mod.run_export()
        assert result["pipeline_summary_written"] is True, (
            "pipeline_summary_written should be True even when step JSONs are missing"
        )

        with open(reports / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        # steps must exist but may be empty or partial
        assert "steps" in data, "pipeline_summary.json must have 'steps' key"
        assert isinstance(data["steps"], dict), "'steps' must be a dict"

    def test_pipeline_summary_partial_steps_includes_available(
        self, tmp_path, monkeypatch
    ):
        """When only some step JSONs exist, only those steps appear in 'steps' — spec §4."""
        processed = tmp_path / "processed"
        exports = tmp_path / "exports"
        reports = tmp_path / "reports"
        logs = tmp_path / "logs"
        for d in (processed, exports, reports, logs):
            d.mkdir(parents=True, exist_ok=True)

        # Write only eda_summary.json
        (reports / "eda_summary.json").write_text(
            json.dumps({"plot_count": 5}), encoding="utf-8"
        )

        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
        monkeypatch.setattr(config, "EXPORTS_DATA_DIR", exports)
        monkeypatch.setattr(config, "REPORTS_DIR", reports)
        monkeypatch.setattr(config, "LOG_DIR", logs)
        monkeypatch.setattr(config, "EXPORT_CSV", False)
        monkeypatch.setattr(config, "EXPORT_EXCEL", False)

        import src.export as export_mod
        importlib.reload(export_mod)

        export_mod.run_export()

        with open(reports / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)

        assert "step_3_eda" in data["steps"], (
            "'step_3_eda' must appear in steps since eda_summary.json was present"
        )
        assert data["steps"]["step_3_eda"]["plot_count"] == 5, (
            "step_3_eda content does not match eda_summary.json"
        )
        # Absent steps must NOT appear
        assert "step_2_cleaning" not in data["steps"], (
            "'step_2_cleaning' must not appear in steps when cleaning_report.json is absent"
        )

    def test_pipeline_summary_run_timestamp_is_string(
        self, upstream_env, monkeypatch
    ):
        """run_timestamp must be a non-empty string — spec §4."""
        _run(monkeypatch)
        with open(upstream_env["reports"] / "pipeline_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("run_timestamp", "")
        assert isinstance(ts, str) and len(ts) > 0, (
            f"run_timestamp must be a non-empty string, got '{ts}'"
        )


# ---------------------------------------------------------------------------
# Class: TestIdempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Validate that a second run overwrites outputs without appending — spec idempotency."""

    def test_second_run_does_not_duplicate_csv_rows(self, upstream_env, monkeypatch):
        """Running run_export() twice must not double the rows in CSV files — spec idempotency."""
        import src.export as export_mod
        importlib.reload(export_mod)
        export_mod.run_export()
        first_count = len(pd.read_csv(upstream_env["exports"] / "prices_clean.csv"))

        importlib.reload(export_mod)
        export_mod.run_export()
        second_count = len(pd.read_csv(upstream_env["exports"] / "prices_clean.csv"))

        assert first_count == second_count, (
            f"Second run changed prices_clean.csv row count: "
            f"{first_count} → {second_count} (expected same)"
        )

    def test_second_run_pipeline_summary_overwritten(
        self, upstream_env, monkeypatch
    ):
        """Running run_export() twice must overwrite pipeline_summary.json — spec idempotency."""
        import src.export as export_mod
        importlib.reload(export_mod)
        export_mod.run_export()

        summary_path = upstream_env["reports"] / "pipeline_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            first_data = json.load(f)

        importlib.reload(export_mod)
        export_mod.run_export()

        with open(summary_path, encoding="utf-8") as f:
            second_data = json.load(f)

        # The structure must be the same; no duplicated keys expected
        assert set(first_data.keys()) == set(second_data.keys()), (
            "pipeline_summary.json top-level keys changed between runs"
        )
        assert second_data["pipeline_version"] == "1.0", (
            "pipeline_version must remain '1.0' after second run"
        )

    def test_second_run_csv_files_written_count_stable(
        self, upstream_env, monkeypatch
    ):
        """csv_files_written in the return dict must be identical across two runs — spec."""
        import src.export as export_mod
        importlib.reload(export_mod)
        r1 = export_mod.run_export()
        importlib.reload(export_mod)
        r2 = export_mod.run_export()
        assert r1["csv_files_written"] == r2["csv_files_written"], (
            f"csv_files_written differed between runs: {r1['csv_files_written']} vs "
            f"{r2['csv_files_written']}"
        )


# ---------------------------------------------------------------------------
# Class: TestNoNetworkNoRealFiles
# ---------------------------------------------------------------------------

class TestNoNetworkNoRealFiles:
    """Guard-rail tests confirming the module never touches real data dirs."""

    def test_no_yfinance_import_in_export_module(self):
        """export.py must not import yfinance — spec 'No network calls' constraint."""
        import src.export as export_mod
        # If yfinance appears in the module's globals, the isolation rule is broken
        assert "yfinance" not in dir(export_mod), (
            "yfinance was found in export module namespace — network isolation violated"
        )

    def test_export_module_does_not_call_print(self):
        """export.py must use loguru for all output; print() is banned — spec conventions."""
        import ast
        import inspect
        import src.export as export_mod

        source = inspect.getsource(export_mod)
        tree = ast.parse(source)

        print_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert len(print_calls) == 0, (
            f"export.py contains {len(print_calls)} print() call(s) — "
            "all output must use loguru.logger"
        )
