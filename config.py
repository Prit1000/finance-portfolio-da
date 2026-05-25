"""
config.py — Central configuration for the Finance Portfolio Analysis Pipeline.
All modules import from here; no hardcoded values elsewhere.
"""

from pathlib import Path

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT"]

# ── Date range ────────────────────────────────────────────────────────────────
DATE_START: str = "2023-01-01"
DATE_END: str = "2024-12-31"

# ── Fetch settings ────────────────────────────────────────────────────────────
FETCH_INTERVAL: str = "1d"
MAX_RETRIES: int = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR: Path = Path("data/raw")
PROCESSED_DATA_DIR: Path = Path("data/processed")
EXPORTS_DATA_DIR: Path = Path("data/exports")
LOG_DIR: Path = Path("logs")
OUTPUTS_DIR: Path = Path("outputs")

# ── Data Cleaning ─────────────────────────────────────────────────────────────
MIN_COVERAGE_PCT: float = 0.80
OUTLIER_RETURN_THRESHOLD: float = 0.25
FILL_METHOD: str = "ffill"
MAX_CONSECUTIVE_FILLS: int = 3
TRADING_CALENDAR: str = "NYSE"

# ── EDA (Step 3) ──────────────────────────────────────────────────────────────
PLOTS_DIR: Path = Path("outputs/plots")
REPORTS_DIR: Path = Path("outputs/reports")
EDA_PLOT_DPI: int = 300
EDA_ROLLING_WINDOWS: list[int] = [20, 50]
EDA_VOL_WINDOW: int = 30
EDA_TRADING_DAYS_PER_YEAR: int = 252
EDA_TOP_N_CORRELATIONS: int = 3
EDA_TOP_N_MOVES: int = 10
EDA_PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
EDA_MIN_QQ_ROWS: int = 30

# ── Metrics (Step 4) ──────────────────────────────────────────────────────────
RISK_FREE_RATE: float = 0.04
# Set to a ticker present in TICKERS (e.g. "^GSPC") to enable beta; None to skip.
BENCHMARK_TICKER: str | None = None
# None → equal-weighted; else must sum to 1.0 and cover all non-benchmark tickers.
PORTFOLIO_WEIGHTS: dict[str, float] | None = None
VAR_CONFIDENCE_LEVELS: list[float] = [0.95, 0.99]
CVAR_CONFIDENCE_LEVELS: list[float] = [0.95]
ROLLING_SHARPE_WINDOW: int = 90
ROLLING_BETA_WINDOW: int = 60
ROLLING_CORR_WINDOW: int = 60
METRICS_TRADING_DAYS_PER_YEAR: int = 252
EXCLUDE_BENCHMARK_FROM_PORTFOLIO: bool = True

# ── Forecasting (Step 5) ──────────────────────────────────────────────────────
SCENARIOS_CSV: Path = Path("scenario_params/scenarios.csv")
FORECAST_HORIZON_DAYS: int = 30
TRAIN_INITIAL_DAYS: int = 252
WALK_FORWARD_STEP_DAYS: int = 30
WALK_FORWARD_EXPANDING: bool = True
FORECAST_CONFIDENCE_LEVEL: float = 0.95
ARIMA_MAX_P: int = 5
ARIMA_MAX_Q: int = 5
ARIMA_MAX_D: int = 2
ARIMA_SEASONAL: bool = False
PROPHET_YEARLY_SEASONALITY: bool = True
PROPHET_WEEKLY_SEASONALITY: bool = True
PROPHET_DAILY_SEASONALITY: bool = False
MIN_OBSERVATIONS_FOR_FORECAST: int = 100
RANDOM_SEED: int = 42
FORECAST_COVERAGE_WARN_BELOW: float = 0.70
FORECAST_COVERAGE_WARN_ABOVE: float = 0.99

# ── Monte Carlo (Step 6) ──────────────────────────────────────────────────────
MC_SCENARIOS_CSV: Path = Path("scenario_params/mc_scenarios.csv")
MC_DEFAULT_N_SIMULATIONS: int = 10000
MC_DEFAULT_HORIZON_DAYS: int = 30
MC_DEFAULT_BLOCK_SIZE: int = 10
MC_RANDOM_SEED: int = 42
MC_PERCENTILES: list[float] = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
MC_VAR_LEVELS: list[float] = [0.95, 0.99]
MC_CVAR_LEVELS: list[float] = [0.95]
MC_SAVE_FULL_PATHS: bool = False
MC_USE_CORRELATION: bool = True
MC_DRIFT_METHOD: str = "historical"
MC_TRADING_DAYS_PER_YEAR: int = 252
MC_PROBABILITY_THRESHOLDS: list[float] = [-0.20, -0.10, 0.0, 0.10, 0.20]
