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
