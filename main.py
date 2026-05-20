"""
main.py — Finance Portfolio Analysis Pipeline
Run: python main.py
"""

from loguru import logger
from pathlib import Path

import config
from src import data_ingestion
# from src import data_cleaning       # TODO: Step 2
# from src import eda                  # TODO: Step 3
# from src import metrics              # TODO: Step 4
# from src import forecasting          # TODO: Step 5
# from src import monte_carlo          # TODO: Step 6
# from src import export               # TODO: Step 7


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        config.LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        level="INFO",
    )
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)

    # ── Step 1: Data Ingestion ────────────────────────────────────────────
    logger.info("STEP 1/7 — Data Ingestion")
    summary = data_ingestion.run_ingestion()
    logger.info(f"Ingestion summary: {summary}")

    # ── Step 2: Data Cleaning ──────── (pending) ──────────────────────────
    # logger.info("STEP 2/7 — Data Cleaning")
    # data_cleaning.run_cleaning()

    # ── Step 3: EDA ──────────────── (pending) ────────────────────────────
    # ── Step 4: Metrics ──────────── (pending) ────────────────────────────
    # ── Step 5: Forecasting ──────── (pending) ────────────────────────────
    # ── Step 6: Monte Carlo ──────── (pending) ────────────────────────────
    # ── Step 7: Export ───────────── (pending) ────────────────────────────

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
