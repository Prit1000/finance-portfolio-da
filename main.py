"""
main.py — Finance Portfolio Analysis Pipeline
Run: python main.py
"""

from loguru import logger
from pathlib import Path

import config
from src import data_ingestion
from src import data_cleaning
from src import eda
from src import metrics
from src import forecasting
from src import monte_carlo
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

    # ── Step 2: Data Cleaning ─────────────────────────────────────────────
    logger.info("STEP 2/7 — Data Cleaning")
    cleaning_summary = data_cleaning.run_cleaning()
    logger.info(f"Cleaning summary: {cleaning_summary}")

    # ── Step 3: EDA ───────────────────────────────────────────────────────
    logger.info("STEP 3/7 — Exploratory Data Analysis")
    eda_summary = eda.run_eda()
    logger.info(f"EDA summary: {eda_summary}")

    # ── Step 4: Metrics ───────────────────────────────────────────────────
    logger.info("STEP 4/7 — Portfolio Metrics")
    metrics_summary = metrics.run_metrics()
    logger.info(f"Metrics summary: {metrics_summary}")

    # ── Step 5: Forecasting ───────────────────────────────────────────────
    logger.info("STEP 5/7 — Forecasting")
    forecasting_summary = forecasting.run_forecasting()
    logger.info(f"Forecasting summary: {forecasting_summary}")

    # ── Step 6: Monte Carlo ───────────────────────────────────────────────
    logger.info("STEP 6/7 — Monte Carlo Simulation")
    mc_summary = monte_carlo.run_monte_carlo()
    logger.info(f"Monte Carlo summary: {mc_summary}")

    # ── Step 7: Export ───────────── (pending) ────────────────────────────

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
