"""
Data ingestion module for pulling Premier League data from FBref.
"""

import logging
from pathlib import Path

import pandas as pd
import pandera as pa
import soccerdata as sd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Define Pandera Schema for FBref Data
FBrefSchema = pa.DataFrameSchema(  # type: ignore
    {
        "date": pa.Column(pa.String, nullable=True),
        "home_team": pa.Column(pa.String, nullable=False),
        "away_team": pa.Column(pa.String, nullable=False),
        "score": pa.Column(pa.String, nullable=True),
    },
    index=pa.Index(int),
    strict=False,  # Allow other columns like xG, attendance, etc.
)


def fetch_fbref_match_history(
    leagues: str = "ENG-Premier League", seasons: str = "2526"
) -> pd.DataFrame:
    """
    Fetch match history including xG and xGA from FBref.

    Args:
        leagues: League identifier for soccerdata.
        seasons: Season identifier (e.g., '2526' for 2025/2026).

    Returns:
        pd.DataFrame containing match schedule and results with xG.
    """
    logger.info(f"Fetching FBref data for {leagues} season {seasons}...")

    try:
        # Initialize the FBref scraper
        fbref = sd.FBref(leagues=leagues, seasons=seasons)

        # Pull the schedule/results
        schedule = fbref.read_schedule()

        if schedule is None or schedule.empty:
            logger.warning("Fetched schedule is empty.")
            return pd.DataFrame()

        # Clean up columns if needed and reset index
        schedule = schedule.reset_index()

        # Validate Schema
        logger.info("Validating schema with Pandera...")
        schedule = FBrefSchema.validate(schedule)

        # Save raw data
        raw_path = RAW_DATA_DIR / f"fbref_schedule_{seasons}.csv"
        schedule.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved to {raw_path}")

        return schedule

    except Exception as e:
        logger.error(f"Failed to fetch FBref data: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting data ingestion...")
    df = fetch_fbref_match_history()
    logger.info(f"Ingested {len(df)} matches.")
