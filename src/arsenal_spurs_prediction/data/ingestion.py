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
        "date": pa.Column(pa.DateTime, nullable=True, required=False),
        "home_team": pa.Column(pa.String, nullable=False),
        "away_team": pa.Column(pa.String, nullable=False),
        "score": pa.Column(pa.String, nullable=True),
    },
    index=pa.Index(int),
    strict=False,  # Allow other columns like xG, attendance, etc.
)


def fetch_fbref_match_history(
    leagues: list[str] | str = "ENG-Premier League", seasons: str = "2526"
) -> pd.DataFrame:
    """
    Fetch match history including xG and xGA from FBref for multiple leagues.

    Args:
        leagues: League identifier or list of identifiers for soccerdata.
        seasons: Season identifier (e.g., '2526' for 2025/2026).

    Returns:
        pd.DataFrame containing combined match schedule and results.
    """
    if isinstance(leagues, str):
        leagues = [leagues]

    logger.info(f"Fetching FBref data for {leagues} season {seasons}...")

    all_schedules = []
    try:
        for league in leagues:
            logger.info(f"Ingesting league: {league}")
            # Initialize the FBref scraper
            fbref = sd.FBref(leagues=league, seasons=seasons)

            # Pull the schedule/results
            schedule = fbref.read_schedule()

            if schedule is None or schedule.empty:
                logger.warning(f"Fetched schedule for {league} is empty.")
                continue

            # Clean up columns if needed and reset index
            schedule = schedule.reset_index()

            # Validate Schema
            logger.info(f"Validating {league} schema with Pandera...")
            schedule = FBrefSchema.validate(schedule)

            # Save raw data for each league
            league_slug = league.lower().replace(" ", "_")
            raw_path = RAW_DATA_DIR / f"fbref_schedule_{league_slug}_{seasons}.csv"
            schedule.to_csv(raw_path, index=False)
            logger.info(f"Raw data for {league} saved to {raw_path}")

            all_schedules.append(schedule)

        if not all_schedules:
            return pd.DataFrame()

        return pd.concat(all_schedules, ignore_index=True)

    except Exception as e:
        logger.error(f"Failed to fetch FBref data: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting multi-league data ingestion...")
    leagues_to_fetch = ["ENG-Premier League", "FRA-Ligue 1"]
    df = fetch_fbref_match_history(leagues=leagues_to_fetch)
    logger.info(f"Ingested {len(df)} matches total from {leagues_to_fetch}.")
