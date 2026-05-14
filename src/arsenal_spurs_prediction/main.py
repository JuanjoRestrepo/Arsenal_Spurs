"""
Main entry point for the Arsenal and Spurs prediction system.
Orchestrates data ingestion, modeling, simulation, and dashboard export.
"""
import logging
from arsenal_spurs_prediction.data.ingestion import fetch_fbref_match_history
from arsenal_spurs_prediction.pipeline import run_pipeline
from arsenal_spurs_prediction.export_dashboard import generate_v2_dashboard

# Configure logging for the main entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Run the full end-to-end prediction workflow.
    """
    try:
        logger.info("Starting end-to-end prediction workflow...")
        
        logger.info("Step 1: Fetching latest match data from FBref...")
        fetch_fbref_match_history()
        
        logger.info("Step 2: Running prediction pipeline (Dixon-Coles + Monte Carlo)...")
        run_pipeline()
        
        logger.info("Step 3: Generating updated HTML dashboard (v2)...")
        generate_v2_dashboard()
        
        logger.info("Workflow completed successfully. Results are available in data/processed/ and the HTML dashboard.")
        
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
