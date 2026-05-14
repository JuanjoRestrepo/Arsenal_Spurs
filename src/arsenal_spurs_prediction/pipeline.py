"""
Main training and prediction pipeline.
"""

import logging
from pathlib import Path

import pandas as pd

from arsenal_spurs_prediction.models.dixon_coles import DixonColesModel
from arsenal_spurs_prediction.simulation.monte_carlo import MonteCarloSimulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    filepath: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw FBref schedule and split into historical results and remaining fixtures.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # Extract goals from score string like "2–1"
    # Handling potential missing values
    scores = df["score"].str.extract(r"(\d+)[^\d]+(\d+)")
    df["home_goals"] = pd.to_numeric(scores[0])
    df["away_goals"] = pd.to_numeric(scores[1])

    played_mask = df["home_goals"].notna() & df["away_goals"].notna()
    played_matches = df[played_mask].copy()
    played_matches["home_goals"] = played_matches["home_goals"].astype(int)
    played_matches["away_goals"] = played_matches["away_goals"].astype(int)

    unplayed_matches = df[~played_mask].copy()

    # Calculate current standings
    standings = []
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())

    for team in teams:
        pts = 0
        gd = 0

        home_games = played_matches[played_matches["home_team"] == team]
        away_games = played_matches[played_matches["away_team"] == team]

        # Points from home games
        for _, row in home_games.iterrows():
            hg, ag = row["home_goals"], row["away_goals"]
            gd += hg - ag
            if hg > ag:
                pts += 3
            elif hg == ag:
                pts += 1

        # Points from away games
        for _, row in away_games.iterrows():
            hg, ag = row["home_goals"], row["away_goals"]
            gd += ag - hg
            if ag > hg:
                pts += 3
            elif ag == hg:
                pts += 1

        standings.append({"team": team, "points": pts, "goal_difference": gd})

    df_standings = pd.DataFrame(standings).sort_values(
        ["points", "goal_difference"], ascending=[False, False]
    )

    return played_matches, unplayed_matches, df_standings


def run_pipeline() -> None:
    data_path = Path("data/raw/fbref_schedule_2526.csv")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}. Please run ingestion.py first.")
        return

    played, remaining, current_standings = load_and_preprocess_data(data_path)

    logger.info("Current Top 5:")
    logger.info(f"\n{current_standings.head()}")

    logger.info("Current Bottom 5:")
    logger.info(f"\n{current_standings.tail()}")

    # Fit Dixon-Coles model
    model = DixonColesModel()
    model.fit(played)
    
    # Export Match Probabilities for Power BI
    remaining_probs = []
    for _, row in remaining.iterrows():
        h, a = row["home_team"], row["away_team"]
        hw, d, aw = model.match_probabilities(h, a)
        remaining_probs.append({
            "home_team": h,
            "away_team": a,
            "home_win_prob": hw,
            "draw_prob": d,
            "away_win_prob": aw
        })
    df_remaining_probs = pd.DataFrame(remaining_probs)
    
    # Ensure processed dir exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Export for Power BI
    current_standings.to_csv("data/processed/current_standings.csv", index=False)
    df_remaining_probs.to_csv("data/processed/remaining_fixtures_probs.csv", index=False)
    
    # Run Simulation
    simulator = MonteCarloSimulator(model=model, n_simulations=100000)
    final_probs = simulator.run(current_standings, remaining)
    
    # Export probabilities for Power BI
    final_probs.to_csv("data/processed/simulation_probabilities.csv")

    # Output Arsenal Title Probability (Pos_1)
    # The columns are Pos_1, Pos_2, etc.
    if "Arsenal" in final_probs.index:
        title_prob = final_probs.loc["Arsenal", "Pos_1"] * 100
        logger.info(f"Arsenal Premier League Title Probability: {title_prob:.2f}%")

    # Output Tottenham Relegation Probability (Pos_18, Pos_19, Pos_20)
    # Bottom 3 in a 20 team league are 18, 19, 20
    if "Tottenham Hotspur" in final_probs.index:
        cols = [c for c in final_probs.columns if int(c.split("_")[1]) >= 18]
        rel_prob = final_probs.loc["Tottenham Hotspur", cols].sum() * 100
        logger.info(f"Tottenham Relegation Probability: {rel_prob:.2f}%")


if __name__ == "__main__":
    run_pipeline()
