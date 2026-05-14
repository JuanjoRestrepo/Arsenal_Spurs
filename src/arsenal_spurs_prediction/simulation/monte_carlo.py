"""
Monte Carlo simulation for season outcomes.
"""

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel

from arsenal_spurs_prediction.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)


class MatchResult(BaseModel):
    """Data structure for a simulated match result."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_points: int
    away_points: int


class MonteCarloSimulator:
    """
    Simulates the remaining fixtures of a season thousands of times
    to estimate probabilities for final standings, titles, and relegation.
    """

    def __init__(self, model: DixonColesModel, n_simulations: int = 10000) -> None:
        """
        Initialize the simulator.

        Args:
            model: A fitted Dixon-Coles model.
            n_simulations: Number of Monte Carlo iterations.
        """
        self.model = model
        self.n_simulations = n_simulations

    def _simulate_match(self, home_team: str, away_team: str) -> tuple[int, int, int, int]:
        """
        Simulate a single match based on the probability matrix.
        Returns: (home_goals, away_goals, home_points, away_points)
        """
        prob_matrix = self.model.predict(home_team, away_team)

        # Flatten and randomly sample based on probabilities
        flat_probs = prob_matrix.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)

        home_goals, away_goals = np.unravel_index(idx, prob_matrix.shape)

        if home_goals > away_goals:
            return int(home_goals), int(away_goals), 3, 0
        elif home_goals < away_goals:
            return int(home_goals), int(away_goals), 0, 3
        else:
            return int(home_goals), int(away_goals), 1, 1

    def run(
        self, current_standings: pd.DataFrame, remaining_fixtures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run the Monte Carlo simulation.

        Args:
            current_standings: DataFrame with ['team', 'points', 'goal_difference']
            remaining_fixtures: DataFrame with ['home_team', 'away_team']

        Returns:
            DataFrame containing probabilities of ending in each position.
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")

        teams = current_standings["team"].tolist()
        n_teams = len(teams)

        # Pre-calculate match probability matrices to save time
        match_matrices = {}
        for _, row in remaining_fixtures.iterrows():
            h, a = row["home_team"], row["away_team"]
            match_matrices[(h, a)] = self.model.predict(h, a).flatten()

        # Initialize tracking arrays
        final_positions = {team: np.zeros(n_teams) for team in teams}

        for _sim in range(self.n_simulations):
            # Make a copy of current points and GD
            sim_pts = dict(
                zip(current_standings["team"], current_standings["points"], strict=False)
            )
            sim_gd = dict(
                zip(current_standings["team"], current_standings["goal_difference"], strict=False)
            )

            # Simulate remaining fixtures
            for (h, a), flat_probs in match_matrices.items():
                idx = np.random.choice(len(flat_probs), p=flat_probs)
                h_goals, a_goals = np.unravel_index(
                    idx, (9, 9)
                )  # assuming max_goals=8 -> shape 9x9

                sim_gd[h] += h_goals - a_goals
                sim_gd[a] += a_goals - h_goals

                if h_goals > a_goals:
                    sim_pts[h] += 3
                elif h_goals < a_goals:
                    sim_pts[a] += 3
                else:
                    sim_pts[h] += 1
                    sim_pts[a] += 1

            # Sort teams based on points, then goal difference
            sorted_teams = sorted(teams, key=lambda x: (sim_pts[x], sim_gd[x]), reverse=True)

            # Record final positions
            for pos, team in enumerate(sorted_teams):
                final_positions[team][pos] += 1

        # Normalize to probabilities
        df_probs = pd.DataFrame(final_positions).T / self.n_simulations
        df_probs.columns = [f"Pos_{i + 1}" for i in range(n_teams)]

        logger.info("Simulation complete.")
        return df_probs
