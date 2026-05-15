"""
Tests for the Dixon-Coles model.
"""

import numpy as np
import pandas as pd

from arsenal_spurs_prediction.models.dixon_coles import DixonColesModel


def test_time_decay_weights() -> None:
    """Test that time decay weights correctly affect the log-likelihood."""
    df = pd.DataFrame(
        {
            "home_team": ["Arsenal", "Spurs"],
            "away_team": ["Spurs", "Arsenal"],
            "home_goals": [2, 1],
            "away_goals": [1, 2],
            "date": pd.to_datetime(["2025-08-15", "2026-05-10"]),
        }
    )

    # Calculate weights manually
    max_date = df["date"].max()
    delta_days = (max_date - df["date"]).dt.days
    alpha = 0.0065
    weights = np.exp(-alpha * delta_days).values

    # Weight for recent match (Spurs vs Arsenal) should be 1.0 (delta = 0)
    assert np.isclose(weights[1], 1.0)

    # Weight for old match should be less than 1.0
    assert weights[0] < 1.0

    # Create model
    model = DixonColesModel()

    # We won't test the full fit because it requires more teams for optimization stability,
    # but we can test the log_likelihood function with and without weights.

    model.teams = ["Arsenal", "Spurs"]
    n_teams = 2
    init_params = np.concatenate(
        [
            [0.2, 0.0],
            np.ones(n_teams) * 0.1,  # attack
            np.ones(n_teams) * -0.1,  # defense
        ]
    )

    llk_unweighted = model._log_likelihood(init_params, df, weights=None)
    llk_weighted = model._log_likelihood(init_params, df, weights=weights)

    # Since weights are <= 1.0 and llk is negative sum,
    # the absolute value of the weighted LLK should be smaller than unweighted
    assert abs(llk_weighted) < abs(llk_unweighted)


def test_predict_probabilities_sum_to_one() -> None:
    """Test that the prediction matrix sums to 1.0 (or very close to it)."""
    model = DixonColesModel()
    model.teams = ["Arsenal", "Spurs"]
    model.params = {
        "Arsenal_att": 1.5,
        "Arsenal_def": -0.5,
        "Spurs_att": 1.0,
        "Spurs_def": 0.0,
    }
    model.home_adv = 0.2
    model.rho = 0.0

    prob_matrix = model.predict("Arsenal", "Spurs", max_goals=5)
    assert np.isclose(np.sum(prob_matrix), 1.0)

    hw, d, aw = model.match_probabilities("Arsenal", "Spurs")
    assert np.isclose(hw + d + aw, 1.0)
