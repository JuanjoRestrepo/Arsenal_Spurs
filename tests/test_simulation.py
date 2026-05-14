import pandas as pd
import pytest

from arsenal_spurs_prediction.simulation.monte_carlo import MonteCarloSimulator


import numpy as np

class DummyModel:
    def predict(self, home_team: str, away_team: str) -> np.ndarray:

        # Return a 2x2 matrix where home team always wins 1-0
        # (prob=1 for home_goals=1, away_goals=0)
        matrix = np.zeros((2, 2))
        matrix[1, 0] = 1.0
        return matrix


@pytest.fixture
def sample_standings() -> pd.DataFrame:
    data = {"team": ["Arsenal", "Spurs"], "points": [80, 40], "goal_difference": [40, -10]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_fixtures() -> pd.DataFrame:
    data = {"home_team": ["Arsenal"], "away_team": ["Spurs"]}
    return pd.DataFrame(data)


def test_monte_carlo_simulator(
    sample_standings: pd.DataFrame, sample_fixtures: pd.DataFrame
) -> None:
    model = DummyModel()
    # Dummy model is not a true DixonColesModel, but we can type-ignore
    # or it fits duck-typing for this test
    simulator = MonteCarloSimulator(model=model, n_simulations=10)

    probs = simulator.run(sample_standings, sample_fixtures)

    # Arsenal should win 100% of the time based on the DummyModel (1-0), getting 3 pts
    # Initial points: Arsenal 80, Spurs 40
    # Final points: Arsenal 83, Spurs 40

    assert "Pos_1" in probs.columns
    assert "Pos_2" in probs.columns

    # Arsenal should be 1st 100% of the time
    assert probs.loc["Arsenal", "Pos_1"] == 1.0
    # Spurs should be 2nd 100% of the time
    assert probs.loc["Spurs", "Pos_2"] == 1.0
