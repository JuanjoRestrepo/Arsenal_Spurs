import numpy as np
import pandas as pd
import pytest

from arsenal_spurs_prediction.models.dixon_coles import DixonColesModel


@pytest.fixture
def sample_data() -> pd.DataFrame:
    data = {
        "home_team": ["Arsenal", "Chelsea", "Arsenal", "Spurs"],
        "away_team": ["Chelsea", "Arsenal", "Spurs", "Arsenal"],
        "home_goals": [2, 1, 3, 0],
        "away_goals": [1, 2, 1, 2],
    }
    return pd.DataFrame(data)


def test_dixon_coles_fit(sample_data: pd.DataFrame) -> None:
    model = DixonColesModel()
    model.fit(sample_data)

    assert len(model.teams) == 3
    assert "Arsenal" in model.teams
    assert "Chelsea" in model.teams
    assert "Spurs" in model.teams

    assert hasattr(model, "home_adv")
    assert hasattr(model, "rho")
    assert "Arsenal_att" in model.params
    assert "Arsenal_def" in model.params


def test_dixon_coles_predict(sample_data: pd.DataFrame) -> None:
    model = DixonColesModel()
    model.fit(sample_data)

    prob_matrix = model.predict("Arsenal", "Chelsea", max_goals=5)

    # Should be a 6x6 matrix (0 to 5 goals)
    assert prob_matrix.shape == (6, 6)

    # Sum of probabilities should be close to 1.0
    assert pytest.approx(np.sum(prob_matrix), 0.01) == 1.0


def test_match_probabilities(sample_data: pd.DataFrame) -> None:
    model = DixonColesModel()
    model.fit(sample_data)

    hw, d, aw = model.match_probabilities("Arsenal", "Spurs")

    assert hw >= 0.0 and hw <= 1.0
    assert d >= 0.0 and d <= 1.0
    assert aw >= 0.0 and aw <= 1.0
    assert pytest.approx(hw + d + aw, 0.01) == 1.0
