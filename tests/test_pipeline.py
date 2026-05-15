"""
Tests for the main pipeline.
"""

import pandas as pd

from arsenal_spurs_prediction.pipeline import load_and_preprocess_data


def test_load_and_preprocess_data_actual_goals(tmp_path) -> None:
    """Test the pipeline loads data and correctly calculates standings using actual goals."""
    # Create mock CSV
    df = pd.DataFrame(
        {
            "date": ["2025-08-15", "2025-08-16", "2026-05-20"],
            "home_team": ["Arsenal", "Spurs", "Arsenal"],
            "away_team": ["Spurs", "Arsenal", "Chelsea"],
            "score": ["2–0", "1–1", None],
            "home_goals": [2, 1, None],
            "away_goals": [0, 1, None],
        }
    )

    file_path = tmp_path / "mock_fbref.csv"
    df.to_csv(file_path, index=False)

    played, unplayed, standings = load_and_preprocess_data(file_path)

    assert len(played) == 2
    assert len(unplayed) == 1

    # Standings
    # Arsenal: Win (3 pts) + Draw (1 pt) = 4 pts
    # Spurs: Loss (0 pts) + Draw (1 pt) = 1 pt
    arsenal_pts = standings[standings["team"] == "Arsenal"]["points"].iloc[0]
    spurs_pts = standings[standings["team"] == "Spurs"]["points"].iloc[0]

    assert arsenal_pts == 4
    assert spurs_pts == 1


def test_load_and_preprocess_data_xg_fallback(tmp_path) -> None:
    """Test that the pipeline uses xG for model training if available, but actual goals for standings."""  # noqa: E501
    df = pd.DataFrame(
        {
            "date": ["2025-08-15"],
            "home_team": ["Arsenal"],
            "away_team": ["Spurs"],
            "score": ["1–0"],
            "home_goals": [1],
            "away_goals": [0],
            "home_xg": [2.5],
            "away_xg": [0.5],
        }
    )

    file_path = tmp_path / "mock_fbref_xg.csv"
    df.to_csv(file_path, index=False)

    played, _, standings = load_and_preprocess_data(file_path)

    # Model goals should be xG
    assert played.iloc[0]["home_goals"] == 2.5
    assert played.iloc[0]["away_goals"] == 0.5

    # Standings should still use actual goals (Arsenal 3 pts for 1-0 win)
    arsenal_pts = standings[standings["team"] == "Arsenal"]["points"].iloc[0]
    assert arsenal_pts == 3
