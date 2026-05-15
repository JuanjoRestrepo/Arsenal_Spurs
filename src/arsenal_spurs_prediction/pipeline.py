"""
Main training and prediction pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from arsenal_spurs_prediction.models.context import (
    MatchContext,
    build_arsenal_cl_context,
    build_arsenal_pl_context,
    build_generic_context,
    build_psg_cl_context,
    build_tottenham_pl_context,
)
from arsenal_spurs_prediction.models.dixon_coles import DixonColesModel
from arsenal_spurs_prediction.simulation.monte_carlo import MonteCarloSimulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(  # noqa: C901
    filepath: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw FBref schedule and split into historical results and remaining fixtures.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # Extract goals from score string like "2-1"
    # Handling potential missing values
    scores = df["score"].str.extract(r"(\d+)[^\d]+(\d+)")
    df["home_goals"] = pd.to_numeric(scores[0])
    df["away_goals"] = pd.to_numeric(scores[1])

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Handle Expected Goals (xG) if available, otherwise fallback to actual goals
    if "home_xg" in df.columns and "away_xg" in df.columns:
        logger.info("Expected Goals (xG) columns found. Using xG for model training.")
        df["home_model_goals"] = pd.to_numeric(df["home_xg"])
        df["away_model_goals"] = pd.to_numeric(df["away_xg"])
    else:
        logger.warning(
            "Expected Goals (xG) not found. Falling back to actual goals for model training."
        )
        df["home_model_goals"] = df["home_goals"]
        df["away_model_goals"] = df["away_goals"]

    played_mask = df["home_goals"].notna() & df["away_goals"].notna()
    played_matches = df[played_mask].copy()
    played_matches["home_goals"] = played_matches["home_goals"].astype(int)
    played_matches["away_goals"] = played_matches["away_goals"].astype(int)

    # Use the model goals (either actual or xG) for the model fit
    played_matches["home_goals"] = played_matches["home_model_goals"]
    played_matches["away_goals"] = played_matches["away_model_goals"]

    unplayed_matches = df[~played_mask].copy()

    # Calculate current standings (always use ACTUAL goals for standings)
    standings = []
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())

    for team in teams:
        pts = 0
        gd = 0

        # Need actual goals for standings, so we re-extract from score string
        # since we overwrote home_goals with xG for the model
        actual_home_games = df[played_mask & (df["home_team"] == team)]
        actual_away_games = df[played_mask & (df["away_team"] == team)]

        # Points from home games
        for _, row in actual_home_games.iterrows():
            scores = str(row["score"]).split("-")
            if len(scores) == 2:
                hg, ag = int(scores[0]), int(scores[1])
                gd += hg - ag
                if hg > ag:
                    pts += 3
                elif hg == ag:
                    pts += 1

        # Points from away games
        for _, row in actual_away_games.iterrows():
            scores = str(row["score"]).split("-")
            if len(scores) == 2:
                hg, ag = int(scores[0]), int(scores[1])
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
    # Support multiple leagues
    leagues = ["eng-premier_league", "fra-ligue_1"]
    all_played = []

    # Load and preprocess all leagues
    for league in leagues:
        data_path = Path(f"data/raw/fbref_schedule_{league}_2526.csv")
        if not data_path.exists():
            logger.warning(f"Data file not found for {league}: {data_path}. Skipping.")
            continue

        played, remaining, current_standings = load_and_preprocess_data(data_path)
        all_played.append(played)

        # We only care about PL standings for the simulation part of PL
        if league == "eng-premier_league":
            pl_remaining = remaining
            pl_standings = current_standings

    if not all_played:
        logger.error("No training data found. Please run ingestion.py first.")
        return

    # Combine all played matches for better parameter estimation
    played_combined = pd.concat(all_played, ignore_index=True)

    logger.info(f"Training global model on {len(played_combined)} matches...")

    # Hyperparameter Tuning for Time Decay (alpha)
    # Use the last 30 matches as a validation set

    played_sorted = played_combined.sort_values("date")
    val_size = 30
    if len(played_sorted) > val_size * 2:
        train_set = played_sorted.iloc[:-val_size]
        val_set = played_sorted.iloc[-val_size:]

        alphas_to_test = [0.001, 0.003, 0.0065, 0.010, 0.015]
        best_alpha = 0.0065
        best_llk = float("-inf")

        logger.info(f"Starting Hyperparameter Tuning for Time Decay Alpha on {len(alphas_to_test)} candidates...")  # noqa: E501
        for alpha in alphas_to_test:
            # Train weights
            max_train_date = train_set["date"].max()
            train_weights = np.exp(-alpha * (max_train_date - train_set["date"]).dt.days).values

            # Fit model on train
            temp_model = DixonColesModel()
            temp_model.fit(train_set, weights=train_weights)

            # Evaluate on val set (pseudo out-of-sample log-likelihood)
            # We don't apply weights to val set evaluation to ensure comparable likelihoods
            init_params = np.concatenate(
                [
                    [temp_model.home_adv, temp_model.rho],
                    [temp_model.params.get(f"{t}_att", 0.1) for t in temp_model.teams],
                    [temp_model.params.get(f"{t}_def", -0.1) for t in temp_model.teams],
                ]
            )
            # Using the _log_likelihood function (returns NEGATIVE llk, so we negate it)
            val_llk = -temp_model._log_likelihood(init_params, val_set, weights=None)

            logger.info(f"Alpha {alpha:.4f} -> Val LLK: {val_llk:.2f}")
            if val_llk > best_llk:
                best_llk = val_llk
                best_alpha = alpha

        logger.info(f"Optimal Alpha Selected: {best_alpha:.4f}")
    else:
        best_alpha = 0.0065
        logger.warning("Not enough data for tuning. Using default alpha.")

    # Final Fit with best alpha
    max_date = played_combined["date"].max()
    delta_days = (max_date - played_combined["date"]).dt.days
    weights = np.exp(-best_alpha * delta_days).values

    model = DixonColesModel()
    model.fit(played_combined, weights=weights)

    # -----------------------------------------------------------------------
    # Premier League: Remaining Fixtures with Contextual Adjustments
    # -----------------------------------------------------------------------
    # Build End-of-Season 2026 scenario contexts
    arsenal_pl_ctx = build_arsenal_pl_context()
    tottenham_ctx = build_tottenham_pl_context()

    logger.info("\n=== END-OF-SEASON CONTEXT SCENARIOS ===")
    logger.info(arsenal_pl_ctx.describe())
    logger.info(tottenham_ctx.describe())

    remaining_probs = []
    for _, row in pl_remaining.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])

        # Apply contextual modifiers for key teams
        home_ctx = (
            arsenal_pl_ctx if h == "Arsenal" else
            tottenham_ctx if h == "Tottenham Hotspur" else
            build_generic_context(h)
        )
        away_ctx = (
            arsenal_pl_ctx if a == "Arsenal" else
            tottenham_ctx if a == "Tottenham Hotspur" else
            build_generic_context(a)
        )
        match_ctx = MatchContext(home_context=home_ctx, away_context=away_ctx)

        hw, d, aw = model.match_probabilities(h, a, context=match_ctx)
        remaining_probs.append(
            {
                "home_team": h,
                "away_team": a,
                "home_win_prob": round(hw, 4),
                "draw_prob": round(d, 4),
                "away_win_prob": round(aw, 4),
            }
        )
    df_remaining_probs = pd.DataFrame(remaining_probs)

    # -----------------------------------------------------------------------
    # Champions League Final: Arsenal vs PSG - Neutral Venue (Puskas Arena)
    # -----------------------------------------------------------------------
    logger.info("\nPredicting CL Final with full contextual adjustments...")

    arsenal_cl_ctx = build_arsenal_cl_context()
    psg_cl_ctx = build_psg_cl_context()

    logger.info(arsenal_cl_ctx.describe())
    logger.info(psg_cl_ctx.describe())

    cl_match_ctx = MatchContext(
        home_context=arsenal_cl_ctx,  # Arsenal treated as 'home side'
        away_context=psg_cl_ctx,
        neutral_venue=True,           # Eliminates home_adv - Puskas Arena
    )

    cl_hw, cl_d, cl_aw = model.match_probabilities(
        "Arsenal", "Paris Saint-Germain", context=cl_match_ctx
    )

    # If the match goes to draw at 90min, Extra Time/Penalties:
    # Assume 50/50 once in ET (conservative; both teams equally capable under pressure)
    ET_SPLIT = 0.50
    p_arsenal_win_full = cl_hw + cl_d * ET_SPLIT
    p_psg_win_full = cl_aw + cl_d * ET_SPLIT

    cl_final_probs = pd.DataFrame([{
        "home_team": "Arsenal",
        "away_team": "Paris Saint-Germain",
        "arsenal_win_90min": round(cl_hw, 4),
        "draw_90min": round(cl_d, 4),
        "psg_win_90min": round(cl_aw, 4),
        "arsenal_win_incl_et": round(p_arsenal_win_full, 4),
        "psg_win_incl_et": round(p_psg_win_full, 4),
        "venue": "Puskas Arena (Neutral)",
        "arsenal_context": "Fatigue 3%, Historic motivation +6%",
        "psg_context": "Fatigue 4%, Desperate for 1st CL +7%",
    }])
    logger.info(
        f"CL Final (90min): Arsenal {cl_hw:.1%} | Draw {cl_d:.1%} | PSG {cl_aw:.1%}"
    )
    logger.info(
        f"CL Final (incl. ET/Pens): Arsenal {p_arsenal_win_full:.1%} | PSG {p_psg_win_full:.1%}"
    )

    # Ensure processed dir exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Export for Power BI
    pl_standings.to_csv("data/processed/current_standings.csv", index=False)
    df_remaining_probs.to_csv("data/processed/remaining_fixtures_probs.csv", index=False)
    cl_final_probs.to_csv("data/processed/cl_final_probs.csv", index=False)

    # Run PL Simulation - pass contexts to simulator via model
    # Contexts are baked into the probability matrices via pre-computed match_matrices
    simulator = MonteCarloSimulator(model=model, n_simulations=100_000)
    final_probs = simulator.run(pl_standings, pl_remaining)

    # Export probabilities for Power BI
    final_probs.to_csv("data/processed/simulation_probabilities.csv")

    # -----------------------------------------------------------------------
    # KEY RESULTS: Extract the three main prediction objectives
    # -----------------------------------------------------------------------
    arsenal_title_prob = 0.0
    tottenham_rel_prob = 0.0

    if "Arsenal" in final_probs.index:
        arsenal_title_prob = float(final_probs.loc["Arsenal", "Pos_1"])

    if "Tottenham Hotspur" in final_probs.index:
        rel_cols = [c for c in final_probs.columns if int(c.split("_")[1]) >= 18]
        tottenham_rel_prob = float(final_probs.loc["Tottenham Hotspur", rel_cols].sum())

    # The Double probability:
    # P(Double) = P(PL Title) * P(Arsenal lifts CL)
    # where P(Arsenal lifts CL) = P(Win 90min) + P(Draw 90min) * 0.50 (ET/Pens)
    double_prob = arsenal_title_prob * p_arsenal_win_full

    # -----------------------------------------------------------------------
    # Executive Summary
    # -----------------------------------------------------------------------
    sep = "=" * 65
    logger.info(f"\n{sep}")
    logger.info("  PREDICTIVE INTELLIGENCE: SEASON 2025/26 FINAL REPORT")
    logger.info(sep)
    logger.info(f"  Arsenal - Premier League Title Probability:  {arsenal_title_prob:>7.2%}")
    logger.info(f"  Arsenal - CL Final Win (incl. ET/Pens):      {p_arsenal_win_full:>7.2%}")
    logger.info(f"  Arsenal - HISTORIC DOUBLE Probability:       {double_prob:>7.2%}")
    logger.info(sep)
    logger.info(f"  Tottenham - Relegation Probability:          {tottenham_rel_prob:>7.2%}")
    logger.info(sep)
    logger.info(f"  CL Final: Arsenal {cl_hw:.1%} | Draw {cl_d:.1%} | PSG {cl_aw:.1%} [90min]")
    logger.info(f"  CL Final: Arsenal {p_arsenal_win_full:.1%} | PSG {p_psg_win_full:.1%} [incl. ET]")  # noqa: E501
    logger.info(sep)

    # Export summary for Power BI
    summary_df = pd.DataFrame([{
        "metric": "Arsenal PL Title Probability",
        "value": round(arsenal_title_prob, 4),
    }, {
        "metric": "Arsenal CL Win (incl. ET/Pens)",
        "value": round(p_arsenal_win_full, 4),
    }, {
        "metric": "Arsenal Historic Double Probability",
        "value": round(double_prob, 4),
    }, {
        "metric": "Tottenham Relegation Probability",
        "value": round(tottenham_rel_prob, 4),
    }])
    summary_df.to_csv("data/processed/executive_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
