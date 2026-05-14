"""
Dixon-Coles Poisson model for football match prediction.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class DixonColesModel:
    """
    Dixon-Coles model for estimating attack, defense, and home advantage parameters.
    Includes the low-scoring match correlation parameter (rho).
    """

    def __init__(self) -> None:
        self.teams: list[str] = []
        self.params: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    def _tau(self, x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
        """
        Dixon-Coles correction factor for low scoring games.
        """
        if x == 0 and y == 0:
            return 1 - lambda_ * mu * rho
        elif x == 0 and y == 1:
            return 1 + lambda_ * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        """
        Calculate negative log-likelihood for the optimization.
        """
        home_adv = params[0]
        rho = params[1]

        # Unpack team parameters
        n_teams = len(self.teams)
        attack_params = params[2 : 2 + n_teams]
        defense_params = params[2 + n_teams :]

        # Maps for quick lookup
        att_map = dict(zip(self.teams, attack_params, strict=False))
        def_map = dict(zip(self.teams, defense_params, strict=False))

        # Expected goals
        lambda_ = np.exp(df["home_team"].map(att_map) + df["away_team"].map(def_map) + home_adv)
        mu = np.exp(df["away_team"].map(att_map) + df["home_team"].map(def_map))

        # Actual goals
        h_goals = df["home_goals"].values
        a_goals = df["away_goals"].values

        # Base Poisson log-likelihood
        llk_poisson = poisson.logpmf(h_goals, lambda_) + poisson.logpmf(a_goals, mu)

        # Dixon-Coles adjustment for rho
        tau_values = np.array(
            [
                self._tau(x, y, lam, m, rho)
                for x, y, lam, m in zip(h_goals, a_goals, lambda_, mu, strict=False)
            ]
        )

        # Prevent log(<=0)
        tau_values = np.maximum(tau_values, 1e-10)

        llk = llk_poisson + np.log(tau_values)
        return float(-np.sum(llk))

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the model on historical match data.
        Expected columns: 'home_team', 'away_team', 'home_goals', 'away_goals'
        """
        logger.info("Fitting Dixon-Coles model...")
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n_teams = len(self.teams)

        # Initial guess: home_adv=0.2, rho=0.0, attack=1.0, defense=-1.0
        init_params = np.concatenate(
            [
                [0.2, 0.0],
                np.ones(n_teams) * 0.1,  # attack
                np.ones(n_teams) * -0.1,  # defense
            ]
        )

        # Constraints: Average of attack parameters must be 1.0 (or sum = n_teams)
        # However, with exponentials, sum of params = 0 is a common constraint
        def constraint_func(params: np.ndarray) -> float:
            return float(np.sum(params[2 : 2 + n_teams]))

        constraints = [{"type": "eq", "fun": constraint_func}]

        bounds = [(None, None), (-0.5, 0.5)] + [(None, None)] * (2 * n_teams)

        res = minimize(
            self._log_likelihood,
            init_params,
            args=(df,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "maxiter": 500},
        )

        if not res.success:
            logger.warning(f"Optimization may have failed: {res.message}")

        # Store results
        self.home_adv = res.x[0]
        self.rho = res.x[1]

        for i, team in enumerate(self.teams):
            self.params[f"{team}_att"] = res.x[2 + i]
            self.params[f"{team}_def"] = res.x[2 + n_teams + i]

        logger.info("Model fitting complete.")

    def predict(self, home_team: str, away_team: str, max_goals: int = 8) -> np.ndarray:
        """
        Predict exact score probabilities for a given matchup.
        Returns a matrix of shape (max_goals+1, max_goals+1) where
        matrix[i, j] is the probability of home_team scoring i and away_team scoring j.
        """
        lambda_ = np.exp(
            self.params[f"{home_team}_att"] + self.params[f"{away_team}_def"] + self.home_adv
        )
        mu = np.exp(self.params[f"{away_team}_att"] + self.params[f"{home_team}_def"])

        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                base_prob = poisson.pmf(h, lambda_) * poisson.pmf(a, mu)
                tau = self._tau(h, a, lambda_, mu, self.rho)
                prob_matrix[h, a] = base_prob * tau

        # Normalize to ensure sum is 1.0 (truncation at max_goals)
        prob_matrix = prob_matrix / np.sum(prob_matrix)
        return prob_matrix

    def match_probabilities(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        """
        Return (Home Win, Draw, Away Win) probabilities.
        """
        matrix = self.predict(home_team, away_team)

        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))

        return float(home_win), float(draw), float(away_win)
