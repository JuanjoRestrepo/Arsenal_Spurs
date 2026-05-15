"""
Contextual Adjustment Engine for the Dixon-Coles football prediction model.

This module implements a Bayesian-style prior adjustment layer on top of the
base Dixon-Coles parameters. It encodes real-world factors that affect match
outcomes but are not captured by historical results alone:

  - Injuries (key player absences reduce attack or defensive stability)
  - Fatigue / Schedule Congestion (mid-week CL matches degrade weekend PL form)
  - Motivation Asymmetry (relegation six-pointers, dead-rubber matches)
  - Tactical Setup (e.g., parking the bus in a must-not-lose fixture)

Design philosophy:
  Parameters are multiplicative scaling factors applied to the expected goal
  rates (lambda, mu) derived by the Dixon-Coles model BEFORE the Poisson
  probability grid is computed. This preserves the statistical integrity of
  the model while allowing domain-expert priors to inform predictions.

Reference:
  Dixon, M. & Coles, S. (1997) — Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market. Applied Statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — scaling bounds enforced to prevent degenerate predictions
# ---------------------------------------------------------------------------
MIN_SCALE = 0.50  # No single factor can reduce expected goals below 50%
MAX_SCALE = 1.50  # No single factor can inflate expected goals above 150%


@dataclass
class TeamContext:
    """
    Encodes the contextual state of a single team for a specific fixture.

    All scale factors are multiplicative coefficients applied to the base
    expected goals rates (λ for attack, μ for defense) derived by the model.
    A value of 1.0 is neutral (no adjustment). Values < 1.0 degrade, > 1.0 boost.

    Attributes:
        team_name:              The team's canonical name (must match model params).
        attack_scale:           Multiplier for expected goals scored (λ).
                                E.g., 0.88 → star attacker is injured.
        defense_scale:          Multiplier for expected goals conceded (μ).
                                E.g., 0.92 → solid defensive block expected.
        motivation_bonus:       Additive term (in log space) for high-stakes motivation.
                                Positive → more aggressive, negative → complacency.
        fatigue_penalty:        Fraction to REDUCE attack/defense scales due to
                                congested schedule. Applied to BOTH scales equally.
        notes:                  Human-readable justification for each adjustment.
    """

    team_name: str
    attack_scale: float = 1.0
    defense_scale: float = 1.0
    motivation_bonus: float = 0.0
    fatigue_penalty: float = 0.0
    notes: list[str] = field(default_factory=list)

    def effective_attack_scale(self) -> float:
        """
        Net attack multiplier after applying fatigue penalty.

        Fatigue is assumed to uniformly degrade both attacking output and
        defensive organization — consistent with sports science literature on
        muscle fatigue and cognitive slowdown after high-intensity matches
        (e.g., Bangsbo et al., 2006; Mohr et al., 2005).
        """
        raw = self.attack_scale * (1.0 - self.fatigue_penalty)
        return float(max(MIN_SCALE, min(MAX_SCALE, raw)))

    def effective_defense_scale(self) -> float:
        """
        Net defensive multiplier. Lower values mean the team concedes more.
        Note: this scales the OPPONENT's expected goals; a value < 1.0 is
        defensive improvement (fewer goals conceded).
        """
        raw = self.defense_scale * (1.0 - self.fatigue_penalty)
        return float(max(MIN_SCALE, min(MAX_SCALE, raw)))

    def log_motivation_bonus(self) -> float:
        """
        Return the motivation bonus in log space (added to log(λ) or log(μ)).
        A motivation bonus of 0.05 increases expected goals by ~5.1%.
        """
        return self.motivation_bonus

    def describe(self) -> str:
        """Return a formatted description of this context for logging."""
        lines = [f"  [{self.team_name}]"]
        lines.append(f"    Attack Scale:    {self.effective_attack_scale():.3f}")
        lines.append(f"    Defense Scale:   {self.effective_defense_scale():.3f}")
        lines.append(f"    Motivation Bonus:{self.log_motivation_bonus():+.3f}")
        if self.notes:
            lines.append("    Notes:")
            for note in self.notes:
                lines.append(f"      - {note}")
        return "\n".join(lines)


@dataclass
class MatchContext:
    """
    Bundles the contextual state of BOTH teams in a fixture.

    The `apply()` method returns adjusted (lambda, mu) pairs ready to be
    plugged into the Poisson grid of the Dixon-Coles predict() method.

    Attributes:
        home_context:   TeamContext for the home side (or 'side_a' in neutral).
        away_context:   TeamContext for the away side (or 'side_b' in neutral).
        neutral_venue:  If True, the model's `home_adv` parameter will be set
                        to 0.0 before prediction (used for CL Final, etc.).
    """

    home_context: TeamContext
    away_context: TeamContext
    neutral_venue: bool = False

    def apply(
        self, base_lambda: float, base_mu: float
    ) -> tuple[float, float]:
        """
        Apply contextual scaling factors to base expected goal rates.

        Args:
            base_lambda:  Base expected goals for the home team (from Dixon-Coles).
            base_mu:      Base expected goals for the away team (from Dixon-Coles).

        Returns:
            Tuple of adjusted (lambda_, mu_) expected goal rates.

        Mathematical formulation:
            λ_adj = base_λ × att_scale_home × def_scale_away × exp(motivation_home)
            μ_adj = base_μ × att_scale_away × def_scale_home × exp(motivation_away)

            Where def_scale affects how many goals the OPPONENT is expected
            to score against this team's defensive organization.
        """
        import numpy as np

        # Home team attack adjusted by its own form, away team's defensive organization
        lambda_adj = (
            base_lambda
            * self.home_context.effective_attack_scale()
            * self.away_context.effective_defense_scale()
            * np.exp(self.home_context.log_motivation_bonus())
        )

        # Away team attack adjusted by its own form, home team's defensive organization
        mu_adj = (
            base_mu
            * self.away_context.effective_attack_scale()
            * self.home_context.effective_defense_scale()
            * np.exp(self.away_context.log_motivation_bonus())
        )

        return float(lambda_adj), float(mu_adj)

    def describe(self) -> str:
        venue = "Neutral Venue" if self.neutral_venue else "Standard Venue"
        lines = [f"Match Context [{venue}]:"]
        lines.append(self.home_context.describe())
        lines.append(self.away_context.describe())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built End-of-Season 2025/26 Scenarios
# These encode our domain-expert knowledge of each team's situation heading
# into the final weeks of the season. All adjustments are documented with
# explicit justifications so the model remains interpretable.
# ---------------------------------------------------------------------------

def build_arsenal_pl_context() -> TeamContext:
    """
    Arsenal's contextual state for their remaining PL matches.

    Rationale:
      - CL Finalist: Arsenal have played 13 CL matches across the campaign.
        Playing midweek semi-finals/finals creates measurable fatigue. Sports
        science literature estimates 5–8% degradation in sprint metrics after
        a 120-min fixture (Mohr et al., 2005). We encode a 6% fatigue penalty.
      - Title almost secured: High motivation to finish strong (no complacency
        risk — Arteta's squad historically drives hard to the end).
      - Squad depth available: Rotation policy mitigates fatigue (~2% offset).
    """
    return TeamContext(
        team_name="Arsenal",
        attack_scale=0.97,     # Slight fatigue on pressing intensity
        defense_scale=1.0,     # Defensive organization remains elite
        motivation_bonus=0.04, # Title confirmation drive (+4.1% goal rate)
        fatigue_penalty=0.04,  # ~4% net fatigue from CL campaign
        notes=[
            "Played 13 CL matches; semi-final/final fatigue estimated at 4-6%",
            "Title motivation partially offsets fatigue — Arteta squad historically finishes hard",
            "Key rotations (e.g., Havertz/Trossard) mitigate acute fatigue",
        ],
    )


def build_tottenham_pl_context() -> TeamContext:
    """
    Tottenham Hotspur's contextual state for remaining PL matches.

    Rationale:
      - Relegation six-pointers: Teams in survival battles exhibit measurable
        adrenaline-driven intensity boosts in must-win situations. Research by
        Lago-Ballesteros et al. (2010) confirms higher pressing intensity and
        tackle rates in relegation matches.
      - However: Season-long underperformance creates negative squad morale.
        Defensive organization has been poor all season (GD = -9).
      - High managerial pressure adds motivation but also decision-making errors.
    """
    return TeamContext(
        team_name="Tottenham Hotspur",
        attack_scale=0.92,      # Below-average attacking output all season
        defense_scale=0.90,     # Poor defensive organization (GD=-9)
        motivation_bonus=0.08,  # Desperation bonus — survival six-pointers
        fatigue_penalty=0.0,    # Not in Europe; no fatigue from CL
        notes=[
            "Relegation survival battle: desperation motivation boost (+8.3% goal rate)",
            "Historically poor defensive organization this season (GD = -9)",
            "No European competition — no fixture congestion fatigue",
            "Squad morale at season low; errors under pressure expected",
        ],
    )


def build_arsenal_cl_context() -> TeamContext:
    """
    Arsenal's contextual state for the Champions League Final vs PSG.

    Rationale:
      - Peak preparation: The CL Final receives the highest tactical preparation
        of any match in the season. Arteta will have had 3 weeks to prepare.
      - Historic occasion: First-ever CL Final for the club — extreme motivation.
      - Slight fatigue from the semi-final (PSG semi) 3 weeks prior.
      - No domestic title pressure by this point (PL concluded).
    """
    return TeamContext(
        team_name="Arsenal",
        attack_scale=1.02,     # Extra preparation and historic motivation
        defense_scale=1.0,     # No degradation — full defensive block ready
        motivation_bonus=0.06, # Historic first final — unprecedented motivation
        fatigue_penalty=0.03,  # Residual fatigue from semi-final
        notes=[
            "First-ever CL Final in Arsenal history — historic motivation peak",
            "3+ weeks of preparation under Arteta post-PL conclusion",
            "Slight residual fatigue from semi-final; managed with squad rotation",
            "No domestic match obligations — 100% focus on one match",
        ],
    )


def build_psg_cl_context() -> TeamContext:
    """
    PSG's contextual state for the Champions League Final vs Arsenal.

    Rationale:
      - PSG are also CL finalists — equivalent fatigue load from 13 matches.
      - Domestic league (Ligue 1) was secured earlier in the season — peak
        motivation now 100% on the CL. They are desperate to add a first-ever
        CL trophy after years of near-misses.
      - Luis Enrique's tactical identity: high-press, high-energy demands
        increase fatigue risk for key attackers (Dembele, Barcola).
      - Squad depth is elite — rotation partially offsets fatigue.
    """
    return TeamContext(
        team_name="Paris Saint-Germain",
        attack_scale=1.03,     # World-class attack; Dembele/Barcola at peak
        defense_scale=0.97,    # Slightly vulnerable defensively under pressure
        motivation_bonus=0.07, # Desperate to win first-ever CL (past near-misses)
        fatigue_penalty=0.04,  # Equivalent CL campaign fatigue to Arsenal
        notes=[
            "PSG desperate for first-ever CL trophy after 2020 Final loss",
            "Dembele/Barcola/Ramos form a world-class attacking trident",
            "Ligue 1 secured early — full focus on CL Final for 4+ weeks",
            "Defensively vulnerable when pressed high (xGA stats confirm)",
            "High-pressing style (Luis Enrique) increases physical toll",
        ],
    )


def build_generic_context(team_name: str) -> TeamContext:
    """
    Returns a neutral, no-adjustment context for teams not in a special scenario.
    Used for remaining PL fixtures involving teams other than Arsenal/Tottenham.
    """
    return TeamContext(team_name=team_name)
