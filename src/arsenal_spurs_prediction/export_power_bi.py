"""
Power BI Data Export Module.

Transforms the output of the machine learning pipeline (Dixon-Coles + Monte Carlo)
into a structured Star Schema (dim/fact tables) saved as CSVs, facilitating
seamless integration with Microsoft Power BI.

Justification for Pandas:
- Pandas is selected for this ETL task as it provides highly efficient vectorized
  mapping, grouping, and shape transformations for medium-sized datasets. It is
  already installed in the project environment and has built-in CSV export
  capabilities that match our needs without adding third-party overhead.
"""

import logging
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for Paths
PROCESSED_DIR = Path("data/processed")
POWERBI_DIR = PROCESSED_DIR / "powerbi"


class PowerBIExporter:
    """Class to manage data transformation and export for Power BI star schema."""

    def __init__(self) -> None:
        """Initialize the exporter and ensure output directories exist."""
        POWERBI_DIR.mkdir(parents=True, exist_ok=True)
        self.teams_df: pd.DataFrame | None = None
        self.team_name_to_id: dict[str, int] = {}

    def build_dim_teams(
        self, standings_df: pd.DataFrame, fixtures_df: pd.DataFrame
    ) -> None:
        """
        Build Dim_Teams dimension table.

        Assigns a unique ID, abbreviation, and hex color code to each team.

        Args:
            standings_df: Current standings dataframe.
            fixtures_df: Remaining fixtures dataframe.
        """
        logger.info("Building Dim_Teams table...")
        # Get unique list of all teams in the system
        all_teams = sorted(
            list(
                set(standings_df["team"].dropna())
                | set(fixtures_df["home_team"].dropna())
                | set(fixtures_df["away_team"].dropna())
                | {"Paris Saint-Germain"}  # Ensure PSG is included for UCL final
            )
        )

        # Brand colors and abbreviations for key teams in the dashboard
        brand_colors = {
            "Arsenal": "#E24B4A",
            "Manchester City": "#378ADD",
            "Tottenham Hotspur": "#7F77DD",
            "Paris Saint-Germain": "#004170",
            "Chelsea": "#034694",
            "Everton": "#003399",
            "Burnley": "#6C1D45",
            "Bournemouth": "#B50E12",
            "West Ham United": "#7A263A",
            "Newcastle United": "#241F20",
            "Aston Villa": "#95BFE5",
            "Leeds United": "#FFCD00",
            "Nottingham Forest": "#DD0000",
            "Liverpool": "#C8102E",
            "Manchester Utd": "#DA291C",
            "Brighton": "#0057B8",
            "Fulham": "#000000",
            "Brentford": "#E30613",
            "Wolves": "#FDB913",
            "Crystal Palace": "#1B458F",
            "Sunderland": "#D71920",
        }

        abbreviations = {
            "Arsenal": "ARS",
            "Manchester City": "MCI",
            "Tottenham Hotspur": "TOT",
            "Paris Saint-Germain": "PSG",
            "Chelsea": "CHE",
            "Everton": "EVE",
            "Burnley": "BUR",
            "Bournemouth": "BOU",
            "West Ham United": "WHU",
            "Newcastle United": "NEW",
            "Aston Villa": "AVL",
            "Leeds United": "LEE",
            "Nottingham Forest": "NFO",
            "Liverpool": "LIV",
            "Manchester Utd": "MUN",
            "Brighton": "BHA",
            "Fulham": "FUL",
            "Brentford": "BRE",
            "Wolves": "WOL",
            "Crystal Palace": "CRY",
            "Sunderland": "SUN",
        }

        teams_data = []
        for i, team in enumerate(all_teams, start=1):
            teams_data.append(
                {
                    "TeamID": i,
                    "TeamName": team,
                    "Abbreviation": abbreviations.get(team, team[:3].upper()),
                    "ColorHex": brand_colors.get(team, "#7F8C8D"),  # Fallback to grey
                }
            )

        self.teams_df = pd.DataFrame(teams_data)
        self.team_name_to_id = dict(
            zip(self.teams_df["TeamName"], self.teams_df["TeamID"], strict=False)
        )

        self.teams_df.to_csv(POWERBI_DIR / "Dim_Teams.csv", index=False)
        logger.info(f"Dim_Teams.csv exported with {len(self.teams_df)} teams.")

    def build_dim_calendar(self, fixtures_df: pd.DataFrame) -> None:
        """
        Build Dim_Calendar dimension table for time intelligence.

        Args:
            fixtures_df: Remaining fixtures dataframe containing 'date' strings.
        """
        logger.info("Building Dim_Calendar table...")
        dates_parsed = pd.to_datetime(fixtures_df["date"], errors="coerce").dropna()
        min_date = dates_parsed.min()
        if pd.isnull(min_date):
            min_date = pd.to_datetime("2025-08-15")  # Fallback to season start
            
        max_date = pd.to_datetime("2026-05-30")  # UCL Final Date
        
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        calendar_data = []
        for dt in all_dates:
            calendar_data.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Year": dt.year,
                    "MonthName": dt.strftime("%B"),
                    "MonthNumber": dt.month,
                    "DayOfWeek": dt.strftime("%A"),
                    "Quarter": f"Q{(dt.month - 1) // 3 + 1}",
                }
            )

        calendar_df = pd.DataFrame(calendar_data)
        calendar_df.to_csv(POWERBI_DIR / "Dim_Calendar.csv", index=False)
        logger.info(f"Dim_Calendar.csv exported with {len(calendar_df)} entries (continuous range).")

    def build_fact_standings(self, standings_df: pd.DataFrame) -> None:
        """
        Build Fact_Standings table.

        Args:
            standings_df: Current standings dataframe.
        """
        logger.info("Building Fact_Standings table...")
        fact_standings = standings_df.copy()
        fact_standings["TeamID"] = fact_standings["team"].map(self.team_name_to_id)
        # Drop raw team name to conform with star schema guidelines (use dimension for names)
        fact_standings = fact_standings.drop(columns=["team"])
        # Reorder columns
        fact_standings = fact_standings[
            ["TeamID", "points", "goal_difference"]
        ].rename(columns={"points": "Points", "goal_difference": "GoalDifference"})

        fact_standings.to_csv(POWERBI_DIR / "Fact_Standings.csv", index=False)
        logger.info("Fact_Standings.csv exported.")

    def build_fact_remaining_fixtures(self, probs_df: pd.DataFrame, schedule_df: pd.DataFrame) -> None:
        """
        Build Fact_Remaining_Fixtures table.

        Enriches remaining fixtures with their date, round, and team IDs.

        Args:
            probs_df: Remaining fixtures win/draw/loss probabilities.
            schedule_df: Raw schedule dataframe to extract match dates.
        """
        logger.info("Building Fact_Remaining_Fixtures table...")

        # Parse schedule date and create lookup
        sched = schedule_df.copy()
        sched["date"] = pd.to_datetime(sched["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        sched_lookup = {}
        for _, r in sched.iterrows():
            sched_lookup[(r["home_team"], r["away_team"])] = r["date"]

        fixtures_data = []
        for i, row in probs_df.iterrows():
            h, a = row["home_team"], row["away_team"]
            h_id = self.team_name_to_id.get(h)
            a_id = self.team_name_to_id.get(a)
            match_date = sched_lookup.get((h, a), "2026-05-24")  # Default to final matchday

            fixtures_data.append(
                {
                    "FixtureID": i + 1,
                    "Date": match_date,
                    "HomeTeamID": h_id,
                    "AwayTeamID": a_id,
                    "HomeWinProb": float(row["home_win_prob"]),
                    "DrawProb": float(row["draw_prob"]),
                    "AwayWinProb": float(row["away_win_prob"]),
                }
            )

        fixtures_df = pd.DataFrame(fixtures_data)
        fixtures_df.to_csv(POWERBI_DIR / "Fact_Remaining_Fixtures.csv", index=False)
        logger.info(f"Fact_Remaining_Fixtures.csv exported with {len(fixtures_df)} rows.")

    def build_fact_monte_carlo_positions(self, sim_probs_df: pd.DataFrame) -> None:
        """
        Build Fact_Monte_Carlo_Positions table.

        Transforms wide-format Monte Carlo positions into a long-format fact table.

        Args:
            sim_probs_df: Wide-format position probabilities table.
        """
        logger.info("Building Fact_Monte_Carlo_Positions table...")
        # Index of sim_probs_df is team name. Let's reset index.
        sim_df = sim_probs_df.reset_index().rename(columns={"index": "TeamName"})

        long_data = []
        for _, row in sim_df.iterrows():
            t_name = row["TeamName"]
            t_id = self.team_name_to_id.get(t_name)
            if not t_id:
                continue

            for col in sim_df.columns:
                if col.startswith("Pos_"):
                    pos_num = int(col.split("_")[1])
                    prob = float(row[col])
                    if prob > 0.00001:  # Skip zero-probability finishes to save space
                        long_data.append(
                            {
                                "TeamID": t_id,
                                "Position": pos_num,
                                "Probability": prob,
                            }
                        )

        fact_mc_df = pd.DataFrame(long_data)
        fact_mc_df.to_csv(POWERBI_DIR / "Fact_Monte_Carlo_Positions.csv", index=False)
        logger.info(
            f"Fact_Monte_Carlo_Positions.csv exported with {len(fact_mc_df)} rows."
        )

    def build_fact_ucl_final(self, cl_probs_df: pd.DataFrame) -> None:
        """
        Build Fact_UCL_Final table.

        Args:
            cl_probs_df: Champions League final probabilities.
        """
        logger.info("Building Fact_UCL_Final table...")
        row = cl_probs_df.iloc[0]
        h_id = self.team_name_to_id.get("Arsenal")
        a_id = self.team_name_to_id.get("Paris Saint-Germain")

        fact_ucl = pd.DataFrame(
            [
                {
                    "HomeTeamID": h_id,
                    "AwayTeamID": a_id,
                    "Date": "2026-05-30",
                    "Venue": row["venue"],
                    "ArsenalWin90min": float(row["arsenal_win_90min"]),
                    "Draw90min": float(row["draw_90min"]),
                    "PSGWin90min": float(row["psg_win_90min"]),
                    "ArsenalWinInclET": float(row["arsenal_win_incl_et"]),
                    "PSGWinInclET": float(row["psg_win_incl_et"]),
                    "ArsenalContext": row["arsenal_context"],
                    "PSGContext": row["psg_context"],
                }
            ]
        )

        fact_ucl.to_csv(POWERBI_DIR / "Fact_UCL_Final.csv", index=False)
        logger.info("Fact_UCL_Final.csv exported.")

    def build_fact_executive_summary(self, summary_df: pd.DataFrame) -> None:
        """
        Build Fact_Executive_Summary table.

        Args:
            summary_df: Executive summary metrics.
        """
        logger.info("Building Fact_Executive_Summary table...")
        summary_df.to_csv(POWERBI_DIR / "Fact_Executive_Summary.csv", index=False)
        logger.info("Fact_Executive_Summary.csv exported.")

    def run_export(self) -> None:
        """Load outputs and compile into Power BI Star Schema CSVs."""
        try:
            logger.info("Reading pipeline files for Power BI export...")
            standings = pd.read_csv(PROCESSED_DIR / "current_standings.csv")
            remaining_probs = pd.read_csv(PROCESSED_DIR / "remaining_fixtures_probs.csv")
            cl_probs = pd.read_csv(PROCESSED_DIR / "cl_final_probs.csv")
            sim_probs = pd.read_csv(PROCESSED_DIR / "simulation_probabilities.csv", index_col=0)
            exec_summary = pd.read_csv(PROCESSED_DIR / "executive_summary.csv")

            # Load raw schedule to extract match dates
            raw_sched = pd.read_csv(Path("data/raw/fbref_schedule_eng-premier_league_2526.csv"))

            # Build star schema tables
            self.build_dim_teams(standings, remaining_probs)
            self.build_dim_calendar(raw_sched)
            self.build_fact_standings(standings)
            self.build_fact_remaining_fixtures(remaining_probs, raw_sched)
            self.build_fact_monte_carlo_positions(sim_probs)
            self.build_fact_ucl_final(cl_probs)
            self.build_fact_executive_summary(exec_summary)

            logger.info("Power BI star schema export completed successfully!")

        except Exception as e:
            logger.error(f"Failed to export Power BI star schema: {e}")
            raise


if __name__ == "__main__":
    exporter = PowerBIExporter()
    exporter.run_export()
