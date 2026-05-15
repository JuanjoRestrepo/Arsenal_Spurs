# Premier League 2025/26: Advanced Predictive Intelligence

![Title Probabilities](images/title_probabilities.png)

A production-grade predictive modeling architecture designed to simulate the outcome of the English Premier League and UEFA Champions League 2025/2026 seasons. Built to the rigorous standards of modern sports data analytics, this project leverages a dynamically tuned, multi-league, time-decayed Dixon-Coles Poisson model alongside strict MLOps principles.

## 🏆 2026 Champions League Final Analysis
The model now supports cross-league team ratings, allowing it to simulate the **UEFA Champions League Final: Arsenal vs Paris Saint-Germain**. Using a neutral-venue calibrated Dixon-Coles model trained on both Premier League and Ligue 1 data, we provide the most accurate probability landscape for the Puskas Arena showdown.

![CL Final Probabilities](images/cl_final_probabilities.png)

## 🧠 Core Architecture

The statistical engine extends the classic Dixon & Coles (1997) methodology with modern enhancements:
1. **Multi-League Time-Decay Log-Likelihood:** The engine now ingest and trains on multiple European leagues (PL, Ligue 1) to derive consistent cross-competition team ratings. Recent form dictates mathematical parameters through an exponential decay function ($e^{-\alpha \cdot t}$).
2. **Neutral Venue Calibration:** Special logic for knockout tournaments (Champions League) that eliminates home field advantage for neutral venue simulations.
3. **Bivariate Poisson Adjustment:** Low-scoring matches (0-0, 1-0, 0-1, 1-1) are structurally adjusted using the $\rho$ parameter to account for under-dispersion in standard Poisson models.

## 🛠️ Data Engineering & MLOps

The repository adheres strictly to elite software engineering standards:
- **Pandera Schema Validation:** All data ingested via FBref is strictly validated at runtime to ensure schema integrity and prevent silent pipeline failures.
- **Strict Typing:** Python 3.12+ with full `mypy` strict mode compliance.
- **Continuous Integration Ready:** Over 80% coverage on core mathematical modules using `pytest`.
- **Formatting:** Handled entirely by `Ruff` for clean, consistent formatting.

## 🚀 Quick Start

Ensure you have `uv` installed, as this project uses strict, lightning-fast dependency management.

```bash
# Clone and setup
git clone https://github.com/JuanjoRestrepo/Arsenal_Spurs.git
cd Arsenal_Spurs

# Install dependencies via uv
uv sync

# 1. Ingest Data (Downloads latest FBref schedules)
uv run python -m arsenal_spurs_prediction.data.ingestion

# 2. Run the Pipeline (Fits model, tunes hyperparams, runs 100k Monte Carlo simulations)
uv run python -m arsenal_spurs_prediction.pipeline

# 3. Export to HTML Dashboard
uv run python -m arsenal_spurs_prediction.export_dashboard

# 4. Generate Visualizations
uv run python src/arsenal_spurs_prediction/visualizations.py
```

## 📊 Outputs & Artifacts

- `data/processed/simulation_probabilities.csv`: Final 100,000 iteration Monte Carlo distributions. Ready for Power BI ingestion.
- `data/processed/remaining_fixtures_probs.csv`: Calibrated 1x2 match probabilities for all remaining matches.
- `images/`: High-fidelity data visualizations of probability landscapes.
- `arsenal_spurs_2026_prediction_dashboard_v2.html`: A localized UI presenting the findings.

## 🔬 Next Steps for V3
- Migration to a fully Bayesian framework using `PyMC` to capture posterior intervals.
- Market calibration using Pinnacle closing lines.

---
*Built for the 2025/2026 Season Analysis.*
