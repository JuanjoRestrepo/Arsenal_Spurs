# Arsenal & Spurs 2026 Prediction System

A production-grade predictive analytics platform for football match forecasting, specializing in Arsenal FC's title race and Tottenham Hotspur's performance metrics.

## Overview

This platform utilizes advanced statistical modeling and stochastic simulations to predict football outcomes. It automates the end-to-end lifecycle from data ingestion to interactive visualization.

### Key Features
- **Data Ingestion**: Automated scraping of FBref data using `soccerdata`.
- **Statistical Modeling**: Implementation of the **Dixon-Coles Poisson model**, calibrated for goal-scoring rates and low-scoring match correlations.
- **Monte Carlo Simulation**: 100,000-iteration season rollout to derive robust probability distributions for league positions.
- **Interactive Dashboards**: High-fidelity HTML dashboards and Power BI-ready data exports.

## Architecture

The system follows a modular architecture:
- `data/`: Managed directory for raw and processed datasets.
- `src/arsenal_spurs_prediction/`: Core logic including models and simulation engines.
- `tests/`: Comprehensive test suite ensuring 80%+ coverage on business logic.

## Installation & Setup

Ensure you have Python 3.12+ and `uv` installed.

```bash
# Clone the repository
cd Arsenal_Spurs

# Create and sync the environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running the Pipeline

You can run the entire workflow (ingestion -> training -> simulation -> export) with a single command:

```bash
uv run predict
```

## Quality Standards

The project enforces high code quality through:
- **Strict Typing**: Mandatory type hints verified by `mypy`.
- **Linting & Formatting**: Automated checks via `Ruff`.
- **Unit Testing**: Robust testing framework using `pytest`.

To run quality checks:
```bash
# Run tests
pytest tests

# Run linter
ruff check .

# Run type checker
mypy src tests
```

## Deliverables

- `arsenal_spurs_2026_prediction_dashboard_v2.html`: Latest predictive dashboard.
- `data/processed/`: CSV exports for Power BI integration.
