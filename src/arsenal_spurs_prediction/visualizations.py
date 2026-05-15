"""
Generate high-fidelity visualizations for the prediction results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_title_probability_chart() -> None:
    # Set seaborn style for professional look
    sns.set_theme(style="whitegrid", palette="deep")

    # Load probabilities
    df = pd.read_csv("data/processed/simulation_probabilities.csv", index_col=0)

    # Extract Title Probabilities (Pos_1)
    if "Pos_1" not in df.columns:
        return

    title_probs = df["Pos_1"].sort_values(ascending=False).head(5) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define team colors
    colors = []
    for team in title_probs.index:
        if team == "Arsenal":
            colors.append("#EF0107")  # Arsenal Red
        elif team == "Manchester City":
            colors.append("#6CABDD")  # Man City Blue
        elif team == "Liverpool":
            colors.append("#C8102E")  # Liverpool Red
        elif team == "Manchester Utd":
            colors.append("#DA291C")  # Man Utd Red
        elif team == "Aston Villa":
            colors.append("#95BFE5")  # Villa Claret/Blue
        else:
            colors.append("#333333")  # Default Dark Gray

    bars = ax.barh(title_probs.index, title_probs.values, color=colors)

    # Formatting
    ax.invert_yaxis()  # Highest probability at top
    ax.set_xlabel("Title Probability (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Premier League Title Probabilities (2025/26 Season)",
        fontsize=16, fontweight="bold", pad=20
    )

    # Add percentage labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%",
                ha="left", va="center", fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save image
    Path("images").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("images/title_probabilities.png", dpi=300, bbox_inches="tight")
    print("Generated images/title_probabilities.png")

def generate_cl_final_chart() -> None:
    # Load CL Final probabilities
    cl_df = pd.read_csv("data/processed/cl_final_probs.csv")
    probs = [  # noqa: E501
        cl_df["arsenal_win_90min"].iloc[0],
        cl_df["draw_90min"].iloc[0],
        cl_df["psg_win_90min"].iloc[0],
    ]
    labels = ["Arsenal Win", "Draw (90m)", "PSG Win"]
    colors = ["#EF0107", "#777777", "#004170"]  # Arsenal Red, Neutral Gray, PSG Blue

    fig, ax = plt.subplots(figsize=(8, 8))

    # Donut chart
    wedges, texts, autotexts = ax.pie(
        probs,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    # Draw circle in center
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title(
        "UEFA Champions League Final 2026 Probabilities\n(90 Minutes)",
        fontsize=16, fontweight="bold"
    )

    plt.savefig("images/cl_final_probabilities.png", dpi=300, bbox_inches="tight")
    print("Generated images/cl_final_probabilities.png")


def generate_executive_summary_chart() -> None:
    """Generate an executive summary dashboard with the four key predictive metrics."""
    summary = pd.read_csv("data/processed/executive_summary.csv")

    metrics = summary["metric"].tolist()
    values = (summary["value"] * 100).tolist()

    # Color coding: Arsenal = red, Tottenham danger = amber, Double = gold
    bar_colors = ["#EF0107", "#1D61A8", "#FFD700", "#E87722"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(metrics, values, color=bar_colors, height=0.5, edgecolor="none")

    # Labels
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", ha="left", va="center",
            color="white", fontsize=13, fontweight="bold"
        )

    ax.set_xlabel("Probability (%)", color="#cccccc", fontsize=11)
    ax.set_title(
        "2025/26 Season Predictive Intelligence — Executive Summary",
        color="white", fontsize=14, fontweight="bold", pad=16
    )
    ax.tick_params(colors="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("images/executive_summary.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Generated images/executive_summary.png")


if __name__ == "__main__":
    generate_title_probability_chart()
    generate_cl_final_chart()
    generate_executive_summary_chart()
