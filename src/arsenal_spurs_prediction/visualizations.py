"""
Generate high-fidelity visualizations for the prediction results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    ax.set_title("Premier League Title Probabilities (2025/26 Season)", fontsize=16, fontweight="bold", pad=20)
    
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

if __name__ == "__main__":
    generate_title_probability_chart()
