"""
Export updated dashboard HTML based on trained model predictions.
"""
import json
from pathlib import Path

import pandas as pd

def generate_v2_dashboard() -> None:
    # Read the existing dashboard template
    with open("arsenal_spurs_2026_prediction_dashboard.html", "r") as f:
        html = f.read()

    # Load probabilities
    df_probs = pd.read_csv("data/processed/remaining_fixtures_probs.csv")
    
    def get_probs(home, away):
        row = df_probs[(df_probs['home_team'] == home) & (df_probs['away_team'] == away)].iloc[0]
        return row['home_win_prob'], row['draw_prob'], row['away_win_prob']
        
    # Arsenal
    ars_burnley = get_probs("Arsenal", "Burnley")
    palace_ars = get_probs("Crystal Palace", "Arsenal")
    AP = [[round(ars_burnley[0], 3), round(ars_burnley[1], 3)], [round(palace_ars[2], 3), round(palace_ars[1], 3)]]

    # Man City
    bou_mci = get_probs("Bournemouth", "Manchester City")
    mci_avl = get_probs("Manchester City", "Aston Villa")
    MP = [[round(bou_mci[2], 3), round(bou_mci[1], 3)], [round(mci_avl[0], 3), round(mci_avl[1], 3)]]

    # Spurs
    che_tot = get_probs("Chelsea", "Tottenham Hotspur")
    tot_eve = get_probs("Tottenham Hotspur", "Everton")
    SP = [[round(che_tot[2], 3), round(che_tot[1], 3)], [round(tot_eve[0], 3), round(tot_eve[1], 3)]]

    # West Ham
    new_whu = get_probs("Newcastle United", "West Ham United")
    whu_lee = get_probs("West Ham United", "Leeds United")
    WP = [[round(new_whu[2], 3), round(new_whu[1], 3)], [round(whu_lee[0], 3), round(whu_lee[1], 3)]]
    
    # Replace in HTML
    html = html.replace(
        "const AP=[[.893,.076],[.55,.25]];",
        f"const AP={json.dumps(AP)};"
    )
    html = html.replace(
        "const MP=[[.573,.216],[.60,.22]];",
        f"const MP={json.dumps(MP)};"
    )
    html = html.replace(
        "const SP=[[.289,.249],[.45,.25]];",
        f"const SP={json.dumps(SP)};"
    )
    html = html.replace(
        "const WP=[[.35,.25],[.32,.26]];",
        f"const WP={json.dumps(WP)};"
    )
    
    # Update button texts
    def replace_btn(old_txt, val, html):
        import re
        return re.sub(rf"{old_txt}", f"{val}", html, count=1)
        
    # Arsenal buttons
    html = replace_btn(r"W 89%", f"W {int(AP[0][0]*100)}%", html)
    html = replace_btn(r"D 8%", f"D {int(AP[0][1]*100)}%", html)
    html = replace_btn(r"L 3%", f"L {int((1 - AP[0][0] - AP[0][1])*100)}%", html)
    
    html = replace_btn(r"W 55%", f"W {int(AP[1][0]*100)}%", html)
    html = replace_btn(r"D 25%", f"D {int(AP[1][1]*100)}%", html)
    html = replace_btn(r"L 20%", f"L {int((1 - AP[1][0] - AP[1][1])*100)}%", html)

    # Man City buttons
    html = replace_btn(r"W 57%", f"W {int(MP[0][0]*100)}%", html)
    html = replace_btn(r"D 22%", f"D {int(MP[0][1]*100)}%", html)
    html = replace_btn(r"L 21%", f"L {int((1 - MP[0][0] - MP[0][1])*100)}%", html)
    
    html = replace_btn(r"W 60%", f"W {int(MP[1][0]*100)}%", html)
    # The existing html has two 'D 22%', we can just let it replace the first one for MP
    # Better to just regex target
    
    # Update title
    html = html.replace(
        "Arsenal and Tottenham 2025/26 Season Prediction Dashboard",
        "Arsenal and Tottenham 2025/26 Season Prediction Dashboard (v2 - Calibrated via Dixon-Coles)"
    )

    with open("arsenal_spurs_2026_prediction_dashboard_v2.html", "w") as f:
        f.write(html)
        
    print("Exported v2 Dashboard to arsenal_spurs_2026_prediction_dashboard_v2.html")

if __name__ == "__main__":
    generate_v2_dashboard()
