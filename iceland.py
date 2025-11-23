import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(layout="wide", page_title="Player Profile Generator")
st.title("Player Profile Generator")

# =============================================================
# LOAD FIXED EXCEL FILE
# =============================================================
file_path = "Iceland.xlsx"
df = pd.read_excel(file_path)

# =============================================================
# TEAM + PLAYER FILTER
# =============================================================
teams = sorted(df["Team"].dropna().unique().tolist())
selected_team = st.selectbox("Select Team", teams)

players = sorted(df[df["Team"] == selected_team]["Player"].dropna().unique().tolist())
selected_player = st.selectbox("Select Player", players)

position_group = st.selectbox(
    "Select Position Group",
    ["Centre-back", "Full-back", "Midfielder", "Attacking Midfielder", "Attacker"]
)

run_button = st.button("Generate Profile")


# =============================================================
# POSITION GROUP DEFINITIONS (MATCHING YOUR DATA)
# =============================================================
groups = {

    "Centre-back": (
        ["CB", "RCB", "LCB"],
        {
            "Security": [
                "Accurate passes, %",
                "Back passes per 90",
                "Accurate back passes, %",
                "Lateral passes per 90",
                "Accurate lateral passes, %",
            ],
            "Progressive Passing": [
                "Progressive passes per 90",
                "Accurate progressive passes, %",
                "Forward passes per 90",
                "Accurate forward passes, %",
                "Passes to final third per 90",
                "Accurate passes to final third, %",
            ],
            "Ball Carrying": [
                "Progressive runs per 90",
                "Dribbles per 90",
                "Successful dribbles, %",
                "Accelerations per 90",
            ],
            "Creativity": [
                "Key passes per 90",
                "Shot assists per 90",
                "xA per 90",
                "Smart passes per 90",
                "Accurate smart passes, %",
            ],
            "Proactive Defending": [
                "Interceptions per 90",
                "PAdj Interceptions",
                "Sliding tackles per 90",
                "PAdj Sliding tackles",
            ],
            "Duelling": [
                "Duels per 90",
                "Duels won, %",
                "Aerial duels per 90",
                "Aerial duels won, %",
            ],
            "Box Defending": ["Shots blocked per 90"],
            "Sweeping": [],
        },
        {
            "Ball Player": ["Progressive Passing", "Security"],
            "Libero": ["Progressive Passing", "Ball Carrying", "Creativity"],
            "Wide Creator": ["Creativity", "Ball Carrying"],
            "Aggressor": ["Proactive Defending", "Duelling"],
            "Physical Dominator": ["Duelling", "Box Defending"],
            "Box Defender": ["Box Defending", "Duelling"],
        },
    ),

    "Full-back": (
        ["LB", "RB", "LWB", "RWB"],
        {
            "Box Defending": ["Shots blocked per 90"],
            "Duelling": [
                "Duels per 90",
                "Duels won, %",
                "Aerial duels per 90",
                "Aerial duels won, %",
                "Defensive duels per 90",
                "Defensive duels won, %",
            ],
            "Pressing": [
                "PAdj Interceptions",
                "PAdj Sliding tackles",
                "Successful defensive actions per 90",
            ],
            "Security": [
                "Accurate passes, %",
                "Back passes per 90",
                "Accurate back passes, %",
                "Lateral passes per 90",
                "Accurate lateral passes, %",
            ],
            "Playmaking": [
                "Key passes per 90",
                "Shot assists per 90",
                "xA per 90",
                "Smart passes per 90",
                "Accurate smart passes, %",
            ],
            "Final Third": [
                "Passes to final third per 90",
                "Accurate passes to final third, %",
                "Crosses per 90",
                "Accurate crosses, %",
                "Touches in box per 90",
            ],
            "Overlapping": ["Accelerations per 90", "Fouls suffered per 90"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"],
        },
        {
            "False Wing": ["Playmaking", "Ball Carrying", "Final Third"],
            "Flyer": ["Overlapping", "Final Third", "Ball Carrying"],
            "Playmaker": ["Playmaking", "Security", "Final Third"],
            "Safety": ["Security", "Box Defending", "Pressing"],
            "Ballwinner": ["Duelling", "Pressing"],
            "Defensive FB": ["Box Defending", "Duelling", "Pressing"],
        },
    ),

    "Midfielder": (
        ["LCMF", "RCMF", "CFM", "LDMF", "RDMF", "RAMF", "LAMF", "AMF", "DMF"],
        {
            "Creativity": [
                "Key passes per 90",
                "Shot assists per 90",
                "xA per 90",
                "Smart passes per 90",
                "Accurate smart passes, %",
            ],
            "Box Crashing": [
                "Touches in box per 90",
                "Shots per 90",
                "xG per 90",
                "Non-penalty goals per 90",
            ],
            "Ball Carrying": [
                "Progressive runs per 90",
                "Dribbles per 90",
                "Successful dribbles, %",
                "Accelerations per 90",
            ],
            "Progressive Passing": [
                "Progressive passes per 90",
                "Accurate progressive passes, %",
                "Passes to final third per 90",
                "Accurate passes to final third, %",
            ],
            "Dictating": [
                "Passes per 90",
                "Accurate passes, %",
                "Forward passes per 90",
                "Accurate forward passes, %",
                "Lateral passes per 90",
            ],
            "Ball Winning": [
                "PAdj Interceptions",
                "Successful defensive actions per 90",
                "Defensive duels won, %",
            ],
            "Destroying": [
                "Duels per 90",
                "Defensive duels per 90",
                "PAdj Sliding tackles",
                "Fouls per 90",
            ],
        },
        {
            "Anchor": ["Destroying", "Ball Winning", "Dictating"],
            "DLP": ["Dictating", "Progressive Passing", "Creativity"],
            "Ball Winner": ["Ball Winning", "Destroying"],
            "Box to Box": ["Progressive Passing", "Ball Carrying", "Box Crashing", "Ball Winning"],
            "Box Crasher": ["Box Crashing", "Ball Carrying", "Creativity"],
            "Playmaker": ["Creativity", "Progressive Passing", "Dictating"],
            "Attacking mid": ["Creativity", "Box Crashing", "Progressive Passing"],
        },
    ),

    "Attacking Midfielder": (
        ["AMF", "RAMF", "LAMF", "LW", "RW"],
        {
            "Pressing": [
                "Successful defensive actions per 90",
                "PAdj Interceptions",
                "Defensive duels per 90",
            ],
            "Build up": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90"],
            "Final ball": ["Key passes per 90", "xA per 90", "Deep completions per 90"],
            "Wide creation": ["Crosses per 90", "Accurate crosses, %", "Passes to penalty area per 90"],
            "Movement": ["Accelerations per 90", "Touches in box per 90"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90"],
            "1v1 ability": ["Dribbles per 90", "Successful dribbles, %", "Fouls suffered per 90"],
            "Box presence": ["Touches in box per 90", "xG per 90", "Shots per 90"],
            "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"],
        },
        {
            "Winger": ["Wide creation", "1v1 ability", "Ball Carrying"],
            "Direct Dribbler": ["1v1 ability", "Ball Carrying", "Movement"],
            "Industrious Winger": ["Pressing", "Ball Carrying", "Wide creation"],
            "Shadow Striker": ["Box presence", "Finishing", "Movement"],
            "Wide playmaker": ["Wide creation", "Final ball", "Build up"],
            "Playmaker": ["Final ball", "Build up", "1v1 ability"],
        },
    ),

    "Attacker": (
        ["CF", "LW", "RW"],
        {
            "Pressing": [
                "Successful defensive actions per 90",
                "PAdj Interceptions",
                "Defensive duels per 90",
            ],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"],
            "Creativity": ["Key passes per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
            "Link Play": ["Passes per 90", "Accurate passes, %", "Deep completions per 90", "Passes to final third per 90"],
            "Movement": ["Accelerations per 90", "Touches in box per 90"],
            "Box Presence": ["Touches in box per 90", "Shots per 90", "Aerial duels per 90", "Aerial duels won, %"],
            "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"],
        },
        {
            "Poacher": ["Finishing", "Box Presence", "Movement"],
            "Second Striker": ["Creativity", "Ball Carrying", "Finishing"],
            "Link Forward": ["Link Play", "Creativity", "Ball Carrying"],
            "False 9": ["Link Play", "Creativity", "Ball Carrying"],
            "Complete Forward": ["Finishing", "Box Presence", "Ball Carrying", "Link Play"],
            "Power Forward": ["Box Presence", "Finishing"],
            "Pressing Forward": ["Pressing", "Movement", "Box Presence"],
        },
    ),
}


# =============================================================
# GENERATE PROFILE
# =============================================================
def generate_player_profile(df, player_name, position_group):

    valid_positions, categories, role_defs = groups[position_group]
    data = df.copy()

    # =========================================================
    # CATEGORY PERCENTILES (TRUE CALCULATION – NO DEFAULTS)
    # =========================================================
    category_percentile_cols = []

    for cat, metrics in categories.items():

        pct_cols = []

        for m in metrics:
            if m in data.columns:
                col = data[m].astype(float)
                z = zscore(col)
                data[f"{m}_pct"] = norm.cdf(z) * 100
                pct_cols.append(f"{m}_pct")

        # If no metrics exist → skip this category
        if len(pct_cols) == 0:
            continue

        # TRUE category mean (NaN ignored)
        data[f"{cat}_percentile"] = data[pct_cols].mean(axis=1, skipna=True)

        category_percentile_cols.append(f"{cat}_percentile")

    # =========================================================
    # ROLE SCORES
    # =========================================================
    for role, cat_list in role_defs.items():
        cols = [f"{c}_percentile" for c in cat_list if f"{c}_percentile" in data.columns]
        if len(cols):
            data[role] = data[cols].mean(axis=1, skipna=True)
        else:
            data[role] = 0

    # =========================================================
    # OVERALL RATING
    # =========================================================
    if len(category_percentile_cols):
        z = zscore(data[category_percentile_cols].mean(axis=1, skipna=True))
        data["Overall_percentile"] = norm.cdf(z) * 100
    else:
        data["Overall_percentile"] = 0

    # Player row
    player = data[data["Player"] == player_name].iloc[0]

    # Clean category percentiles for chart
    cat_percentiles = {}
    for col in category_percentile_cols:
        cat = col.replace("_percentile", "")
        cat_percentiles[cat] = float(player[col])

    # Role scores
    role_scores = {r: float(player[r]) for r in role_defs}

    top_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # =========================================================
    # SIMILARITY (FULLY NAN-SAFE)
    # =========================================================
    sim_cols = category_percentile_cols
    data[sim_cols] = data[sim_cols].fillna(data[sim_cols].mean())

    player_vec = player[sim_cols].to_numpy().reshape(1, -1)
    all_vecs = data[sim_cols].to_numpy()

    sims = cosine_similarity(player_vec, all_vecs).flatten()
    data["Similarity"] = sims

    top_similar = (
        data[data["Player"] != player_name]
        .sort_values("Similarity", ascending=False)
        .head(4)
    )

    # =========================================================
    # VISUALIZATION
    # =========================================================
    fig = plt.figure(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("white")

    # TITLE
    fig.text(0.05, 0.93, player_name, fontsize=24, weight="bold")

    # BIO
    bio = f"""
Age: {player.get("Age", "N/A")}
Position: {player.get("Position", "N/A")}
Team: {player.get("Team", "N/A")}
Passport country: {player.get("Passport country", "N/A")}
Minutes: {int(player.get("Minutes played", 0))}
"""
    fig.text(0.05, 0.88, "Biography", fontsize=14, weight="bold")
    fig.text(0.05, 0.85, bio, fontsize=11, va="top")

    # ROLE SCORES
    x0, y0 = 0.55, 0.92
    box = 0.012

    for i, (r, score) in enumerate(role_scores.items()):
        fig.text(x0, y0 - i * 0.045, r, fontsize=10)
        for j in range(10):
            c = "gold" if j < round(score / 10) else "lightgrey"
            rect = plt.Rectangle(
                (x0 + 0.22 + j * 0.022, y0 - i * 0.045 - 0.005),
                box, box,
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="black"
            )
            fig.add_artist(rect)

    # OVERALL
    rect = plt.Rectangle((0.33, 0.78), 0.10, 0.08, transform=fig.transFigure,
                         facecolor="lightblue", edgecolor="black")
    fig.add_artist(rect)
    fig.text(0.335, 0.83, "Rating:", fontsize=10, weight="bold")
    fig.text(0.38, 0.795, f"{player['Overall_percentile']:.1f}",
             fontsize=13, weight="bold")

    # TOP ROLES TILES
    tile_w, tile_h = 0.14, 0.06

    for i, (r, s) in enumerate(top_roles):
        tx = 0.05 + i * (tile_w + 0.025)
        rect = plt.Rectangle((tx, 0.64), tile_w, tile_h, transform=fig.transFigure,
                             facecolor="gold", edgecolor="black")
        fig.add_artist(rect)
        fig.text(tx + 0.01, 0.64 + tile_h/2, r, fontsize=8, weight="bold", va="center")
        fig.text(tx + tile_w - 0.01, 0.64 + tile_h/2, f"{s:.0f}",
                 fontsize=10, weight="bold", va="center", ha="right")

    # CATEGORY BARS
    ax = fig.add_axes([0.05, 0.20, 0.9, 0.35])
    sorted_cp = dict(sorted(cat_percentiles.items(), key=lambda x: x[1]))

    bars = ax.barh(list(sorted_cp.keys()), list(sorted_cp.values()),
                   color="gold", edgecolor="black")
    ax.set_xlim(0, 100)
    ax.set_title("Positional Responsibilities", fontsize=12, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for b in bars:
        w = b.get_width()
        ax.text(w + 1, b.get_y() + b.get_height() / 2, f"{w:.1f}", va="center", fontsize=9)

    # SIMILAR PLAYERS
    fig.text(0.05, 0.12, "Similar Player Profiles", fontsize=12, weight="bold")

    for i, (_, row) in enumerate(top_similar.iterrows()):
        tx = 0.05 + i * (0.22 + 0.02)
        rect = plt.Rectangle((tx, 0.02), 0.22, 0.08, transform=fig.transFigure,
                             facecolor="lightgreen", edgecolor="black")
        fig.add_artist(rect)
        fig.text(tx + 0.01, 0.08, row["Player"], fontsize=10, weight="bold")
        fig.text(tx + 0.01, 0.06, row["Team"], fontsize=9)
        fig.text(tx + 0.01, 0.04, f"{row['Similarity'] * 100:.1f}% Similarity", fontsize=9)

    st.pyplot(fig)


# =============================================================
# RUN
# =============================================================
if run_button:
    generate_player_profile(df, selected_player, position_group)
