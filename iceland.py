import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(layout="wide", page_title="Player Profile Generator")

# ---------------------- LOAD DATA ------------------------
file_path = "Iceland.xlsx"
df = pd.read_excel(file_path)

# ---------------------- TITLE ----------------------------
st.title("Player Profile Generator")

# ---------------------- SIDEBAR FILTERS ------------------
st.sidebar.header("Filters")

teams = sorted(df["Team"].dropna().unique())
selected_team = st.sidebar.selectbox("Select Team", teams)

players = sorted(df[df["Team"] == selected_team]["Player"].dropna().unique())
selected_player = st.sidebar.selectbox("Select Player", players)

position_group = st.sidebar.selectbox(
    "Select Position Group",
    ['Centre-back', 'Full-back', 'Midfielder', 'Attacking Midfielder', 'Attacker']
)

run_button = st.sidebar.button("Generate Profile")


# ---------------------- GROUP DEFINITIONS ----------------
groups = {}

groups['Centre-back'] = (
    ['CB', 'RCB', 'LCB'],
    {
        "Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"],
        "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Forward passes per 90", "Accurate forward passes, %", "Passes to final third per 90", "Accurate passes to final third, %"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"],
        "Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
        "Proactive Defending": ["Interceptions per 90", "PAdj Interceptions", "Sliding tackles per 90", "PAdj Sliding tackles"],
        "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %"],
        "Box Defending": ["Shots blocked per 90"],
        "Sweeping": []
    },
    {
        "Ball Player": ["Progressive Passing", "Security"],
        "Libero": ["Progressive Passing", "Ball Carrying", "Creativity"],
        "Wide Creator": ["Creativity", "Ball Carrying"],
        "Aggressor": ["Proactive Defending", "Duelling"],
        "Physical Dominator": ["Duelling", "Box Defending"],
        "Box Defender": ["Box Defending", "Duelling"]
    }
)

groups['Full-back'] = (
    ['LB', 'RB', 'LWB', 'RWB'],
    {
        "Box Defending": ["Shots blocked per 90"],
        "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %", "Defensive duels per 90", "Defensive duels won, %"],
        "Pressing": ["PAdj Interceptions", "PAdj Sliding tackles", "Counterpressing recoveries per 90"],
        "Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"],
        "Playmaking": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
        "Final Third": ["Passes to final third per 90", "Accurate passes to final third, %", "Crosses per 90", "Accurate crosses, %", "Touches in box per 90"],
        "Overlapping": ["Accelerations per 90", "Fouls suffered per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"]
    },
    {
        "False Wing": ["Playmaking", "Ball Carrying", "Final Third"],
        "Flyer": ["Overlapping", "Final Third", "Ball Carrying"],
        "Playmaker": ["Playmaking", "Security", "Final Third"],
        "Safety": ["Security", "Box Defending", "Pressing"],
        "Ballwinner": ["Duelling", "Pressing"],
        "Defensive FB": ["Box Defending", "Duelling", "Pressing"]
    }
)

groups['Midfielder'] = (
    ['LCMF', 'RCMF', 'CFM', 'LDMF', 'RDMF', 'RAMF', 'LAMF', 'AMF', 'DMF'],
    {
        "Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
        "Box Crashing": ["Touches in box per 90", "Shots per 90", "xG per 90", "Non-penalty goals per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"],
        "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Passes to final third per 90", "Accurate passes to final third, %"],
        "Dictating": ["Passes per 90", "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %", "Lateral passes per 90"],
        "Ball Winning": ["PAdj Interceptions", "Counterpressing recoveries per 90", "Defensive duels won, %"],
        "Destroying": ["Duels per 90", "Defensive duels per 90", "PAdj Sliding tackles", "Fouls per 90"]
    },
    {
        "Anchor": ["Destroying", "Ball Winning", "Dictating"],
        "DLP": ["Dictating", "Progressive Passing", "Creativity"],
        "Ball Winner": ["Ball Winning", "Destroying"],
        "Box to Box": ["Progressive Passing", "Ball Carrying", "Box Crashing", "Ball Winning"],
        "Box Crasher": ["Box Crashing", "Ball Carrying", "Creativity"],
        "Playmaker": ["Creativity", "Progressive Passing", "Dictating"],
        "Attacking mid": ["Creativity", "Box Crashing", "Progressive Passing"]
    }
)

groups['Attacking Midfielder'] = (
    ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'],
    {
        "Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"],
        "Build up": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90"],
        "Final ball": ["Key passes per 90", "xA per 90", "Deep completions per 90"],
        "Wide creation": ["Crosses per 90", "Accurate crosses, %", "Passes to penalty area per 90"],
        "Movement": ["Accelerations per 90", "Touches in box per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90"],
        "1v1 ability": ["Dribbles per 90", "Successful dribbles, %", "Fouls suffered per 90"],
        "Box presence": ["Touches in box per 90", "xG per 90", "Shots per 90"],
        "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]
    },
    {
        "Winger": ["Wide creation", "1v1 ability", "Ball Carrying"],
        "Direct Dribbler": ["1v1 ability", "Ball Carrying", "Movement"],
        "Industrious Winger": ["Pressing", "Ball Carrying", "Wide creation"],
        "Shadow Striker": ["Box presence", "Finishing", "Movement"],
        "Wide playmaker": ["Wide creation", "Final ball", "Build up"],
        "Playmaker": ["Final ball", "Build up", "1v1 ability"]
    }
)

groups['Attacker'] = (
    ['CF', 'LW', 'RW'],
    {
        "Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"],
        "Creativity": ["Key passes per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
        "Link Play": ["Passes per 90", "Accurate passes, %", "Deep completions per 90", "Passes to final third per 90"],
        "Movement": ["Accelerations per 90", "Touches in box per 90"],
        "Box Presence": ["Touches in box per 90", "Shots per 90", "Aerial duels per 90", "Aerial duels won, %"],
        "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]
    },
    {
        "Poacher": ["Finishing", "Box Presence", "Movement"],
        "Second Striker": ["Creativity", "Ball Carrying", "Finishing"],
        "Link Forward": ["Link Play", "Creativity", "Ball Carrying"],
        "False 9": ["Link Play", "Creativity", "Ball Carrying"],
        "Complete Forward": ["Finishing", "Box Presence", "Ball Carrying", "Link Play"],
        "Power Forward": ["Box Presence", "Finishing"],
        "Pressing Forward": ["Pressing", "Movement", "Box Presence"]
    }
)


# ---------------------- GENERATE PROFILE FUNCTION ----------------
def generate_player_profile(df, player_name, position_group):
    df = df.copy()
    df.rename(columns={"Nationality": "Passport country"}, inplace=True)

    valid_positions, categories, roles = groups[position_group]

    # ---------------- FILTER PLAYER ----------------
    player_row_original = df[df["Player"] == player_name].iloc[0]

    # ---------------- CALCULATE CATEGORY PERCENTILES ----------------
    data = df.copy()

    for cat, metrics in categories.items():
        pct_cols = []
        for metric in metrics:
            if metric in data.columns:
                data[f"{metric}_z"] = zscore(data[metric].astype(float))
                data[f"{metric}_pct"] = data[f"{metric}_z"].apply(lambda x: norm.cdf(x) * 100)
                pct_cols.append(f"{metric}_pct")
        if pct_cols:
            data[f"{cat}_percentile"] = data[pct_cols].mean(axis=1)

    # ---------------- ROLE SCORES ----------------
    for role, cat_list in roles.items():
        cat_pcts = [f"{c}_percentile" for c in cat_list if f"{c}_percentile" in data.columns]
        if cat_pcts:
            data[role] = data[cat_pcts].mean(axis=1)

    # ---------------- OVERALL RATING ----------------
    category_percentile_cols = [f"{cat}_percentile" for cat in categories if f"{cat}_percentile" in data.columns]

    if category_percentile_cols:
        data["Overall_mean"] = data[category_percentile_cols].mean(axis=1)
        data["Overall_z"] = zscore(data["Overall_mean"])
        data["Overall_percentile"] = data["Overall_z"].apply(lambda x: norm.cdf(x) * 100)
        avg_rating = data.loc[data["Player"] == player_name, "Overall_percentile"].values[0]
    else:
        avg_rating = np.nan

    # Retrieve processed player row
    player_row = data[data["Player"] == player_name].iloc[0]

    # ---------------- PREP VISUAL DATA ----------------
    age = player_row_original.get("Age", "N/A")
    position = player_row_original.get("Position", "N/A")
    team = player_row_original.get("Team", "N/A")
    nation = player_row_original.get("Passport country", "N/A")
    minutes = player_row_original.get("Minutes played", 0)

    cat_percentiles = {
        cat: player_row.get(f"{cat}_percentile", 0)
        for cat in categories
        if not pd.isna(player_row.get(f"{cat}_percentile", None))
    }

    role_scores = {role: player_row.get(role, 0) for role in roles}
    top_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # ---------------- SIMILAR PLAYERS (FIXED FOR NAN) ----------------
    category_cols = [col for col in category_percentile_cols if col in data.columns]

    # Fill NaNs with column means
    data[category_cols] = data[category_cols].fillna(data[category_cols].mean())

    # Player vector
    player_vector = player_row[category_cols].values.astype(float)
    player_vector = np.nan_to_num(player_vector, nan=np.nanmean(player_vector))
    player_vector = player_vector.reshape(1, -1)

    all_vectors = data[category_cols].values.astype(float)

    similarities = cosine_similarity(player_vector, all_vectors).flatten()

    sim_df = data.copy()
    sim_df["Similarity"] = similarities
    sim_df = sim_df[sim_df["Player"] != player_name]
    top_similar = sim_df.sort_values("Similarity", ascending=False).head(4)

    # ---------------- VISUAL ----------------
    fig = plt.figure(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.05, 0.93, player_name, fontsize=24, weight="bold")

    # Biography
    fig.text(0.05, 0.88, "Biography", fontsize=14, weight="bold")
    bio_text = (
        f"Age: {age}\n"
        f"Position: {position}\n"
        f"Team: {team}\n"
        f"Passport country: {nation}\n"
        f"Minutes: {int(minutes)}"
    )
    fig.text(0.05, 0.85, bio_text, fontsize=11, va="top", linespacing=1.5)

    # Role score bars
    x0, y0 = 0.55, 0.92
    box_size = 0.012
    for i, (role, score) in enumerate(sorted(role_scores.items(), key=lambda x: list(roles.keys()).index(x[0]))):
        fig.text(x0, y0 - i * 0.045, role, fontsize=10)
        for j in range(10):
            color = "gold" if j < round(score / 10) else "lightgrey"
            rect = plt.Rectangle(
                (x0 + 0.22 + j * 0.022, y0 - i * 0.045 - 0.005),
                box_size, box_size,
                transform=fig.transFigure,
                facecolor=color, edgecolor="black", lw=0.3,
            )
            fig.add_artist(rect)

    # Overall rating
    if not pd.isna(avg_rating):
        rect = plt.Rectangle((0.33, 0.78), 0.10, 0.08, transform=fig.transFigure,
                             facecolor="lightblue", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(0.335, 0.83, "Rating:", fontsize=10, weight="bold")
        fig.text(0.38, 0.795, f"{avg_rating:.1f}", fontsize=13, weight="bold")

    # Top roles
    tile_w, tile_h, tile_y = 0.14, 0.06, 0.64
    for i, (role, score) in enumerate(top_roles):
        tx = 0.05 + i * (tile_w + 0.025)
        rect = plt.Rectangle((tx, tile_y), tile_w, tile_h, transform=fig.transFigure,
                             facecolor="gold", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(tx + 0.01, tile_y + tile_h / 2, role, fontsize=8, weight="bold",
                 ha="left", va="center")
        fig.text(tx + tile_w - 0.01, tile_y + tile_h / 2, f"{score:.0f}",
                 fontsize=10, weight="bold", ha="right", va="center")

    # Category bars
    ax_bar = fig.add_axes([0.05, 0.20, 0.9, 0.35])
    bar_data = dict(sorted(cat_percentiles.items(), key=lambda x: x[1]))
    bars = ax_bar.barh(list(bar_data.keys()), list(bar_data.values()),
                       color="gold", edgecolor="black")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_title("Positional Responsibilities", fontsize=12, weight="bold", loc="left")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax_bar.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}",
                    va="center", fontsize=9)

    # Similar players
    fig.text(0.05, 0.12, "Similar Player Profiles", fontsize=12, weight="bold")
    for i, (_, row) in enumerate(top_similar.iterrows()):
        tx = 0.05 + i * (0.22 + 0.02)
        rect = plt.Rectangle((tx, 0.02), 0.22, 0.08, transform=fig.transFigure,
                             facecolor="lightgreen", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(tx + 0.01, 0.08, row["Player"], fontsize=10, weight="bold")
        fig.text(tx + 0.01, 0.06, row["Team"], fontsize=9)
        fig.text(tx + 0.01, 0.04, f"{row['Similarity'] * 100:.1f}% Similarity", fontsize=9)

    # Streamlit output
    st.pyplot(fig)


# ---------------------- RUN APP ----------------------
if run_button:
    generate_player_profile(df, selected_player, position_group)
