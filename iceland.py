import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(layout="wide", page_title="Player Profile Generator")


# ============================================================
# LOAD DATA
# ============================================================
file_path = "Iceland.xlsx"
df = pd.read_excel(file_path)


# ============================================================
# SIDEBAR UI
# ============================================================
st.title("Player Profile Generator")
st.sidebar.header("Filters")

teams = sorted(df["Team"].dropna().unique())
selected_team = st.sidebar.selectbox("Select Team", teams)

players = sorted(df[df["Team"] == selected_team]["Player"].dropna().unique())
selected_player = st.sidebar.selectbox("Select Player", players)

position_group = st.sidebar.selectbox(
    "Select Position Group",
    [
        "Centre-back",
        "Full-back",
        "Midfielder",
        "Attacking Midfielder",
        "Attacker",
    ],
)

run_button = st.sidebar.button("Generate Profile")


# ============================================================
# GROUP DEFINITIONS
# ============================================================
groups = {}

# ---- Centre-back ----
groups["Centre-back"] = (
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
)

# ---- Full-back ----
groups["Full-back"] = (
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
            "Counterpressing recoveries per 90",
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
        "Ball Carrying": [
            "Progressive runs per 90",
            "Dribbles per 90",
            "Successful dribbles, %",
        ],
    },
    {
        "False Wing": ["Playmaking", "Ball Carrying", "Final Third"],
        "Flyer": ["Overlapping", "Final Third", "Ball Carrying"],
        "Playmaker": ["Playmaking", "Security", "Final Third"],
        "Safety": ["Security", "Box Defending", "Pressing"],
        "Ballwinner": ["Duelling", "Pressing"],
        "Defensive FB": ["Box Defending", "Duelling", "Pressing"],
    },
)

# ---- Midfielder ----
groups["Midfielder"] = (
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
            "Counterpressing recoveries per 90",
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
        "Box to Box": [
            "Progressive Passing",
            "Ball Carrying",
            "Box Crashing",
            "Ball Winning",
        ],
        "Box Crasher": ["Box Crashing", "Ball Carrying", "Creativity"],
        "Playmaker": ["Creativity", "Progressive Passing", "Dictating"],
        "Attacking mid": ["Creativity", "Box Crashing", "Progressive Passing"],
    },
)

# ---- Attacking Midfielder ----
groups["Attacking Midfielder"] = (
    ["AMF", "RAMF", "LAMF", "LW", "RW"],
    {
        "Pressing": [
            "Counterpressing recoveries per 90",
            "PAdj Interceptions",
            "Defensive duels per 90",
        ],
        "Build up": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90"],
        "Final ball": ["Key passes per 90", "xA per 90", "Deep completions per 90"],
        "Wide creation": [
            "Crosses per 90",
            "Accurate crosses, %",
            "Passes to penalty area per 90",
        ],
        "Movement": ["Accelerations per 90", "Touches in box per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90"],
        "1v1 ability": [
            "Dribbles per 90",
            "Successful dribbles, %",
            "Fouls suffered per 90",
        ],
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
)

# ---- Attacker ----
groups["Attacker"] = (
    ["CF", "LW", "RW"],
    {
        "Pressing": [
            "Counterpressing recoveries per 90",
            "PAdj Interceptions",
            "Defensive duels per 90",
        ],
        "Ball Carrying": [
            "Progressive runs per 90",
            "Dribbles per 90",
            "Successful dribbles, %",
        ],
        "Creativity": [
            "Key passes per 90",
            "xA per 90",
            "Smart passes per 90",
            "Accurate smart passes, %",
        ],
        "Link Play": [
            "Passes per 90",
            "Accurate passes, %",
            "Deep completions per 90",
            "Passes to final third per 90",
        ],
        "Movement": ["Accelerations per 90", "Touches in box per 90"],
        "Box Presence": [
            "Touches in box per 90",
            "Shots per 90",
            "Aerial duels per 90",
            "Aerial duels won, %",
        ],
        "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"],
    },
    {
        "Poacher": ["Finishing", "Box Presence", "Movement"],
        "Second Striker": ["Creativity", "Ball Carrying", "Finishing"],
        "Link Forward": ["Link Play", "Creativity", "Ball Carrying"],
        "False 9": ["Link Play", "Creativity", "Ball Carrying"],
        "Complete Forward": [
            "Finishing",
            "Box Presence",
            "Ball Carrying",
            "Link Play",
        ],
        "Power Forward": ["Box Presence", "Finishing"],
        "Pressing Forward": ["Pressing", "Movement", "Box Presence"],
    },
)


# ============================================================
# MAIN PROFILE FUNCTION
# ============================================================
def generate_player_profile(df, player_name, position_group):

    df = df.copy()
    df.rename(columns={"Nationality": "Passport country"}, inplace=True)

    valid_positions, categories, roles = groups[position_group]

    # ---------------- SELECT PLAYER ----------------
    player_row_original = df[df["Player"] == player_name].iloc[0]

    # ---------------- COMPUTE CATEGORY PERCENTILES ----------------
    data = df.copy()

    for cat, metrics in categories.items():
        pct_cols = []
        for metric in metrics:
            if metric in data.columns:
                col = data[metric].astype(float)
                if col.isna().all():
                    continue
                data[f"{metric}_z"] = zscore(col)
                data[f"{metric}_pct"] = data[f"{metric}_z"].apply(
                    lambda x: norm.cdf(x) * 100
                )
                pct_cols.append(f"{metric}_pct")

        if pct_cols:
            data[f"{cat}_percentile"] = data[pct_cols].mean(axis=1)

    # ---------------- ROLE SCORES ----------------
    for role, cat_list in roles.items():
        cols = [
            f"{c}_percentile"
            for c in cat_list
            if f"{c}_percentile" in data.columns
        ]
        if cols:
            data[role] = data[cols].mean(axis=1)

    # ---------------- OVERALL RATING ----------------
    category_percentile_cols = [
        f"{cat}_percentile"
        for cat in categories
        if f"{cat}_percentile" in data.columns
    ]

    if category_percentile_cols:
        data["Overall_mean"] = data[category_percentile_cols].mean(axis=1)
        data["Overall_z"] = zscore(data["Overall_mean"])
        data["Overall_percentile"] = data["Overall_z"].apply(
            lambda x: norm.cdf(x) * 100
        )
        avg_rating = float(
            data[data["Player"] == player_name]["Overall_percentile"].iloc[0]
        )
    else:
        avg_rating = np.nan

    # Updated player row
    player_row = data[data["Player"] == player_name].iloc[0]

    # ---------------- CATEGORY PERCENTILES FOR VISUAL ----------------
    cat_percentiles = {
        cat: player_row.get(f"{cat}_percentile", 0)
        for cat in categories
        if f"{cat}_percentile" in data.columns
    }

    cat_percentiles = {
        cat: (0 if pd.isna(val) else float(val))
        for cat, val in cat_percentiles.items()
    }

    # ---------------- CLEAN ROLE SCORES ----------------
    role_scores = {role: player_row.get(role, 0) for role in roles}

    role_scores = {
        role: (0 if pd.isna(score) else float(score))
        for role, score in role_scores.items()
    }

    top_roles = sorted(
        role_scores.items(), key=lambda x: x[1], reverse=True
    )[:3]

    # ---------------- SIMILAR PLAYERS (NaN-safe) ----------------
    category_cols = [
        col for col in category_percentile_cols if col in data.columns
    ]

    if len(category_cols) == 0:
        top_similar = pd.DataFrame(
            columns=["Player", "Team", "Similarity"]
        )
    else:
        for col in category_cols:
            data[col] = data[col].astype(float).fillna(data[col].mean())

        player_vector = (
            player_row[category_cols].astype(float).values
        )
        player_vector = np.nan_to_num(
            player_vector, nan=np.nanmean(player_vector)
        ).reshape(1, -1)

        all_vectors = data[category_cols].astype(float).values
        all_vectors = np.nan_to_num(
            all_vectors, nan=np.nanmean(all_vectors)
        )

        sims = cosine_similarity(player_vector, all_vectors).flatten()

        sim_df = data.copy()
        sim_df["Similarity"] = sims
        sim_df = sim_df[sim_df["Player"] != player_name]
        top_similar = (
            sim_df.sort_values("Similarity", ascending=False).head(4)
        )

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig = plt.figure(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.05, 0.93, player_name, fontsize=24, weight="bold")

    # BIO
    bio = (
        f"Age: {player_row_original.get('Age', 'N/A')}\n"
        f"Position: {player_row_original.get('Position', 'N/A')}\n"
        f"Team: {player_row_original.get('Team', 'N/A')}\n"
        f"Passport country: {player_row_original.get('Passport country', 'N/A')}\n"
        f"Minutes: {int(player_row_original.get('Minutes played', 0))}"
    )

    fig.text(0.05, 0.88, "Biography", fontsize=14, weight="bold")
    fig.text(0.05, 0.85, bio, fontsize=11, linespacing=1.5)

    # ROLE BARS
    x0, y0 = 0.55, 0.92
    box = 0.012

    ordered_roles = sorted(
        role_scores.items(),
        key=lambda x: list(roles.keys()).index(x[0]),
    )

    for i, (role, score) in enumerate(ordered_roles):
        fig.text(x0, y0 - i * 0.045, role, fontsize=10)

        score_10 = int(round(score / 10))

        for j in range(10):
            color = "gold" if j < score_10 else "lightgrey"
            r = plt.Rectangle(
                (
                    x0 + 0.22 + j * 0.022,
                    y0 - i * 0.045 - 0.005,
                ),
                box,
                box,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="black",
                lw=0.3,
            )
            fig.add_artist(r)

    # OVERALL TILE
    if not pd.isna(avg_rating):
        r = plt.Rectangle(
            (0.33, 0.78),
            0.10,
            0.08,
            transform=fig.transFigure,
            facecolor="lightblue",
            edgecolor="black",
            lw=1,
        )
        fig.add_artist(r)
        fig.text(0.335, 0.83, "Rating:", fontsize=10, weight="bold")
        fig.text(
            0.38,
            0.795,
            f"{avg_rating:.1f}",
            fontsize=13,
            weight="bold",
        )

    # TOP ROLES
    tw, th, ty = 0.14, 0.06, 0.64
    for i, (role, score) in enumerate(top_roles):
        tx = 0.05 + i * (tw + 0.025)
        r = plt.Rectangle(
            (tx, ty),
            tw,
            th,
            transform=fig.transFigure,
            facecolor="gold",
            edgecolor="black",
            lw=1,
        )
        fig.add_artist(r)
        fig.text(
            tx + 0.01,
            ty + th / 2,
            role,
            fontsize=8,
            weight="bold",
            va="center",
        )
        fig.text(
            tx + tw - 0.01,
            ty + th / 2,
            f"{score:.0f}",
            fontsize=10,
            weight="bold",
            ha="right",
            va="center",
        )

    # CATEGORY BAR CHART
    ax = fig.add_axes([0.05, 0.20, 0.9, 0.35])

    cp_sorted = dict(
        sorted(cat_percentiles.items(), key=lambda x: x[1])
    )

    bars = ax.barh(
        list(cp_sorted.keys()),
        list(cp_sorted.values()),
        color="gold",
        edgecolor="black",
    )
    ax.set_xlim(0, 100)
    ax.set_title(
        "Positional Responsibilities",
        fontsize=12,
        weight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for b in bars:
        w = b.get_width()
        ax.text(
            w + 1,
            b.get_y() + b.get_height() / 2,
            f"{w:.1f}",
            fontsize=9,
            va="center",
        )

    # SIMILAR PLAYERS
    fig.text(0.05, 0.12, "Similar Player Profiles", fontsize=12, weight="bold")

    for i, (_, row) in enumerate(top_similar.iterrows()):
        tx = 0.05 + i * (0.22 + 0.02)

        r = plt.Rectangle(
            (tx, 0.02),
            0.22,
            0.08,
            transform=fig.transFigure,
            facecolor="lightgreen",
            edgecolor="black",
            lw=1,
        )
        fig.add_artist(r)

        fig.text(
            tx + 0.01,
            0.08,
            row["Player"],
            fontsize=10,
            weight="bold",
        )
        fig.text(
            tx + 0.01,
            0.06,
            row["Team"],
            fontsize=9,
        )
        fig.text(
            tx + 0.01,
            0.04,
            f"{row['Similarity']*100:.1f}% Similarity",
            fontsize=9,
        )

    st.pyplot(fig)


# ============================================================
# RUN
# ============================================================
if run_button:
    generate_player_profile(df, selected_player, position_group)
