import math

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

from mplsoccer import PyPizza  # make sure mplsoccer is installed

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(page_title="FiveThirtyEight Scouting – Full Expanded", layout="wide")

# =============================================================
# DARK THEME CSS
# =============================================================
st.markdown("""
<style>

:root {
    --bg: #000000;
    --fg: #FFFFFF;
    --card: #111111;
    --border: #303030;
    --accent: #FF5C35;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
}

[data-testid="stSidebar"] {
    background-color: #0C0C0C !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--fg) !important;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--fg);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.3rem;
    margin-top: 1.4rem;
}

.result-card {
    background: var(--card);
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 6px;
}

.result-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #CCCCCC;
}

.result-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--fg);
}

svg text { fill: var(--fg) !important; }

.dataframe td, .dataframe th {
    color: var(--fg) !important;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD & CLEAN DATA
# =============================================================
DATA_PATH = "Iceland.xlsx"  # adjust path if needed
df = pd.read_excel(DATA_PATH).copy()

# Force Team and Position to strings (avoid sorting errors)
df["Team"] = df["Team"].astype(str)
df["Position"] = df["Position"].astype(str)

# Numeric columns used in scoring & pizza
numeric_cols = [
    "Goals per 90", "xG per 90", "Shots per 90",
    "Assists per 90", "xA per 90",
    "PAdj Interceptions", "PAdj Sliding tackles",
    "Aerial duels won, %", "Defensive duels won, %",
    "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df = df[df["Player"].notna()]  # drop empty rows

# =============================================================
# SCORING FUNCTIONS
# =============================================================
def percentile(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100

df["Offensive Score"] = percentile(
    df[["Goals per 90", "xG per 90", "Shots per 90",
        "Assists per 90", "xA per 90"]].mean(axis=1)
)

df["Defensive Score"] = percentile(
    df[["PAdj Interceptions", "PAdj Sliding tackles",
        "Aerial duels won, %", "Defensive duels won, %",
        "Shots blocked per 90"]].mean(axis=1)
)

df["Key Passing Score"] = percentile(
    df[["Key passes per 90", "Through passes per 90",
        "Assists per 90", "xA per 90",
        "Passes to final third per 90", "Passes to penalty area per 90"]].mean(axis=1)
)

# =============================================================
# PLAYER PROFILE PANEL
# =============================================================
def show_player_profile(row: pd.Series):
    st.markdown(f"## {row['Player']}")
    st.markdown(
        f"**Team:** {row['Team']} — {row['Position']}<br>"
        f"**Minutes:** {int(row['Minutes played'])}",
        unsafe_allow_html=True,
    )

    metrics = {
        "Off": row["Offensive Score"],
        "Def": row["Defensive Score"],
        "KeyP": row["Key Passing Score"],
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(5, 5),
        subplot_kw=dict(polar=True),
        facecolor="#000",
    )
    ax.set_facecolor("#000")

    ax.plot(angles, values, color="#FF5C35", linewidth=2)
    ax.fill(angles, values, color="#FF5C35", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels([])
    st.pyplot(fig)

# =============================================================
# ROLE NAMING LOGIC FOR K-MEANS CLUSTERS
# =============================================================
def assign_role_names(df_roles: pd.DataFrame, feature_cols, kmeans_model: KMeans) -> pd.DataFrame:
    """
    Assign readable football role names based on cluster centroids in feature_cols.
    """
    centroids = pd.DataFrame(kmeans_model.cluster_centers_, columns=feature_cols)

    names = []
    for _, row in centroids.iterrows():
        off = row["Offensive Score"]
        deff = row["Defensive Score"]
        kp = row["Key Passing Score"]

        # basic shape logic
        if off > kp and off > deff:
            role = "Attacking Forward"
        elif kp > off and kp > deff:
            role = "Advanced Creator"
        elif deff > off and deff > kp:
            role = "Defensive Destroyer"
        elif off > 0.7 and kp > 0.7:
            role = "Attacking Playmaker"
        elif deff > 0.7 and kp > 0.7:
            role = "Deep Progressor"
        elif deff > 0.7 and off > 0.7:
            role = "Box-to-Box Hybrid"
        else:
            role = "All-Round Player"

        names.append(role)

    # map cluster index → role name
    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: names[x])
    return df_roles

# =============================================================
# PIZZA CHART FUNCTION
# =============================================================
def show_pizza_chart(df_all: pd.DataFrame, player_name: str, minute_threshold: int = 900):
    """
    Streamlit-friendly PyPizza chart like your example, using league percentiles.
    Filters by minutes >= minute_threshold to define population.
    """
    # Filter population for percentiles
    df_p = df_all[df_all["Minutes played"] >= minute_threshold].copy()
    if len(df_p) == 0:
        st.warning("No players meet the minutes threshold for pizza chart.")
        return

    # Parameters (you can adjust order / content here)
    params = [
        "Goals per 90", "Shots per 90", "Assists per 90", "xG per 90", "xA per 90",
        "Key passes per 90", "Through passes per 90",
        "Passes to final third per 90", "Passes to penalty area per 90",
        "PAdj Interceptions", "PAdj Sliding tackles",
        "Defensive duels won, %", "Aerial duels won, %",
        "Shots blocked per 90",
    ]

    # Ensure columns exist
    params = [p for p in params if p in df_p.columns]

    # Make numeric
    for p in params:
        df_p[p] = pd.to_numeric(df_p[p], errors="coerce")

    if player_name not in df_p["Player"].values:
        st.warning("Selected player does not meet the minute/position filter or isn't in the dataset.")
        return

    player_row = df_p[df_p["Player"] == player_name].iloc[0]

    # Percentiles
    values = []
    for p in params:
        percentile_val = stats.percentileofscore(df_p[p].dropna(), player_row[p])
        percentile_val = 99 if percentile_val == 100 else math.floor(percentile_val)
        values.append(percentile_val)

    # Build pizza
    baker = PyPizza(
        params=params,
        straight_line_color="white",
        straight_line_lw=1.5,
        last_circle_lw=6,
        other_circle_lw=2.5,
        other_circle_ls="-.",
        inner_circle_size=15,
    )

    # Slice colors: attacking / defending / key passing grouping (roughly)
    n = len(params)
    # Simple scheme: first 5 attacking, next 4 key passing, rest defensive
    slice_colors = []
    for i, p in enumerate(params):
        if i < 5:
            slice_colors.append("#598BAF")   # attacking
        elif i < 9:
            slice_colors.append("#ffa600")   # key passing
        else:
            slice_colors.append("#ff6361")   # defending

    text_colors = ["white"] * n

    fig, ax = baker.make_pizza(
        values,
        figsize=(12, 12),
        param_location=110,
        color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_slices=dict(
            edgecolor="black",
            zorder=2,
            linewidth=2,
        ),
        kwargs_params=dict(
            color="white",
            fontsize=12,
            weight="bold",
            fontname="Arial",
            va="center",
            alpha=.9,
        ),
        kwargs_values=dict(
            color="white",
            fontsize=10,
            weight="bold",
            fontname="Arial",
            zorder=3,
            bbox=dict(
                edgecolor="white",
                facecolor="#1a1a1a",
                boxstyle="round,pad=0.3",
                lw=1,
            ),
        ),
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Title & subtitle
    fig.text(
        0.5, 0.97,
        f"{player_name}",
        size=26, ha="center", color="white",
        weight="bold", fontname="Arial",
    )

    # Simple subtitle: team, minutes
    fig.text(
        0.5, 0.94,
        f"{player_row['Team']} | Minutes: {int(player_row['Minutes played'])}",
        size=14, ha="center", color="white", fontname="Arial",
    )

    # Category labels & colored rectangles like your example
    fig.text(
        0.35, 0.9,
        "Attacking     Key passing     Defending",
        size=14, color="white", fontname="Arial",
    )

    fig.patches.extend([
        plt.Rectangle((0.32, 0.897), 0.018, 0.018, fill=True, color="#598BAF", transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.46, 0.897), 0.018, 0.018, fill=True, color="#ffa600", transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.60, 0.897), 0.018, 0.018, fill=True, color="#ff6361", transform=fig.transFigure, figure=fig),
    ])

    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.title("Filters")

search = st.sidebar.text_input("Search Player")

teams = ["All"] + sorted(df["Team"].unique().tolist())
team_filter = st.sidebar.selectbox("Team", teams)

positions = sorted(df["Position"].unique().tolist())
position_filter = st.sidebar.multiselect("Position", positions, positions)

minutes = st.sidebar.slider(
    "Minutes Played",
    int(df["Minutes played"].min()),
    int(df["Minutes played"].max()),
    (200, int(df["Minutes played"].max())),
)

st.sidebar.subheader("Score Filters")
min_off = st.sidebar.slider("Min Offensive", 0, 100, 0)
min_def = st.sidebar.slider("Min Defensive", 0, 100, 0)
min_key = st.sidebar.slider("Min Key Passing", 0, 100, 0)

# =============================================================
# APPLY FILTERS
# =============================================================
df_f = df.copy()

if search:
    df_f = df_f[df_f["Player"].str.contains(search, case=False, na=False)]

if team_filter != "All":
    df_f = df_f[df_f["Team"] == team_filter]

df_f = df_f[df_f["Position"].isin(position_filter)]

df_f = df_f[
    (df_f["Minutes played"] >= minutes[0]) &
    (df_f["Minutes played"] <= minutes[1])
]

df_f = df_f[
    (df_f["Offensive Score"] >= min_off) &
    (df_f["Defensive Score"] >= min_def) &
    (df_f["Key Passing Score"] >= min_key)
]

# =============================================================
# NAVIGATION
# =============================================================
mode = st.radio(
    "Mode",
    [
        "Player Explorer",
        "Player Comparison",
        "Team Dashboard",
        "Style Map (PCA)",
        "Role Clustering",
        "Pizza Chart",
    ],
    horizontal=True,
)

# =============================================================
# MODE 1 — PLAYER EXPLORER
# =============================================================
if mode == "Player Explorer":
    st.markdown("## Player Explorer")

    st.dataframe(
        df_f[[
            "Player", "Team", "Position", "Minutes played",
            "Offensive Score", "Defensive Score", "Key Passing Score",
        ]],
        hide_index=True,
        use_container_width=True,
    )

    choice = st.selectbox("Select player", ["None"] + df_f["Player"].tolist())
    if choice != "None":
        show_player_profile(df_f[df_f["Player"] == choice].iloc[0])

# =============================================================
# MODE 2 — PLAYER COMPARISON
# =============================================================
elif mode == "Player Comparison":
    st.markdown("## Player Comparison")

    players = df_f["Player"].tolist()
    if len(players) < 2:
        st.warning("Not enough players after filters to compare.")
    else:
        p1 = st.selectbox("Player 1", players, index=0)
        p2 = st.selectbox("Player 2", players, index=1 if len(players) > 1 else 0)

        if p1 != p2:
            c1, c2 = st.columns(2)
            with c1:
                show_player_profile(df_f[df_f["Player"] == p1].iloc[0])
            with c2:
                show_player_profile(df_f[df_f["Player"] == p2].iloc[0])

# =============================================================
# MODE 3 — TEAM DASHBOARD
# =============================================================
elif mode == "Team Dashboard":
    st.markdown("## Team Dashboard")

    team = st.selectbox("Select Team", sorted(df["Team"].unique().tolist()))
    df_team = df[df["Team"] == team]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Top Offensive")
        st.table(df_team.nlargest(5, "Offensive Score")[["Player", "Offensive Score"]])

    with c2:
        st.markdown("### Top Defensive")
        st.table(df_team.nlargest(5, "Defensive Score")[["Player", "Defensive Score"]])

    with c3:
        st.markdown("### Top Creators")
        st.table(df_team.nlargest(5, "Key Passing Score")[["Player", "Key Passing Score"]])

# =============================================================
# MODE 4 — PCA STYLE MAP
# =============================================================
elif mode == "Style Map (PCA)":
    st.markdown("## Player Style Map (PCA)")

    if len(df_f) < 2:
        st.warning("Not enough players after filters to compute PCA.")
    else:
        X = df_f[["Offensive Score", "Defensive Score", "Key Passing Score"]].copy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)

        df_f["PC1"] = coords[:, 0]
        df_f["PC2"] = coords[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#000")
        ax.set_facecolor("#000")
        ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.84)

        for _, row in df_f.iterrows():
            ax.text(row["PC1"], row["PC2"], row["Player"], fontsize=8, color="white")

        ax.set_xlabel("PC1", color="white")
        ax.set_ylabel("PC2", color="white")
        ax.tick_params(colors="white")
        ax.grid(color="#333333", alpha=0.4)

        st.pyplot(fig)

# =============================================================
# MODE 5 — ROLE CLUSTERING (WITH NAMES)
# =============================================================
elif mode == "Role Clustering":
    st.markdown("## Role Clustering with Named Roles")

    if len(df_f) < 2:
        st.warning("Not enough players after filters to cluster.")
    else:
        feature_cols = ["Offensive Score", "Defensive Score", "Key Passing Score"]
        features = df_f[feature_cols]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        k = st.slider("Number of roles (clusters)", 2, 8, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_f["Role"] = kmeans.fit_predict(X)

        df_roles = assign_role_names(df_f.copy(), feature_cols, kmeans)

        st.dataframe(
            df_roles[[
                "Player", "Team", "Role", "Role Name",
                "Offensive Score", "Defensive Score", "Key Passing Score",
            ]],
            hide_index=True,
            use_container_width=True,
        )

# =============================================================
# MODE 6 — PIZZA CHART
# =============================================================
elif mode == "Pizza Chart":
    st.markdown("## Player Pizza Chart")

    # Use unfiltered df so a player doesn't disappear due to sidebar filters
    player_choice = st.selectbox("Select player", sorted(df["Player"].unique().tolist()))

    min_thresh = st.slider("Minutes threshold for comparison population", 0, 2000, 900, step=100)

    if player_choice:
        show_pizza_chart(df, player_choice, minute_threshold=min_thresh)
