import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from mplsoccer import PyPizza

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(page_title="Pro Scouting Platform", layout="wide")

# =============================================================
# DARK PROFESSIONAL THEME
# =============================================================
st.markdown("""
<style>
:root {
    --bg: #000000;
    --fg: #FFFFFF;
    --card: #121212;
    --border: #303030;
    --accent: #FF5C35;
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
}
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--fg) !important;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--fg);
    margin-top: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}
.result-card {
    background: var(--card);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
}
.tabTitle {
    font-size: 1.3rem;
    font-weight: bold;
    color: var(--accent);
    margin-bottom: 0.8rem;
}
svg text { fill: var(--fg) !important; }
.dataframe td, .dataframe th { color: var(--fg) !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# HELPER: SAFE PLAYER LOOKUP
# =============================================================
def safe_get_player(df: pd.DataFrame, player_name: str):
    """Return row for player_name or None if not found."""
    sub = df[df["Player"] == player_name]
    if len(sub) == 0:
        return None
    return sub.iloc[0]

# =============================================================
# LOAD DATA
# =============================================================
DATA_PATH = "Iceland.xlsx"  # adjust path if needed
df = pd.read_excel(DATA_PATH).copy()

# Ensure these are strings to avoid sort/type issues
df["Team"] = df["Team"].astype(str)
df["Position"] = df["Position"].astype(str)

numeric_cols = [
    "Goals per 90", "xG per 90", "Shots per 90",
    "Assists per 90", "xA per 90",
    "PAdj Interceptions", "PAdj Sliding tackles",
    "Aerial duels won, %", "Defensive duels won, %",
    "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90",
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df["Minutes played"] = df["Minutes played"].fillna(0)
df = df[df["Player"].notna()]  # drop fully empty rows

# =============================================================
# SCORING
# =============================================================
def pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100

df["Offensive Score"] = pct(
    df[["Goals per 90", "xG per 90", "Shots per 90",
        "Assists per 90", "xA per 90"]].mean(axis=1)
)

df["Defensive Score"] = pct(
    df[["PAdj Interceptions", "PAdj Sliding tackles",
        "Aerial duels won, %", "Defensive duels won, %",
        "Shots blocked per 90"]].mean(axis=1)
)

df["Key Passing Score"] = pct(
    df[["Key passes per 90", "Through passes per 90",
        "Assists per 90", "xA per 90",
        "Passes to final third per 90",
        "Passes to penalty area per 90"]].mean(axis=1)
)

# =============================================================
# PLAYER PROFILE RADAR
# =============================================================
def show_profile(row: pd.Series):
    st.markdown(f"<div class='tabTitle'>{row['Player']}</div>", unsafe_allow_html=True)
    st.write(
        f"**Team:** {row['Team']} | **Position:** {row['Position']} | "
        f"**Minutes:** {int(row['Minutes played'])}"
    )

    labels = ["Off", "Def", "KeyP"]
    vals = [row["Offensive Score"], row["Defensive Score"], row["Key Passing Score"]]
    vals += vals[:1]
    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(5, 5), facecolor="#000")
    ax.set_facecolor("#000")
    ax.plot(ang, vals, color="#FF5C35", linewidth=2)
    ax.fill(ang, vals, color="#FF5C35", alpha=0.25)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels([])
    st.pyplot(fig)

# =============================================================
# ROLE NAMING
# =============================================================
def assign_roles(df_roles: pd.DataFrame, feature_cols, km: KMeans) -> pd.DataFrame:
    centroids = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    role_names = []

    for _, c in centroids.iterrows():
        off = c["Offensive Score"]
        deff = c["Defensive Score"]
        kp = c["Key Passing Score"]

        if off > kp and off > deff:
            rn = "Attacking Forward"
        elif kp > off and kp > deff:
            rn = "Advanced Creator"
        elif deff > 65:
            rn = "Defensive Destroyer"
        elif deff > 50 and kp > 50:
            rn = "Deep-Lying Progressor"
        elif off > 60 and kp > 60:
            rn = "Attacking Playmaker"
        elif c.mean() > 70:
            rn = "Elite All-Rounder"
        else:
            rn = "Balanced Player"

        role_names.append(rn)

    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: role_names[x])
    return df_roles

# =============================================================
# PIZZA CHART FUNCTION
# =============================================================
def pizza(df_all: pd.DataFrame, player_name: str, min_thresh: int = 900):
    """
    Pro-style pizza chart with percentiles and group coloring.
    """
    dfp = df_all[df_all["Minutes played"] >= min_thresh].copy()
    if len(dfp) == 0:
        st.warning("No players meet the minutes threshold for pizza chart.")
        return

    row = safe_get_player(dfp, player_name)
    if row is None:
        st.error("Player not available for pizza chart (minutes or filters).")
        return

    params = [
        "Goals per 90","Shots per 90","Assists per 90","xG per 90","xA per 90",
        "Key passes per 90","Through passes per 90","Passes to final third per 90",
        "Passes to penalty area per 90","PAdj Interceptions","PAdj Sliding tackles",
        "Defensive duels won, %","Aerial duels won, %","Shots blocked per 90"
    ]
    params = [p for p in params if p in dfp.columns]

    # Percentiles
    values = []
    for p in params:
        col = pd.to_numeric(dfp[p], errors="coerce").dropna()
        if len(col) == 0:
            values.append(50)
            continue
        perc = stats.percentileofscore(col, row[p])
        values.append(min(99, int(perc)))

    # Color grouping
    slice_colors = []
    for i, p in enumerate(params):
        if i < 5:
            slice_colors.append("#598BAF")  # attacking
        elif i < 9:
            slice_colors.append("#ffa600")  # key passing
        else:
            slice_colors.append("#ff6361")  # defending

    baker = PyPizza(
        params=params,
        straight_line_color="white",
        straight_line_lw=1.5,
        last_circle_lw=5,
        other_circle_lw=2,
        inner_circle_size=15
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(12, 12),
        color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_params=dict(color="white", fontsize=12),
        kwargs_values=dict(color="white", fontsize=10),
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5, 0.97, player_name, ha="center", size=26,
             color="white", weight="bold")

    fig.text(
        0.5, 0.94,
        f"{row['Team']} | Minutes: {int(row['Minutes played'])}",
        ha="center", size=14, color="white",
    )

    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.header("Filters")

search = st.sidebar.text_input("Search Player")

teams = ["All"] + sorted(df["Team"].unique().tolist())
team = st.sidebar.selectbox("Team", teams)

positions = sorted(df["Position"].unique().tolist())
pos_sel = st.sidebar.multiselect("Position", positions, positions)

mins = st.sidebar.slider(
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

if team != "All":
    df_f = df_f[df_f["Team"] == team]

df_f = df_f[df_f["Position"].isin(pos_sel)]

df_f = df_f[
    (df_f["Minutes played"] >= mins[0]) &
    (df_f["Minutes played"] <= mins[1])
]

df_f = df_f[
    (df_f["Offensive Score"] >= min_off) &
    (df_f["Defensive Score"] >= min_def) &
    (df_f["Key Passing Score"] >= min_key)
]

# If no players after filters, stop early
if len(df_f) == 0:
    st.markdown("## Pro Scouting Platform")
    st.warning("No players match the current filters. Relax filters in the sidebar.")
else:
    # =========================================================
    # MAIN TABS (BUTTON-LIKE TOP NAV)
    # =========================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔎 Player Explorer",
        "🆚 Player Comparison",
        "🍕 Pizza Chart",
        "🧠 Role Clustering",
        "🏟️ Team Dashboard",
        "🌍 Style Map (PCA)"
    ])

    # =========================================================
    # TAB 1 — PLAYER EXPLORER
    # =========================================================
    with tab1:
        st.markdown("<div class='tabTitle'>Player Explorer</div>", unsafe_allow_html=True)

        st.dataframe(
            df_f[[
                "Player", "Team", "Position", "Minutes played",
                "Offensive Score", "Defensive Score", "Key Passing Score"
            ]],
            hide_index=True,
            use_container_width=True,
        )

        sel = st.selectbox("Select player", ["None"] + df_f["Player"].tolist())
        if sel != "None":
            row = safe_get_player(df_f, sel)
            if row is not None:
                show_profile(row)
            else:
                st.error("Player not available under current filters.")

    # =========================================================
    # TAB 2 — PLAYER COMPARISON
    # =========================================================
    with tab2:
        st.markdown("<div class='tabTitle'>Player Comparison</div>", unsafe_allow_html=True)

        players = df_f["Player"].tolist()
        if len(players) < 2:
            st.warning("Not enough players for comparison.")
        else:
            p1 = st.selectbox("Player 1", players, index=0)
            p2 = st.selectbox("Player 2", players, index=min(1, len(players)-1))

            if p1 != p2:
                row1 = safe_get_player(df_f, p1)
                row2 = safe_get_player(df_f, p2)
                if row1 is not None and row2 is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        show_profile(row1)
                    with c2:
                        show_profile(row2)
                else:
                    st.error("One of the players is not available under current filters.")

    # =========================================================
    # TAB 3 — PIZZA CHART
    # =========================================================
    with tab3:
        st.markdown("<div class='tabTitle'>Pizza Chart</div>", unsafe_allow_html=True)

        all_players = sorted(df["Player"].unique().tolist())
        pc = st.selectbox("Player", all_players)
        threshold = st.slider("Minutes threshold (for comparison population)", 0, 2000, 900, step=100)

        if pc:
            pizza(df, pc, threshold)

    # =========================================================
    # TAB 4 — ROLE CLUSTERING
    # =========================================================
    with tab4:
        st.markdown("<div class='tabTitle'>Role Clustering</div>", unsafe_allow_html=True)

        if len(df_f) < 3:
            st.warning("Not enough players for clustering.")
        else:
            feature_cols = ["Offensive Score", "Defensive Score", "Key Passing Score"]
            X = StandardScaler().fit_transform(df_f[feature_cols])

            k = st.slider("Number of roles", 2, 8, 4)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            df_f["Role"] = km.fit_predict(X)

            df_roles = assign_roles(df_f.copy(), feature_cols, km)

            st.dataframe(
                df_roles[[
                    "Player", "Team", "Role", "Role Name",
                    "Offensive Score", "Defensive Score", "Key Passing Score"
                ]],
                hide_index=True,
                use_container_width=True,
            )

    # =========================================================
    # TAB 5 — TEAM DASHBOARD
    # =========================================================
    with tab5:
        st.markdown("<div class='tabTitle'>Team Dashboard</div>", unsafe_allow_html=True)

        teams_all = sorted(df["Team"].unique().tolist())
        t = st.selectbox("Team", teams_all)
        df_t = df[df["Team"] == t]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("### Offensive")
            st.table(df_t.nlargest(5, "Offensive Score")[["Player", "Offensive Score"]])
        with c2:
            st.write("### Defensive")
            st.table(df_t.nlargest(5, "Defensive Score")[["Player", "Defensive Score"]])
        with c3:
            st.write("### Creators")
            st.table(df_t.nlargest(5, "Key Passing Score")[["Player", "Key Passing Score"]])

    # =========================================================
    # TAB 6 — PCA STYLE MAP
    # =========================================================
    with tab6:
        st.markdown("<div class='tabTitle'>PCA Style Map</div>", unsafe_allow_html=True)

        if len(df_f) < 2:
            st.warning("Not enough players after filters.")
        else:
            feats = df_f[["Offensive Score", "Defensive Score", "Key Passing Score"]]
            X = StandardScaler().fit_transform(feats)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)

            df_f["PC1"] = coords[:, 0]
            df_f["PC2"] = coords[:, 1]

            fig, ax = plt.subplots(figsize=(8, 6), facecolor="#000")
            ax.set_facecolor("#000")

            ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.85)
            for _, row in df_f.iterrows():
                ax.text(row["PC1"], row["PC2"], row["Player"], fontsize=8, color="white")

            ax.set_xlabel("PC1", color="white")
            ax.set_ylabel("PC2", color="white")
            ax.tick_params(colors="white")
            ax.grid(color="#333333", alpha=0.4)

            st.pyplot(fig)
