import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# LOAD DATA
# =============================================================
df = pd.read_excel("Iceland.xlsx").copy()

numeric_cols = [
    "Goals per 90","xG per 90","Shots per 90",
    "Assists per 90","xA per 90",
    "PAdj Interceptions","PAdj Sliding tackles",
    "Aerial duels won, %","Defensive duels won, %",
    "Shots blocked per 90",
    "Key passes per 90","Through passes per 90",
    "Passes to final third per 90","Passes to penalty area per 90"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df = df[df["Player"].notna()]

# =============================================================
# SCORING
# =============================================================
def percentile(s): return s.rank(pct=True) * 100

df["Offensive Score"] = percentile(df[["Goals per 90","xG per 90","Shots per 90",
                                      "Assists per 90","xA per 90"]].mean(axis=1))

df["Defensive Score"] = percentile(df[["PAdj Interceptions","PAdj Sliding tackles",
                                      "Aerial duels won, %","Defensive duels won, %",
                                      "Shots blocked per 90"]].mean(axis=1))

df["Key Passing Score"] = percentile(df[["Key passes per 90","Through passes per 90",
                                        "Assists per 90","xA per 90",
                                        "Passes to final third per 90","Passes to penalty area per 90"]].mean(axis=1))

# =============================================================
# PLAYER PROFILE
# =============================================================
def show_player_profile(row):
    st.markdown(f"## {row['Player']}")
    st.markdown(f"**Team:** {row['Team']} — {row['Position']}<br>**Minutes:** {int(row['Minutes played'])}",
                unsafe_allow_html=True)

    metrics = {
        "Off": row["Offensive Score"],
        "Def": row["Defensive Score"],
        "KeyP": row["Key Passing Score"]
    }

    labels = list(metrics.keys())
    stats = list(metrics.values())
    stats += stats[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(5, 5), facecolor="#000")
    ax.set_facecolor("#000")

    ax.plot(angles, stats, color="#FF5C35", linewidth=2)
    ax.fill(angles, stats, color="#FF5C35", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels([])
    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.header("Filters")

search = st.sidebar.text_input("Search")

teams = ["All"] + sorted(df["Team"].unique())
team_filter = st.sidebar.selectbox("Team", teams)

positions = sorted(df["Position"].unique())
position_filter = st.sidebar.multiselect("Position", positions, positions)

min_minutes = st.sidebar.slider("Minutes Played",
                                int(df["Minutes played"].min()),
                                int(df["Minutes played"].max()),
                                (200, int(df["Minutes played"].max())))

st.sidebar.subheader("Score Filters")
min_off = st.sidebar.slider("Min Offensive", 0, 100, 0)
min_def = st.sidebar.slider("Min Defensive", 0, 100, 0)
min_key = st.sidebar.slider("Min Key Passing", 0, 100, 0)

# =============================================================
# FILTER DATA
# =============================================================
df_f = df.copy()

if search:
    df_f = df_f[df_f["Player"].str.contains(search, case=False)]

if team_filter != "All":
    df_f = df_f[df_f["Team"] == team_filter]

df_f = df_f[df_f["Position"].isin(position_filter)]

df_f = df_f[(df_f["Minutes played"] >= min_minutes[0]) &
            (df_f["Minutes played"] <= min_minutes[1])]

df_f = df_f[(df_f["Offensive Score"] >= min_off) &
            (df_f["Defensive Score"] >= min_def) &
            (df_f["Key Passing Score"] >= min_key)]

# =============================================================
# NAVIGATION MODES
# =============================================================
mode = st.radio("Mode",
                ["Player Explorer",
                 "Player Comparison",
                 "Team Dashboard",
                 "Style Map (PCA)",
                 "Role Clustering"],
                horizontal=True)

# =============================================================
# MODE: PLAYER EXPLORER
# =============================================================
if mode == "Player Explorer":
    st.markdown("## Player Explorer")

    st.dataframe(df_f[[
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]], hide_index=True, use_container_width=True)

    player_choice = st.selectbox("Inspect Player", ["None"] + df_f["Player"].tolist())
    if player_choice != "None":
        show_player_profile(df_f[df_f["Player"] == player_choice].iloc[0])

# =============================================================
# MODE: PLAYER COMPARISON
# =============================================================
elif mode == "Player Comparison":
    st.markdown("## Player Comparison")

    players = df_f["Player"].tolist()
    p1 = st.selectbox("Player 1", players)
    p2 = st.selectbox("Player 2", players)

    if p1 and p2 and p1 != p2:
        r1 = df_f[df_f["Player"] == p1].iloc[0]
        r2 = df_f[df_f["Player"] == p2].iloc[0]

        c1, c2 = st.columns(2)
        with c1: show_player_profile(r1)
        with c2: show_player_profile(r2)

        st.markdown("### Attribute Gaps")

        compare_df = pd.DataFrame({
            "Metric": ["Offense", "Defense", "Key Passing"],
            p1: [r1["Offensive Score"], r1["Defensive Score"], r1["Key Passing Score"]],
            p2: [r2["Offensive Score"], r2["Defensive Score"], r2["Key Passing Score"]],
        })

        compare_df["Difference"] = compare_df[p1] - compare_df[p2]
        st.table(compare_df)

# =============================================================
# MODE: TEAM DASHBOARD
# =============================================================
elif mode == "Team Dashboard":
    st.markdown("## Team Dashboard")

    team = st.selectbox("Select team", sorted(df["Team"].unique()))
    df_t = df[df["Team"] == team]

    st.markdown(f"### Top Players – {team}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Offensive")
        st.table(df_t.nlargest(5, "Offensive Score")[["Player", "Offensive Score"]])

    with c2:
        st.markdown("#### Defensive")
        st.table(df_t.nlargest(5, "Defensive Score")[["Player", "Defensive Score"]])

    with c3:
        st.markdown("#### Key Passing")
        st.table(df_t.nlargest(5, "Key Passing Score")[["Player", "Key Passing Score"]])

# =============================================================
# MODE: STYLE MAP (PCA)
# =============================================================
elif mode == "Style Map (PCA)":
    st.markdown("## Player Style Map (PCA)")

    data = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df_f["PC1"] = components[:,0]
    df_f["PC2"] = components[:,1]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#000")
    ax.set_facecolor("#000")

    ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.85)

    for _, r in df_f.iterrows():
        ax.text(r["PC1"], r["PC2"], r["Player"], fontsize=8, color="white")

    ax.set_xlabel("PC1 (style)", color="white")
    ax.set_ylabel("PC2 (style)", color="white")

    st.pyplot(fig)

# =============================================================
# MODE: ROLE CLUSTERING
# =============================================================
elif mode == "Role Clustering":
    st.markdown("## Role Clustering (K-Means)")

    features = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    k = st.slider("Number of clusters", 2, 8, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_f["Cluster"] = kmeans.fit_predict(X)

    st.dataframe(df_f[["Player","Team","Cluster",
                       "Offensive Score","Defensive Score","Key Passing Score"]],
                 hide_index=True)

