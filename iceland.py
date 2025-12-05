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
</style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_excel("Iceland.xlsx").copy()

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
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")

# =============================================================
# SCORING
# =============================================================
def pct(s): return s.rank(pct=True) * 100

df["Offensive Score"] = pct(df[["Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90"]].mean(axis=1))
df["Defensive Score"] = pct(df[["PAdj Interceptions","PAdj Sliding tackles","Aerial duels won, %","Defensive duels won, %","Shots blocked per 90"]].mean(axis=1))
df["Key Passing Score"] = pct(df[["Key passes per 90","Through passes per 90","Assists per 90","xA per 90","Passes to final third per 90","Passes to penalty area per 90"]].mean(axis=1))

df = df[df["Player"].notna()]

# =============================================================
# PLAYER PROFILE RADAR
# =============================================================
def show_profile(r):
    st.markdown(f"<div class='tabTitle'>{r['Player']}</div>", unsafe_allow_html=True)
    st.write(f"**Team:** {r['Team']} | **Position:** {r['Position']} | **Minutes:** {int(r['Minutes played'])}")

    labels = ["Off", "Def", "KeyP"]
    vals = [r["Offensive Score"], r["Defensive Score"], r["Key Passing Score"]]
    vals += vals[:1]
    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(subplot_kw={'polar':True}, figsize=(5,5), facecolor="#000")
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
def assign_roles(df_roles, feature_cols, km):
    centroids = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    role_names = []
    for _, c in centroids.iterrows():
        if c["Offensive Score"] > c["Key Passing Score"] and c["Offensive Score"] > c["Defensive Score"]:
            rn = "Attacking Forward"
        elif c["Key Passing Score"] > c["Offensive Score"]:
            rn = "Advanced Creator"
        elif c["Defensive Score"] > 65:
            rn = "Defensive Destroyer"
        elif c.mean() > 70:
            rn = "Elite All-Rounder"
        elif c["Defensive Score"] > 50 and c["Key Passing Score"] > 50:
            rn = "Deep-Lying Progressor"
        else:
            rn = "Balanced Player"
        role_names.append(rn)

    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: role_names[x])
    return df_roles

# =============================================================
# PIZZA CHART FUNCTION
# =============================================================
def pizza(df_all, player_name, min_thresh=900):
    dfp = df_all[df_all["Minutes played"] >= min_thresh].copy()

    params = [
        "Goals per 90","Shots per 90","Assists per 90","xG per 90","xA per 90",
        "Key passes per 90","Through passes per 90","Passes to final third per 90",
        "Passes to penalty area per 90","PAdj Interceptions","PAdj Sliding tackles",
        "Defensive duels won, %","Aerial duels won, %","Shots blocked per 90"
    ]
    params = [p for p in params if p in dfp.columns]

    player_row = dfp[dfp["Player"] == player_name].iloc[0]

    vals = []
    for p in params:
        score = stats.percentileofscore(dfp[p].dropna(), player_row[p])
        vals.append(min(99, int(score)))

    colors = []
    for i, p in enumerate(params):
        if i < 5:
            colors.append("#598BAF")
        elif i < 9:
            colors.append("#ffa600")
        else:
            colors.append("#ff6361")

    baker = PyPizza(params=params, straight_line_color="white",
                    last_circle_lw=5, other_circle_lw=2, inner_circle_size=15)

    fig, ax = baker.make_pizza(vals,
        figsize=(12,12),
        slice_colors=colors,
        color_blank_space="same",
        kwargs_params=dict(color="white", fontsize=12),
        kwargs_values=dict(color="white", fontsize=10)
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5,0.97,player_name,ha="center",size=26,color="white",weight="bold")

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

mins = st.sidebar.slider("Minutes Played",
                         int(df["Minutes played"].min()),
                         int(df["Minutes played"].max()),
                         (200,int(df["Minutes played"].max())))

min_off = st.sidebar.slider("Min Offensive",0,100,0)
min_def = st.sidebar.slider("Min Defensive",0,100,0)
min_key = st.sidebar.slider("Min Key Passing",0,100,0)

# APPLY FILTERS
df_f = df.copy()
if search:
    df_f = df_f[df_f["Player"].str.contains(search, case=False)]
if team!="All":
    df_f = df_f[df_f["Team"]==team]
df_f = df_f[df_f["Position"].isin(pos_sel)]
df_f = df_f[(df_f["Minutes played"]>=mins[0])&(df_f["Minutes played"]<=mins[1])]
df_f = df_f[(df_f["Offensive Score"]>=min_off)&
            (df_f["Defensive Score"]>=min_def)&
            (df_f["Key Passing Score"]>=min_key)]

# =============================================================
# MAIN TABS (Professional top navigation)
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔎 Player Explorer",
    "🆚 Player Comparison",
    "🍕 Pizza Chart",
    "🧠 Role Clustering",
    "🏟️ Team Dashboard",
    "🌍 Style Map (PCA)"
])

# =============================================================
# TAB 1 — PLAYER EXPLORER
# =============================================================
with tab1:
    st.markdown("<div class='tabTitle'>Player Explorer</div>", unsafe_allow_html=True)

    st.dataframe(df_f[[
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]], hide_index=True)

    sel = st.selectbox("Select player", ["None"] + df_f["Player"].tolist())
    if sel != "None":
        show_profile(df_f[df_f["Player"]==sel].iloc[0])

# =============================================================
# TAB 2 — PLAYER COMPARISON
# =============================================================
with tab2:
    st.markdown("<div class='tabTitle'>Player Comparison</div>", unsafe_allow_html=True)

    players = df_f["Player"].tolist()
    if len(players) < 2:
        st.warning("Not enough players for comparison.")
    else:
        p1 = st.selectbox("Player 1", players)
        p2 = st.selectbox("Player 2", players, index=1)

        if p1 != p2:
            c1, c2 = st.columns(2)
            with c1: show_profile(df_f[df_f["Player"]==p1].iloc[0])
            with c2: show_profile(df_f[df_f["Player"]==p2].iloc[0])

# =============================================================
# TAB 3 — PIZZA CHART
# =============================================================
with tab3:
    st.markdown("<div class='tabTitle'>Pizza Chart</div>", unsafe_allow_html=True)

    all_players = sorted(df["Player"].unique().tolist())
    pc = st.selectbox("Player", all_players)
    threshold = st.slider("Minutes threshold", 0, 2000, 900, step=100)

    if pc:
        pizza(df, pc, threshold)

# =============================================================
# TAB 4 — ROLE CLUSTERING
# =============================================================
with tab4:
    st.markdown("<div class='tabTitle'>Role Clustering</div>", unsafe_allow_html=True)

    if len(df_f) < 3:
        st.warning("Not enough players for clustering.")
    else:
        feats = ["Offensive Score","Defensive Score","Key Passing Score"]
        X = StandardScaler().fit_transform(df_f[feats])

        k = st.slider("Number of roles", 2, 8, 4)
        km = KMeans(n_clusters=k, n_init=10)
        df_f["Role"] = km.fit_predict(X)

        df_roles = assign_roles(df_f.copy(), feats, km)

        st.dataframe(df_roles[[
            "Player","Team","Role","Role Name",
            "Offensive Score","Defensive Score","Key Passing Score"
        ]], hide_index=True)

# =============================================================
# TAB 5 — TEAM DASHBOARD
# =============================================================
with tab5:
    st.markdown("<div class='tabTitle'>Team Dashboard</div>", unsafe_allow_html=True)

    t = st.selectbox("Team", sorted(df["Team"].unique().tolist()))
    df_t = df[df["Team"] == t]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### Offensive")
        st.table(df_t.nlargest(5,"Offensive Score")[["Player","Offensive Score"]])
    with col2:
        st.write("### Defensive")
        st.table(df_t.nlargest(5,"Defensive Score")[["Player","Defensive Score"]])
    with col3:
        st.write("### Creators")
        st.table(df_t.nlargest(5,"Key Passing Score")[["Player","Key Passing Score"]])

# =============================================================
# TAB 6 — PCA STYLE MAP
# =============================================================
with tab6:
    st.markdown("<div class='tabTitle'>PCA Style Map</div>", unsafe_allow_html=True)

    if len(df_f) < 2:
        st.warning("Not enough players after filters.")
    else:
        feats = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]
        X = StandardScaler().fit_transform(feats)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)

        df_f["PC1"] = coords[:,0]
        df_f["PC2"] = coords[:,1]

        fig, ax = plt.subplots(figsize=(8,6), facecolor="#000")
        ax.set_facecolor("#000")

        ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.85)
        for _, row in df_f.iterrows():
            ax.text(row["PC1"], row["PC2"], row["Player"], fontsize=8, color="white")

        st.pyplot(fig)
