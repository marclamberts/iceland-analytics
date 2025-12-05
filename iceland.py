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
st.set_page_config(page_title="Professional Scouting Platform", layout="wide")

# =============================================================
# PREMIUM TIGHT DARK THEME
# =============================================================
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #222 !important;
}

[data-testid="stSidebar"] * {
    color: #EEE !important;
    font-size: 0.85rem !important;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    padding-bottom: 4px;
    margin: 0 0 10px 0;
    border-bottom: 1px solid #222;
}

.compact-card {
    background-color: #111;
    border: 1px solid #222;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 0.7rem;
    color: #AAA;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFF;
}

.dataframe td, .dataframe th {
    color: #EEE !important;
    font-size: 0.8rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem !important;
    padding: 6px 16px !important;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# SAFETY WRAPPER FOR PLAYERS
# =============================================================
def safe_get_player(df, name):
    sub = df[df["Player"] == name]
    return None if sub.empty else sub.iloc[0]

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_excel("Iceland.xlsx")
df["Team"] = df["Team"].astype(str)
df["Position"] = df["Position"].astype(str)

# Numeric cleaning
numeric_cols = [
    "Goals per 90", "xG per 90", "Shots per 90", "Assists per 90",
    "xA per 90", "PAdj Interceptions", "PAdj Sliding tackles",
    "Aerial duels won, %", "Defensive duels won, %", "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df = df[df["Player"].notna()]

# =============================================================
# SCORING
# =============================================================
def pct(s): return s.rank(pct=True) * 100

df["Offensive Score"] = pct(df[[
    "Goals per 90", "xG per 90", "Shots per 90", "Assists per 90", "xA per 90"
]].mean(axis=1))

df["Defensive Score"] = pct(df[[
    "PAdj Interceptions", "PAdj Sliding tackles", "Aerial duels won, %",
    "Defensive duels won, %", "Shots blocked per 90"
]].mean(axis=1))

df["Key Passing Score"] = pct(df[[
    "Key passes per 90", "Through passes per 90", "Assists per 90", "xA per 90",
    "Passes to final third per 90", "Passes to penalty area per 90"
]].mean(axis=1))

# =============================================================
# PLAYER RADAR
# =============================================================
def show_profile(row):
    st.markdown(f"### {row['Player']}")
    st.caption(f"{row['Team']} • {row['Position']} • {int(row['Minutes played'])} mins")

    vals = [
        row["Offensive Score"],
        row["Defensive Score"],
        row["Key Passing Score"]
    ]
    vals += vals[:1]
    labels = ["Off", "Def", "Key"]
    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True), facecolor="#000")
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
def assign_roles(df_roles, cols, km):
    cen = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = []
    for _, c in cen.iterrows():
        if c["Offensive Score"] > c["Key Passing Score"] and c["Offensive Score"] > c["Defensive Score"]:
            names.append("Attacking Forward")
        elif c["Key Passing Score"] > c["Offensive Score"]:
            names.append("Advanced Creator")
        elif c["Defensive Score"] > 65:
            names.append("Defensive Anchor")
        elif c.mean() > 70:
            names.append("Elite All-Rounder")
        else:
            names.append("Hybrid Profile")
    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: names[x])
    return df_roles

# =============================================================
# PIZZA CHART
# =============================================================
def pizza(df_all, player, min_thresh=900):
    pool = df_all[df_all["Minutes played"] >= min_thresh]
    row = safe_get_player(pool, player)
    if row is None:
        st.error("Player unavailable under filters.")
        return

    params = [
        "Goals per 90","Shots per 90","Assists per 90","xG per 90","xA per 90",
        "Key passes per 90","Through passes per 90","Passes to final third per 90",
        "Passes to penalty area per 90","PAdj Interceptions",
        "PAdj Sliding tackles","Defensive duels won, %",
        "Aerial duels won, %","Shots blocked per 90"
    ]

    values = []
    for p in params:
        col = pool[p].dropna()
        perc = stats.percentileofscore(col, row[p])
        values.append(min(99, int(perc)))

    colors = ["#598BAF"]*5 + ["#ffa600"]*4 + ["#ff6361"]*5

    baker = PyPizza(params=params, straight_line_color="white",
                    last_circle_lw=5, other_circle_lw=2, inner_circle_size=15)

    fig, ax = baker.make_pizza(
        values, figsize=(10,10), slice_colors=colors,
        color_blank_space="same",
        kwargs_params=dict(color="white", fontsize=10),
        kwargs_values=dict(color="white", fontsize=9),
    )
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5,0.97, player, ha="center", size=22, weight="bold", color="white")
    fig.text(0.5,0.94, f"{row['Team']} | {int(row['Minutes played'])} mins", 
             ha="center", size=12, color="white")

    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
with st.sidebar:
    st.header("Filters")

    search = st.text_input("Search", "")

    teams = ["All"] + sorted(df["Team"].unique())
    team = st.selectbox("Team", teams)

    positions = sorted(df["Position"].unique())
    pos_sel = st.multiselect("Positions", positions, positions)

    mins = st.slider("Minutes", 0, int(df["Minutes played"].max()), (300, int(df["Minutes played"].max())))

    min_off = st.slider("Min Offensive", 0, 100, 0)
    min_def = st.slider("Min Defensive", 0, 100, 0)
    min_key = st.slider("Min Key Passing", 0, 100, 0)

# APPLY FILTERS
df_f = df.copy()
if search:
    df_f = df_f[df_f["Player"].str.contains(search, case=False)]
if team != "All":
    df_f = df_f[df_f["Team"] == team]
df_f = df_f[df_f["Position"].isin(pos_sel)]
df_f = df_f[(df_f["Minutes played"] >= mins[0]) & (df_f["Minutes played"] <= mins[1])]
df_f = df_f[
    (df_f["Offensive Score"] >= min_off) &
    (df_f["Defensive Score"] >= min_def) &
    (df_f["Key Passing Score"] >= min_key)
]

# =============================================================
# TABS AS PRO BUTTONS
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Player Explorer",
    "Comparison",
    "Pizza Chart",
    "Role Clustering",
    "Team Dashboard",
    "Style Map"
])

# =============================================================
# TAB 1 — PLAYER EXPLORER
# =============================================================
with tab1:
    st.markdown("<div class='section-title'>Player Explorer</div>", unsafe_allow_html=True)

    st.dataframe(df_f[[
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]], hide_index=True)

    player = st.selectbox("Select player", [""] + df_f["Player"].tolist())
    if player:
        row = safe_get_player(df_f, player)
        if row is not None:
            show_profile(row)

# =============================================================
# TAB 2 — PLAYER COMPARISON
# =============================================================
with tab2:
    st.markdown("<div class='section-title'>Player Comparison</div>", unsafe_allow_html=True)

    players = df_f["Player"].tolist()
    if len(players) >= 2:
        p1 = st.selectbox("Player 1", players)
        p2 = st.selectbox("Player 2", players, index=1)

        if p1 != p2:
            r1, r2 = safe_get_player(df_f, p1), safe_get_player(df_f, p2)
            if r1 and r2:
                c1, c2 = st.columns(2)
                with c1: show_profile(r1)
                with c2: show_profile(r2)

# =============================================================
# TAB 3 — PIZZA CHART
# =============================================================
with tab3:
    st.markdown("<div class='section-title'>Pizza Chart</div>", unsafe_allow_html=True)

    all_players = sorted(df["Player"].unique())
    p = st.selectbox("Player", all_players)
    threshold = st.slider("Minutes threshold", 0, 2000, 900, 50)

    if p:
        pizza(df, p, threshold)

# =============================================================
# TAB 4 — ROLE CLUSTERING
# =============================================================
with tab4:
    st.markdown("<div class='section-title'>Role Clustering</div>", unsafe_allow_html=True)

    if len(df_f) >= 3:
        feats = ["Offensive Score", "Defensive Score", "Key Passing Score"]
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
    st.markdown("<div class='section-title'>Team Dashboard</div>", unsafe_allow_html=True)

    t = st.selectbox("Team", sorted(df["Team"].unique()))
    dft = df[df["Team"] == t]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Top Offensive**")
        st.table(dft.nlargest(5, "Offensive Score")[["Player","Offensive Score"]])
    with c2:
        st.write("**Top Defensive**")
        st.table(dft.nlargest(5, "Defensive Score")[["Player","Defensive Score"]])
    with c3:
        st.write("**Top Creators**")
        st.table(dft.nlargest(5, "Key Passing Score")[["Player","Key Passing Score"]])

# =============================================================
# TAB 6 — STYLE MAP
# =============================================================
with tab6:
    st.markdown("<div class='section-title'>Style Map (PCA)</div>", unsafe_allow_html=True)

    if len(df_f) >= 2:
        feats = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]
        X = StandardScaler().fit_transform(feats)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        df_f["PC1"], df_f["PC2"] = coords[:,0], coords[:,1]

        fig, ax = plt.subplots(figsize=(7,5), facecolor="#000")
        ax.set_facecolor("#000")

        ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.8)
        for _, r in df_f.iterrows():
            ax.text(r["PC1"], r["PC2"], r["Player"], fontsize=7, color="white")

        ax.set_xlabel("PC1", color="white")
        ax.set_ylabel("PC2", color="white")
        ax.grid(color="#222")
        st.pyplot(fig)
