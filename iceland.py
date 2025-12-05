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
# DARK UI THEME
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
    margin: 0 0 12px 0;
    border-bottom: 1px solid #222;
}

.dataframe td, .dataframe th {
    color: #EEE !important;
    font-size: 0.82rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem !important;
    padding: 6px 16px !important;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# SAFE PLAYER LOOKUP
# =============================================================
def safe_get_player(df, name):
    sub = df[df["Player"] == name]
    return None if sub.empty else sub.iloc[0]

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_excel("Iceland.xlsx").copy()

df["Team"] = df["Team"].astype(str)
df["Position"] = df["Position"].astype(str)

numeric_cols = [
    "Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90",
    "PAdj Interceptions","PAdj Sliding tackles",
    "Aerial duels won, %","Defensive duels won, %",
    "Shots blocked per 90","Key passes per 90",
    "Through passes per 90","Passes to final third per 90","Passes to penalty area per 90"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df = df[df["Player"].notna()]

# =============================================================
# COMPOSITE SCORING
# =============================================================
def pct(s): return s.rank(pct=True) * 100

df["Offensive Score"] = pct(df[[
    "Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90"
]].mean(axis=1))

df["Defensive Score"] = pct(df[[
    "PAdj Interceptions","PAdj Sliding tackles","Aerial duels won, %",
    "Defensive duels won, %","Shots blocked per 90"
]].mean(axis=1))

df["Key Passing Score"] = pct(df[[
    "Key passes per 90","Through passes per 90","Assists per 90","xA per 90",
    "Passes to final third per 90","Passes to penalty area per 90"
]].mean(axis=1))

# =============================================================
# PLAYER RADAR (3-axis)
# =============================================================
def show_profile(row):
    st.markdown(f"### {row['Player']}")
    st.caption(f"{row['Team']} • {row['Position']} • {int(row['Minutes played'])} mins")

    vals = [
        row["Offensive Score"],
        row["Defensive Score"],
        row["Key Passing Score"],
        row["Offensive Score"],
    ]

    labels = ["Off","Def","Key"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'polar': True}, facecolor="#000")
    ax.set_facecolor("#000")
    ax.plot(angles, vals, color="#FF5C35", linewidth=2)
    ax.fill(angles, vals, color="#FF5C35", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels([])
    st.pyplot(fig)

# =============================================================
# NAIVE ROLE LABELS
# =============================================================
def assign_roles(df_roles, cols, km):
    centers = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = []
    for _, c in centers.iterrows():
        o, d, k = c["Offensive Score"], c["Defensive Score"], c["Key Passing Score"]
        if o > d and o > k:
            rn = "Attacking Forward"
        elif k > o:
            rn = "Advanced Creator"
        elif d > 65:
            rn = "Defensive Anchor"
        elif o > 60 and k > 60:
            rn = "Attacking Playmaker"
        elif np.mean([o, d, k]) > 70:
            rn = "Elite All-Rounder"
        else:
            rn = "Hybrid Profile"
        names.append(rn)
    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: names[x])
    return df_roles

# =============================================================
# SINGLE PLAYER PIZZA
# =============================================================
def pizza(df_all, player, min_thresh=900):
    pool = df_all[df_all["Minutes played"] >= min_thresh]
    row = safe_get_player(pool, player)
    if row is None:
        st.error("Player not in population.")
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
        val = row[p]
        if pd.isna(val):
            values.append(0)
            continue
        pop = pool[p].dropna()
        if pop.empty:
            values.append(0)
            continue
        perc = stats.percentileofscore(pop, val)
        if np.isnan(perc):
            perc = 0
        if perc == 100:
            perc = 99
        values.append(int(perc))

    baker = PyPizza(
        params=params,
        straight_line_color="white",
        last_circle_lw=5,
        other_circle_lw=2,
        inner_circle_size=15,
    )

    fig, ax = baker.make_pizza(
        values, figsize=(10, 10), slice_colors=["#598BAF"] * len(params),
        color_blank_space="same",
        kwargs_params=dict(color="white", fontsize=10),
        kwargs_values=dict(color="white", fontsize=9),
    )
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5, 0.97, player, ha="center", size=22, weight="bold", color="white")
    st.pyplot(fig)

# =============================================================
# TWO PLAYER COMPARISON PIZZA
# =============================================================
def comparison_pizza(df_all, p1, p2, min_thresh=1500):
    pool = df_all[df_all["Minutes played"] >= min_thresh]
    r1 = safe_get_player(pool, p1)
    r2 = safe_get_player(pool, p2)
    if r1 is None or r2 is None:
        st.error("One or both players missing from filtered dataset.")
        return

    df_metrics = pool.drop(
        columns=["Team","Position","Age","Matches played","Minutes played"],
        errors="ignore"
    ).reset_index(drop=True)

    params = [c for c in df_metrics.columns if c not in ["Player","index"]]

    def get_vals(name):
        row = df_metrics[df_metrics["Player"] == name]
        if row.empty:
            return None
        vals_raw = row.iloc[0][params].values
        result = []
        for val, param in zip(vals_raw, params):
            if pd.isna(val):
                result.append(0)
                continue
            pop = df_metrics[param].dropna()
            if pop.empty:
                result.append(0)
                continue
            perc = stats.percentileofscore(pop, val)
            if np.isnan(perc):
                perc = 0
            if perc == 100:
                perc = 99
            result.append(int(perc))
        return result

    vals1 = get_vals(p1)
    vals2 = get_vals(p2)

    baker = PyPizza(
        params=params,
        straight_line_color="white",
        straight_line_lw=1.5,
        last_circle_lw=6,
        other_circle_lw=2.5,
        inner_circle_size=15,
    )

    fig, ax = baker.make_pizza(
        vals1,
        compare_values=vals2,
        figsize=(12, 12),
        param_location=110,
        color_blank_space="same",

        kwargs_slices=dict(facecolor="#598BAF", edgecolor="black", linewidth=2),
        kwargs_compare=dict(facecolor="#ff6361", edgecolor="black", linewidth=2),

        kwargs_params=dict(color="white", fontsize=14, weight="bold"),
        kwargs_values=dict(color="white", fontsize=11,
                           bbox=dict(edgecolor="white", facecolor="#1a1a1a",
                                     boxstyle="round,pad=0.3")),
        kwargs_compare_values=dict(color="white", fontsize=11,
                                   bbox=dict(edgecolor="white", facecolor="#333333",
                                             boxstyle="round,pad=0.3")),
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5, 0.98, f"{p1} vs {p2}", ha="center", size=24, weight="bold", color="white")
    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
with st.sidebar:
    st.header("Filters")
    search = st.text_input("Search Player")
    teams = ["All"] + sorted(df["Team"].unique())
    team = st.selectbox("Team", teams)
    positions = sorted(df["Position"].unique())
    pos_sel = st.multiselect("Positions", positions, positions)
    mins = st.slider("Minutes Played", 0, int(df["Minutes played"].max()),
                     (300, int(df["Minutes played"].max())))
    st.subheader("Score Filters")
    min_off = st.slider("Offensive ≥", 0, 100, 0)
    min_def = st.slider("Defensive ≥", 0, 100, 0)
    min_key = st.slider("Key Passing ≥", 0, 100, 0)

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
# MAIN TABS
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Player Explorer",
    "Comparison Radar",
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

    p = st.selectbox("Select Player", [""] + df_f["Player"].tolist())
    if p:
        row = safe_get_player(df_f, p)
        if row:
            show_profile(row)

# =============================================================
# TAB 2 — CLEAN TWO-PLAYER COMPARISON RADAR
# =============================================================
with tab2:
    st.markdown("<div class='section-title'>Two-Player Comparison Radar</div>", unsafe_allow_html=True)

    all_players = sorted(df["Player"].unique())

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, key="cmp_p1")
    with col2:
        p2 = st.selectbox("Player 2", all_players, key="cmp_p2")

    th = st.slider("Minutes Threshold (Population)", 0, 2000, 1500, step=100)

    if p1 != p2:
        comparison_pizza(df, p1, p2, min_thresh=th)
    else:
        st.warning("Choose two different players.")

# =============================================================
# TAB 3 — SINGLE PLAYER PIZZA
# =============================================================
with tab3:
    st.markdown("<div class='section-title'>Player Pizza Chart</div>", unsafe_allow_html=True)
    p = st.selectbox("Player", sorted(df["Player"].unique()), key="pizza_select")
    th = st.slider("Minutes Threshold", 0, 2000, 900, step=50)
    if p:
        pizza(df, p, th)

# =============================================================
# TAB 4 — ROLE CLUSTERING
# =============================================================
with tab4:
    st.markdown("<div class='section-title'>Role Clustering</div>", unsafe_allow_html=True)
    if len(df_f) >= 3:
        feats = ["Offensive Score","Defensive Score","Key Passing Score"]
        X = StandardScaler().fit_transform(df_f[feats])
        k = st.slider("Number of Roles", 2, 8, 4)
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
# TAB 6 — PCA STYLE MAP
# =============================================================
with tab6:
    st.markdown("<div class='section-title'>Style Map (PCA)</div>", unsafe_allow_html=True)
    if len(df_f) >= 2:
        feats = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]
        X = StandardScaler().fit_transform(feats)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        df_f["PC1"], df_f["PC2"] = coords[:,0], coords[:,1]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#000")
        ax.set_facecolor("#000")
        ax.scatter(df_f["PC1"], df_f["PC2"], c="#FF5C35", alpha=0.8)

        for _, r in df_f.iterrows():
            ax.text(r["PC1"], r["PC2"], r["Player"], fontsize=7, color="white")

        ax.set_xlabel("PC1", color="white")
        ax.set_ylabel("PC2", color="white")
        ax.grid(color="#222")
        st.pyplot(fig)
