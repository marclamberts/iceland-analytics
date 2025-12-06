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
# LOGIN SYSTEM
# =============================================================
VALID_USER = "lamberts"
VALID_PASS = "lamberts"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "home"

def login_screen():
    st.title("🔐 Professional Scouting Platform")
    st.subheader("Login")
    st.write("Please enter your credentials to continue.")

    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == VALID_USER and password == VALID_PASS:
            st.session_state.logged_in = True
            st.session_state.page = "home"
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password.")

def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "home"
        st.experimental_rerun()

if not st.session_state.logged_in:
    login_screen()
    st.stop()

# =============================================================
# TOP BAR (TITLE + LOGOUT)
# =============================================================
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.title("⚽ Professional Scouting Platform")
with top_col2:
    logout_button()

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
def pct(s):
    return s.rank(pct=True) * 100

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
# SAFE PLAYER LOOKUP
# =============================================================
def safe_get_player(df_local, name):
    sub = df_local[df_local["Player"] == name]
    return None if sub.empty else sub.iloc[0]

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
# ROLE LABELLING
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
# SHARED PIZZA METRICS
# =============================================================
PIZZA_PARAMS = [
    "Goals per 90","Shots per 90","Assists per 90","xG per 90","xA per 90",
    "Key passes per 90","Through passes per 90","Passes to final third per 90",
    "Passes to penalty area per 90","PAdj Interceptions",
    "PAdj Sliding tackles","Defensive duels won, %",
    "Aerial duels won, %","Shots blocked per 90"
]

# =============================================================
# SAFE PERCENTILES FOR PIZZA
# =============================================================
def safe_percentiles(row, pool, params):
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

    return values

# =============================================================
# SINGLE PLAYER PIZZA
# =============================================================
def pizza(df_all, player, min_thresh=900):
    pool = df_all[df_all["Minutes played"] >= min_thresh]
    row = safe_get_player(pool, player)
    if row is None:
        st.error("Player not in population.")
        return

    values = safe_percentiles(row, pool, PIZZA_PARAMS)

    baker = PyPizza(
        params=PIZZA_PARAMS,
        straight_line_color="white",
        last_circle_lw=5,
        other_circle_lw=2,
        inner_circle_size=15,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(10, 10),
        slice_colors=["#598BAF"] * len(PIZZA_PARAMS),
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
        st.error("One or both players missing in filtered population.")
        return

    vals1 = safe_percentiles(r1, pool, PIZZA_PARAMS)
    vals2 = safe_percentiles(r2, pool, PIZZA_PARAMS)

    baker = PyPizza(
        params=PIZZA_PARAMS,
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
# SIDEBAR FILTERS (USED BY MULTIPLE PAGES)
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
# PLATFORM SCREEN WITH 6 TILES
# =============================================================
st.markdown("### Select a module:")

tile_css = """
<style>
.stButton > button {
    border-radius: 14px;
    border: 1px solid #444444;
    background-color: #111111;
    color: #FFFFFF;
    padding: 16px 8px;
    font-size: 0.95rem;
    font-weight: 600;
    transition: 0.2s;
}
.stButton > button:hover {
    background-color: #222222;
    transform: scale(1.03);
    border-color: #888888;
}
</style>
"""
st.markdown(tile_css, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

def tile(label, page_key):
    if st.button(label):
        st.session_state.page = page_key

with col1:
    tile("📊 Data Scouting", "data")
    tile("📈 Outlier Scouting", "outliers")

with col2:
    tile("🔥 Overperformance (Attackers)", "attackers")
    tile("🆚 Comparison", "compare")

with col3:
    tile("🍕 Pizza Plots", "pizza")
    tile("📉 Bar Graphs", "bars")

st.divider()

# =============================================================
# PAGE LOGIC
# =============================================================

# 1) DATA SCOUTING
if st.session_state.page == "data" or st.session_state.page == "home":
    st.markdown("<div class='section-title'>Data Scouting</div>", unsafe_allow_html=True)

    st.subheader("Player Explorer")
    st.dataframe(df_f[[
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]], hide_index=True)

    p = st.selectbox("Select Player", [""] + df_f["Player"].tolist(), key="ds_player")
    if p:
        row = safe_get_player(df_f, p)
        if row is not None:
            show_profile(row)

    st.markdown("---")
    st.subheader("Role Clustering")

    if len(df_f) >= 3:
        feats = ["Offensive Score","Defensive Score","Key Passing Score"]
        X = StandardScaler().fit_transform(df_f[feats])
        k = st.slider("Number of Roles", 2, 8, 4, key="role_k")
        km = KMeans(n_clusters=k, n_init=10)
        df_f_roles = df_f.copy()
        df_f_roles["Role"] = km.fit_predict(X)
        df_roles = assign_roles(df_f_roles, feats, km)

        st.dataframe(df_roles[[
            "Player","Team","Role","Role Name",
            "Offensive Score","Defensive Score","Key Passing Score"
        ]], hide_index=True)

    st.markdown("---")
    st.subheader("Team Dashboard")

    t = st.selectbox("Team", sorted(df["Team"].unique()), key="team_dash")
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

# 2) OUTLIER SCOUTING
elif st.session_state.page == "outliers":
    st.markdown("<div class='section-title'>Outlier Scouting</div>", unsafe_allow_html=True)
    st.write("Identify statistical outliers based on composite scores.")

    threshold = st.slider("Z-score threshold (absolute value)", 1.5, 3.5, 2.0, 0.1)

    scores = ["Offensive Score","Defensive Score","Key Passing Score"]
    df_out = df_f.copy()
    for sc in scores:
        m = df_out[sc].mean()
        s = df_out[sc].std(ddof=0)
        if s == 0 or np.isnan(s):
            df_out[f"{sc} Z"] = 0
        else:
            df_out[f"{sc} Z"] = (df_out[sc] - m) / s

    cols_z = [f"{s} Z" for s in scores]
    df_out["Max |Z|"] = df_out[cols_z].abs().max(axis=1)
    outliers = df_out[df_out["Max |Z|"] >= threshold].sort_values("Max |Z|", ascending=False)

    st.write(f"Players with |z| ≥ {threshold}:")
    if outliers.empty:
        st.info("No outliers found with the current threshold and filters.")
    else:
        st.dataframe(outliers[[
            "Player","Team","Position","Minutes played",
            "Offensive Score","Defensive Score","Key Passing Score","Max |Z|"
        ] + cols_z], hide_index=True)

# 3) OVERPERFORMANCE SCOUTING (ATTACKERS)
elif st.session_state.page == "attackers":
    st.markdown("<div class='section-title'>Overperformance Scouting (Attackers)</div>", unsafe_allow_html=True)
    st.write("Attackers who are overperforming their expected goals (Goals per 90 vs xG per 90).")

    min_mins = st.slider("Minimum minutes played", 0, int(df["Minutes played"].max()), 600, 50)
    min_diff = st.slider("Minimum (Goals - xG) per 90", 0.0, 1.5, 0.3, 0.05)

    # Simple position filter for forwards/attackers (adapt if needed)
    attackers = df[df["Position"].str.contains("FW|ST|ATT|W", case=False, na=False)].copy()
    attackers = attackers[attackers["Minutes played"] >= min_mins]

    attackers["G - xG per 90"] = attackers["Goals per 90"] - attackers["xG per 90"]
    overperf = attackers[attackers["G - xG per 90"] >= min_diff]
    overperf = overperf.sort_values("G - xG per 90", ascending=False)

    if overperf.empty:
        st.info("No attackers match the overperformance criteria.")
    else:
        st.dataframe(overperf[[
            "Player","Team","Position","Minutes played",
            "Goals per 90","xG per 90","G - xG per 90",
            "Offensive Score"
        ]], hide_index=True)

# 4) COMPARISON
elif st.session_state.page == "compare":
    st.markdown("<div class='section-title'>Comparison</div>", unsafe_allow_html=True)

    st.subheader("Two-Player Pizza Comparison")
    all_players = sorted(df["Player"].unique())
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, key="cmp_p1")
    with col2:
        p2 = st.selectbox("Player 2", all_players, key="cmp_p2")

    th = st.slider("Minutes Threshold (Population)", 0, 2000, 1500, step=100)

    if p1 == p2:
        st.warning("Choose two different players.")
    else:
        comparison_pizza(df, p1, p2, min_thresh=th)

    st.markdown("---")
    st.subheader("Style Map (PCA)")

    if len(df_f) >= 2:
        feats = df_f[["Offensive Score","Defensive Score","Key Passing Score"]]
        X = StandardScaler().fit_transform(feats)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        df_pca = df_f.copy()
        df_pca["PC1"], df_pca["PC2"] = coords[:,0], coords[:,1]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#000")
        ax.set_facecolor("#000")
        ax.scatter(df_pca["PC1"], df_pca["PC2"], c="#FF5C35", alpha=0.8)

        for _, r in df_pca.iterrows():
            ax.text(r["PC1"], r["PC2"], r["Player"], fontsize=7, color="white")

        ax.set_xlabel("PC1", color="white")
        ax.set_ylabel("PC2", color="white")
        ax.grid(color="#222")
        st.pyplot(fig)
    else:
        st.info("Not enough players in the filtered set for PCA.")

# 5) PIZZAPLOTS
elif st.session_state.page == "pizza":
    st.markdown("<div class='section-title'>Pizza Plots</div>", unsafe_allow_html=True)
    st.subheader("Single Player Pizza Chart")

    p = st.selectbox("Player", sorted(df["Player"].unique()), key="pizza_select")
    th = st.slider("Minutes Threshold", 0, 2000, 900, step=50)
    if p:
        pizza(df, p, th)

# 6) BAR GRAPHS
elif st.session_state.page == "bars":
    st.markdown("<div class='section-title'>Bar Graphs</div>", unsafe_allow_html=True)
    st.write("Create bar graphs for any metric.")

    metrics = [
        "Offensive Score","Defensive Score","Key Passing Score"
    ] + PIZZA_PARAMS

    metric = st.selectbox("Metric", metrics, key="bar_metric")
    group_by = st.radio("Group by", ["Team","Player"], key="bar_group")

    if group_by == "Team":
        df_bar = df_f.groupby("Team")[metric].mean().reset_index()
        df_bar = df_bar.sort_values(metric, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#000")
        ax.set_facecolor("#000")
        ax.bar(df_bar["Team"], df_bar[metric])
        ax.set_xticklabels(df_bar["Team"], rotation=45, ha="right", color="white")
        ax.set_ylabel(metric, color="white")
        ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)

    else:  # Player
        top_n = st.slider("Top N players", 5, 30, 15, key="bar_topn")
        df_bar = df_f[["Player", metric]].dropna()
        df_bar = df_bar.sort_values(metric, ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#000")
        ax.set_facecolor("#000")
        ax.bar(df_bar["Player"], df_bar[metric])
        ax.set_xticklabels(df_bar["Player"], rotation=45, ha="right", color="white")
        ax.set_ylabel(metric, color="white")
        ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)
