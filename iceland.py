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
# PREMIUM DARK THEME (TIGHT)
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
    "Goals per 90", "xG per 90", "Shots per 90",
    "Assists per 90", "xA per 90",
    "PAdj Interceptions", "PAdj Sliding tackles",
    "Aerial duels won, %", "Defensive duels won, %",
    "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df = df[df["Player"].notna()]

# =============================================================
# SCORING FUNCTIONS
# =============================================================
def pct(s): 
    return s.rank(pct=True) * 100

df["Offensive Score"] = pct(
    df[["Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90"]].mean(axis=1)
)
df["Defensive Score"] = pct(
    df[["PAdj Interceptions","PAdj Sliding tackles","Aerial duels won, %","Defensive duels won, %","Shots blocked per 90"]].mean(axis=1)
)
df["Key Passing Score"] = pct(
    df[["Key passes per 90","Through passes per 90","Assists per 90","xA per 90",
        "Passes to final third per 90","Passes to penalty area per 90"]].mean(axis=1)
)

# =============================================================
# PLAYER RADAR
# =============================================================
def show_profile(row):
    st.markdown(f"### {row['Player']}")
    st.caption(f"{row['Team']} • {row['Position']} • {int(row['Minutes played'])} mins")

    vals = [
        row["Offensive Score"],
        row["Defensive Score"],
        row["Key Passing Score"],
    ]
    vals += vals[:1]

    labels = ["Off", "Def", "Key"]
    ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={'polar':True}, facecolor="#000")
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
        off, deff, kp = c["Offensive Score"], c["Defensive Score"], c["Key Passing Score"]

        if off > kp and off > deff:
            rn = "Attacking Forward"
        elif kp > off:
            rn = "Advanced Creator"
        elif deff > 65:
            rn = "Defensive Anchor"
        elif off > 60 and kp > 60:
            rn = "Attacking Playmaker"
        elif np.mean([off, deff, kp]) > 70:
            rn = "Elite All-Rounder"
        else:
            rn = "Hybrid Profile"

        names.append(rn)

    df_roles["Role Name"] = df_roles["Role"].apply(lambda x: names[x])
    return df_roles

# =============================================================
# SINGLE-PLAYER PIZZA
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

    colors = ["#598BAF"]*5 + ["#ffa600"]*4 + ["#ff6361"]*5  # att / key / def

    baker = PyPizza(params=params, straight_line_color="white", last_circle_lw=5,
                    other_circle_lw=2, inner_circle_size=15)

    fig, ax = baker.make_pizza(
        values, figsize=(10,10), slice_colors=colors,
        color_blank_space="same",
        kwargs_params=dict(color="white", fontsize=10),
        kwargs_values=dict(color="white", fontsize=9),
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(0.5, 0.97, player, ha="center", size=22, weight="bold", color="white")
    fig.text(0.5, 0.94, f"{row['Team']} • {int(row['Minutes played'])} mins",
             ha="center", size=11, color="white")

    st.pyplot(fig)

# =============================================================
# TWO-PLAYER COMPARISON PIZZA
# =============================================================
def comparison_pizza(df_all, player1, player2, min_thresh=900):
    """
    Draw a comparison pizza like your example:
    - Both players on the same pizza
    - Population = df_all filtered by minutes >= min_thresh
    """
    pool = df_all[df_all["Minutes played"] >= min_thresh].copy()

    row1 = safe_get_player(pool, player1)
    row2 = safe_get_player(pool, player2)

    if row1 is None or row2 is None:
        st.error("One of the players is not available in the comparison pool (minutes filter).")
        return

    # Drop non-metric columns similar to your script
    df_metrics = pool.drop(
        columns=["Team", "Position", "Age", "Matches played", "Minutes played"],
        errors="ignore"
    ).reset_index(drop=True)

    # Parameters = all columns except 'Player' and technical index columns
    params = [c for c in df_metrics.columns if c not in ["Player", "index"]]

    def get_percentiles_for(player_name: str):
        p_df = df_metrics[df_metrics["Player"] == player_name].reset_index(drop=True)
        if p_df.empty:
            return None

        player_vals = p_df.loc[0, params].values
        vals = []
        for i, param in enumerate(params):
            vals.append(
                99 if stats.percentileofscore(df_metrics[param], player_vals[i]) == 100
                else math.floor(stats.percentileofscore(df_metrics[param], player_vals[i]))
            )
        return vals

    vals1 = get_percentiles_for(player1)
    vals2 = get_percentiles_for(player2)

    if vals1 is None or vals2 is None:
        st.error("Could not compute percentiles for one of the players.")
        return

    baker = PyPizza(
        params=params,
        straight_line_color="white",
        straight_line_lw=1.5,
        last_circle_lw=6,
        other_circle_lw=2.5,
        other_circle_ls="-.",
        inner_circle_size=15
    )

    fig, ax = baker.make_pizza(
        vals1,
        compare_values=vals2,
        figsize=(12, 12),
        param_location=110,
        color_blank_space="same",

        kwargs_slices=dict(
            facecolor="#598BAF",   # player 1 (blue)
            edgecolor="black",
            zorder=2,
            linewidth=2
        ),
        kwargs_compare=dict(
            facecolor="#ff6361",   # player 2 (red)
            edgecolor="black",
            zorder=3,
            linewidth=2
        ),

        kwargs_params=dict(
            color="white", fontsize=14, weight='bold',
            fontname="Arial", va="center", alpha=.9,
        ),

        kwargs_values=dict(
            color="white", fontsize=11, weight='bold', fontname="Arial",
            zorder=4,
            bbox=dict(edgecolor="white", facecolor="#1a1a1a",
                      boxstyle="round,pad=0.3", lw=1)
        ),

        kwargs_compare_values=dict(
            color="white", fontsize=11, weight='bold', fontname="Arial",
            zorder=5,
            bbox=dict(edgecolor="white", facecolor="#333333",
                      boxstyle="round,pad=0.3", lw=1)
        )
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    fig.text(
        0.5, 0.98,
        f"{player1}  vs  {player2}",
        size=24, ha="center", color="white", weight="bold", fontname="Arial"
    )

    fig.text(
        0.5, 0.945,
        f"{row1['Team']}  |  {row2['Team']}   •   Min. {min_thresh} mins (population)",
        size=11, ha="center", color="white", fontname="Arial"
    )

    # Legend rectangles
    legend_y = 0.905
    fig.patches.extend([
        plt.Rectangle((0.35, legend_y - 0.012), 0.02, 0.02,
                      transform=fig.transFigure, fill=True, color="#598BAF"),
        plt.Rectangle((0.55, legend_y - 0.012), 0.02, 0.02,
                      transform=fig.transFigure, fill=True, color="#ff6361")
    ])
    fig.text(0.38, legend_y, player1, ha="left", va="center",
             color="white", fontsize=12, fontname="Arial")
    fig.text(0.58, legend_y, player2, ha="left", va="center",
             color="white", fontsize=12, fontname="Arial")

    st.pyplot(fig)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
with st.sidebar:
    st.header("Filters")

    search = st.text_input("Search")

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

# APPLY FILTERS (for table / explorer)
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
# MAIN TABS (PRO NAV)
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
# TAB 1 — EXPLORER
# =============================================================
with tab1:
    st.markdown("<div class='section-title'>Player Explorer</div>", unsafe_allow_html=True)

    st.dataframe(df_f[[
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]], hide_index=True)

    p = st.selectbox("Select player", [""] + df_f["Player"].tolist())
    if p:
        row = safe_get_player(df_f, p)
        if row is not None:
            show_profile(row)

# =============================================================
# TAB 2 — COMPARISON (PIZZA STYLE)
# =============================================================
with tab2:
    st.markdown("<div class='section-title'>Player Comparison (Pizza)</div>", unsafe_allow_html=True)

    all_players = sorted(df["Player"].unique())
    if len(all_players) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            p1 = st.selectbox("Player 1", all_players, index=0)
        with col_b:
            p2 = st.selectbox("Player 2", all_players, index=1)

        comp_min_thresh = st.slider(
            "Minutes threshold for comparison population", 0, 2000, 1500, step=100
        )

        if p1 != p2:
            comparison_pizza(df, p1, p2, min_thresh=comp_min_thresh)
        else:
            st.info("Select two different players to compare.")

# =============================================================
# TAB 3 — SINGLE PIZZA
# =============================================================
with tab3:
    st.markdown("<div class='section-title'>Pizza Chart</div>", unsafe_allow_html=True)

    all_players = sorted(df["Player"].unique())
    p = st.selectbox("Player", all_players, key="pizza_player")
    th = st.slider("Minutes threshold", 0, 2000, 900, 50, key="pizza_thresh")

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

        k = st.slider("Number of roles", 2, 8, 4)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
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
