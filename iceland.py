import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(page_title="FiveThirtyEight Scouting – Enhanced", layout="wide")


# =============================================================
# DARK THEME CSS
# =============================================================
st.markdown("""
<style>

:root {
    --bg: #000000;
    --fg: #FFFFFF;
    --card: #111111;
    --border: #333333;
    --accent: #FF5C35; /* FiveThirtyEight orange */
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--fg) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--fg) !important;
}

/* Headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--fg) !important;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.25rem;
    margin-top: 1.4rem;
}

h1, h2, h3 {
    color: var(--fg) !important;
}

/* Cards */
.result-card {
    background: var(--card) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.result-title {
    font-size: 0.75rem;
    color: #CCCCCC !important;
    text-transform: uppercase;
}

.result-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--fg) !important;
}

/* Player Cards */
.player-card {
    background: var(--card);
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 10px;
}

.player-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--fg);
}

.player-team {
    font-size: 0.85rem;
    color: #AAAAAA;
}

/* Table */
.dataframe td, .dataframe th {
    color: var(--fg) !important;
}

.dataframe thead th {
    background-color: #111 !important;
}

/* Plot text */
svg text { fill: var(--fg) !important; }

</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD & CLEAN DATA
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
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")


# =============================================================
# SCORING
# =============================================================
def percentile(s): return s.rank(pct=True) * 100

df["Offensive Score"] = percentile(df[["Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90"]].mean(axis=1))
df["Defensive Score"] = percentile(df[["PAdj Interceptions","PAdj Sliding tackles","Aerial duels won, %","Defensive duels won, %","Shots blocked per 90"]].mean(axis=1))
df["Key Passing Score"] = percentile(df[["Key passes per 90","Through passes per 90","Assists per 90","xA per 90","Passes to final third per 90","Passes to penalty area per 90"]].mean(axis=1))


# =============================================================
# SIDEBAR – ENHANCED FILTERS
# =============================================================
st.sidebar.title("Filters")

search = st.sidebar.text_input("Search Player")

teams = ["All"] + sorted(df["Team"].dropna().unique())
team_filter = st.sidebar.selectbox("Team", teams)

positions = sorted(df["Position"].dropna().unique())
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
# APPLY FILTERS
# =============================================================
df_filtered = df.copy()

if search:
    df_filtered = df_filtered[df_filtered["Player"].str.contains(search, case=False)]

if team_filter != "All":
    df_filtered = df_filtered[df_filtered["Team"] == team_filter]

df_filtered = df_filtered[df_filtered["Position"].isin(position_filter)]
df_filtered = df_filtered[(df_filtered["Minutes played"] >= min_minutes[0]) &
                          (df_filtered["Minutes played"] <= min_minutes[1])]

df_filtered = df_filtered[
    (df_filtered["Offensive Score"] >= min_off) &
    (df_filtered["Defensive Score"] >= min_def) &
    (df_filtered["Key Passing Score"] >= min_key)
]


# =============================================================
# MAIN TITLE
# =============================================================
st.markdown("## FiveThirtyEight Player Scouting – Enhanced Dark Mode")


# =============================================================
# SUMMARY CARDS
# =============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Players</div>
        <div class="result-value">{len(df_filtered)}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Avg Offensive</div>
        <div class="result-value">{df_filtered["Offensive Score"].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Avg Defensive</div>
        <div class="result-value">{df_filtered["Defensive Score"].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# VIEW MODE SWITCHER
# =============================================================
view = st.radio(
    "View Mode",
    ["Table View", "Player Cards", "Scatter: Offense vs Key Passing", "Distributions"],
    horizontal=True
)


# =============================================================
# TABLE VIEW
# =============================================================
if view == "Table View":
    st.markdown('<div class="section-title">Players</div>', unsafe_allow_html=True)

    cols = [
        "Player","Team","Position","Minutes played",
        "Offensive Score","Defensive Score","Key Passing Score"
    ]

    st.dataframe(
        df_filtered[cols].sort_values("Offensive Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# =============================================================
# PLAYER CARDS VIEW
# =============================================================
elif view == "Player Cards":
    st.markdown('<div class="section-title">Player Cards</div>', unsafe_allow_html=True)

    for _, row in df_filtered.iterrows():
        st.markdown(f"""
        <div class="player-card">
            <div class="player-name">{row['Player']}</div>
            <div class="player-team">{row['Team']} — {row['Position']}</div>
            <br>
            <b>Offensive:</b> {row['Offensive Score']:.1f}<br>
            <b>Defensive:</b> {row['Defensive Score']:.1f}<br>
            <b>Key Passing:</b> {row['Key Passing Score']:.1f}
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# SCATTERPLOT VIEW
# =============================================================
elif view == "Scatter: Offense vs Key Passing":
    st.markdown('<div class="section-title">Offensive vs Key Passing</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8,5), facecolor="#000000")
    ax.scatter(
        df_filtered["Offensive Score"],
        df_filtered["Key Passing Score"],
        c="#FF5C35", alpha=0.8
    )
    ax.set_facecolor("#000000")
    ax.set_xlabel("Offensive Score", color="white")
    ax.set_ylabel("Key Passing Score", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")

    st.pyplot(fig)


# =============================================================
# DISTRIBUTIONS VIEW
# =============================================================
elif view == "Distributions":
    st.markdown('<div class="section-title">Distributions</div>', unsafe_allow_html=True)

    metric = st.selectbox(
        "Choose metric",
        ["Goals per 90","Assists per 90","Shots per 90","Key passes per 90",
         "PAdj Interceptions","Defensive duels won, %"]
    )

    fig, ax = plt.subplots(figsize=(8,5), facecolor="#000")
    ax.hist(df_filtered[metric].dropna(), bins=20, color="#FF5C35")
    ax.set_facecolor("#000")
    ax.set_title(metric, color="white")
    ax.tick_params(colors="white")

    st.pyplot(fig)
