import streamlit as st
import pandas as pd
import numpy as np

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(page_title="FiveThirtyEight Scouting – Dark Mode", layout="wide")

# =============================================================
# DARK THEME CSS (black background, white text)
# =============================================================
st.markdown("""
<style>

:root {
    --bg: #000000;
    --text: #FFFFFF;
    --card: #111111;
    --border: #333333;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Headers */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text) !important;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.25rem;
    margin-top: 1.4rem;
}

h1, h2, h3, h4, h5 {
    color: var(--text) !important;
}

/* Cards */
.result-card {
    background: var(--card) !important;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 16px;
    margin-bottom: 12px;
}

.result-title {
    font-size: 0.8rem;
    color: #BBBBBB !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.result-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text) !important;
}

/* Tables */
.dataframe, .dataframe td, .dataframe th {
    color: var(--text) !important;
}

.dataframe thead th {
    background-color: #111111 !important;
    border-bottom: 1px solid var(--border) !important;
}

/* Inputs */
input, select, textarea {
    background-color: #222 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

svg text { fill: var(--text) !important; }

</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD DATA
# =============================================================
DATA_PATH = "Iceland.xlsx"
df_raw = pd.read_excel(DATA_PATH)
df = df_raw.copy()

required_cols = [
    "Player", "Team", "Position", "Minutes played",
    "Goals per 90", "xG per 90", "Shots per 90",
    "Assists per 90", "xA per 90",
    "PAdj Interceptions", "PAdj Sliding tackles",
    "Aerial duels won, %", "Defensive duels won, %",
    "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


# =============================================================
# SCORING FUNCTIONS
# =============================================================
def percentile(series):
    return series.rank(pct=True) * 100

def add_outlier_scores(df):
    df = df.copy()

    offensive = ["Goals per 90","xG per 90","Shots per 90","Assists per 90","xA per 90"]
    defensive = ["PAdj Interceptions","PAdj Sliding tackles","Aerial duels won, %","Defensive duels won, %","Shots blocked per 90"]
    keypass = ["Key passes per 90","Through passes per 90","Assists per 90","xA per 90","Passes to final third per 90","Passes to penalty area per 90"]

    df["Offensive Score"] = percentile(df[offensive].mean(axis=1))
    df["Defensive Score"] = percentile(df[defensive].mean(axis=1))
    df["Key Passing Score"] = percentile(df[keypass].mean(axis=1))

    return df

df = add_outlier_scores(df)


# =============================================================
# FILTER PANEL (SIDEBAR)
# =============================================================
def filters(df):
    st.sidebar.markdown("### Filters")

    # Search
    search = st.sidebar.text_input("Search Player")

    # Team
    teams = ["All"] + sorted(df["Team"].unique())
    team = st.sidebar.selectbox("Team", teams)

    # Position
    positions = sorted(df["Position"].unique())
    pos = st.sidebar.multiselect("Position", positions, positions)

    # Minutes
    min_m, max_m = int(df["Minutes played"].min()), int(df["Minutes played"].max())
    mins = st.sidebar.slider("Minutes Played", min_m, max_m, (min_m, max_m))

    # Scores
    st.sidebar.markdown("### Score Filters")
    off = st.sidebar.slider("Min Offensive Score", 0, 100, 0)
    deff = st.sidebar.slider("Min Defensive Score", 0, 100, 0)
    keyp = st.sidebar.slider("Min Key Passing Score", 0, 100, 0)

    df_f = df.copy()

    if search:
        df_f = df_f[df_f["Player"].str.contains(search, case=False)]

    if team != "All":
        df_f = df_f[df_f["Team"] == team]

    df_f = df_f[df_f["Position"].isin(pos)]
    df_f = df_f[(df_f["Minutes played"] >= mins[0]) & (df_f["Minutes played"] <= mins[1])]
    df_f = df_f[
        (df_f["Offensive Score"] >= off) &
        (df_f["Defensive Score"] >= deff) &
        (df_f["Key Passing Score"] >= keyp)
    ]

    return df_f


# =============================================================
# RESULTS PANEL (MAIN)
# =============================================================
def results(df_f):
    st.markdown("## FiveThirtyEight Scouting – Dark Mode")

    # Summary metrics
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="result-card"><div class="result-title">Players</div>'
                    f'<div class="result-value">{len(df_f)}</div></div>', 
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="result-card"><div class="result-title">Avg Offensive</div>'
                    f'<div class="result-value">{df_f["Offensive Score"].mean():.1f}</div></div>', 
                    unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="result-card"><div class="result-title">Avg Defensive</div>'
                    f'<div class="result-value">{df_f["Defensive Score"].mean():.1f}</div></div>', 
                    unsafe_allow_html=True)

    st.markdown('<div class="section-title">Players</div>', unsafe_allow_html=True)

    columns = [
        "Player", "Team", "Position", "Minutes played",
        "Offensive Score", "Defensive Score", "Key Passing Score"
    ]

    st.dataframe(
        df_f[columns].sort_values("Offensive Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================
# MAIN
# =============================================================
df_filtered = filters(df)

if len(df_filtered) == 0:
    st.warning("No players match the current filters.")
else:
    results(df_filtered)
