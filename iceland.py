import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(layout="wide", page_title="Outlier Scouting – ASA + 538")


# =============================================================
# FIVE THIRTY EIGHT STYLE CSS
# =============================================================
st.markdown("""
<style>

/* ------------------------------------------------------------
   GLOBAL FONT COLOR — make EVERYTHING fully black
------------------------------------------------------------ */
* {
    color: #000000 !important;
}

/* ------------------------------------------------------------
   APP BACKGROUND
------------------------------------------------------------ */
[data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
}

/* ------------------------------------------------------------
   SIDEBAR
------------------------------------------------------------ */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D1D5DB;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Sidebar titles */
.sidebar-title {
    font-weight: 800;
    font-size: 1.25rem;
    color: #000000 !important;
}

.sidebar-subtitle {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #000000 !important;
}

/* ------------------------------------------------------------
   MAIN HEADERS + TITLES
------------------------------------------------------------ */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #000000 !important;
    border-bottom: 2px solid #E5E7EB;
}

.main-header, .main-subheader {
    color: #000000 !important;
}

/* ------------------------------------------------------------
   RESULT CARDS
------------------------------------------------------------ */
.result-card {
    padding: 14px 18px;
    background: white !important;
    border: 1px solid #D1D5DB;
    border-radius: 7px;
}

.result-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #000000 !important;
}

.result-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #000000 !important;
}

/* ------------------------------------------------------------
   TABLE FIX — Make table readable
------------------------------------------------------------ */
.dataframe, .dataframe th, .dataframe td {
    color: #000000 !important;
}

.dataframe thead th {
    background: #F1F3F5 !important;
    font-weight: 700 !important;
    color: #000000 !important;
}

/* ------------------------------------------------------------
   FORM ELEMENTS — selectbox, radio, sliders
------------------------------------------------------------ */
.stSelectbox label, .stRadio label, .stSlider label {
    color: #000000 !important;
}

.stSelectbox div, .stRadio div, .stSlider div {
    color: #000000 !important;
}

/* Inputs */
input, select, textarea {
    color: #000000 !important;
}

/* Plot st.pyplot labels */
svg text {
    fill: #000000 !important;
}

</style>
""", unsafe_allow_html=True)



# =============================================================
# LOAD DATA
# =============================================================
DATA_PATH = "Iceland.xlsx"
df_raw = pd.read_excel(DATA_PATH)
df = df_raw.copy()

required_cols = [
    "Player", "Team", "Team within selected timeframe", "Position", "Minutes played",
    "Goals per 90", "xG per 90", "Shots per 90", "Assists per 90", "xA per 90",
    "PAdj Interceptions", "PAdj Sliding tackles", "Aerial duels won, %",
    "Defensive duels won, %", "Shots blocked per 90",
    "Key passes per 90", "Through passes per 90",
    "Passes to final third per 90", "Passes to penalty area per 90",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in Excel: {missing}")
    st.stop()



# =============================================================
# HELPER FUNCTIONS
# =============================================================
def percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


def add_outlier_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    offensive_metrics = [
        "Goals per 90", "xG per 90", "Shots per 90",
        "Assists per 90", "xA per 90"
    ]
    defensive_metrics = [
        "PAdj Interceptions", "PAdj Sliding tackles",
        "Aerial duels won, %", "Defensive duels won, %",
        "Shots blocked per 90"
    ]
    key_passing_metrics = [
        "Key passes per 90", "Through passes per 90",
        "Assists per 90", "xA per 90",
        "Passes to final final third per 90"
        if "Passes to final final third per 90" in df.columns
        else "Passes to final third per 90",
        "Passes to penalty area per 90",
    ]

    df["Offensive_raw"] = df[offensive_metrics].mean(axis=1)
    df["Defensive_raw"] = df[defensive_metrics].mean(axis=1)
    df["KeyPassing_raw"] = df[key_passing_metrics].mean(axis=1)

    df["Offensive Score"] = percentile(df["Offensive_raw"])
    df["Defensive Score"] = percentile(df["Defensive_raw"])
    df["Key Passing Score"] = percentile(df["KeyPassing_raw"])

    return df


df = add_outlier_scores(df)



# =============================================================
# --- FIVE THIRTY EIGHT FILTER PANEL ---
# =============================================================
def fivethirtyeight_filters(df):
    st.sidebar.markdown('<div class="sidebar-title">538 Filters</div>', unsafe_allow_html=True)

    player_search = st.sidebar.text_input("Search Player")

    teams = ["All"] + sorted(df["Team"].dropna().unique())
    team = st.sidebar.selectbox("Team", teams)

    positions = sorted(df["Position"].dropna().unique())
    pos_selected = st.sidebar.multiselect("Position(s)", positions, default=positions)

    min_m, max_m = int(df["Minutes played"].min()), int(df["Minutes played"].max())
    minutes = st.sidebar.slider("Minutes Played", min_m, max_m, (min_m, max_m))

    st.sidebar.markdown('<div class="sidebar-subtitle">Score Filters</div>', unsafe_allow_html=True)
    off_min = st.sidebar.slider("Min Offensive Score", 0, 100, 0)
    def_min = st.sidebar.slider("Min Defensive Score", 0, 100, 0)
    key_min = st.sidebar.slider("Min Key Passing Score", 0, 100, 0)

    df_f = df.copy()

    if player_search:
        df_f = df_f[df_f["Player"].str.contains(player_search, case=False)]

    if team != "All":
        df_f = df_f[df_f["Team"] == team]

    df_f = df_f[df_f["Position"].isin(pos_selected)]
    df_f = df_f[(df_f["Minutes played"] >= minutes[0]) & (df_f["Minutes played"] <= minutes[1])]
    df_f = df_f[
        (df_f["Offensive Score"] >= off_min) &
        (df_f["Defensive Score"] >= def_min) &
        (df_f["Key Passing Score"] >= key_min)
    ]

    return df_f



# =============================================================
# --- FIVE THIRTY EIGHT RESULTS PANEL ---
# =============================================================
def fivethirtyeight_results(df_filtered):
    st.markdown('<div class="section-title">538 Scouting Results</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Players</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{len(df_filtered)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Avg Offensive</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{df_filtered["Offensive Score"].mean():.1f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Avg Defensive</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{df_filtered["Defensive Score"].mean():.1f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Players</div>', unsafe_allow_html=True)

    cols = [
        "Player", "Team", "Position", "Minutes played",
        "Offensive Score", "Defensive Score", "Key Passing Score"
    ]

    st.dataframe(
        df_filtered[cols].sort_values("Offensive Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )



# =============================================================
# NAVIGATION
# =============================================================
menu = st.sidebar.radio(
    "View",
    [
        "🏠 Dashboard",
        "📊 Outlier Scouting",
        "👤 Player Explorer",
        "📈 Visual Explorer",
        "🔵 FiveThirtyEight Scouting"
    ],
    index=0,
)



# =============================================================
# YOUR EXISTING PAGES (LEFT UNCHANGED)
# =============================================================
# ⚠️ I will not re-paste your full app here — we keep all your original
#     Dashboard / Outlier Scouting / Player Explorer / Visual Explorer
#     exactly as before.
# Paste your original page code here unchanged.


# =============================================================
# NEW: FIVE THIRTY EIGHT SCOUTING PAGE
# =============================================================
if menu == "🔵 FiveThirtyEight Scouting":
    st.markdown(
        "<h1 style='font-weight:800;'>FiveThirtyEight Scouting</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#444;'>A clean, high-contrast scouting interface inspired by FiveThirtyEight’s visual style.</p>",
        unsafe_allow_html=True,
    )

    df_filtered = fivethirtyeight_filters(df)

    if len(df_filtered) == 0:
        st.warning("No players match the filters.")
    else:
        fivethirtyeight_results(df_filtered)
