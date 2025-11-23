import streamlit as st
import pandas as pd
import numpy as np

# =============================================================
# PAGE SETTINGS
# =============================================================
st.set_page_config(layout="wide", page_title="Outlier Scouting Platform")
st.title("Outlier Scouting Platform")

# =============================================================
# LOAD DATA
# =============================================================
file_path = "Iceland.xlsx"
df = pd.read_excel(file_path)

# =============================================================
# CREATE OUTLIER SCORES (percentiles)
# =============================================================

def percentile(series):
    """Convert column to percentile ranking."""
    return series.rank(pct=True) * 100

# ---- Offensive Score ----
offensive_metrics = [
    "Goals per 90",
    "xG per 90",
    "Shots per 90",
    "Assists per 90",
    "xA per 90",
]

df["Offensive Score"] = percentile(df[offensive_metrics].mean(axis=1, skipna=True))

# ---- Defensive Score ----
defensive_metrics = [
    "PAdj Interceptions",
    "PAdj Sliding tackles",
    "Aerial duels won, %",
    "Defensive duels won, %",
    "Shots blocked per 90",
]

df["Defensive Score"] = percentile(df[defensive_metrics].mean(axis=1, skipna=True))

# ---- Key Passing Score ----
key_passing_metrics = [
    "Key passes per 90",
    "Through passes per 90",
    "Assists per 90",
    "xA per 90",
    "Passes to final third per 90",
    "Passes to penalty area per 90",
]

df["Key Passing Score"] = percentile(df[key_passing_metrics].mean(axis=1, skipna=True))


# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.header("Filters")

# Player search
player_search = st.sidebar.text_input("Search Player Name")

# Team within timeframe
teams = sorted(df["Team within selected timeframe"].dropna().unique().tolist())
team_filter = st.sidebar.selectbox("Team within timeframe", ["All"] + teams)

# Minutes played
min_min = int(df["Minutes played"].min())
max_min = int(df["Minutes played"].max())
min_minutes, max_minutes = st.sidebar.slider(
    "Minutes Played Range",
    min_value=min_min,
    max_value=max_min,
    value=(min_min, max_min)
)

# Position
positions = sorted(df["Position"].dropna().unique().tolist())
position_filter = st.sidebar.multiselect(
    "Position",
    options=positions,
    default=positions
)

# =============================================================
# OUTLIER SCORE FILTERS
# =============================================================
st.sidebar.header("Outlier Score Filters")

min_offensive = st.sidebar.slider(
    "Minimum Offensive Score (0–100)",
    min_value=0, max_value=100, value=0
)

min_defensive = st.sidebar.slider(
    "Minimum Defensive Score (0–100)",
    min_value=0, max_value=100, value=0
)

min_keypassing = st.sidebar.slider(
    "Minimum Key Passing Score (0–100)",
    min_value=0, max_value=100, value=0
)


# =============================================================
# APPLY FILTERS
# =============================================================
filtered = df.copy()

# Player search
if player_search.strip() != "":
    filtered = filtered[filtered["Player"].str.contains(player_search, case=False, na=False)]

# Team filter
if team_filter != "All":
    filtered = filtered[filtered["Team within selected timeframe"] == team_filter]

# Minutes filter
filtered = filtered[
    (filtered["Minutes played"] >= min_minutes) &
    (filtered["Minutes played"] <= max_minutes)
]

# Position filter
filtered = filtered[filtered["Position"].isin(position_filter)]

# Outlier score filters
filtered = filtered[
    (filtered["Offensive Score"] >= min_offensive) &
    (filtered["Defensive Score"] >= min_defensive) &
    (filtered["Key Passing Score"] >= min_keypassing)
]


# =============================================================
# DISPLAY TABLE
# =============================================================
st.subheader("Filtered Player List")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

st.write(f"**Players Found:** {len(filtered)}")


# =============================================================
# OPTIONAL: Show Top Outliers per Category
# =============================================================
st.subheader("Top Outliers")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔥 Offensive Outliers")
    st.dataframe(
        df.sort_values("Offensive Score", ascending=False).head(10),
        use_container_width=True, hide_index=True
    )

with col2:
    st.markdown("### 🛡 Defensive Outliers")
    st.dataframe(
        df.sort_values("Defensive Score", ascending=False).head(10),
        use_container_width=True, hide_index=True
    )

with col3:
    st.markdown("### 🎯 Key Passing Outliers")
    st.dataframe(
        df.sort_values("Key Passing Score", ascending=False).head(10),
        use_container_width=True, hide_index=True
    )
