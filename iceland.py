import streamlit as st
import pandas as pd
import numpy as np

# =============================================================
# PAGE SETTINGS
# =============================================================
st.set_page_config(layout="wide", page_title="Outlier Scouting Platform")
st.title("Outlier Scouting Platform")

# =============================================================
# LOAD DATA (your uploaded file)
# =============================================================
file_path = "Iceland.xlsx"
df = pd.read_excel(file_path)

# =============================================================
# SIDEBAR FILTERS
# =============================================================
st.sidebar.header("Filters")

# --- Player search
player_search = st.sidebar.text_input("Search Player Name")

# --- Team filter (team within selected timeframe)
teams = sorted(df["Team within selected timeframe"].dropna().unique().tolist())
team_filter = st.sidebar.selectbox("Team (within selected timeframe)", ["All"] + teams)

# --- Minutes played filter
min_minutes = int(df["Minutes played"].min())
max_minutes = int(df["Minutes played"].max())

minutes_range = st.sidebar.slider(
    "Minutes Played",
    min_value=min_minutes,
    max_value=max_minutes,
    value=(min_minutes, max_minutes)
)

# --- Position filter
positions = sorted(df["Position"].dropna().unique().tolist())
position_filter = st.sidebar.multiselect(
    "Position",
    options=positions,
    default=positions
)

# =============================================================
# APPLY FILTERS
# =============================================================
filtered = df.copy()

# Filter: Player search
if player_search.strip() != "":
    filtered = filtered[filtered["Player"].str.contains(player_search, case=False, na=False)]

# Filter: Team
if team_filter != "All":
    filtered = filtered[filtered["Team within selected timeframe"] == team_filter]

# Filter: Minutes
filtered = filtered[
    (filtered["Minutes played"] >= minutes_range[0]) &
    (filtered["Minutes played"] <= minutes_range[1])
]

# Filter: Position
filtered = filtered[filtered["Position"].isin(position_filter)]

# =============================================================
# OUTLIER SCOUTING TABLE
# =============================================================

st.subheader("Filtered Player Dataset")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

st.write(f"**Total Players Found:** {len(filtered)}")

# =============================================================
# OPTIONAL: Quick stats for initial outlier detection
# =============================================================
st.subheader("Quick Outlier Indicators (z-score > 2 or < -2)")

numeric_cols = filtered.select_dtypes(include=[np.number]).columns

if len(filtered) > 0:
    zscores = filtered[numeric_cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    outlier_mask = (zscores > 2) | (zscores < -2)

    outliers = filtered[outlier_mask.any(axis=1)]

    st.write("Players with ≥ 1 extreme metric:")
    st.dataframe(outliers, use_container_width=True, hide_index=True)
else:
    st.info("No players match the current filters.")
