import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(layout="wide", page_title="Outlier Scouting Platform")

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
DATA_PATH = "Iceland.xlsx"  # your uploaded file
df_raw = pd.read_excel(DATA_PATH)

# Make a working copy
df = df_raw.copy()

# Ensure expected columns exist (will raise clear error if missing)
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
    """Convert values to percentiles (0-100)."""
    return series.rank(pct=True) * 100


def add_outlier_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add Offensive/Defensive/Key Passing composite scores as percentiles."""
    df = df.copy()

    offensive_metrics = [
        "Goals per 90",
        "xG per 90",
        "Shots per 90",
        "Assists per 90",
        "xA per 90",
    ]
    defensive_metrics = [
        "PAdj Interceptions",
        "PAdj Sliding tackles",
        "Aerial duels won, %",
        "Defensive duels won, %",
        "Shots blocked per 90",
    ]
    key_passing_metrics = [
        "Key passes per 90",
        "Through passes per 90",
        "Assists per 90",
        "xA per 90",
        "Passes to final third per 90",
        "Passes to penalty area per 90",
    ]

    # Composite raw scores (mean of available metrics)
    df["Offensive_raw"] = df[offensive_metrics].mean(axis=1, skipna=True)
    df["Defensive_raw"] = df[defensive_metrics].mean(axis=1, skipna=True)
    df["KeyPassing_raw"] = df[key_passing_metrics].mean(axis=1, skipna=True)

    # Convert to percentiles
    df["Offensive Score"] = percentile(df["Offensive_raw"])
    df["Defensive Score"] = percentile(df["Defensive_raw"])
    df["Key Passing Score"] = percentile(df["KeyPassing_raw"])

    return df


def radar_chart(player_row: pd.Series, metrics: dict, title: str = ""):
    """Draw a radar chart for given metric -> label mapping on one player."""
    labels = list(metrics.keys())
    cols = list(metrics.values())

    values = []
    for c in cols:
        if c in player_row.index and not pd.isna(player_row[c]):
            values.append(float(player_row[c]))
        else:
            values.append(0.0)  # fallback

    # Close the circle
    values += values[:1]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.1)
    ax.grid(True)

    st.pyplot(fig)


def zscore_series(series: pd.Series) -> pd.Series:
    """Safe zscore for a numeric series."""
    if series.std(ddof=0) == 0 or series.isna().all():
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / series.std(ddof=0)


# Add scores once globally
df = add_outlier_scores(df)


# =============================================================
# GLOBAL SIDEBAR NAVIGATION
# =============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Outlier Scouting", "👤 Player Explorer", "📈 Visual Explorer"]
)


# =============================================================
# COMMON FILTER UI (function to reuse)
# =============================================================
def filter_panel(df: pd.DataFrame):
    """Return filtered dataframe and filter selections."""
    st.sidebar.header("Base Filters")

    # Player search
    player_search = st.sidebar.text_input("Search Player Name")

    # Team within timeframe
    teams = sorted(df["Team within selected timeframe"].dropna().unique().tolist())
    team = st.sidebar.selectbox("Team within selected timeframe", ["All"] + teams)

    # Minutes played
    min_min = int(df["Minutes played"].min())
    max_min = int(df["Minutes played"].max())
    minutes_range = st.sidebar.slider(
        "Minutes Played",
        min_value=min_min,
        max_value=max_min,
        value=(min_min, max_min),
    )

    # Position
    positions = sorted(df["Position"].dropna().unique().tolist())
    pos_selected = st.sidebar.multiselect(
        "Position(s)",
        options=positions,
        default=positions,
    )

    df_f = df.copy()

    if player_search.strip():
        df_f = df_f[df_f["Player"].str.contains(player_search, case=False, na=False)]

    if team != "All":
        df_f = df_f[df_f["Team within selected timeframe"] == team]

    df_f = df_f[
        (df_f["Minutes played"] >= minutes_range[0])
        & (df_f["Minutes played"] <= minutes_range[1])
    ]

    df_f = df_f[df_f["Position"].isin(pos_selected)]

    return df_f


# =============================================================
# PAGE: HOME
# =============================================================
if page == "🏠 Home":
    st.header("🏠 Home")
    st.markdown(
        """
Welcome to the **Outlier Scouting Platform**.

Use the navigation on the left to:

- **📊 Outlier Scouting** – filter by team, minutes, position, and outlier scores.
- **👤 Player Explorer** – inspect an individual player in detail (including radar chart).
- **📈 Visual Explorer** – see scatter plots & distributions to spot outliers visually.
"""
    )

    st.subheader("Dataset Overview")
    st.write(f"Total players in dataset: **{len(df)}**")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.subheader("Score Distributions (Offensive / Defensive / Key Passing)")
    cols = ["Offensive Score", "Defensive Score", "Key Passing Score"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col in zip(axes, cols):
        ax.hist(df[col].fillna(0), bins=20)
        ax.set_title(col)
    st.pyplot(fig)


# =============================================================
# PAGE: OUTLIER SCOUTING
# =============================================================
elif page == "📊 Outlier Scouting":
    st.header("📊 Outlier Scouting")

    filtered = filter_panel(df)

    st.sidebar.header("Outlier Score Thresholds")

    min_off = st.sidebar.slider(
        "Min Offensive Score", 0.0, 100.0, 0.0, step=1.0
    )
    min_def = st.sidebar.slider(
        "Min Defensive Score", 0.0, 100.0, 0.0, step=1.0
    )
    min_key = st.sidebar.slider(
        "Min Key Passing Score", 0.0, 100.0, 0.0, step=1.0
    )

    filtered = filtered[
        (filtered["Offensive Score"] >= min_off)
        & (filtered["Defensive Score"] >= min_def)
        & (filtered["Key Passing Score"] >= min_key)
    ]

    st.subheader("Filtered Players")

    columns_to_show = [
        "Player", "Team", "Team within selected timeframe", "Position",
        "Minutes played",
        "Offensive Score", "Defensive Score", "Key Passing Score",
    ]
    columns_to_show = [c for c in columns_to_show if c in filtered.columns]

    st.dataframe(
        filtered[columns_to_show].sort_values("Offensive Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.write(f"**Players found:** {len(filtered)}")

    # Tabs: lists of top outliers in each category
    st.subheader("Top Outliers by Composite Score")

    tab1, tab2, tab3 = st.tabs(["🔥 Offensive", "🛡 Defensive", "🎯 Key Passing"])

    with tab1:
        st.write("Top 15 Offensive Outliers")
        st.dataframe(
            df.sort_values("Offensive Score", ascending=False).head(15)[columns_to_show],
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.write("Top 15 Defensive Outliers")
        st.dataframe(
            df.sort_values("Defensive Score", ascending=False).head(15)[columns_to_show],
            use_container_width=True, hide_index=True
        )

    with tab3:
        st.write("Top 15 Key Passing Outliers")
        st.dataframe(
            df.sort_values("Key Passing Score", ascending=False).head(15)[columns_to_show],
            use_container_width=True, hide_index=True
        )


# =============================================================
# PAGE: PLAYER EXPLORER
# =============================================================
elif page == "👤 Player Explorer":
    st.header("👤 Player Explorer")

    filtered = filter_panel(df)

    if len(filtered) == 0:
        st.warning("No players match the current filters.")
    else:
        # choose player from filtered list
        player_name = st.selectbox("Select Player", sorted(filtered["Player"].unique()))

        player_row = filtered[filtered["Player"] == player_name].iloc[0]

        st.subheader(f"{player_name} – Summary")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Bio / Context**")
            st.write(f"**Team:** {player_row['Team']}")
            st.write(f"**Team (selected timeframe):** {player_row['Team within selected timeframe']}")
            st.write(f"**Position:** {player_row['Position']}")
            st.write(f"**Minutes played:** {int(player_row['Minutes played'])}")

            st.write("**Composite Scores**")
            st.write(f"Offensive Score: `{player_row['Offensive Score']:.1f}`")
            st.write(f"Defensive Score: `{player_row['Defensive Score']:.1f}`")
            st.write(f"Key Passing Score: `{player_row['Key Passing Score']:.1f}`")

        with colB:
            st.markdown("**Radar – Outlier Profile**")

            radar_metrics = {
                "Offense": "Offensive Score",
                "Defense": "Defensive Score",
                "Key Pass": "Key Passing Score",
            }
            radar_chart(player_row, radar_metrics, title="Composite Outlier Scores")

        st.subheader("Detailed Numbers (Key Metrics)")

        metric_cols = [
            "Goals per 90", "xG per 90", "Shots per 90",
            "Assists per 90", "xA per 90",
            "PAdj Interceptions", "PAdj Sliding tackles",
            "Aerial duels won, %", "Defensive duels won, %",
            "Shots blocked per 90",
            "Key passes per 90", "Through passes per 90",
            "Passes to final third per 90", "Passes to penalty area per 90",
        ]
        metric_cols = [c for c in metric_cols if c in filtered.columns]

        detail_df = (
            pd.DataFrame(player_row[metric_cols])
            .reset_index()
            .rename(columns={"index": "Metric", 0: "Value"})
        )

        st.dataframe(detail_df, use_container_width=True, hide_index=True)


# =============================================================
# PAGE: VISUAL EXPLORER
# =============================================================
elif page == "📈 Visual Explorer":
    st.header("📈 Visual Explorer")

    filtered = filter_panel(df)

    if len(filtered) == 0:
        st.warning("No players match the current filters.")
        st.stop()

    tab_scatter, tab_dist = st.tabs(["Scatter Outliers", "Distributions / Boxplots"])

    # ---------------- SCATTER TAB ----------------
    with tab_scatter:
        st.subheader("Scatter Plots for Outlier Detection")

        plot_type = st.selectbox(
            "Choose scatter",
            [
                "Offense: Goals per 90 vs xG per 90",
                "Defense: PAdj Interceptions vs Shots blocked per 90",
                "Key Passing: Key passes per 90 vs Passes to penalty area per 90",
            ],
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        if plot_type == "Offense: Goals per 90 vs xG per 90":
            xcol = "xG per 90"
            ycol = "Goals per 90"
            color = filtered["Offensive Score"]
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            sc = ax.scatter(filtered[xcol], filtered[ycol], c=color, cmap="viridis")
            ax.set_title("Offensive Outliers")
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Offensive Score")

        elif plot_type == "Defense: PAdj Interceptions vs Shots blocked per 90":
            xcol = "PAdj Interceptions"
            ycol = "Shots blocked per 90"
            color = filtered["Defensive Score"]
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            sc = ax.scatter(filtered[xcol], filtered[ycol], c=color, cmap="plasma")
            ax.set_title("Defensive Outliers")
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Defensive Score")

        else:  # Key Passing
            xcol = "Key passes per 90"
            ycol = "Passes to penalty area per 90"
            color = filtered["Key Passing Score"]
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            sc = ax.scatter(filtered[xcol], filtered[ycol], c=color, cmap="magma")
            ax.set_title("Key Passing Outliers")
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Key Passing Score")

        st.pyplot(fig)

    # ---------------- DISTRIBUTION TAB ----------------
    with tab_dist:
        st.subheader("Metric Distributions & Boxplots")

        metric = st.selectbox(
            "Select metric",
            [
                "Goals per 90",
                "xG per 90",
                "Shots per 90",
                "Assists per 90",
                "xA per 90",
                "PAdj Interceptions",
                "PAdj Sliding tackles",
                "Aerial duels won, %",
                "Defensive duels won, %",
                "Shots blocked per 90",
                "Key passes per 90",
                "Through passes per 90",
                "Passes to final third per 90",
                "Passes to penalty area per 90",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Distribution of **{metric}**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(filtered[metric].dropna(), bins=20)
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.write(f"Boxplot of **{metric}** (detect extremes)")
            fig, ax = plt.subplots(figsize=(3, 4))
            ax.boxplot(filtered[metric].dropna(), vert=True)
            ax.set_ylabel(metric)
            st.pyplot(fig)
