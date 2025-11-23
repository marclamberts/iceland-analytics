import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(layout="wide", page_title="Outlier Scouting – ASA Style")

# =============================================================
# GLOBAL STYLE (ASA-LIKE)
# =============================================================
st.markdown(
    """
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background-color: #F2F3F5;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A2540;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

.sidebar-title {
    font-weight: 800;
    font-size: 1.25rem;
    color: #F9FAFB;
    margin-bottom: 0.5rem;
}

.sidebar-subtitle {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9CA3AF;
}

/* Main headers */
.main-header {
    font-size: 2rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.25rem;
}

.main-subheader {
    font-size: 0.9rem;
    color: #6B7280;
    margin-bottom: 1.25rem;
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin: 1.2rem 0 0.4rem 0;
}

/* Cards */
.metric-card {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 8px rgba(15,23,42,0.08);
    border: 1px solid #E5E7EB;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6B7280;
}

.metric-value-lg {
    font-size: 1.35rem;
    font-weight: 800;
    color: #111827;
}

.metric-value-sm {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
}

/* Dataframe tweaks */
.dataframe td, .dataframe th {
    font-size: 0.9rem;
}

/* Reduce top padding */
.block-container {
    padding-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================
# LOAD DATA
# =============================================================
DATA_PATH = "Iceland.xlsx"
df_raw = pd.read_excel(DATA_PATH)
df = df_raw.copy()

# Required columns for scores
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

    df["Offensive_raw"] = df[offensive_metrics].mean(axis=1, skipna=True)
    df["Defensive_raw"] = df[defensive_metrics].mean(axis=1, skipna=True)
    df["KeyPassing_raw"] = df[key_passing_metrics].mean(axis=1, skipna=True)

    df["Offensive Score"] = percentile(df["Offensive_raw"])
    df["Defensive Score"] = percentile(df["Defensive_raw"])
    df["Key Passing Score"] = percentile(df["KeyPassing_raw"])

    return df


def radar_chart(player_row: pd.Series, metrics: dict, title: str = ""):
    """Draw a radar chart for given metric -> column mapping."""
    labels = list(metrics.keys())
    cols = list(metrics.values())

    values = []
    for c in cols:
        v = player_row.get(c, np.nan)
        values.append(0 if pd.isna(v) else float(v))

    # Close the circle
    values += values[:1]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4.8, 4.8))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.08)
    ax.grid(True, alpha=0.4)

    st.pyplot(fig)


# Apply scoring globally
df = add_outlier_scores(df)


# =============================================================
# FILTER PANEL (REUSABLE)
# =============================================================
def filter_panel(df: pd.DataFrame):
    """Sidebar base filters used on all pages."""
    st.sidebar.markdown('<div class="sidebar-title">Outlier Scouting</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-subtitle">Base Filters</div>',
        unsafe_allow_html=True,
    )

    # Player search
    player_search = st.sidebar.text_input("Player search")

    # Team within timeframe
    teams = sorted(df["Team within selected timeframe"].dropna().unique().tolist())
    team = st.sidebar.selectbox("Team (selected timeframe)", ["All"] + teams)

    # Minutes played
    min_min = int(df["Minutes played"].min())
    max_min = int(df["Minutes played"].max())
    minutes_range = st.sidebar.slider(
        "Minutes played",
        min_value=min_min,
        max_value=max_min,
        value=(min_min, max_min),
    )

    # Position
    positions = sorted(df["Position"].dropna().unique().tolist())
    pos_selected = st.sidebar.multiselect(
        "Position(s)", options=positions, default=positions
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
# NAVIGATION
# =============================================================
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "View",
    ["🏠 Dashboard", "📊 Outlier Scouting", "👤 Player Explorer", "📈 Visual Explorer"],
    index=0,
)


# =============================================================
# DASHBOARD PAGE
# =============================================================
if menu == "🏠 Dashboard":
    st.markdown('<div class="main-header">Outlier Scouting Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subheader">American Soccer Analysis–style scouting environment for quickly finding statistical outliers.</div>',
        unsafe_allow_html=True,
    )

    # Top-level cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Players</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value-lg">{len(df)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Teams</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value-lg">{df["Team"].nunique()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Median Minutes</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value-lg">{int(df["Minutes played"].median())}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Positions</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value-lg">{df["Position"].nunique()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Score Distributions</div>', unsafe_allow_html=True)

    cols = ["Offensive Score", "Defensive Score", "Key Passing Score"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col in zip(axes, cols):
        ax.hist(df[col].fillna(0), bins=20)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown('<div class="section-title">Sample of Player Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# =============================================================
# OUTLIER SCOUTING PAGE
# =============================================================
elif menu == "📊 Outlier Scouting":
    st.markdown('<div class="main-header">Outlier Scouting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subheader">Filter by minutes, team, position and composite scores to surface statistical outliers.</div>',
        unsafe_allow_html=True,
    )

    filtered = filter_panel(df)

    st.sidebar.markdown('<div class="sidebar-subtitle">Score thresholds</div>', unsafe_allow_html=True)

    min_off = st.sidebar.slider("Min Offensive Score", 0.0, 100.0, 0.0, step=1.0)
    min_def = st.sidebar.slider("Min Defensive Score", 0.0, 100.0, 0.0, step=1.0)
    min_key = st.sidebar.slider("Min Key Passing Score", 0.0, 100.0, 0.0, step=1.0)

    filtered = filtered[
        (filtered["Offensive Score"] >= min_off)
        & (filtered["Defensive Score"] >= min_def)
        & (filtered["Key Passing Score"] >= min_key)
    ]

    st.markdown('<div class="section-title">Filtered Players</div>', unsafe_allow_html=True)

    cols_to_show = [
        "Player", "Team", "Team within selected timeframe", "Position",
        "Minutes played", "Offensive Score", "Defensive Score", "Key Passing Score",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    st.dataframe(
        filtered[cols_to_show].sort_values("Offensive Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.write(f"**Players found:** {len(filtered)}")

    st.markdown('<div class="section-title">Top Outliers by Composite Score</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔥 Offensive", "🛡 Defensive", "🎯 Key Passing"])

    base_cols = cols_to_show

    with tab1:
        st.write("Top 15 offensive outliers in full dataset")
        st.dataframe(
            df.sort_values("Offensive Score", ascending=False).head(15)[base_cols],
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.write("Top 15 defensive outliers in full dataset")
        st.dataframe(
            df.sort_values("Defensive Score", ascending=False).head(15)[base_cols],
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.write("Top 15 key passing outliers in full dataset")
        st.dataframe(
            df.sort_values("Key Passing Score", ascending=False).head(15)[base_cols],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================
# PLAYER EXPLORER PAGE
# =============================================================
elif menu == "👤 Player Explorer":
    st.markdown('<div class="main-header">Player Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subheader">Dive into an individual player with composite scores, radar visual, and key metrics.</div>',
        unsafe_allow_html=True,
    )

    filtered = filter_panel(df)

    if len(filtered) == 0:
        st.warning("No players match the current filters.")
    else:
        player_name = st.selectbox("Select player", sorted(filtered["Player"].unique()))
        player_row = filtered[filtered["Player"] == player_name].iloc[0]

        # Top section: bio + scores
        colA, colB = st.columns([1.1, 1])

        with colA:
            st.markdown('<div class="section-title">Profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-value-lg'>{player_name}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='metric-value-sm'>{player_row['Team']} – {player_row['Position']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='metric-label'>Minutes played</div><div class='metric-value-sm'>{int(player_row['Minutes played'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Composite score cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Offensive score</div>', unsafe_allow_html=True)
                st.markdown(
                    f"<div class='metric-value-lg'>{player_row['Offensive Score']:.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Defensive score</div>', unsafe_allow_html=True)
                st.markdown(
                    f"<div class='metric-value-lg'>{player_row['Defensive Score']:.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Key passing score</div>', unsafe_allow_html=True)
                st.markdown(
                    f"<div class='metric-value-lg'>{player_row['Key Passing Score']:.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="section-title">Outlier radar</div>', unsafe_allow_html=True)
            radar_metrics = {
                "Offense": "Offensive Score",
                "Defense": "Defensive Score",
                "Key Pass": "Key Passing Score",
            }
            radar_chart(player_row, radar_metrics, title="Composite percentile profile")

        # Key metrics table
        st.markdown('<div class="section-title">Key metrics</div>', unsafe_allow_html=True)
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
# VISUAL EXPLORER PAGE
# =============================================================
elif menu == "📈 Visual Explorer":
    st.markdown('<div class="main-header">Visual Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subheader">Use scatter plots and distributions to visually spot statistical outliers.</div>',
        unsafe_allow_html=True,
    )

    filtered = filter_panel(df)

    if len(filtered) == 0:
        st.warning("No players match the current filters.")
    else:
        tab_scatter, tab_dist = st.tabs(["Scatter outliers", "Distributions & boxplots"])

        # --------------- SCATTER TAB ---------------
        with tab_scatter:
            st.markdown('<div class="section-title">Scatter views</div>', unsafe_allow_html=True)

            scatter_type = st.selectbox(
                "Scatter type",
                [
                    "Offense: Goals per 90 vs xG per 90",
                    "Defense: PAdj Interceptions vs Shots blocked per 90",
                    "Key passing: Key passes per 90 vs Passes to penalty area per 90",
                ],
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            if scatter_type.startswith("Offense"):
                xcol, ycol = "xG per 90", "Goals per 90"
                scores = filtered["Offensive Score"]
                ax.set_title("Offensive outliers")
            elif scatter_type.startswith("Defense"):
                xcol, ycol = "PAdj Interceptions", "Shots blocked per 90"
                scores = filtered["Defensive Score"]
                ax.set_title("Defensive outliers")
            else:
                xcol, ycol = "Key passes per 90", "Passes to penalty area per 90"
                scores = filtered["Key Passing Score"]
                ax.set_title("Key passing outliers")

            sc = ax.scatter(filtered[xcol], filtered[ycol], c=scores, alpha=0.85)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(alpha=0.3)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Composite score")
            st.pyplot(fig)

        # --------------- DISTRIBUTION TAB ---------------
        with tab_dist:
            st.markdown('<div class="section-title">Distributions & boxplots</div>', unsafe_allow_html=True)

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

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"**Distribution – {metric}**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(filtered[metric].dropna(), bins=20)
                ax.set_xlabel(metric)
                ax.set_ylabel("Count")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

            with c2:
                st.markdown(f"**Boxplot – {metric}**")
                fig, ax = plt.subplots(figsize=(3, 4))
                ax.boxplot(filtered[metric].dropna(), vert=True)
                ax.set_ylabel(metric)
                ax.grid(alpha=0.3)
                st.pyplot(fig)
