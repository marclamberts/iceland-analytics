import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ------------------- Streamlit UI -------------------
st.sidebar.header("Data Upload")
st.sidebar.write("Upload your NWSL dataset to begin.")
# Load Excel file from same folder
file_path = "Iceland.xlsx"

player_name = st.text_input("Player Name")
position_group = st.selectbox(
    "Select Position Group",
    ['Centre-back', 'Full-back', 'Midfielder', 'Attacking Midfielder', 'Attacker']
)

run_button = st.button("Generate Profile")("Generate Profile")
st.set_page_config(layout="wide", page_title="Player Profile Generator")
st.title("Player Profile Generator")

uploaded_file = st.file_uploader("Upload NWSL 2025 Excel File", type=["xlsx"])
player_name = st.text_input("Player Name")
position_group = st.selectbox(
    "Select Position Group",
    ['Centre-back', 'Full-back', 'Midfielder', 'Attacking Midfielder', 'Attacker']
)

run_button = st.button("Generate Profile")

if run_button:
    # call your function here
    generate_player_profile(df, player_name, position_group)

# ------------------- Function -------------------
def generate_player_profile(df, player_name, position_group):
    df = df.copy()
    df.rename(columns={"Nationality": "Passport country"}, inplace=True)

    # ---- Define Groups ----
    groups = {}
    groups['Centre-back'] = (
        ['CB', 'RCB', 'LCB'],
        {
            "Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"],
            "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Forward passes per 90", "Accurate forward passes, %", "Passes to final third per 90", "Accurate passes to final third, %"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"],
            "Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
            "Proactive Defending": ["Interceptions per 90", "PAdj Interceptions", "Sliding tackles per 90", "PAdj Sliding tackles"],
            "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %"],
            "Box Defending": ["Shots blocked per 90"],
            "Sweeping": []
        },
        {
            "Ball Player": ["Progressive Passing", "Security"],
            "Libero": ["Progressive Passing", "Ball Carrying", "Creativity"],
            "Wide Creator": ["Creativity", "Ball Carrying"],
            "Aggressor": ["Proactive Defending", "Duelling"],
            "Physical Dominator": ["Duelling", "Box Defending"],
            "Box Defender": ["Box Defending", "Duelling"]
        }
    )

    groups['Full-back'] = (
        ['LB', 'RB', 'LWB', 'RWB'],
        {
            "Box Defending": ["Shots blocked per 90"],
            "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %", "Defensive duels per 90", "Defensive duels won, %"],
            "Pressing": ["PAdj Interceptions", "PAdj Sliding tackles", "Counterpressing recoveries per 90"],
            "Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"],
            "Playmaking": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
            "Final Third": ["Passes to final third per 90", "Accurate passes to final third, %", "Crosses per 90", "Accurate crosses, %", "Touches in box per 90"],
            "Overlapping": ["Accelerations per 90", "Fouls suffered per 90"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"]
        },
        {
            "False Wing": ["Playmaking", "Ball Carrying", "Final Third"],
            "Flyer": ["Overlapping", "Final Third", "Ball Carrying"],
            "Playmaker": ["Playmaking", "Security", "Final Third"],
            "Safety": ["Security", "Box Defending", "Pressing"],
            "Ballwinner": ["Duelling", "Pressing"],
            "Defensive FB": ["Box Defending", "Duelling", "Pressing"]
        }
    )

    groups['Midfielder'] = (
        ['LCMF', 'RCMF', 'CFM', 'LDMF', 'RDMF', 'RAMF', 'LAMF', 'AMF', 'DMF'],
        {
            "Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
            "Box Crashing": ["Touches in box per 90", "Shots per 90", "xG per 90", "Non-penalty goals per 90"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"],
            "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Passes to final third per 90", "Accurate passes to final third, %"],
            "Dictating": ["Passes per 90", "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %", "Lateral passes per 90"],
            "Ball Winning": ["PAdj Interceptions", "Counterpressing recoveries per 90", "Defensive duels won, %"],
            "Destroying": ["Duels per 90", "Defensive duels per 90", "PAdj Sliding tackles", "Fouls per 90"]
        },
        {
            "Anchor": ["Destroying", "Ball Winning", "Dictating"],
            "DLP": ["Dictating", "Progressive Passing", "Creativity"],
            "Ball Winner": ["Ball Winning", "Destroying"],
            "Box to Box": ["Progressive Passing", "Ball Carrying", "Box Crashing", "Ball Winning"],
            "Box Crasher": ["Box Crashing", "Ball Carrying", "Creativity"],
            "Playmaker": ["Creativity", "Progressive Passing", "Dictating"],
            "Attacking mid": ["Creativity", "Box Crashing", "Progressive Passing"]
        }
    )

    groups['Attacking Midfielder'] = (
        ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'],
        {
            "Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"],
            "Build up": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90"],
            "Final ball": ["Key passes per 90", "xA per 90", "Deep completions per 90"],
            "Wide creation": ["Crosses per 90", "Accurate crosses, %", "Passes to penalty area per 90"],
            "Movement": ["Accelerations per 90", "Touches in box per 90"],
            "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90"],
            "1v1 ability": ["Dribbles per 90", "Successful dribbles, %", "Fouls suffered per 90"],
            "Box presence": ["Touches in box per 90", "xG per 90", "Shots per 90"],
            "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]
        },
        {
            "Winger": ["Wide creation", "1v1 ability", "Ball Carrying"],
            "Direct Dribbler": ["1v1 ability", "Ball Carrying", "Movement"],
            "Industrious Winger": ["Pressing", "Ball Carrying", "Wide creation"],
            "Shadow Striker": ["Box presence", "Finishing", "Movement"],
            "Wide playmaker": ["Wide creation", "Final ball", "Build up"],
            "Playmaker": ["Final ball", "Build up", "1v1 ability"]
        }
    )

    groups['Attacker'] = (
        ['CF', 'LW', 'RW'],
        {
        "Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"],
        "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"],
        "Creativity": ["Key passes per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"],
        "Link Play": ["Passes per 90", "Accurate passes, %", "Deep completions per 90", "Passes to final third per 90"],
        "Movement": ["Accelerations per 90", "Touches in box per 90"],
        "Box Presence": ["Touches in box per 90", "Shots per 90", "Aerial duels per 90", "Aerial duels won, %"],
        "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]
        },
        {
        "Poacher": ["Finishing", "Box Presence", "Movement"],
        "Second Striker": ["Creativity", "Ball Carrying", "Finishing"],
        "Link Forward": ["Link Play", "Creativity", "Ball Carrying"],
        "False 9": ["Link Play", "Creativity", "Ball Carrying"],
        "Complete Forward": ["Finishing", "Box Presence", "Ball Carrying", "Link Play"],
        "Power Forward": ["Box Presence", "Finishing"],
        "Pressing Forward": ["Pressing", "Movement", "Box Presence"]
        }
        )
       



