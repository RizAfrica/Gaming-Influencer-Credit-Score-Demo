"""Gaming Engine: chess-based credit scoring demo.

Three input modes: manual game entry, sample players, batch CSV upload.
All modes accept raw game data and run feature engineering internally.
"""
import sys
import os
import io
import uuid
import base64
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from src.gaming_inference import ChessCreditScorer


st.set_page_config(
    page_title="Gaming Credit Engine",
    page_icon="♟️",
    layout="wide",
)


# ============================================================
# Background image setup
# ============================================================
def set_background(image_path):
    """Apply a background image with a light overlay so text stays readable."""
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(255, 255, 255, 0.88),
                                rgba(255, 255, 255, 0.88)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp > header {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background("artifacts/chess.png")


# ============================================================
# Cached resources
# ============================================================
@st.cache_resource
def load_scorer():
    return ChessCreditScorer()


@st.cache_data
def load_sample_players():
    return pd.read_parquet("artifacts/sample_games.parquet")


scorer = load_scorer()


# ============================================================
# Header
# ============================================================
st.title("♟️ Gaming Engine: Psychometric Credit Scoring")
st.caption("Model B from the Alternative Credit Scoring proposal")

with st.expander("About this engine", expanded=False):
    st.markdown(
        """
        **Premise:** how a person plays games reveals psychometric traits relevant
        to financial behavior.

        **What you input:** raw game history (one row per game).

        **What happens behind the scenes:** Behavioral features are computed
        automatically from your input (engagement intensity, clock management,
        opening repertoire concentration, schedule discipline, and so on).

        **What you get:** a credit score (300 to 850), a risk band, and the top
        factors that drove the score. 

        **Note:** demo on real but limited data (one month, elite players).
        Production deployment would require multi-month longitudinal data and
        validation against actual loan outcomes.
        """
    )

st.divider()


# ============================================================
# Shared rendering helpers
# ============================================================
BAND_COLORS = {
    "Very Low Risk":  "#1b8a3a",
    "Low Risk":       "#5ba85b",
    "Moderate":       "#d9a44a",
    "High Risk":      "#d96c4a",
    "Very High Risk": "#b53030",
}


def render_score_card(result):
    score = result["credit_score"]
    band = result["risk_band"]
    color = BAND_COLORS.get(band, "#888")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Credit Score", f"{score}",
                  help="FICO style scale (300 to 850)")
    with c2:
        st.markdown(
            f"""
            <div style="background:{color}; color:white; padding:14px;
                        border-radius:6px; text-align:center; margin-top:6px;">
                <div style="font-size:12px; opacity:0.85;">RISK BAND</div>
                <div style="font-size:22px; font-weight:600;">{band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.get("warning"):
        st.warning(result["warning"])


def render_top_factors(result):
    st.subheader("Top 5 factors driving this score")
    st.caption("How each factor moved the score, in points (out of 550 total range).")

    top_five = result["top_factors"][:5]

    for i, factor in enumerate(top_five, 1):
        pts = factor["score_points"]
        if pts >= 0:
            sign = "+"
            bar_color = "#5ba85b"
        else:
            sign = ""
            bar_color = "#d96c4a"

        magnitude = min(abs(pts) / 50.0, 1.0)
        bar_width = int(magnitude * 100)

        st.markdown(
            f"""
            <div style="margin-bottom: 14px; padding: 12px;
                        background: rgba(248, 248, 248, 0.92); border-radius: 6px;
                        border-left: 4px solid {bar_color};">
                <div style="display: flex; justify-content: space-between;
                            align-items: center;">
                    <div style="font-size: 14px; color: #222;">
                        <b>{i}. {factor['description']}</b>
                    </div>
                    <div style="font-size: 18px; font-weight: 700;
                                color: {bar_color};">
                        {sign}{pts:.0f} pts
                    </div>
                </div>
                <div style="margin-top: 10px; height: 5px; background: #e0e0e0;
                            border-radius: 3px;">
                    <div style="width: {bar_width}%; height: 100%;
                                background: {bar_color}; border-radius: 3px;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def score_and_render(games_df):
    """Run inference and render the full result UI."""
    result = scorer.score(games_df)
    render_score_card(result)
    st.divider()
    render_top_factors(result)


# ============================================================
# Opening options for the manual editor
# ============================================================
COMMON_OPENINGS = [
    ("Sicilian Defense", "B20"),
    ("Italian Game", "C50"),
    ("Ruy Lopez", "C60"),
    ("French Defense", "C00"),
    ("Caro-Kann Defense", "B10"),
    ("Queen's Gambit", "D06"),
    ("King's Indian Defense", "E60"),
    ("English Opening", "A10"),
    ("Scandinavian Defense", "B01"),
    ("Pirc Defense", "B07"),
    ("Slav Defense", "D10"),
    ("Nimzo-Indian Defense", "E20"),
    ("Catalan Opening", "E00"),
    ("Reti Opening", "A04"),
    ("London System", "D02"),
    ("Vienna Game", "C25"),
    ("Scotch Game", "C44"),
    ("Four Knights Game", "C46"),
    ("Petroff Defense", "C42"),
    ("Alekhine Defense", "B02"),
]

OPENING_CHOICES = [f"{name} ({eco})" for name, eco in COMMON_OPENINGS]


def parse_opening_choice(choice):
    """Split 'Sicilian Defense (B20)' back into ('Sicilian Defense', 'B20')."""
    if "(" not in choice or not choice.endswith(")"):
        return choice, ""
    name = choice.rsplit("(", 1)[0].strip()
    eco = choice.rsplit("(", 1)[1].rstrip(")").strip()
    return name, eco


def build_default_games(n=20):
    """Pre-populated sensible default games for the manual editor."""
    base_time = datetime(2024, 4, 1, 12, 0, 0)
    outcomes_cycle = ["win", "loss", "draw", "win", "loss"]
    rows = []
    for i in range(n):
        opening_choice = OPENING_CHOICES[i % len(OPENING_CHOICES)]
        outcome = outcomes_cycle[i % len(outcomes_cycle)]
        rating_diff = {"win": 7, "loss": -7, "draw": 0}[outcome]
        rows.append({
            "elo": 2500,
            "rating_diff": rating_diff,
            "color": "white" if i % 2 == 0 else "black",
            "outcome": outcome,
            "termination": "Normal" if i % 7 != 0 else "Time forfeit",
            "time_control": "180+0",
            "opening_full": opening_choice,
            "timestamp": (base_time + timedelta(hours=i * 8)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows)


# ============================================================
# Three input tabs
# ============================================================
tab1, tab2 = st.tabs([
    "Enter games manually",
    "Upload a CSV (batch)",
])


# ----- Tab 1: Manual entry -----
with tab1:
    st.markdown(
        """
        Enter at least 20 games below. The form is pre-filled with example values.
        Edit them to match the player you want to score.

        **The model never sees a player name.** We generate an anonymous internal
        ID for you. Only the game data matters.
        """
    )

    with st.expander("What does each column mean?", expanded=False):
        st.markdown(
            """
            - **elo**: the player's Elo rating during this game (e.g. 2400 to 2700)
            - **rating_diff**: rating change after this game (positive after win,
              negative after loss; typically -10 to +10)
            - **color**: which side the player had, `white` or `black`
            - **outcome**: result for this player, `win`, `loss`, or `draw`
            - **termination**: how the game ended. `Normal` means resignation or
              checkmate. `Time forfeit` means lost on the clock.
            - **time_control**: Lichess format string like `180+0` (3 minutes, no
              increment) or `300+3` (5 minutes plus 3 second increment)
            - **opening**: pick from the dropdown. The list includes the 20 most
              common openings at the elite level. The ECO code (an academic
              classification) is attached to each entry automatically.
            - **timestamp**: when the game was played, format `YYYY-MM-DD HH:MM:SS`
            """
        )

    default_games = build_default_games(20)
    edited = st.data_editor(
        default_games,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "elo": st.column_config.NumberColumn(
                "Elo",
                help="Player's Elo rating during this game",
                min_value=1000, max_value=3500, step=10,
            ),
            "rating_diff": st.column_config.NumberColumn(
                "Rating change",
                help="Rating change after this game",
                min_value=-50, max_value=50, step=1,
            ),
            "color": st.column_config.SelectboxColumn(
                "Color",
                options=["white", "black"],
                required=True,
            ),
            "outcome": st.column_config.SelectboxColumn(
                "Outcome",
                options=["win", "loss", "draw"],
                required=True,
            ),
            "termination": st.column_config.SelectboxColumn(
                "Termination",
                options=["Normal", "Time forfeit"],
                required=True,
            ),
            "time_control": st.column_config.TextColumn(
                "Time control",
                help="e.g. 180+0 means 3 min, no increment",
            ),
            "opening_full": st.column_config.SelectboxColumn(
                "Opening",
                options=OPENING_CHOICES,
                required=True,
                help="Choose the opening played. ECO code is included automatically.",
            ),
            "timestamp": st.column_config.TextColumn(
                "Timestamp",
                help="Format: YYYY-MM-DD HH:MM:SS",
            ),
        },
        hide_index=True,
        key="manual_editor",
    )

    if st.button("Calculate credit score", type="primary", key="btn_manual"):
        if len(edited) < 20:
            st.error(f"Need at least 20 games. You provided {len(edited)}.")
        else:
            games_input = edited.copy()
            parsed = games_input["opening_full"].apply(parse_opening_choice)
            games_input["opening"] = parsed.apply(lambda x: x[0])
            games_input["eco"] = parsed.apply(lambda x: x[1])
            games_input = games_input.drop(columns=["opening_full"])

            games_input["timestamp"] = pd.to_datetime(games_input["timestamp"],
                                                     errors="coerce")
            games_input["player"] = f"manual_{uuid.uuid4().hex[:8]}"

            if games_input["timestamp"].isna().any():
                st.error(
                    "Some timestamps couldn't be parsed. Make sure they're in "
                    "`YYYY-MM-DD HH:MM:SS` format."
                )
            else:
                st.divider()
                score_and_render(games_input)

# ----- Tab 3: Batch CSV upload -----
with tab2:
    st.markdown(
        """
        Upload a CSV with raw game history for one or more players. The CSV must
        have these columns:

        `player, elo, rating_diff, color, outcome, termination, time_control, opening, eco, timestamp`

        Players with fewer than 20 games will still be scored, but with a warning.
        """
    )

    example_df = build_default_games(5).assign(player="example_player_001")
    parsed_ex = example_df["opening_full"].apply(parse_opening_choice)
    example_df["opening"] = parsed_ex.apply(lambda x: x[0])
    example_df["eco"] = parsed_ex.apply(lambda x: x[1])
    example_df = example_df.drop(columns=["opening_full"])
    example_csv = example_df.to_csv(index=False)
    st.download_button(
        "Download example CSV format",
        data=example_csv,
        file_name="example_games.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        upload_key = uploaded.name + str(uploaded.size)
        if st.session_state.get("batch_upload_key") != upload_key:
            st.session_state["batch_upload_key"] = upload_key
            st.session_state.pop("batch_results_df", None)
            st.session_state.pop("batch_games_df", None)

        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        required_cols = {"player", "elo", "rating_diff", "color", "outcome",
                         "termination", "time_control", "opening", "eco", "timestamp"}
        missing = required_cols - set(batch_df.columns)
        if missing:
            st.error(f"CSV is missing columns: {sorted(missing)}")
            st.stop()

        batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], errors="coerce")
        if batch_df["timestamp"].isna().any():
            st.warning(
                f"{batch_df['timestamp'].isna().sum()} rows have unparseable "
                "timestamps and will be excluded."
            )
            batch_df = batch_df.dropna(subset=["timestamp"])

        players = batch_df["player"].unique()
        st.success(f"Loaded {len(batch_df):,} games across {len(players)} players.")

        if st.button("Score all players", type="primary", key="btn_batch"):
            results_rows = []
            progress = st.progress(0)
            for i, player in enumerate(players):
                player_games = batch_df[batch_df["player"] == player]
                result = scorer.score(player_games)
                results_rows.append({
                    "player": player,
                    "n_games": result["features"]["n_games"],
                    "credit_score": result["credit_score"],
                    "risk_band": result["risk_band"],
                    "warning": result["warning"] or "",
                })
                progress.progress((i + 1) / len(players))

            st.session_state["batch_results_df"] = pd.DataFrame(results_rows)
            st.session_state["batch_games_df"] = batch_df

    if "batch_results_df" in st.session_state:
        results_df = st.session_state["batch_results_df"]
        batch_games_df = st.session_state["batch_games_df"]

        st.subheader("Batch results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored CSV",
            data=csv_bytes,
            file_name="scored_players.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Drill into one player")
        drill_player = st.selectbox(
            "Pick a player to see full breakdown:",
            results_df["player"].tolist(),
            key="batch_drill",
        )
        if drill_player:
            drill_games = batch_games_df[batch_games_df["player"] == drill_player]
            score_and_render(drill_games)


st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:12px; margin-top:32px;">
        Built with the Alternative Credit Scoring framework.
        Model trained on real Lichess data, April 2024.
    </div>
    """,
    unsafe_allow_html=True,
)