"""Social Media Engine: YouTube influencer credit scoring demo.

Three input modes: manual channel entry, sample channels, batch CSV upload.
All modes accept raw channel data and run feature engineering internally.
"""
import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from src.social_inference import SocialCreditScorer


st.set_page_config(
    page_title="Social Media Credit Engine",
    page_icon="📺",
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


set_background("artifacts/social_media.png")

# ============================================================
# Cached resources
# ============================================================
@st.cache_resource
def load_scorer():
    return SocialCreditScorer()


@st.cache_data
def load_sample_channels():
    return pd.read_parquet("artifacts/sample_channels.parquet")


scorer = load_scorer()
KNOWN_COUNTRIES = scorer.known_countries()

CHANNEL_TYPES = [
    "Entertainment", "Music", "Gaming", "Education", "Film",
    "Sports", "News", "Tech", "Comedy", "Howto", "People",
    "Animals", "Autos", "Travel", "Nonprofit", "Trailers",
]


# ============================================================
# Header
# ============================================================
st.title("Social Media Engine: Influencer Credit Scoring")
st.caption("Model A from the Alternative Credit Scoring proposal")

with st.expander("About this engine", expanded=False):
    st.markdown(
        """
        **Premise:** creator economy signals (subscriber base, upload cadence,
        audience loyalty, country macro environment) predict credit risk for
        influencers.

        **What you input:** raw channel statistics (subscribers, uploads,
        views, country, etc.).

        **What you get:** a credit score (300 to 850), a risk band, and the
        top factors that drove the score.
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


def score_and_render(raw_input):
    """Run inference and render the full result UI."""
    result = scorer.score(raw_input)
    render_score_card(result)
    st.divider()
    render_top_factors(result)


# ============================================================
# Three input tabs
# ============================================================
tab1, tab2 = st.tabs([
    "Enter channel manually",
    "Upload a CSV (batch)",
])


# ----- Tab 1: Manual entry -----
with tab1:
    st.markdown(
        """
        Enter the channel's statistics below. The model never sees the
        channel's name; only the underlying metrics matter for scoring.
        """
    )

    with st.expander("What does each field mean?", expanded=False):
        st.markdown(
            """
            - **Subscribers**: total subscriber count for this channel.
            - **Year channel started**: the year the channel was created on YouTube.
            - **Total uploads**: how many videos the channel has published in total.
            - **Total video views**: lifetime view count across all videos.
            - **Global video views rank**: this channel's rank globally by total
              views (lower = bigger). If you don't know, leave the default.
            - **Country**: the channel's primary country.
            - **Country rank**: the channel's rank within its country (lower = bigger).
            - **Channel type**: the content category. We use this to detect whether
              it's a brand channel (Music, Entertainment, Education, Film tend to
              be brand-operated).
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        subscribers = st.number_input(
            "Subscribers", min_value=1000, max_value=300_000_000,
            value=250_000, step=10_000,
        )
        channel_started_year = st.number_input(
            "Year channel started", min_value=2005, max_value=2024,
            value=2019, step=1,
        )
        uploads = st.number_input(
            "Total uploads", min_value=1, max_value=100_000,
            value=320, step=10,
        )
        video_views = st.number_input(
            "Total video views (lifetime)", min_value=1000,
            max_value=300_000_000_000, value=18_000_000, step=100_000,
        )

    with col2:
        video_views_rank = st.number_input(
            "Global video views rank", min_value=1, max_value=10_000_000,
            value=45_000, step=1_000,
            help="Lower = bigger channel. Leave default if unknown.",
        )

        default_country_idx = KNOWN_COUNTRIES.index("United States") \
            if "United States" in KNOWN_COUNTRIES else 0
        country = st.selectbox(
            "Country", KNOWN_COUNTRIES, index=default_country_idx,
        )

        country_rank = st.number_input(
            "Country rank", min_value=1, max_value=1_000_000,
            value=12_000, step=500,
            help="Lower = bigger channel in this country.",
        )
        channel_type = st.selectbox(
            "Channel type", CHANNEL_TYPES, index=0,
        )

    if st.button("Calculate credit score", type="primary", key="btn_manual"):
        # Derive is_brand_channel from channel_type (matches notebook logic)
        is_brand = channel_type in ["Music", "Entertainment", "Education", "Film"]

        raw_input = {
            "subscribers": subscribers,
            "channel_started_year": channel_started_year,
            "uploads": uploads,
            "video_views": video_views,
            "video_views_rank": video_views_rank,
            "country": country,
            "country_rank": country_rank,
            "channel_type": channel_type,
            "is_brand_channel": is_brand,
        }

        st.divider()
        score_and_render(raw_input)


# ----- Tab 2: Batch CSV upload -----
with tab2:
    st.markdown(
        """
        Upload a CSV with one row per channel. The CSV must have these columns:

        `channel_name, subscribers, channel_started_year, uploads, video_views, video_views_rank, country, country_rank, channel_type`

        Channels with missing required fields will be skipped.
        """
    )

    # Example CSV
    example_csv = pd.DataFrame([
        {
            "channel_name": "ExampleChannel_001",
            "subscribers": 250_000,
            "channel_started_year": 2019,
            "uploads": 320,
            "video_views": 18_000_000,
            "video_views_rank": 45_000,
            "country": "United States",
            "country_rank": 12_000,
            "channel_type": "Entertainment",
        },
        {
            "channel_name": "ExampleChannel_002",
            "subscribers": 5_000_000,
            "channel_started_year": 2015,
            "uploads": 1_200,
            "video_views": 800_000_000,
            "video_views_rank": 800,
            "country": "United Kingdom",
            "country_rank": 100,
            "channel_type": "Music",
        },
    ]).to_csv(index=False)

    st.download_button(
        "Download example CSV format",
        data=example_csv,
        file_name="example_channels.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        upload_key = uploaded.name + str(uploaded.size)
        if st.session_state.get("social_upload_key") != upload_key:
            st.session_state["social_upload_key"] = upload_key
            st.session_state.pop("social_results_df", None)
            st.session_state.pop("social_input_df", None)

        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        required_cols = {"channel_name", "subscribers", "channel_started_year",
                         "uploads", "video_views", "video_views_rank",
                         "country", "country_rank", "channel_type"}
        missing = required_cols - set(batch_df.columns)
        if missing:
            st.error(f"CSV is missing columns: {sorted(missing)}")
            st.stop()

        batch_df = batch_df.dropna(subset=list(required_cols))
        st.success(f"Loaded {len(batch_df)} channels.")

        if st.button("Score all channels", type="primary", key="btn_batch_social"):
            results_rows = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(batch_df.iterrows()):
                channel_type = str(row["channel_type"])
                is_brand = channel_type in ["Music", "Entertainment", "Education", "Film"]
                raw_input = {
                    "subscribers": int(row["subscribers"]),
                    "channel_started_year": int(row["channel_started_year"]),
                    "uploads": int(row["uploads"]),
                    "video_views": int(row["video_views"]),
                    "video_views_rank": int(row["video_views_rank"]),
                    "country": str(row["country"]),
                    "country_rank": int(row["country_rank"]),
                    "channel_type": channel_type,
                    "is_brand_channel": is_brand,
                }
                result = scorer.score(raw_input)
                results_rows.append({
                    "channel_name": row["channel_name"],
                    "credit_score": result["credit_score"],
                    "risk_band": result["risk_band"],
                    "country_used": result["country_used"],
                    "warning": result["warning"] or "",
                })
                progress.progress((i + 1) / len(batch_df))

            st.session_state["social_results_df"] = pd.DataFrame(results_rows)
            st.session_state["social_input_df"] = batch_df.reset_index(drop=True)

    if "social_results_df" in st.session_state:
        results_df = st.session_state["social_results_df"]
        input_df = st.session_state["social_input_df"]

        st.subheader("Batch results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored CSV",
            data=csv_bytes,
            file_name="scored_channels.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Drill into one channel")
        drill_channel = st.selectbox(
            "Pick a channel to see full breakdown:",
            results_df["channel_name"].tolist(),
            key="batch_drill_social",
        )
        if drill_channel:
            row = input_df[input_df["channel_name"] == drill_channel].iloc[0]
            channel_type = str(row["channel_type"])
            is_brand = channel_type in ["Music", "Entertainment", "Education", "Film"]
            raw_input = {
                "subscribers": int(row["subscribers"]),
                "channel_started_year": int(row["channel_started_year"]),
                "uploads": int(row["uploads"]),
                "video_views": int(row["video_views"]),
                "video_views_rank": int(row["video_views_rank"]),
                "country": str(row["country"]),
                "country_rank": int(row["country_rank"]),
                "channel_type": channel_type,
                "is_brand_channel": is_brand,
            }
            score_and_render(raw_input)


st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:12px; margin-top:32px;">
        Built with the Alternative Credit Scoring framework.
        Model trained on the Global YouTube Statistics dataset.
    </div>
    """,
    unsafe_allow_html=True,
)