---
title: Alternative Credit Scoring Engines
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# Alternative Credit Scoring: Gaming and Influencer Engines

Two production-style demo engines that build FICO-scale credit scores from non-traditional data sources. Both follow the same architecture: take raw behavioral data as input, compute behavioral features internally, run an XGBoost classifier, return a credit score with per-feature explainability.

## The two engines

### Gaming Engine (Model B)

Credit scoring from chess gameplay behavior. Built on the Lichess Elite Database (April 2024, 4,482 players with 20+ games per month).

Given a player's raw chess game history, computes 14 behavioral features covering clock management, opening repertoire concentration, schedule discipline, late-night play patterns, and time-control preference. Returns a 300 to 850 credit score with the top 5 contributing factors.

- Entry point: `app.py`
- Training notebook: `Notebooks/GamingEngine.ipynb`
- Inference module: `src/gaming_inference.py`

### Social Media Engine (Model A)

Credit scoring from YouTube creator behavior. Built on the Global YouTube Statistics dataset.

Given a channel's raw statistics (subscribers, uploads, views, country, channel type), computes 12 engineered features covering channel size, age, upload cadence, audience loyalty, ranking, and country macro environment. Returns a 300 to 850 credit score with the top 5 contributing factors.

- Entry point: `social_app.py`
- Training notebook: `Notebooks/SocialMediaEngine.ipynb`
- Inference module: `src/social_inference.py`

## Three input modes per engine

- **Manual entry**: enter raw data through forms or a table; features are engineered internally
- **Sample players/channels**: pick from real entries in the training data
- **Batch CSV upload**: upload multiple records at once, get back scored results plus per-record drill-down

## Repository structure

```
├── app.py                          Gaming engine Streamlit app
├── social_app.py                   Social media engine Streamlit app
├── requirements.txt
├── README.md
├── artifacts/
│   ├── xgb_model.pkl               Gaming engine model
│   ├── logreg_pipe.pkl             Gaming engine LogReg pipeline
│   ├── feature_metadata.pkl        Gaming engine metadata
│   ├── sample_games.parquet        Gaming engine sample data
│   ├── social_xgb_model.pkl        Social engine model
│   ├── social_logreg_pipe.pkl      Social engine LogReg pipeline
│   ├── social_feature_metadata.pkl Social engine metadata
│   ├── sample_channels.parquet     Social engine sample data
│   ├── country_macros.pkl          Country macro lookup table
│   └── chess.png                   Gaming engine background image
├── src/
│   ├── __init__.py
│   ├── gaming_inference.py         Gaming engine inference pipeline
│   ├── social_inference.py         Social engine inference pipeline
│   └── build_sample_data.py        Gaming engine sample data builder
└── Notebooks/
    ├── GamingEngine.ipynb
    └── SocialMediaEngine.ipynb
```

## Run locally

```bash
pip install -r requirements.txt

# Gaming engine
streamlit run app.py

# Social media engine
streamlit run social_app.py
```

Each app starts on port 8501. To run them side by side, pass `--server.port 8502` to the second one.

## How both engines work

1. **Raw input**: caller provides domain-specific raw data (chess games for gaming, channel stats for social). The user never enters engineered feature values.
2. **Feature engineering**: happens internally inside the inference module. Behavioral aggregates are computed from raw records.
3. **Model inference**: XGBoost classifier outputs a probability of high credit risk.
4. **Score mapping**: probability is mapped to the 300 to 850 FICO scale.
5. **Per-prediction explainability**: XGBoost's pred_contribs returns SHAP-equivalent values per feature, converted to credit score points for human-readable explanations.

## Data sources

- **Gaming engine**: Lichess Elite Database, April 2024 monthly snapshot. Filtered to 2400+ rated players with 20+ games per month. Source: https://database.nikonoel.fr/
- **Social media engine**: Global YouTube Statistics dataset, single snapshot covering creator metrics, monthly engagement, and country macro indicators.

## Limitations

Both engines are demos built on real but limited data. Production deployment would require:

- Multi-month longitudinal data (single-snapshot training cannot detect behavioral drift)
- Broader sample tiers (gaming model trained on elite players only; social model trained on top-tier creators only)
- Validation against actual loan repayment outcomes (current targets are proxy composites)
- Bot/automation filters (especially for the gaming engine)

See each notebook for the full training pipeline, leakage diagnostics, and methodology.