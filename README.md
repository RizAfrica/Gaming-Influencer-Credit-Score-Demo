---
title: Gaming Credit Engine
emoji: ♟️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---

# Gaming Engine: Psychometric Credit Scoring

Demo of Model B from the Alternative Credit Scoring framework. Builds a credit risk score from real chess gameplay behavior on Lichess.

## What it does

Given a player's raw chess game history, computes 14 behavioral features (clock management, opening repertoire concentration, schedule discipline, late-night play patterns, and so on), runs them through an XGBoost classifier, and returns a FICO style 300 to 850 credit score with a breakdown of the top factors driving the result.

## Three input modes

- **Manual entry**: edit a table of 20 default games to score a hypothetical player
- **Sample players**: pick from real players in the training data
- **Batch CSV**: upload a CSV with raw game histories for multiple players

## Data

Lichess Elite Database, April 2024 (4,482 active players with 20+ games).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

This is a demo on real but limited data (one month, elite players only). Production deployment would require multi-month longitudinal data and validation against actual loan outcomes. See `Notebooks/GamingEngine.ipynb` for the full training pipeline and methodology.