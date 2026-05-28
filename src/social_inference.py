"""Inference pipeline for the social media (influencer) credit scoring model.

Input: raw channel stats and country name.
Output: credit score with top contributing factors. Feature engineering happens
internally; the caller never sees engineered features.
"""

import os
import numpy as np
import pandas as pd
import joblib

DEFAULT_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts"
)

FEATURE_DESCRIPTIONS = {
    "log_subscribers":           "Subscriber count (channel size)",
    "channel_age_years":         "How long the channel has existed",
    "uploads_per_year":          "Upload frequency",
    "views_per_subscriber":      "View-to-subscriber ratio (audience loyalty)",
    "views_per_upload":          "Average views per video",
    "is_brand_channel":          "Brand channel vs personal creator",
    "log_country_rank":          "Country-level ranking position",
    "log_views_rank":            "Global views ranking position",
    "unemployment_rate":         "Country unemployment rate",
    "education_rate":            "Country tertiary education enrollment",
    "country_pop_millions":      "Country population (millions)",
    "urbanization_pct":          "Country urbanization percentage",
    "earnings_originally_zero":  "Demonetization or zero-earnings flag",
}


def _engineer_features(raw_input, country_macros, snapshot_year):
    """Turn raw channel stats into the engineered feature vector.

    Parameters
    ----------
    raw_input : dict
        Required keys:
            subscribers (int)
            channel_started_year (int, e.g. 2018)
            uploads (int, total uploads since channel started)
            video_views_rank (int, global rank)
            country (str)
            country_rank (int)
            channel_type (str, e.g. 'Music', 'Entertainment')
        Optional keys (default to False/0 when missing):
            is_brand_channel (bool)
            earnings_originally_zero (bool)

    country_macros : dict
        {'lookup': {country_name: {macro_dict}}, 'defaults': {macro_dict}}

    snapshot_year : int
        The year the model was trained on (used to compute channel age).

    Returns
    -------
    dict of engineered features matching the model's input schema.
    """
    subscribers = max(int(raw_input.get("subscribers", 0)), 1)
    started_year = int(raw_input.get("channel_started_year", snapshot_year - 1))
    uploads = max(int(raw_input.get("uploads", 0)), 1)
    video_views = int(raw_input.get("video_views", subscribers * 100))
    country_rank = max(int(raw_input.get("country_rank", 100000)), 1)
    views_rank = max(int(raw_input.get("video_views_rank", 100000)), 1)
    country = str(raw_input.get("country", "")).strip()

    is_brand = bool(raw_input.get("is_brand_channel", False))
    earnings_zero = bool(raw_input.get("earnings_originally_zero", False))

    channel_age = max(snapshot_year - started_year, 1)
    uploads_per_year = uploads / channel_age
    views_per_subscriber = video_views / subscribers
    views_per_upload = video_views / uploads

    macros = country_macros["lookup"].get(country, country_macros["defaults"])

    return {
        "log_subscribers": float(np.log1p(subscribers)),
        "channel_age_years": float(channel_age),
        "uploads_per_year": float(uploads_per_year),
        "views_per_subscriber": float(views_per_subscriber),
        "views_per_upload": float(views_per_upload),
        "is_brand_channel": int(is_brand),
        "log_country_rank": float(np.log1p(country_rank)),
        "log_views_rank": float(np.log1p(views_rank)),
        "unemployment_rate": float(macros.get("unemployment_rate", 0)),
        "education_rate": float(macros.get("education_rate", 0)),
        "country_pop_millions": float(macros.get("country_pop_millions", 0)),
        "urbanization_pct": float(macros.get("urbanization_pct", 0)),
        "earnings_originally_zero": int(earnings_zero),
    }


class SocialCreditScorer:
    """Inference pipeline for the YouTube influencer credit scoring model.

    Usage:
        scorer = SocialCreditScorer()
        result = scorer.score({
            "subscribers": 250000,
            "channel_started_year": 2019,
            "uploads": 320,
            "video_views": 18000000,
            "video_views_rank": 45000,
            "country": "United States",
            "country_rank": 12000,
            "channel_type": "Entertainment",
            "is_brand_channel": False,
        })
        print(result["credit_score"], result["risk_band"])
        for factor in result["top_factors"]:
            print(factor["description"], factor["score_points"])
    """

    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS_DIR):
        self.model = joblib.load(os.path.join(artifacts_dir, "social_xgb_model.pkl"))
        meta = joblib.load(os.path.join(artifacts_dir, "social_feature_metadata.pkl"))
        self.country_macros = joblib.load(os.path.join(artifacts_dir, "country_macros.pkl"))
        self.feature_cols = meta["feature_cols"]
        self.feature_medians = meta["feature_medians"]
        self.snapshot_year = meta.get("snapshot_year", 2024)

    def _impute_and_order(self, features):
        row = {}
        for col in self.feature_cols:
            val = features.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = self.feature_medians.get(col, 0)
            row[col] = val
        return pd.DataFrame([row], columns=self.feature_cols)

    @staticmethod
    def _to_credit_score(probability):
        return int(round(300 + (1 - probability) * 550))

    @staticmethod
    def _risk_band(score):
        if score >= 750: return "Very Low Risk"
        if score >= 650: return "Low Risk"
        if score >= 550: return "Moderate"
        if score >= 450: return "High Risk"
        return "Very High Risk"

    def _explain(self, X, probability):
        """Per-feature contributions, converted to credit score points."""
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X.values, feature_names=self.feature_cols)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)[0]
        feature_contribs = contribs[:-1]
        bias = contribs[-1]

        sigmoid_slope = probability * (1 - probability)
        score_range = 550

        explanations = []
        for col, val, contrib in zip(self.feature_cols, X.iloc[0].values, feature_contribs):
            score_pts = -float(contrib) * sigmoid_slope * score_range
            explanations.append({
                "feature": col,
                "description": FEATURE_DESCRIPTIONS.get(col, col),
                "value": round(float(val), 3),
                "contribution": round(float(contrib), 4),
                "score_points": round(score_pts, 1),
                "direction": "boosts score" if contrib < 0 else "lowers score",
            })

        explanations.sort(key=lambda e: abs(e["score_points"]), reverse=True)
        return explanations, float(bias)

    def known_countries(self):
        """Return the sorted list of countries with macro data available."""
        return sorted(self.country_macros["lookup"].keys())

    def score(self, raw_input, top_n_explanations=5):
        """Score a channel from raw input.

        Parameters
        ----------
        raw_input : dict with required keys:
            subscribers, channel_started_year, uploads, video_views,
            video_views_rank, country, country_rank
          Optional keys:
            is_brand_channel (default False)
            earnings_originally_zero (default False)
            channel_type

        Returns
        -------
        dict with: credit_score, risk_band, risk_probability,
                   features (engineered), top_factors, all_factors,
                   country_used, country_was_known, warning
        """
        warning = None
        country = str(raw_input.get("country", "")).strip()
        country_known = country in self.country_macros["lookup"]
        if not country_known and country:
            warning = (f"Country '{country}' not in our lookup. Using global "
                       f"defaults for unemployment, education, population, and "
                       f"urbanization.")

        features = _engineer_features(raw_input, self.country_macros, self.snapshot_year)
        X = self._impute_and_order(features)
        proba = float(self.model.predict_proba(X)[0, 1])
        score = self._to_credit_score(proba)
        band = self._risk_band(score)
        explanations, baseline = self._explain(X, proba)

        return {
            "credit_score": score,
            "risk_band": band,
            "risk_probability": round(proba, 4),
            "features": features,
            "top_factors": explanations[:top_n_explanations],
            "all_factors": explanations,
            "country_used": country if country_known else "Global default",
            "country_was_known": country_known,
            "warning": warning,
        }