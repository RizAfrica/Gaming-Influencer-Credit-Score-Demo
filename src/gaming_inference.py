"""Inference pipeline for the chess credit scoring model.

Input: a player's raw game history.
Output: credit score plus per-feature explanation of why the model scored them this way.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy

DEFAULT_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts"
)

FEATURE_DESCRIPTIONS = {
    "n_games":              "Total games played in the month",
    "elo_mean":             "Average Elo rating",
    "win_rate":             "Fraction of games won",
    "draw_rate":            "Fraction of games drawn",
    "rating_diff_volatility": "Per-game rating change variance",
    "rapid_share":          "Share of rapid-format games (vs blitz)",
    "classical_share":      "Share of classical-format games (vs blitz)",
    "tc_diversity":         "Number of distinct time controls played",
    "opening_entropy":      "Opening repertoire diversity",
    "top_opening_share":    "Reliance on top-1 opening",
    "color_skill_gap":      "Difference in white vs black win rate",
    "late_night_share":     "Share of games played 0am to 6am",
    "active_days":          "Distinct days played in the month",
    "gap_mean_hours":       "Mean hours between consecutive games",
}


def _parse_time_control(tc):
    if not tc or "+" not in str(tc):
        return "unknown"
    try:
        base, inc = tc.split("+")
        base, inc = int(base), int(inc)
    except (ValueError, AttributeError):
        return "unknown"
    estimated = base + 40 * inc
    if estimated < 180: return "bullet"
    if estimated < 480: return "blitz"
    if estimated < 1500: return "rapid"
    return "classical"


def _engineer_features(games):
    g = games.sort_values("timestamp").copy()
    n = len(g)

    g["tc_category"] = g["time_control"].apply(_parse_time_control)
    g["hour"] = pd.to_datetime(g["timestamp"]).dt.hour

    outcomes = g["outcome"].value_counts(normalize=True)
    tc_counts = g["tc_category"].value_counts(normalize=True)
    opening_counts = g["eco"].value_counts(normalize=True)

    wgames = g[g["color"] == "white"]
    bgames = g[g["color"] == "black"]
    wwr = (wgames["outcome"] == "win").mean() if len(wgames) > 0 else np.nan
    bwr = (bgames["outcome"] == "win").mean() if len(bgames) > 0 else np.nan
    color_gap = abs(wwr - bwr) if not (np.isnan(wwr) or np.isnan(bwr)) else np.nan

    times = pd.to_datetime(g["timestamp"]).dropna().sort_values()
    if len(times) > 2:
        gaps = times.diff().dt.total_seconds().dropna() / 3600
        gap_mean = gaps.mean()
    else:
        gap_mean = np.nan

    return {
        "n_games": n,
        "elo_mean": g["elo"].mean(),
        "win_rate": outcomes.get("win", 0),
        "draw_rate": outcomes.get("draw", 0),
        "rating_diff_volatility": g["rating_diff"].std() if n > 1 else 0,
        "rapid_share": tc_counts.get("rapid", 0),
        "classical_share": tc_counts.get("classical", 0),
        "tc_diversity": g["tc_category"].nunique(),
        "opening_entropy": entropy(opening_counts.values) if len(opening_counts) > 0 else 0,
        "top_opening_share": opening_counts.iloc[0] if len(opening_counts) > 0 else 0,
        "color_skill_gap": color_gap,
        "late_night_share": ((g["hour"] >= 0) & (g["hour"] < 6)).mean(),
        "active_days": pd.to_datetime(g["timestamp"]).dt.date.nunique(),
        "gap_mean_hours": gap_mean,
    }


class ChessCreditScorer:
    """Inference pipeline with per-prediction feature explanations."""

    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS_DIR, model_name="xgb"):
        if model_name != "xgb":
            raise ValueError(
                "Per-prediction explanations are only supported for XGBoost. "
                "Pass model_name='xgb'."
            )
        self.model = joblib.load(os.path.join(artifacts_dir, "xgb_model.pkl"))
        meta = joblib.load(os.path.join(artifacts_dir, "feature_metadata.pkl"))
        self.feature_cols = meta["feature_cols"]
        self.feature_medians = meta["feature_medians"]
        self.min_games = meta["min_games_threshold"]

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
        """Compute per-feature contributions in both log-odds and score points.

        Score points are signed: negative = pushes score down (risky), positive = pushes score up (safe).
        """
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

    def score(self, games_df, top_n_explanations=5):
        """Score a player and explain the result.

        Parameters
        ----------
        games_df : pd.DataFrame
            One row per game played, with columns:
            elo, rating_diff, color, outcome, termination,
            time_control, opening, eco, timestamp.
        top_n_explanations : int
            How many top contributing features to return (sorted by impact).

        Returns
        -------
        dict with keys:
            credit_score, risk_band, risk_probability, features,
            top_factors, all_factors, warning
        """
        features = _engineer_features(games_df)

        warning = None
        if features["n_games"] < self.min_games:
            warning = (f"Player has only {features['n_games']} games. Model was "
                       f"trained on players with at least {self.min_games}. "
                       f"Score is unreliable.")

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
            "baseline_log_odds": round(baseline, 4),
            "warning": warning,
        }

    def format_explanation(self, result):
        """Return a human-readable string of the score and its top factors."""
        lines = []
        lines.append(f"Credit score: {result['credit_score']}  ({result['risk_band']})")
        lines.append(f"Risk probability: {result['risk_probability']:.1%}")
        if result["warning"]:
            lines.append(f"Warning: {result['warning']}")
        lines.append("")
        lines.append("Top factors driving this score:")
        for i, factor in enumerate(result["top_factors"], 1):
            arrow = "↑" if factor["contribution"] > 0 else "↓"
            lines.append(
                f"  {i}. {factor['description']}: {factor['value']} "
                f"{arrow} {factor['direction']} "
                f"(impact: {factor['contribution']:+.3f})"
            )
        return "\n".join(lines)