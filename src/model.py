"""Modeling pipeline for predicting automation tipping points.

Trains and evaluates models to predict which occupations will shift from
augmentation-dominant to automation-dominant AI use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

# Features used for predicting automation velocity / tipping
FEATURE_COLS = [
    "initial_automation_share",
    "initial_augmentation_share",
    "initial_directive",
    "initial_feedback_loop",
    "initial_task_iteration",
    "initial_learning",
    "initial_validation",
    "task_count",
    "hhi",
    "log_salary",
    "jobzone",
    "chanceauto",
    "observed_exposure",
    "log_conversations",
    "ai_autonomy_mean",
    "success_rate",
]


@dataclass
class ModelResult:
    """Container for model evaluation results."""

    name: str
    target: str
    metrics: dict[str, float]
    feature_importance: pd.DataFrame | None = None
    predictions: pd.Series | None = None
    cv_scores: np.ndarray | None = None
    model: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "automation_velocity",
    min_releases: int = 2,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare feature matrix and target, handling missing values.

    Returns (X, y, used_features) where used_features is the subset of
    feature_cols that actually have data.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Filter to occupations with enough temporal observations
    df = df[df["n_releases"] >= min_releases].copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Select available features
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target_col].copy()

    # Impute missing features with median
    for col in available:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y, available


def train_velocity_models(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train regression models to predict automation velocity.

    Returns a list of ModelResult objects for comparison.
    """
    X, y, used_features = _prepare_features(
        features, feature_cols, target_col="automation_velocity"
    )

    if len(X) < 20:
        raise ValueError(f"Too few samples ({len(X)}) for reliable modeling.")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=used_features, index=X.index)

    results = []

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X, y)
    preds_xgb = xgb.predict(X)
    cv_xgb = cross_val_score(xgb, X, y, cv=5, scoring="r2")
    importance_xgb = pd.DataFrame({
        "feature": used_features,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)

    results.append(ModelResult(
        name="XGBoost",
        target="automation_velocity",
        metrics={
            "r2_train": r2_score(y, preds_xgb),
            "r2_cv_mean": cv_xgb.mean(),
            "r2_cv_std": cv_xgb.std(),
            "mae": mean_absolute_error(y, preds_xgb),
            "rmse": np.sqrt(mean_squared_error(y, preds_xgb)),
        },
        feature_importance=importance_xgb,
        predictions=pd.Series(preds_xgb, index=X.index),
        cv_scores=cv_xgb,
        model=xgb,
    ))

    # Gradient Boosting (sklearn)
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbr.fit(X, y)
    preds_gbr = gbr.predict(X)
    cv_gbr = cross_val_score(gbr, X, y, cv=5, scoring="r2")
    importance_gbr = pd.DataFrame({
        "feature": used_features,
        "importance": gbr.feature_importances_,
    }).sort_values("importance", ascending=False)

    results.append(ModelResult(
        name="GradientBoosting",
        target="automation_velocity",
        metrics={
            "r2_train": r2_score(y, preds_gbr),
            "r2_cv_mean": cv_gbr.mean(),
            "r2_cv_std": cv_gbr.std(),
            "mae": mean_absolute_error(y, preds_gbr),
            "rmse": np.sqrt(mean_squared_error(y, preds_gbr)),
        },
        feature_importance=importance_gbr,
        predictions=pd.Series(preds_gbr, index=X.index),
        cv_scores=cv_gbr,
        model=gbr,
    ))

    return results


def train_tipping_models(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train classification models to predict binary tipping (crossed 50% automation).

    Since tipping is rare, also trains on a continuous proxy: predicting whether
    automation_delta > median.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Use automation_delta > 0 as a more balanced binary target
    df = features[features["n_releases"] >= 2].dropna(subset=["automation_delta"]).copy()
    df["shifting_toward_automation"] = (df["automation_delta"] > 0).astype(int)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = X[col].fillna(X[col].median())

    y = df["shifting_toward_automation"]

    if len(X) < 20 or y.nunique() < 2:
        raise ValueError("Insufficient data for classification modeling.")

    results = []

    # Logistic Regression
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available, index=X.index)

    lr = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    lr.fit(X_scaled, y)
    preds_lr = lr.predict(X_scaled)
    proba_lr = lr.predict_proba(X_scaled)[:, 1]
    cv_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")

    coef_df = pd.DataFrame({
        "feature": available,
        "importance": np.abs(lr.coef_[0]),
        "coefficient": lr.coef_[0],
    }).sort_values("importance", ascending=False)

    results.append(ModelResult(
        name="LogisticRegression",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds_lr),
            "auc_train": roc_auc_score(y, proba_lr),
            "auc_cv_mean": cv_lr.mean(),
            "auc_cv_std": cv_lr.std(),
            "report": classification_report(y, preds_lr, output_dict=True),
        },
        feature_importance=coef_df,
        predictions=pd.Series(proba_lr, index=X.index),
        cv_scores=cv_lr,
        model=lr,
    ))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
    )
    rf.fit(X, y)
    preds_rf = rf.predict(X)
    proba_rf = rf.predict_proba(X)[:, 1]
    cv_rf = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")

    rf_importance = pd.DataFrame({
        "feature": available,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    results.append(ModelResult(
        name="RandomForest",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds_rf),
            "auc_train": roc_auc_score(y, proba_rf),
            "auc_cv_mean": cv_rf.mean(),
            "auc_cv_std": cv_rf.std(),
            "report": classification_report(y, preds_rf, output_dict=True),
        },
        feature_importance=rf_importance,
        predictions=pd.Series(proba_rf, index=X.index),
        cv_scores=cv_rf,
        model=rf,
    ))

    # XGBoost Classifier
    xgbc = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgbc.fit(X, y)
    preds_xgbc = xgbc.predict(X)
    proba_xgbc = xgbc.predict_proba(X)[:, 1]
    cv_xgbc = cross_val_score(xgbc, X, y, cv=5, scoring="roc_auc")

    xgbc_importance = pd.DataFrame({
        "feature": available,
        "importance": xgbc.feature_importances_,
    }).sort_values("importance", ascending=False)

    results.append(ModelResult(
        name="XGBoost_Classifier",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds_xgbc),
            "auc_train": roc_auc_score(y, proba_xgbc),
            "auc_cv_mean": cv_xgbc.mean(),
            "auc_cv_std": cv_xgbc.std(),
            "report": classification_report(y, preds_xgbc, output_dict=True),
        },
        feature_importance=xgbc_importance,
        predictions=pd.Series(proba_xgbc, index=X.index),
        cv_scores=cv_xgbc,
        model=xgbc,
    ))

    return results


def rank_tipping_candidates(
    features: pd.DataFrame,
    model_result: ModelResult,
) -> pd.DataFrame:
    """Rank currently augmentation-dominant occupations by predicted automation probability.

    Returns a DataFrame sorted by predicted probability of shifting toward automation,
    filtered to occupations that are currently below 50% automation share.
    """
    available = model_result.feature_importance["feature"].tolist()
    candidates = features[features["currently_augmentation_dominant"] == 1].copy()

    if candidates.empty:
        return pd.DataFrame()

    X = candidates[available].copy()
    for col in available:
        X[col] = X[col].fillna(X[col].median())

    model = model_result.model

    # Handle scaled models (logistic regression)
    if model_result.name == "LogisticRegression":
        scaler = StandardScaler()
        full_X = features[available].copy()
        for col in available:
            full_X[col] = full_X[col].fillna(full_X[col].median())
        scaler.fit(full_X)
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]

    ranking = candidates[["soc_code", "title", "latest_automation_share", "mediansalary"]].copy()
    ranking["predicted_automation_probability"] = proba
    ranking["gap_to_tipping"] = 0.5 - ranking["latest_automation_share"]

    return ranking.sort_values("predicted_automation_probability", ascending=False).reset_index(
        drop=True
    )
