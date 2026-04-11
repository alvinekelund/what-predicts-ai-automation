"""Modeling pipeline for predicting automation tipping points.

Trains and evaluates models to predict which occupations will shift from
augmentation-dominant to automation-dominant AI use.

Key insight from EDA: there is strong mean reversion in automation share.
Occupations that start highly automated tend to decelerate. The model needs
to account for this to avoid trivially predicting "things stay the same."
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


# Features used for prediction — chosen based on data availability and
# domain relevance. Excludes latest_automation_share (target leakage)
# and automation_delta (mechanically linked to target).
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
    scaler: object = field(default=None, repr=False)


def _prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "automation_velocity",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare feature matrix and target, handling missing values."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = df.dropna(subset=[target_col])
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target_col].copy()

    for col in available:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y, available


def train_velocity_models(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train regression models to predict automation velocity."""
    X, y, used_features = _prepare_features(
        features, feature_cols, target_col="automation_velocity"
    )

    if len(X) < 30:
        raise ValueError(f"Too few samples ({len(X)}) for reliable modeling.")

    results = []

    # XGBoost — tends to handle mixed feature types well
    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb.fit(X, y)
    preds = xgb.predict(X)
    cv = cross_val_score(xgb, X, y, cv=5, scoring="r2")

    results.append(ModelResult(
        name="XGBoost",
        target="automation_velocity",
        metrics={
            "r2_train": r2_score(y, preds),
            "r2_cv_mean": cv.mean(),
            "r2_cv_std": cv.std(),
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
        },
        feature_importance=pd.DataFrame({
            "feature": used_features, "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(preds, index=X.index),
        cv_scores=cv,
        model=xgb,
    ))

    # Gradient Boosting (sklearn) — for comparison
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbr.fit(X, y)
    preds = gbr.predict(X)
    cv = cross_val_score(gbr, X, y, cv=5, scoring="r2")

    results.append(ModelResult(
        name="GradientBoosting",
        target="automation_velocity",
        metrics={
            "r2_train": r2_score(y, preds),
            "r2_cv_mean": cv.mean(),
            "r2_cv_std": cv.std(),
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
        },
        feature_importance=pd.DataFrame({
            "feature": used_features, "importance": gbr.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(preds, index=X.index),
        cv_scores=cv,
        model=gbr,
    ))

    return results


def train_tipping_models(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train classification models to predict automation direction.

    Target: whether automation_delta > 0 (shifting toward automation).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = features.dropna(subset=["automation_delta"]).copy()
    df["shifting_toward_automation"] = (df["automation_delta"] > 0).astype(int)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = X[col].fillna(X[col].median())

    y = df["shifting_toward_automation"]

    if len(X) < 30 or y.nunique() < 2:
        raise ValueError("Insufficient data for classification.")

    results = []

    # Logistic Regression — interpretable baseline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    lr.fit(X_scaled, y)
    preds = lr.predict(X_scaled)
    proba = lr.predict_proba(X_scaled)[:, 1]
    cv = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")

    results.append(ModelResult(
        name="LogisticRegression",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds),
            "auc_train": roc_auc_score(y, proba),
            "auc_cv_mean": cv.mean(),
            "auc_cv_std": cv.std(),
        },
        feature_importance=pd.DataFrame({
            "feature": available,
            "importance": np.abs(lr.coef_[0]),
            "coefficient": lr.coef_[0],
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(proba, index=X.index),
        cv_scores=cv,
        model=lr,
        scaler=scaler,
    ))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
    )
    rf.fit(X, y)
    preds = rf.predict(X)
    proba = rf.predict_proba(X)[:, 1]
    cv = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")

    results.append(ModelResult(
        name="RandomForest",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds),
            "auc_train": roc_auc_score(y, proba),
            "auc_cv_mean": cv.mean(),
            "auc_cv_std": cv.std(),
        },
        feature_importance=pd.DataFrame({
            "feature": available, "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(proba, index=X.index),
        cv_scores=cv,
        model=rf,
    ))

    # XGBoost Classifier
    xgbc = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42, use_label_encoder=False, eval_metric="logloss",
    )
    xgbc.fit(X, y)
    preds = xgbc.predict(X)
    proba = xgbc.predict_proba(X)[:, 1]
    cv = cross_val_score(xgbc, X, y, cv=5, scoring="roc_auc")

    results.append(ModelResult(
        name="XGBoost_Classifier",
        target="shifting_toward_automation",
        metrics={
            "accuracy": accuracy_score(y, preds),
            "auc_train": roc_auc_score(y, proba),
            "auc_cv_mean": cv.mean(),
            "auc_cv_std": cv.std(),
        },
        feature_importance=pd.DataFrame({
            "feature": available, "importance": xgbc.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(proba, index=X.index),
        cv_scores=cv,
        model=xgbc,
    ))

    return results


def train_task_autonomy_models(
    tasks: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train regression models to predict task-level AI autonomy.

    With ~3,000 tasks (10x more than occupation-level), this should achieve
    meaningfully positive R² where the occupation model could not.
    """
    if feature_cols is None:
        feature_cols = [
            "human_education_years", "ai_education_years", "skill_compression",
            "human_only_time", "human_with_ai_time", "time_ratio",
            "success_rate", "human_only_ability_pct", "multitasking_pct",
            "use_case_work", "use_case_personal", "use_case_coursework",
            "conversation_count",
        ]

    X, y, used_features = _prepare_features(
        tasks, feature_cols, target_col="ai_autonomy_mean"
    )

    if len(X) < 50:
        raise ValueError(f"Too few samples ({len(X)}) for task-level modeling.")

    results = []

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb.fit(X, y)
    preds = xgb.predict(X)
    cv = cross_val_score(xgb, X, y, cv=5, scoring="r2")

    results.append(ModelResult(
        name="XGBoost",
        target="ai_autonomy_mean",
        metrics={
            "r2_train": r2_score(y, preds),
            "r2_cv_mean": cv.mean(),
            "r2_cv_std": cv.std(),
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
        },
        feature_importance=pd.DataFrame({
            "feature": used_features, "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(preds, index=X.index),
        cv_scores=cv,
        model=xgb,
    ))

    # Gradient Boosting
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbr.fit(X, y)
    preds = gbr.predict(X)
    cv = cross_val_score(gbr, X, y, cv=5, scoring="r2")

    results.append(ModelResult(
        name="GradientBoosting",
        target="ai_autonomy_mean",
        metrics={
            "r2_train": r2_score(y, preds),
            "r2_cv_mean": cv.mean(),
            "r2_cv_std": cv.std(),
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
        },
        feature_importance=pd.DataFrame({
            "feature": used_features, "importance": gbr.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(preds, index=X.index),
        cv_scores=cv,
        model=gbr,
    ))

    return results


def train_time_savings_models(
    tasks: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[ModelResult]:
    """Train regression models to predict time savings ratio per task."""
    if feature_cols is None:
        feature_cols = [
            "ai_autonomy_mean", "human_education_years", "ai_education_years",
            "skill_compression", "success_rate", "human_only_ability_pct",
            "multitasking_pct", "use_case_work", "conversation_count",
        ]

    X, y, used_features = _prepare_features(
        tasks, feature_cols, target_col="time_ratio"
    )

    if len(X) < 50:
        raise ValueError(f"Too few samples ({len(X)}).")

    results = []

    xgb = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb.fit(X, y)
    preds = xgb.predict(X)
    cv = cross_val_score(xgb, X, y, cv=5, scoring="r2")

    results.append(ModelResult(
        name="XGBoost",
        target="time_ratio",
        metrics={
            "r2_train": r2_score(y, preds),
            "r2_cv_mean": cv.mean(),
            "r2_cv_std": cv.std(),
            "mae": mean_absolute_error(y, preds),
        },
        feature_importance=pd.DataFrame({
            "feature": used_features, "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False),
        predictions=pd.Series(preds, index=X.index),
        cv_scores=cv,
        model=xgb,
    ))

    return results


def rank_tipping_candidates(
    features: pd.DataFrame,
    model_result: ModelResult,
) -> pd.DataFrame:
    """Rank augmentation-dominant occupations by predicted automation probability."""
    available = model_result.feature_importance["feature"].tolist()
    candidates = features[features["currently_augmentation_dominant"] == 1].copy()

    if candidates.empty:
        return pd.DataFrame()

    X = candidates[available].copy()
    for col in available:
        X[col] = X[col].fillna(X[col].median())

    if model_result.scaler is not None:
        proba = model_result.model.predict_proba(model_result.scaler.transform(X))[:, 1]
    else:
        proba = model_result.model.predict_proba(X.values)[:, 1]

    ranking = candidates[["soc_code", "title", "latest_automation_share", "mediansalary",
                           "conversation_count", "task_count"]].copy()
    ranking["predicted_automation_probability"] = proba
    ranking["gap_to_tipping"] = 0.5 - ranking["latest_automation_share"]

    return ranking.sort_values("predicted_automation_probability", ascending=False).reset_index(drop=True)
