"""Feature engineering for the automation tipping-point model.

Transforms the raw occupation panel into a feature matrix suitable for
predicting which occupations will shift from augmentation to automation.

Key design decisions:
- We filter aggressively for data quality: minimum 200 conversations,
  3+ matched tasks, and 3+ releases. This cuts the dataset from 633 to
  ~320 occupations, but the remaining occupations have reliable signals.
- We track mean reversion explicitly: occupations that start highly automated
  tend to decelerate. This is the single strongest pattern in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    RELEASE_ORDER,
    build_occupation_panel,
    load_unified_release,
    extract_autonomy_from_unified,
    extract_task_success_from_unified,
    extract_task_counts_from_unified,
    _build_task_to_soc,
)

# Quality thresholds — occupations below these are too noisy to model
MIN_CONVERSATIONS = 200
MIN_TASKS = 3
MIN_RELEASES = 3


def compute_automation_velocity(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the rate of change in automation share across releases.

    For each occupation, fits a simple slope (automation_share ~ release_index).
    Occupations appearing in fewer than 2 releases get NaN velocity.
    """
    release_idx = {r: i for i, r in enumerate(RELEASE_ORDER)}
    panel = panel.copy()
    panel["release_idx"] = panel["release"].map(release_idx)

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        x = group["release_idx"].values.astype(float)
        y = group["automation_share"].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        x, y = x[mask], y[mask]
        return np.polyfit(x, y, 1)[0]

    velocity = (
        panel.groupby("soc_code")[["release_idx", "automation_share"]]
        .apply(_slope)
        .reset_index()
        .rename(columns={0: "automation_velocity"})
    )
    return velocity


def compute_task_concentration(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute task concentration (inverse task count) for each occupation."""
    latest = panel[panel["release"] == panel["release"].max()].copy()
    latest["hhi"] = 1.0 / latest["task_count"].clip(lower=1)
    return latest[["soc_code", "task_count", "hhi"]].drop_duplicates(subset=["soc_code"])


def compute_initial_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the earliest available observation per occupation."""
    release_idx = {r: i for i, r in enumerate(RELEASE_ORDER)}
    panel = panel.copy()
    panel["release_idx"] = panel["release"].map(release_idx)

    earliest = panel.sort_values("release_idx").groupby("soc_code").first().reset_index()
    return earliest.rename(
        columns={
            "automation_share": "initial_automation_share",
            "augmentation_share": "initial_augmentation_share",
            "directive": "initial_directive",
            "feedback_loop": "initial_feedback_loop",
            "task_iteration": "initial_task_iteration",
            "learning": "initial_learning",
            "validation": "initial_validation",
        }
    )


def compute_latest_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the latest observation per occupation."""
    release_idx = {r: i for i, r in enumerate(RELEASE_ORDER)}
    panel = panel.copy()
    panel["release_idx"] = panel["release"].map(release_idx)

    latest = panel.sort_values("release_idx").groupby("soc_code").last().reset_index()
    return latest.rename(
        columns={
            "automation_share": "latest_automation_share",
            "augmentation_share": "latest_augmentation_share",
        }
    )


def enrich_with_economic_primitives(features: pd.DataFrame) -> pd.DataFrame:
    """Add AI autonomy and task success features from the latest unified release."""
    task_to_soc = _build_task_to_soc()

    for release in ["2026_03", "2026_01"]:
        try:
            raw = load_unified_release(release)
            break
        except Exception:
            continue
    else:
        return features

    # AI autonomy by occupation
    auto_df = extract_autonomy_from_unified(raw)
    if not auto_df.empty and "value" in auto_df.columns:
        auto_df = auto_df.rename(columns={"value": "ai_autonomy_mean"})
        auto_df["task_name"] = auto_df["task_name"].str.lower().str.strip()
        auto_merged = auto_df.merge(task_to_soc, on="task_name", how="inner")
        occ_auto = auto_merged.groupby("soc_code")["ai_autonomy_mean"].mean().reset_index()
        features = features.merge(occ_auto, on="soc_code", how="left")

    # Task success by occupation
    success_df = extract_task_success_from_unified(raw)
    if not success_df.empty and "success_rate" in success_df.columns:
        success_df["task_name"] = success_df["task_name"].str.lower().str.strip()
        success_merged = success_df.merge(task_to_soc, on="task_name", how="inner")
        occ_success = success_merged.groupby("soc_code")["success_rate"].mean().reset_index()
        features = features.merge(occ_success, on="soc_code", how="left")

    # Conversation volume by occupation
    counts_df = extract_task_counts_from_unified(raw)
    if not counts_df.empty:
        counts_df["task_name"] = counts_df["task_name"].str.lower().str.strip()
        counts_merged = counts_df.merge(task_to_soc, on="task_name", how="inner")
        occ_volume = counts_merged.groupby("soc_code")["conversation_count"].sum().reset_index()
        features = features.merge(occ_volume, on="soc_code", how="left")

    return features


def build_feature_matrix(apply_quality_filter: bool = True) -> pd.DataFrame:
    """Build the complete feature matrix for modeling.

    Args:
        apply_quality_filter: If True (default), removes occupations with
            insufficient data (< 200 conversations, < 3 tasks, < 3 releases).
            Set to False to get the unfiltered matrix for data quality analysis.

    Each row is an occupation with:
    - Target: automation_velocity (change in automation share over time)
    - Features from initial state, economic data, task characteristics
    """
    panel = build_occupation_panel()

    # Compute core features
    velocity = compute_automation_velocity(panel)
    concentration = compute_task_concentration(panel)
    initial = compute_initial_state(panel)
    latest = compute_latest_state(panel)

    # Build the feature matrix
    features = velocity.merge(concentration, on="soc_code", how="left")

    # Add initial-state features
    initial_cols = [
        "soc_code", "title", "initial_automation_share", "initial_augmentation_share",
        "initial_directive", "initial_feedback_loop", "initial_task_iteration",
        "initial_learning", "initial_validation",
        "mediansalary", "jobzone", "chanceauto", "jobforecast", "observed_exposure",
    ]
    initial_cols = [c for c in initial_cols if c in initial.columns]
    features = features.merge(initial[initial_cols], on="soc_code", how="left")

    # Add latest state
    latest_cols = ["soc_code", "latest_automation_share", "latest_augmentation_share"]
    latest_cols = [c for c in latest_cols if c in latest.columns]
    features = features.merge(latest[latest_cols], on="soc_code", how="left")

    # Add economic primitives
    features = enrich_with_economic_primitives(features)

    # Derived features
    features["automation_delta"] = (
        features["latest_automation_share"] - features["initial_automation_share"]
    )
    features["tipped"] = (
        (features["initial_automation_share"] < 0.5)
        & (features["latest_automation_share"] >= 0.5)
    ).astype(int)
    features["currently_augmentation_dominant"] = (
        features["latest_automation_share"] < 0.5
    ).astype(int)

    if "mediansalary" in features.columns:
        features["log_salary"] = np.log1p(features["mediansalary"].fillna(0))
    if "conversation_count" in features.columns:
        features["log_conversations"] = np.log1p(features["conversation_count"].fillna(0))

    # Release count per occupation
    release_counts = panel.groupby("soc_code")["release"].nunique().reset_index()
    release_counts.columns = ["soc_code", "n_releases"]
    features = features.merge(release_counts, on="soc_code", how="left")

    # SOC major group for grouping analysis
    features["major_group"] = features["soc_code"].str[:2]

    if apply_quality_filter:
        n_before = len(features)
        features = features[
            (features["conversation_count"] >= MIN_CONVERSATIONS)
            & (features["task_count"] >= MIN_TASKS)
            & (features["n_releases"] >= MIN_RELEASES)
        ].copy()
        n_after = len(features)
        if n_before > n_after:
            import logging
            logging.getLogger(__name__).info(
                "Quality filter: %d → %d occupations (removed %d with insufficient data)",
                n_before, n_after, n_before - n_after,
            )

    return features.reset_index(drop=True)
