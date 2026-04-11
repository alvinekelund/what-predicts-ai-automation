"""Data acquisition and loading for the Jagged Adoption Frontier.

Downloads Anthropic Economic Index releases from HuggingFace and merges them
with O*NET and BLS data into a unified panel dataset for analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

REPO_ID = "Anthropic/EconomicIndex"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Release manifest — maps logical names to HuggingFace paths
# ---------------------------------------------------------------------------

RELEASE_FILES: dict[str, str] = {
    # Initial release (2025-02-10)
    "task_mappings_v1": "release_2025_02_10/onet_task_mappings.csv",
    "collab_aggregate_v1": "release_2025_02_10/automation_vs_augmentation.csv",
    "onet_tasks": "release_2025_02_10/onet_task_statements.csv",
    "soc_structure": "release_2025_02_10/SOC_Structure.csv",
    "bls_employment": "release_2025_02_10/bls_employment_may_2023.csv",
    "wage_data": "release_2025_02_10/wage_data.csv",
    # Cluster-level release (2025-03-27)
    "task_pct_v1": "release_2025_03_27/task_pct_v1.csv",
    "task_pct_v2": "release_2025_03_27/task_pct_v2.csv",
    "collab_by_task": "release_2025_03_27/automation_vs_augmentation_by_task.csv",
    "collab_aggregate_v1_r2": "release_2025_03_27/automation_vs_augmentation_v1.csv",
    "collab_aggregate_v2": "release_2025_03_27/automation_vs_augmentation_v2.csv",
    # Unified-schema releases
    "aei_claude_ai_2025_09": "release_2025_09_15/data/intermediate/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv",
    "aei_api_2025_09": "release_2025_09_15/data/intermediate/aei_raw_1p_api_2025-08-04_to_2025-08-11.csv",
    "aei_claude_ai_2026_01": "release_2026_01_15/data/intermediate/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv",
    "aei_api_2026_01": "release_2026_01_15/data/intermediate/aei_raw_1p_api_2025-11-13_to_2025-11-20.csv",
    "aei_claude_ai_2026_03": "release_2026_03_24/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv",
    "aei_api_2026_03": "release_2026_03_24/data/aei_raw_1p_api_2026-02-05_to_2026-02-12.csv",
    # Labor market impacts
    "job_exposure": "labor_market_impacts/job_exposure.csv",
    "task_penetration": "labor_market_impacts/task_penetration.csv",
}

# Temporal ordering of releases for panel construction
RELEASE_ORDER = ["2025_03", "2025_09", "2026_01", "2026_03"]

# Collaboration modes → automation vs augmentation
AUTOMATION_MODES = {"directive", "feedback_loop"}
AUGMENTATION_MODES = {"validation", "task_iteration", "learning"}
ALL_COLLAB_MODES = AUTOMATION_MODES | AUGMENTATION_MODES


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(name: str, force: bool = False) -> Path:
    """Download a single file from the HuggingFace dataset repo."""
    hf_path = RELEASE_FILES[name]
    local_path = DATA_DIR / hf_path.replace("/", "_")

    if local_path.exists() and not force:
        return local_path

    logger.info("Downloading %s → %s", hf_path, local_path.name)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        repo_type="dataset",
        local_dir=DATA_DIR / "_hf_cache",
    )
    local_path.write_bytes(Path(downloaded).read_bytes())
    return local_path


def download_all(force: bool = False) -> dict[str, Path]:
    """Download all data files. Returns a dict of name → local path."""
    paths = {}
    for name in RELEASE_FILES:
        try:
            paths[name] = download_file(name, force=force)
        except Exception as e:
            logger.warning("Failed to download %s: %s", name, e)
    return paths


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """Load a named dataset as a DataFrame, downloading if needed."""
    path = download_file(name)
    return pd.read_csv(path, **kwargs)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, underscores, no dashes."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("*", "", regex=False)
    )
    return df


def load_onet_tasks() -> pd.DataFrame:
    """Load O*NET task statements with clean column names."""
    df = _clean_columns(load_csv("onet_tasks"))
    if "onet_soc_code" in df.columns:
        pass  # already clean
    elif "onetsoc_code" in df.columns:
        df = df.rename(columns={"onetsoc_code": "onet_soc_code"})
    return df


def load_soc_structure() -> pd.DataFrame:
    """Load SOC hierarchy."""
    return _clean_columns(load_csv("soc_structure"))


def load_collaboration_by_task() -> pd.DataFrame:
    """Load per-task collaboration mode breakdown (2025-03 release)."""
    return _clean_columns(load_csv("collab_by_task"))


def load_unified_release(release: str, platform: str = "claude_ai") -> pd.DataFrame:
    """Load a unified-schema release (2025-09 onward)."""
    name = f"aei_{platform}_{release}"
    df = _clean_columns(load_csv(name))
    df["release"] = release
    return df


def load_wage_data() -> pd.DataFrame:
    """Load O*NET wage data."""
    return _clean_columns(load_csv("wage_data"))


def load_bls_employment() -> pd.DataFrame:
    """Load BLS employment statistics by SOC major group."""
    return _clean_columns(load_csv("bls_employment"))


def load_job_exposure() -> pd.DataFrame:
    """Load occupation-level observed exposure scores."""
    return _clean_columns(load_csv("job_exposure"))


def load_task_penetration() -> pd.DataFrame:
    """Load task-level penetration data."""
    return _clean_columns(load_csv("task_penetration"))


# ---------------------------------------------------------------------------
# Extraction from unified-schema releases
# ---------------------------------------------------------------------------


def _extract_facet(
    df: pd.DataFrame,
    facet: str,
    variable: str,
    value_col: str = "value",
) -> pd.DataFrame:
    """Generic extractor for intersection facets in unified schema.

    For facets like "onet_task::collaboration", splits cluster_name on "::"
    into (task_name, category) and pivots into wide format.
    """
    mask = (df["facet"] == facet) & (df["variable"] == variable) & (df["geography"] == "global")
    subset = df.loc[mask, ["cluster_name", value_col]].copy()

    if subset.empty:
        return pd.DataFrame()

    parts = subset["cluster_name"].str.split("::", n=1, expand=True)
    subset["task_name"] = parts[0]
    subset["category"] = parts[1]

    pivot = subset.pivot_table(
        index="task_name", columns="category", values=value_col, aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    return pivot


def extract_collaboration_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-task collaboration mode shares from a unified release.

    Returns one row per task with columns for each collaboration mode (0–1 scale).
    """
    pivot = _extract_facet(df, "onet_task::collaboration", "onet_task_collaboration_pct")
    if pivot.empty:
        return pivot

    # Normalize mode names to match 2025-03 column style
    pivot = pivot.rename(columns={"task iteration": "task_iteration", "feedback loop": "feedback_loop"})
    # Convert from percentage (0–100) to proportion (0–1)
    num_cols = [c for c in pivot.columns if c != "task_name"]
    pivot[num_cols] = pivot[num_cols] / 100.0
    return pivot


def extract_autonomy_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-task AI autonomy scores (mean) from a unified release."""
    pivot = _extract_facet(df, "onet_task::ai_autonomy", "onet_task_ai_autonomy_mean")
    if not pivot.empty and "value" in pivot.columns:
        pivot = pivot.rename(columns={"value": "ai_autonomy_mean"})
    return pivot


def extract_task_success_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-task success rates from a unified release."""
    pivot = _extract_facet(df, "onet_task::task_success", "onet_task_task_success_pct")
    if not pivot.empty and "yes" in pivot.columns:
        pivot["success_rate"] = pivot["yes"] / 100.0
    return pivot


def extract_task_counts_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-task conversation counts from a unified release."""
    mask = (
        (df["facet"] == "onet_task")
        & (df["variable"] == "onet_task_count")
        & (df["geography"] == "global")
    )
    subset = df.loc[mask, ["cluster_name", "value"]].copy()
    subset = subset.rename(columns={"cluster_name": "task_name", "value": "conversation_count"})
    return subset.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel dataset construction
# ---------------------------------------------------------------------------


def build_task_collaboration_panel() -> pd.DataFrame:
    """Build a panel of task-level collaboration modes across all available releases.

    Returns a DataFrame with columns:
        task_name, release, directive, feedback_loop, validation,
        task_iteration, learning, automation_share, augmentation_share
    """
    frames = []

    # 2025-03 release — direct per-task file
    try:
        early = load_collaboration_by_task()
        early["release"] = "2025_03"
        frames.append(early)
    except Exception as e:
        logger.warning("Could not load 2025-03 collaboration data: %s", e)

    # Unified releases
    for release in ["2025_09", "2026_01", "2026_03"]:
        try:
            raw = load_unified_release(release, platform="claude_ai")
            collab = extract_collaboration_from_unified(raw)
            if not collab.empty:
                collab["release"] = release
                frames.append(collab)
        except Exception as e:
            logger.warning("Could not load %s collaboration data: %s", release, e)

    if not frames:
        raise RuntimeError("No collaboration data could be loaded.")

    panel = pd.concat(frames, ignore_index=True)

    # Ensure all mode columns exist
    for col in ALL_COLLAB_MODES:
        if col not in panel.columns:
            panel[col] = 0.0

    panel["automation_share"] = panel[list(AUTOMATION_MODES)].fillna(0).sum(axis=1)
    panel["augmentation_share"] = panel[list(AUGMENTATION_MODES)].fillna(0).sum(axis=1)

    return panel


def _build_task_to_soc() -> pd.DataFrame:
    """Create a mapping from lowercase task description to SOC code and title."""
    onet = load_onet_tasks()
    mapping = onet[["onet_soc_code", "task", "title"]].copy()
    mapping["task_name"] = mapping["task"].str.lower().str.strip()
    mapping["soc_code"] = mapping["onet_soc_code"].str[:7]
    return mapping[["task_name", "soc_code", "onet_soc_code", "title"]].drop_duplicates()


def build_occupation_panel() -> pd.DataFrame:
    """Build an occupation-level panel by aggregating task data to SOC codes.

    Joins task collaboration data with O*NET task statements to map tasks
    to occupations, then merges with wage and employment data.
    """
    task_panel = build_task_collaboration_panel()
    task_to_soc = _build_task_to_soc()

    # Normalize task names for matching
    task_panel["task_name"] = task_panel["task_name"].str.lower().str.strip()

    # One task can appear in multiple occupations — this is intentional
    merged = task_panel.merge(task_to_soc, on="task_name", how="inner")

    # Aggregate to occupation × release level
    mode_aggs = {m: (m, "mean") for m in ALL_COLLAB_MODES}
    occ = (
        merged.groupby(["soc_code", "title", "release"])
        .agg(
            automation_share=("automation_share", "mean"),
            augmentation_share=("augmentation_share", "mean"),
            task_count=("task_name", "nunique"),
            **mode_aggs,
        )
        .reset_index()
    )

    # Merge wage data (SOC code at detailed level)
    try:
        wages = load_wage_data()
        wages["soc_code"] = wages["soccode"].astype(str).str[:7]
        wage_cols = ["soc_code", "mediansalary", "jobzone", "chanceauto", "jobforecast"]
        wage_cols = [c for c in wage_cols if c in wages.columns or c == "soc_code"]
        wages_dedup = wages[wage_cols].drop_duplicates(subset=["soc_code"])
        occ = occ.merge(wages_dedup, on="soc_code", how="left")
    except Exception as e:
        logger.warning("Could not merge wage data: %s", e)

    # Merge job exposure data
    try:
        exposure = load_job_exposure()
        exposure["soc_code"] = exposure["occ_code"].astype(str).str[:7]
        occ = occ.merge(
            exposure[["soc_code", "observed_exposure"]].drop_duplicates(subset=["soc_code"]),
            on="soc_code",
            how="left",
        )
    except Exception as e:
        logger.warning("Could not merge job exposure data: %s", e)

    return occ
