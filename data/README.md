# Data Sources

This directory is populated by running `notebooks/01_data_acquisition.ipynb` or calling
`src.data.download_all()`. Data files are not checked into version control.

## Primary Sources

### Anthropic Economic Index
- **Source**: [HuggingFace: Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex)
- **License**: Released by Anthropic for research use
- **Releases used**:
  - `2025-02-10` — Initial release with task mappings and aggregate collaboration modes
  - `2025-03-27` — Per-task collaboration breakdowns, model version comparisons
  - `2025-09-15` — Unified schema with geographic and API data
  - `2026-01-15` — Economic primitives (autonomy, education, time estimates)
  - `2026-03-24` — Latest release with learning curves

### O\*NET Database
- **Source**: [O\*NET OnLine](https://www.onetonline.org/)
- **Content**: Task statements, occupation descriptions, skill requirements
- **Included in**: Anthropic Economic Index releases (pre-merged)

### BLS Occupational Employment and Wage Statistics
- **Source**: [Bureau of Labor Statistics](https://www.bls.gov/oes/)
- **Content**: Employment counts and wage data by SOC code
- **Included in**: Anthropic Economic Index initial release
