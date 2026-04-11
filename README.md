# The Jagged Adoption Frontier

**An empirical analysis of Anthropic's Economic Index: what predicts whether AI augments or automates an occupation?**

An honest investigation using 12 months of [Anthropic Economic Index](https://www.anthropic.com/economic-index) data across 319 quality-filtered occupations. The answer: less than you'd think.

## Key Findings

### 1. Most AI use remains augmentative

After filtering to occupations with reliable data (>=200 conversations, >=3 tasks, >=3 releases), the vast majority of AI use on Claude is collaborative rather than autonomous. Only a small fraction of occupations are automation-dominant.

### 2. Mean reversion dominates trajectories

The single strongest signal in the data: occupations that start with high automation shares tend to *decelerate*, while those starting low tend to *accelerate*. This convergence toward the mean means naive "more automation begets more automation" predictions are wrong.

### 3. The frontier is genuinely jagged

Wage — the most common proxy for task complexity — explains almost nothing about which occupations automate. The adoption frontier is jagged and unpredictable from simple economic proxies, echoing Dell'Acqua et al.'s (2023) finding.

### 4. The channel matters more than the occupation

API usage (programmatic pipelines) is dramatically more automative than interactive Claude.ai usage. The same occupation looks very different depending on whether humans are using Claude conversationally or embedding it in automated workflows.

### 5. Individual trajectories are hard to predict

Our best velocity model achieves **R^2 ~ 0** after proper cross-validation — individual occupation automation trajectories are essentially unpredictable from current features. Direction classification achieves a modest **AUC ~ 0.71**, enough to identify associated features but not enough for reliable prediction. Only **3 occupations** genuinely tipped from augmentation to automation during the observation window.

## Why the honest null result matters

A project that claims "we can predict automation with R^2 = 0.33" by overfitting to noisy data helps no one. The honest finding — that occupation-level automation trajectories are hard to predict — is itself important:

- It suggests the augmentation-to-automation transition depends on idiosyncratic factors (individual behavior, organizational decisions, specific tool availability) more than on occupation-level characteristics
- It challenges deterministic narratives about which jobs AI will automate
- It highlights that **how** AI is deployed (API vs. interactive) matters more than **where** it's deployed

## Data

All data is publicly available and downloaded automatically when running the notebooks.

| Source | Description |
|--------|-------------|
| [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) | 4 releases (Mar 2025 -- Mar 2026) with per-task collaboration modes, AI autonomy, task success rates |
| [O\*NET](https://www.onetonline.org/) | 19,530 task statements mapped to 974 occupations via SOC codes |
| [BLS OEWS](https://www.bls.gov/oes/) | Wage and employment data by occupation |

The Anthropic Economic Index classifies every Claude conversation into one of five collaboration modes:

| Mode | Category | Description |
|------|----------|-------------|
| **Directive** | Automation | AI executes autonomously |
| **Feedback loop** | Automation | Iterative AI-driven refinement |
| **Validation** | Augmentation | Human validates AI output |
| **Task iteration** | Augmentation | Human iterates on AI drafts |
| **Learning** | Augmentation | Human learns from AI |

## Methodology

**Quality filtering.** Raw data covers 633 occupations, but only ~319 have sufficient data for reliable analysis. We require >=200 conversations, >=3 matched O\*NET tasks, and data in >=3 of 4 releases. This eliminates noise from occupations like "Dancers" (17 conversations, 1 task).

**Panel construction.** Task-level collaboration mode shares are mapped to occupations via O\*NET SOC codes across four releases. For each occupation x release, we compute the share of automative vs. augmentative AI use.

**Feature engineering.** Per occupation: automation velocity (polyfit slope), initial collaboration mode distribution, task characteristics (count, concentration, success rate), and economic characteristics (wage, job zone, AI exposure).

**Modeling.** XGBoost, Gradient Boosting, Logistic Regression, and Random Forest models, validated via 5-fold CV. Features from early releases, targets from the full time series.

## Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_acquisition.ipynb   # Download, quality assessment, API vs Claude.ai
│   ├── 02_exploration.ipynb        # Mean reversion, jagged frontier, occupation groups
│   ├── 03_modeling.ipynb           # Model training with honest evaluation
│   └── 04_analysis.ipynb          # Tipping candidates, synthesis, final rankings
├── src/
│   ├── data.py                     # Data download, loading, panel construction
│   ├── features.py                 # Feature engineering with quality filtering
│   └── model.py                    # Model training, evaluation, ranking
├── data/
│   └── README.md                   # Data source documentation
└── figures/                        # Generated visualizations (14 figures)
```

## Reproducing

```bash
git clone https://github.com/alvinekelund/AI-vin-Index.git
cd AI-vin-Index
pip install -r requirements.txt
```

Run the notebooks in order — data downloads automatically on first execution:

```bash
jupyter nbconvert --execute notebooks/01_data_acquisition.ipynb --to notebook
jupyter nbconvert --execute notebooks/02_exploration.ipynb --to notebook
jupyter nbconvert --execute notebooks/03_modeling.ipynb --to notebook
jupyter nbconvert --execute notebooks/04_analysis.ipynb --to notebook
```

## References

- Anthropic. (2025--2026). *The Anthropic Economic Index.* [anthropic.com/economic-index](https://www.anthropic.com/economic-index)
- Dell'Acqua, F., et al. (2023). *Navigating the Jagged Technological Frontier.* Harvard Business School Working Paper 24-013.
- Acemoglu, D. (2024). *The Simple Macroeconomics of AI.* NBER Working Paper 32487.
- Brynjolfsson, E., Li, D., & Raymond, L. R. (2023). *Generative AI at Work.* NBER Working Paper 31161.
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). *GPTs are GPTs.* arXiv:2303.10130.

## License

MIT License. The underlying data is provided by Anthropic, O\*NET, and BLS under their respective terms.
