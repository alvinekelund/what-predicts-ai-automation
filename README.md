# The Jagged Adoption Frontier

**What determines whether people use AI as an autonomous tool or as a collaborator?**

Using Anthropic's [Economic Index](https://www.anthropic.com/economic-index) — which classifies a sample of Claude conversations by O\*NET occupational task and collaboration mode — we analyze 3,259 tasks to understand the structure of AI adoption. Three patterns emerge.

## Findings

### 1. Output type predicts automation; education does not

Tasks that produce **artifacts** (reports, transcripts, data transformations) are automated at roughly twice the rate of tasks that require **human interaction** (advising, teaching, negotiating). Education requirements have no measurable effect — high-education and low-education tasks automate at the same rate.

![Education vs Output Type](figures/04_task_categories.png)

### 2. Deployment context swings automation more than the task itself

The same O\*NET task shows radically different automation rates on the API (programmatic) vs. Claude.ai (interactive). Across 2,429 matched tasks, the API averages +25pp higher automation. Some tasks go from 0% automated on Claude.ai to 100% on API. The model is the same — what changes is how organizations deploy it.

![Platform Gap](figures/05_platform_gap.png)

### 3. Tasks within the same occupation span the full spectrum

Tasks within a single occupation range from fully automated to fully augmented. This within-occupation variation is why occupation-level prediction fails and why "will AI automate X?" is the wrong question.

![Within-Occupation Variation](figures/06_within_occupation.png)

## Data

All data downloads automatically from [Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex) on HuggingFace.

The Economic Index classifies each Claude conversation into one of five collaboration modes:

| Mode | Category | Description |
|------|----------|-------------|
| Directive | Automation | AI executes autonomously |
| Feedback loop | Automation | AI-driven iteration with minimal human input |
| Task iteration | Augmentation | Human iterates on AI drafts |
| Validation | Augmentation | Human checks AI output |
| Learning | Augmentation | Human learns from AI |

Four releases span March 2025 to March 2026. The January and March 2026 releases add per-task continuous measures (AI autonomy, education years, success rates) beyond the original collaboration mode shares. We primarily use the March 2026 release for the task-level analysis and all four releases for the collaboration mode panel. O\*NET task statements provide the occupation mapping; BLS provides wage data.

## Limitations

- **Single platform.** All data is from Claude. Patterns may differ on other AI systems.
- **Observational.** We document associations between task characteristics and collaboration mode usage. We do not claim these are causal.
- **Task classification is coarse.** Categorizing tasks by their leading verb is a rough heuristic. It works well enough to reveal the output-type pattern but misclassifies some tasks.
- **12-month window.** These patterns may shift as AI capabilities and user behavior evolve.

## Reproducing

```bash
git clone https://github.com/alvinekelund/AI-vin-Index.git
cd AI-vin-Index
pip install -r requirements.txt
jupyter nbconvert --execute notebooks/01_data_acquisition.ipynb --to notebook
jupyter nbconvert --execute notebooks/02_skill_compression.ipynb --to notebook
jupyter nbconvert --execute notebooks/03_task_level_analysis.ipynb --to notebook
```

## Structure

```
notebooks/
  01_data_acquisition.ipynb     Data download and structure
  02_skill_compression.ipynb    Output type vs. education as predictors
  03_task_level_analysis.ipynb  Platform gap and within-occupation variation
src/
  data.py                       Data pipeline
  features.py                   Feature engineering
  model.py                      Predictive models
```

## License

MIT. Underlying data provided by Anthropic, O\*NET, and BLS under their respective terms.
