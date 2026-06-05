# Experiment Results

## Goal

Can a system automatically improve its own prompts for financial question answering? We tested three strategies for deciding which prompt rewrites are actually better, across three financial QA datasets and three compute budgets, with 5 replications each (135 experiments total).

## Methods

Starting from a seed prompt, the system iteratively generates candidate rewrites, evaluates them against RAG-based QA samples using BEM scoring, and keeps the best candidates. Three evaluator strategies control how candidates are scored:

- **UCB** (Upper Confidence Bound): a bandit algorithm that balances exploring new prompts with exploiting known good ones.
- **PPO** (Proximal Policy Optimization): a reinforcement learning approach that learns a scoring policy over time.
- **DPO** (Direct Preference Optimization): a preference learning method that ranks prompts based on pairwise comparisons.

Each was tested at three evaluation budgets (small, medium, large) on FinanceBench (84 PDFs), FinDoc-RAG (46 banking documents), and FinQA (8,281 structured financial reports).

## Results

See [analysis.ipynb](analysis.ipynb) for the full analysis with charts and tables.

**Key numbers (mean final accuracy across 5 seeds):**

| Dataset | Best evaluator at Budget 1 | Best at Budget 2 | Best at Budget 3 |
|---|---|---|---|
| FinanceBench | Tied (all ~30%) | Tied (all ~33%) | Tied (all ~40%) |
| FinDoc-RAG | UCB (49.6%) | Tied (~61%) | UCB (72.0%) |
| FinQA | DPO (35.1%) | UCB (49.8%) | UCB (49.8%) |

## Takeaway

Automatic prompt optimization reliably improves financial QA accuracy (up to +28 percentage points on FinQA, up to +23 on FinanceBench, up to +14 on FinDoc-RAG). No single evaluator strategy wins everywhere: UCB is the safest general choice, particularly on larger datasets, but all three evaluators converge to similar performance on smaller benchmarks. The evaluation budget you allocate matters at least as much as which strategy you pick.

## Contents

- `analysis.ipynb`: analysis notebook with summary tables, bar charts, and progression plots.
- `experiments/`: raw output files for all 135 experiments (3 datasets x 3 evaluators x 3 budgets x 5 seeds).
