# Automatic Prompt Optimization with RAG

ProTeGi-style automatic prompt optimization for retrieval-augmented financial question answering. The system generates candidate prompts, evaluates them against document retrieval + QA samples using BEM scoring, and iteratively improves them over multiple rounds.

## Datasets

- **FinanceBench** - Financial document QA with PDF-based retrieval
- **FinQA** - Financial QA with structured context (pre_text, table, post_text)
- **FinDoc-RAG** - Financial document QA with markdown reference files

## Evaluator Strategies

Three evaluator strategies are compared across different compute budgets:

- **UCB** (Upper Confidence Bound) - Multi-armed bandit approach to prompt selection
- **PPO** (Proximal Policy Optimization) - Reinforcement learning-based evaluator
- **DPO** (Direct Preference Optimization) - Preference learning-based evaluator

## Project Structure

```
main.py              # Optimization loop with per-round checkpointing
prepare_data.py      # Dataset preparation (parquet generation)
vectorize.py         # Chroma vectorstore builder and retriever factory
generate.py          # Answer generation using optimized prompts
evaluate.py          # Answer grading (GPT or BEM judge)
predictors.py        # QA inference with retrieval
optimizers.py        # ProTeGi-style prompt expansion and gradient generation
evaluators.py        # UCB, PPO, DPO evaluator implementations
scorers.py           # BEM scoring
tasks.py             # Dataset loading and evaluation harness
utils.py             # OpenAI SDK wrapper
paths.py             # Central path resolver (APO_ROOT env var)
run_experiments.py   # Batch experiment runner with completion tracking
setup_gcp.sh         # GCP VM setup script
```

## Running on a GCP VM

### Setup

```bash
# Create a VM
gcloud compute instances create apo-runner --machine-type=e2-standard-2 --zone=us-central1-a --boot-disk-size=50GB

# SSH in
gcloud compute ssh apo-runner --zone=us-central1-a

# Install git and run the setup script
sudo apt-get update && sudo apt-get install -y git
curl -sL https://raw.githubusercontent.com/CyprianBohojlo/APO/gcp-port/setup_gcp.sh | bash
```

### Data Transfer

Upload experiment data to a GCS bucket, then download to the VM:

```bash
# On the VM
gcloud storage cp -r gs://YOUR_BUCKET/vectorstores/FinQa ~/experiments/vectorstores/
gcloud storage cp gs://YOUR_BUCKET/dataset_prepared.parquet ~/experiments/data/FinQa/
gcloud storage cp gs://YOUR_BUCKET/basic.txt ~/experiments/prompts/
```

### Running Experiments

```bash
tmux
source ~/apo-venv/bin/activate
export OPENAI_API_KEY="sk-..."
export APO_ROOT=~/experiments
cd ~/APO
python run_experiments.py
```

Detach from tmux with `Ctrl+B` then `D`. Reconnect later with `tmux attach`.

### Checkpointing and Rate Limits

Experiments save a checkpoint after each completed round. If the process is interrupted or the OpenAI daily rate limit is reached, the experiment stops cleanly and saves progress. Re-running `python run_experiments.py` resumes from the last checkpoint.

Completed experiments (7 rounds in the output file) are automatically skipped.

### Stopping the VM

Stop the VM when experiments are paused to avoid charges:

```bash
gcloud compute instances stop apo-runner --zone=us-central1-a
```

Restart with:

```bash
gcloud compute instances start apo-runner --zone=us-central1-a
```

## Running on Google Colab

A bootstrap notebook is available on the `colab-port` branch. Open it directly:

```
https://colab.research.google.com/github/CyprianBohojlo/APO/blob/colab-port/colab_bootstrap.ipynb
```

Colab Pro is recommended for longer experiments. Set `APO_ROOT` to your Google Drive path and store `OPENAI_API_KEY` in Colab Secrets.

## Configuration

All scripts resolve paths from the `APO_ROOT` environment variable. If unset, it falls back to the repository directory.

Expected layout under `APO_ROOT`:

```
data/
  FinQa/dataset_prepared.parquet
  FinanceBench/dataset_prepared.parquet
vectorstores/
  FinQa/finqa/<doc_id>/chroma.sqlite3
  FinanceBench/financebench/<doc_id>/chroma.sqlite3
results/
prompts/
  basic.txt
```

## Requirements

Install with:

```bash
pip install -r requirements-gcp.txt    # GCP VM (all dependencies)
pip install -r requirements-colab.txt  # Colab (Colab pre-installs most packages)
```
