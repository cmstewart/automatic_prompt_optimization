#!/usr/bin/env bash
# setup_vm.sh - Complete one-shot setup for a fresh GCP VM.
# Usage: curl -sL https://raw.githubusercontent.com/CyprianBohojlo/APO/gcp-port/setup_vm.sh | bash
set -euo pipefail

APO_ROOT="$HOME/experiments"
REPO_DIR="$HOME/APO"
VENV_DIR="$HOME/apo-venv"
GCS_BUCKET="gs://apo-experiment-data"

echo "=== APO VM Setup ==="

# 1. System packages
echo "Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y git python3.11-venv tmux

# 2. File descriptor limit
echo "Setting file descriptor limit..."
sudo sh -c 'echo "* soft nofile 65536" >> /etc/security/limits.conf'
sudo sh -c 'echo "* hard nofile 65536" >> /etc/security/limits.conf'

# 3. Clone repo
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists. Pulling latest..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    git clone -b gcp-port https://github.com/CyprianBohojlo/APO.git "$REPO_DIR"
fi

# 4. Virtualenv and dependencies
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv "$VENV_DIR"
fi
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements-gcp.txt" pyarrow "transformers==4.44.2"

# 5. Directory structure
echo "Creating experiment directories..."
for subdir in data/FinQa references vectorstores results prompts .cache/hf; do
    mkdir -p "$APO_ROOT/$subdir"
done

# 6. Corrected seed prompt
echo "Writing corrected seed prompt..."
cat > "$APO_ROOT/prompts/basic.txt" << 'PROMPT'
# Task
You are a financial QA assistant.
Answer the question using only the information in the context.


# Output format
Return your answer as free text. If it's a number, just output the number (don't wrap in words). Otherwise, output a concise sentence.

# Prediction
Context: {context}

Question: {question}

Answer:
PROMPT

# 7. Download data from GCS
echo "Downloading dataset..."
gcloud storage cp "$GCS_BUCKET/dataset_prepared.parquet" "$APO_ROOT/data/FinQa/"

echo "Downloading vectorstores (this will take a while)..."
gcloud storage cp -r "$GCS_BUCKET/vectorstores/FinQa" "$APO_ROOT/vectorstores/"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run experiments:"
echo "  tmux"
echo "  ulimit -n 65536"
echo "  source ~/apo-venv/bin/activate"
echo "  export OPENAI_API_KEY=\"sk-...\""
echo "  export APO_ROOT=~/experiments"
echo "  cd ~/APO"
echo "  python run_experiments.py E20 E22 E25    # (adjust IDs per VM)"
