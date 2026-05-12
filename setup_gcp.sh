#!/usr/bin/env bash
# setup_gcp.sh - Set up a GCP VM for running APO experiments.
# Run this ON the VM after SSH-ing in. Safe to re-run (idempotent).
set -euo pipefail

APO_ROOT="${APO_ROOT:-$HOME/experiments}"
REPO_DIR="$HOME/APO"
VENV_DIR="$HOME/apo-venv"

echo "=== APO GCP Setup ==="
echo "APO_ROOT : $APO_ROOT"
echo "REPO_DIR : $REPO_DIR"
echo "VENV_DIR : $VENV_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Ensure a suitable Python (3.10+) is available
# ------------------------------------------------------------------
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        major=$("$candidate" -c "import sys; print(sys.version_info[0])")
        minor=$("$candidate" -c "import sys; print(sys.version_info[1])")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$(command -v "$candidate")"
            echo "Found suitable Python: $PYTHON ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No Python 3.10+ found. Installing Python 3.12 via apt..."
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON="$(command -v python3.12)"
    echo "Installed Python 3.12: $PYTHON"
fi

# ------------------------------------------------------------------
# 2. Clone (or update) the repo
# ------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already cloned at $REPO_DIR. Pulling latest..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    git clone -b gcp-port https://github.com/CyprianBohojlo/APO.git "$REPO_DIR"
fi

# ------------------------------------------------------------------
# 3. Create virtualenv and install dependencies
# ------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "Virtualenv already exists at $VENV_DIR."
else
    echo "Creating virtualenv..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Activating virtualenv and installing dependencies..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements-colab.txt"

# ------------------------------------------------------------------
# 4. Create experiment directory structure
# ------------------------------------------------------------------
echo "Creating directory structure under $APO_ROOT..."
for subdir in data data/FinQa references vectorstores vectorstores/FinQa results prompts .cache/hf; do
    mkdir -p "$APO_ROOT/$subdir"
done
echo "Directory structure ready."

# ------------------------------------------------------------------
# 5. Print next steps
# ------------------------------------------------------------------
cat <<'INSTRUCTIONS'

=== Setup complete ===

Next steps:

1. Set your OpenAI API key:
     export OPENAI_API_KEY="sk-..."

2. Set APO_ROOT (add to ~/.bashrc for persistence):
     export APO_ROOT="$HOME/experiments"

3. Copy your data files to the VM. You will need:
     $APO_ROOT/data/FinQa/dataset_prepared.parquet
     $APO_ROOT/prompts/basic.txt
     $APO_ROOT/vectorstores/FinQa/finqa/  (per-document Chroma stores)

   Example using gcloud from your local machine:
     gcloud compute scp --recurse /local/path/to/FinQa vm-name:~/experiments/data/FinQa
     gcloud compute scp /local/path/to/basic.txt vm-name:~/experiments/prompts/basic.txt
     gcloud compute scp --recurse /local/path/to/vectorstores/FinQa vm-name:~/experiments/vectorstores/FinQa

4. Activate the virtualenv before running experiments:
     source ~/apo-venv/bin/activate

5. Run the experiments:
     cd ~/APO
     python run_experiments.py

INSTRUCTIONS
