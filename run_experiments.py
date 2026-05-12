#!/usr/bin/env python3
"""Run the remaining FinQA experiments (E22-E27) sequentially on a GCP VM.

Usage:
    export APO_ROOT=~/experiments
    export OPENAI_API_KEY=sk-...
    cd ~/APO
    python run_experiments.py

The script checks each experiment's output file for completion (7 ROUND
entries) and skips any that are already done. Safe to re-run after a crash.
"""

import os
import pathlib
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APO_ROOT = pathlib.Path(os.environ.get("APO_ROOT", os.path.expanduser("~/experiments")))
MAX_THREADS = int(os.environ.get("MAX_THREADS", "8"))
EXPECTED_ROUNDS = 7  # rounds 0 through 6

EXPERIMENTS = [
    {
        "id": "E22",
        "evaluator": "ucb",
        "samples_per_eval": 36,
        "eval_rounds": 4,
        "eval_prompts_per_round": 4,
        "top_k": 3,
    },
    {
        "id": "E23",
        "evaluator": "ppo",
        "samples_per_eval": 36,
        "eval_rounds": 4,
        "eval_prompts_per_round": 4,
        "top_k": 3,
    },
    {
        "id": "E24",
        "evaluator": "dpo",
        "samples_per_eval": 36,
        "eval_rounds": 4,
        "eval_prompts_per_round": 4,
        "top_k": 3,
    },
    {
        "id": "E25",
        "evaluator": "ucb",
        "samples_per_eval": 48,
        "eval_rounds": 8,
        "eval_prompts_per_round": 8,
        "top_k": 6,
    },
    {
        "id": "E26",
        "evaluator": "ppo",
        "samples_per_eval": 48,
        "eval_rounds": 8,
        "eval_prompts_per_round": 8,
        "top_k": 6,
    },
    {
        "id": "E27",
        "evaluator": "dpo",
        "samples_per_eval": 48,
        "eval_rounds": 8,
        "eval_prompts_per_round": 8,
        "top_k": 6,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_rounds(filepath: pathlib.Path) -> int:
    """Count the number of ROUND entries in an experiment output file."""
    if not filepath.exists():
        return 0
    return filepath.read_text().count("======== ROUND")


def run_experiment(exp: dict, data_dir: str, prompt_file: str, out_path: str) -> bool:
    """Run a single experiment via subprocess. Returns True on success."""
    env = {**os.environ, "APO_ROOT": str(APO_ROOT)}

    cmd = [
        sys.executable, "main.py",
        "--task", "finqa",
        "--data_dir", data_dir,
        "--prompts", prompt_file,
        "--evaluator", exp["evaluator"],
        "--samples_per_eval", str(exp["samples_per_eval"]),
        "--eval_rounds", str(exp["eval_rounds"]),
        "--eval_prompts_per_round", str(exp["eval_prompts_per_round"]),
        "--top_k", str(exp["top_k"]),
        "--max_threads", str(MAX_THREADS),
        "--out", out_path,
    ]

    # Stream stdout/stderr directly to the terminal so progress is visible.
    result = subprocess.run(cmd, env=env)
    if result.returncode == 2:
        return "rate_limited"
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = str(APO_ROOT / "data" / "FinQa")
    prompt_file = str(APO_ROOT / "prompts" / "basic.txt")

    # Preflight checks
    if not pathlib.Path(data_dir).exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    if not pathlib.Path(prompt_file).exists():
        print(f"ERROR: Prompt file not found: {prompt_file}")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    print(f"APO_ROOT    : {APO_ROOT}")
    print(f"MAX_THREADS : {MAX_THREADS}")
    print(f"Experiments : {len(EXPERIMENTS)}")
    print()

    completed = 0
    skipped = 0
    failed = 0

    for exp in EXPERIMENTS:
        out_path = str(APO_ROOT / "results" / f"{exp['id']}_finqa_{exp['evaluator']}.txt")
        rounds_done = count_rounds(pathlib.Path(out_path))

        if rounds_done >= EXPECTED_ROUNDS:
            print(f"SKIP {exp['id']} ({exp['evaluator']}, samples={exp['samples_per_eval']}): "
                  f"already complete ({rounds_done} rounds)")
            skipped += 1
            continue

        if rounds_done > 0:
            print(f"NOTE: {exp['id']} has {rounds_done}/{EXPECTED_ROUNDS} rounds but is incomplete. "
                  f"Will resume from checkpoint if available.")

        print(f"\n{'=' * 60}")
        print(f"STARTING {exp['id']}: finqa / {exp['evaluator']} / "
              f"samples={exp['samples_per_eval']} / eval_rounds={exp['eval_rounds']} / "
              f"top_k={exp['top_k']} / threads={MAX_THREADS}")
        print(f"{'=' * 60}")

        start = time.time()
        result = run_experiment(exp, data_dir, prompt_file, out_path)
        elapsed = time.time() - start

        if result == "rate_limited":
            print(f"\nRATE LIMITED during {exp['id']} after {elapsed / 60:.1f} min.")
            print("Daily request limit reached. Re-run this script after the limit resets.")
            break
        elif result:
            completed += 1
            print(f"\nCOMPLETED {exp['id']} in {elapsed / 60:.1f} min. Output: {out_path}")
        else:
            failed += 1
            print(f"\nFAILED {exp['id']} after {elapsed / 60:.1f} min.")

    print(f"\n{'=' * 60}")
    print(f"Done. {completed} completed, {skipped} skipped, {failed} failed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
