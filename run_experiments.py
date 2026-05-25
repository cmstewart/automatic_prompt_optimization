#!/usr/bin/env python3
"""Run ProTeGi-style prompt optimization experiments on a GCP VM.

Usage:
    export APO_ROOT=~/experiments
    export OPENAI_API_KEY=sk-...
    cd ~/APO

    # Run all experiments for one dataset with 3 seeds:
    python run_experiments.py --dataset finqa --seeds 1 2 3

    # Run a specific budget level only:
    python run_experiments.py --dataset financebench --seeds 1 2 3 --budgets 2 3

    # Run a specific evaluator only:
    python run_experiments.py --dataset findoc --seeds 1 --evaluators ucb

    # Run everything (all datasets, all seeds):
    python run_experiments.py --seeds 1 2 3

The script checks each experiment's output file for completion (7 ROUND
entries) and skips any that are already done. Safe to re-run after a crash.
"""

import argparse
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
EXPECTED_ROUNDS = 7

DATASETS = {
    "financebench": {
        "task": "financebench",
        "data_subdir": "data/FinanceBench",
    },
    "findoc": {
        "task": "findoc",
        "data_subdir": "data/FinDoc",
    },
    "finqa": {
        "task": "finqa",
        "data_subdir": "data/FinQa",
    },
}

EVALUATORS = ["ucb", "ppo", "dpo"]

BUDGETS = {
    1: {"samples_per_eval": 24, "eval_rounds": 2, "eval_prompts_per_round": 2, "top_k": 1},
    2: {"samples_per_eval": 36, "eval_rounds": 4, "eval_prompts_per_round": 4, "top_k": 3},
    3: {"samples_per_eval": 48, "eval_rounds": 8, "eval_prompts_per_round": 8, "top_k": 6},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_name(dataset, evaluator, budget, seed):
    prefix = {"financebench": "FB", "findoc": "FD", "finqa": "FQ"}[dataset]
    return f"{prefix}_{evaluator}_B{budget}_s{seed}_{dataset}_{evaluator}.txt"


def count_rounds(filepath: pathlib.Path) -> int:
    if not filepath.exists():
        return 0
    return filepath.read_text().count("======== ROUND")


def run_experiment(dataset_cfg, evaluator, budget_cfg, seed, out_path) -> str:
    env = {**os.environ, "APO_ROOT": str(APO_ROOT)}
    data_dir = str(APO_ROOT / dataset_cfg["data_subdir"])
    prompt_file = str(APO_ROOT / "prompts" / "basic.txt")

    cmd = [
        sys.executable, "main.py",
        "--task", dataset_cfg["task"],
        "--data_dir", data_dir,
        "--prompts", prompt_file,
        "--evaluator", evaluator,
        "--samples_per_eval", str(budget_cfg["samples_per_eval"]),
        "--eval_rounds", str(budget_cfg["eval_rounds"]),
        "--eval_prompts_per_round", str(budget_cfg["eval_prompts_per_round"]),
        "--top_k", str(budget_cfg["top_k"]),
        "--max_threads", str(MAX_THREADS),
        "--test_seed", str(seed),
        "--out", out_path,
    ]

    result = subprocess.run(cmd, env=env)
    if result.returncode == 2:
        return "rate_limited"
    return "success" if result.returncode == 0 else "failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt optimization experiments")
    parser.add_argument("--dataset", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Which dataset(s) to run")
    parser.add_argument("--evaluators", nargs="+", choices=EVALUATORS,
                        default=EVALUATORS,
                        help="Which evaluator(s) to run")
    parser.add_argument("--budgets", nargs="+", type=int, choices=[1, 2, 3],
                        default=[1, 2, 3],
                        help="Which budget level(s) to run")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Random seeds for replication")
    args = parser.parse_args()

    prompt_file = APO_ROOT / "prompts" / "basic.txt"
    if not prompt_file.exists():
        print(f"ERROR: Prompt file not found: {prompt_file}")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    # Build experiment list ordered by budget (lightest first).
    experiments = []
    for budget in sorted(args.budgets):
        for evaluator in args.evaluators:
            for dataset in args.dataset:
                for seed in args.seeds:
                    experiments.append({
                        "dataset": dataset,
                        "evaluator": evaluator,
                        "budget": budget,
                        "seed": seed,
                    })

    total = len(experiments)
    print(f"APO_ROOT    : {APO_ROOT}")
    print(f"MAX_THREADS : {MAX_THREADS}")
    print(f"Datasets    : {args.dataset}")
    print(f"Evaluators  : {args.evaluators}")
    print(f"Budgets     : {args.budgets}")
    print(f"Seeds       : {args.seeds}")
    print(f"Total runs  : {total}")
    print()

    # Preflight: check data directories exist.
    for dataset in args.dataset:
        data_dir = APO_ROOT / DATASETS[dataset]["data_subdir"]
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            sys.exit(1)

    completed = 0
    skipped = 0
    failed = 0

    for i, exp in enumerate(experiments):
        dataset_cfg = DATASETS[exp["dataset"]]
        budget_cfg = BUDGETS[exp["budget"]]
        out_name = make_output_name(exp["dataset"], exp["evaluator"], exp["budget"], exp["seed"])
        out_path = str(APO_ROOT / "results" / out_name)
        rounds_done = count_rounds(pathlib.Path(out_path))

        label = f"{exp['dataset']}/{exp['evaluator']}/B{exp['budget']}/s{exp['seed']}"

        if rounds_done >= EXPECTED_ROUNDS:
            print(f"SKIP [{i+1}/{total}] {label}: already complete ({rounds_done} rounds)")
            skipped += 1
            continue

        if rounds_done > 0:
            print(f"NOTE: {label} has {rounds_done}/{EXPECTED_ROUNDS} rounds. Resuming from checkpoint.")

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{total}] STARTING {label}")
        print(f"  samples={budget_cfg['samples_per_eval']} eval_rounds={budget_cfg['eval_rounds']} "
              f"top_k={budget_cfg['top_k']} threads={MAX_THREADS}")
        print(f"{'=' * 60}")

        start = time.time()
        result = run_experiment(dataset_cfg, exp["evaluator"], budget_cfg, exp["seed"], out_path)
        elapsed = time.time() - start

        if result == "rate_limited":
            print(f"\nRATE LIMITED during {label} after {elapsed / 60:.1f} min.")
            print("Daily request limit reached. Re-run this script after the limit resets.")
            break
        elif result == "success":
            completed += 1
            print(f"\nCOMPLETED {label} in {elapsed / 60:.1f} min.")
        else:
            failed += 1
            print(f"\nFAILED {label} after {elapsed / 60:.1f} min.")

    print(f"\n{'=' * 60}")
    print(f"Done. {completed} completed, {skipped} skipped, {failed} failed out of {total}.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
