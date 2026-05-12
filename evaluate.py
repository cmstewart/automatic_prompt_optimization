"""
evaluate.py  – FinanceBench ProTeGi fork
----------------------------------------
Grades model answers into:
  • "CORRECT"
  • "INCORRECT"
  • "REFUSED"

Choose the judge with --metric {gpt|bem}
"""

import argparse, json, os, sys, pathlib
from collections import Counter
from tqdm import tqdm

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# LOCAL IMPORTS
# ----------------------------------------------------------------------
from scorers import BEMScorer     # reuse shared implementation
from paths import ROOT

# ----------------------------------------------------------------------
# CONFIG PATHS
# ----------------------------------------------------------------------
load_dotenv()

ANSWERS  = ROOT / "results" / "answers.jsonl"
GRADES   = ROOT / "results" / "grades_ce.jsonl"
PLOT_PATH = ROOT / "results" / "verdict_distribution.png"

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Grade FinanceBench answers")
    p.add_argument("--metric", choices=["gpt", "bem", "checkembed"], default="gpt",
                   help="Which judge to use.")
    p.add_argument("--threshold", type=float, default=0.56,
                   help="BEM equivalence probability threshold.")
    # NEW: allow selecting custom input/output files (keeps old defaults)
    p.add_argument("--answers", type=pathlib.Path, default=ANSWERS,
                   help="Path to answers JSONL to grade (default: results/answers.jsonl)")
    p.add_argument("--out", type=pathlib.Path, default=GRADES,
                   help="Path to write graded JSONL (default: results/grades_ce.jsonl)")
    return p.parse_args()

# ----------------------------------------------------------------------
# GPT-BASED JUDGE
# ----------------------------------------------------------------------
judge = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=64)
SYS = (
    "You are a strict grader.\n"
    "Classify the candidate answer as one of:\n"
    "- CORRECT (factual and matches the gold answer)\n"
    "- INCORRECT (plausible but wrong)\n"
    "- REFUSED (the model refused to answer or hallucinated)\n"
    "Respond ONLY with one of those labels."
)

def verdict_gpt(question: str, gold: str, cand: str) -> str:
    prompt = f"{SYS}\nQ: {question}\nGold: {gold}\nCandidate: {cand}\nVerdict:"
    resp = judge.predict(prompt).strip().upper()
    if resp.startswith("CORRECT"):
        return "CORRECT"
    if resp.startswith("INCORRECT"):
        return "INCORRECT"
    if resp.startswith("REFUSED"):
        return "REFUSED"
    return "INCORRECT"

# ----------------------------------------------------------------------
# BEM-BASED JUDGE
# ----------------------------------------------------------------------
bem_judge = BEMScorer(None)

def verdict_bem(question: str, gold: str, cand: str, tau: float) -> str:
    # NOTE: tau is exposed for compatibility; BEMScorer may handle its own thresholding internally.
    if bem_judge.pair_equivalent(cand, gold, question):
        return "CORRECT"
    if cand.strip().lower() in {"", "i don't know", "idk"}:
        return "REFUSED"
    return "INCORRECT"

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = get_args()

    # Validate input answers file
    if not args.answers.exists():
        sys.exit(f"Answers file not found: {args.answers}\n"
                 f"Provide --answers path or run your generator first.")

    # ---------- progress bar total -----------------------------------
    total_qas = sum(1 for _ in args.answers.open("r", encoding="utf-8"))

    graded = []
    with args.answers.open("r", encoding="utf-8") as fin, \
         tqdm(total=total_qas, desc="Grading", unit="qa") as pbar:
        for line in fin:
            rec = json.loads(line)
            if args.metric == "gpt":
                v = verdict_gpt(rec["question"], rec["gold_answer"], rec["model_answer"])
            elif args.metric == "bem" or args.metric == "checkembed":
                # treat "checkembed" same as BEM-backed judgment for now
                v = verdict_bem(rec["question"], rec["gold_answer"],
                                rec["model_answer"], args.threshold)
            else:
                # fallback to incorrect if unknown metric (shouldn't happen due to choices)
                v = "INCORRECT"
            graded.append({**rec, "verdict": v})
            pbar.update(1)

    # ---------- write grades -----------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for g in graded:
            fout.write(json.dumps(g, ensure_ascii=False) + "\n")

    counts = Counter(g["verdict"] for g in graded)

    # ---------- bar plot ---------------------------------------------
    labels = ["CORRECT", "INCORRECT", "REFUSED"]
    values = [counts.get(l, 0) for l in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(f"Verdict distribution ({args.metric.upper()})")
    plt.ylabel("Count")
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Plot saved to {PLOT_PATH}")

    total = len(graded)
    acc = counts.get("CORRECT", 0) / total if total else 0.0
    print(f"Accuracy ({args.metric}): {acc:.2%}   Counts: {dict(counts)}")
    print(f"Grades saved to {args.out}")

if __name__ == "__main__":
    main()
