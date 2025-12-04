"""Quick helper to export DocFinQA samples to HTML."""
import argparse
import pathlib

import pandas as pd


BASE_PATH = "hf://datasets/kensho/DocFinQA/"
SPLITS = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
COLUMN_ORDER = [
    "question",
    "query",
    "answer",
    "answers",
    "context",
    "context_text",
    "evidence",
    "passage",
    "split",
]


def parse_args():
    p = argparse.ArgumentParser("Inspect DocFinQA split head")
    p.add_argument("--split", choices=list(SPLITS.keys()), default="train")
    p.add_argument(
        "--base_path",
        default=BASE_PATH,
        help="Base path to the DocFinQA dataset (default is Hugging Face hub)",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows to export (default: 10)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="HTML file to write (default: docfinqa_<split>_preview.html)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    target = args.base_path.rstrip("/") + "/" + SPLITS[args.split]
    df = pd.read_json(target)
    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(f"docfinqa_{args.split}_preview.html")
    cols = [c for c in COLUMN_ORDER if c in df.columns]
    preview = df.head(args.rows)
    if cols:
        preview = preview[cols]
    preview.to_html(out_path, index=False, justify="center")
    print(f"Saved {len(preview)} DocFinQA rows to {out_path}")


if __name__ == "__main__":
    main()
