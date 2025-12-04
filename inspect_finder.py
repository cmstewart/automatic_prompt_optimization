"""Quick helper to export the first FinDER rows as an HTML table."""
import argparse
import pathlib

import pandas as pd


DEFAULT_SOURCE = "hf://datasets/Linq-AI-Research/FinDER/data/train-00000-of-00001.parquet"
DEFAULT_OUT = pathlib.Path("finder_preview.html")
COLUMN_ORDER = ["_id", "text", "reasoning", "category", "references", "answer", "type"]


def parse_args():
    p = argparse.ArgumentParser("Inspect FinDER dataset head")
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Parquet file to read (defaults to the HF FinDER training shard)",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=10,
        help="How many rows to export (default: 10)",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=f"HTML file to write (default: {DEFAULT_OUT})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.source)
    cols = [c for c in COLUMN_ORDER if c in df.columns]
    preview = df.head(args.rows)
    if cols:
        preview = preview[cols]
    preview.to_html(args.out, index=False, justify="center")
    print(f"Saved {len(preview)} FinDER rows to {args.out}")


if __name__ == "__main__":
    main()
