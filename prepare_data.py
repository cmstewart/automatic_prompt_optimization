"""Dataset preparation helper for FinanceBench, FinQA, and FinDoc-RAG."""
import argparse
import json
import pathlib

import pandas as pd

from paths import ROOT
DEFAULT_DATA_DIRS = {
    "financebench": ROOT / "data" / "FinanceBench",
    "finqa": ROOT / "data" / "FinQa",
    "findoc": ROOT / "data" / "Findoc",
}


def _default_out_dir(task: str, override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    return DEFAULT_DATA_DIRS[task]


def prepare_financebench(out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    q  = pd.read_json(ROOT / "data" / "FinanceBench" / "financebench_open_source.jsonl", lines=True)
    di = pd.read_json(ROOT / "data" / "FinanceBench" / "financebench_document_information.jsonl", lines=True)
    df = q.merge(di, on="doc_name").sort_values("doc_name")
    dest = out_dir / "dataset_prepared.parquet"
    df.to_parquet(dest, index=False)
    print(f"FinanceBench dataset saved to {dest}  ({len(df)} rows, {df.doc_name.nunique()} PDFs)")


def prepare_finqa(out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.json"
    dev_path = out_dir / "dev.json"
    test_path = out_dir / "test.json"

    train_df = pd.read_json(train_path)
    dev_df = pd.read_json(dev_path)
    test_df = pd.read_json(test_path)
    df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    def to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(to_text(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    qa_col = df["qa"] if "qa" in df.columns else pd.Series([{}] * len(df), index=df.index)

    def qa_item(value) -> dict:
        if isinstance(value, list):
            value = value[0] if value else {}
        return value if isinstance(value, dict) else {}

    qa_norm = qa_col.apply(qa_item)
    df["question"] = qa_norm.apply(lambda x: x.get("question", "")).astype(str)
    df["answer"] = qa_norm.apply(lambda x: x.get("exe_ans", "")).astype(str)

    pre_series = (df["pre_text"] if "pre_text" in df.columns else pd.Series([""] * len(df), index=df.index)).apply(to_text)
    table_series = (df["table"] if "table" in df.columns else pd.Series([""] * len(df), index=df.index)).apply(to_text)
    post_series = (df["post_text"] if "post_text" in df.columns else pd.Series([""] * len(df), index=df.index)).apply(to_text)
    df["context_text"] = (pre_series + "\n" + table_series + "\n" + post_series).str.strip()

    if "id" in df.columns:
        df["doc_name"] = df["id"].astype(str)
    else:
        df["doc_name"] = df.index.map(lambda x: f"finqa_{x}")

    # Keep original FinQA structured columns but store them as text so parquet
    # writing is robust across mixed nested types.
    for col in ("qa", "table", "pre_text", "post_text"):
        if col in df.columns:
            df[col] = df[col].apply(to_text)

    df.to_parquet(out_dir / "dataset_prepared.parquet", index=False)
    print(f"FinQA dataset saved to {out_dir / 'dataset_prepared.parquet'} ({len(df)} rows)")


def prepare_findoc(out_dir: pathlib.Path, qa_path: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if qa_path is None:
        raise ValueError("--findoc_qa is required for task=findoc")
    df = pd.read_json(qa_path)
    # Keep columns simple and aligned with the pipeline
    df = df.rename(columns={"QUESTION": "question", "ANSWER": "answer"})
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    # DOCS can be a list; take the first entry as the doc identifier
    df["doc_name"] = [
        docs[0] if isinstance(docs, list) and len(docs) > 0 else str(idx)
        for idx, docs in enumerate(df.get("DOCS", []))
    ]
    if "DOCS" not in df.columns:
        df["doc_name"] = df.index.astype(str)

    df.to_parquet(out_dir / "dataset_prepared.parquet", index=False)
    print(f"Findoc dataset saved to {out_dir / 'dataset_prepared.parquet'} ({len(df)} rows)")


def parse_args():
    p = argparse.ArgumentParser("Prepare datasets for ProTeGi")
    p.add_argument("--task", choices=("financebench", "finqa", "findoc"), default="financebench")
    p.add_argument("--out_dir", help="Destination folder that will contain dataset_prepared.parquet", default=None)
    p.add_argument("--findoc_qa", help="Path to findoc qa.json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task.lower()
    out_dir = _default_out_dir(task, args.out_dir)
    if task == "financebench":
        prepare_financebench(out_dir)
    elif task == "finqa":
        prepare_finqa(out_dir)
    elif task == "findoc":
        prepare_findoc(out_dir, args.findoc_qa)
    else:
        raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
