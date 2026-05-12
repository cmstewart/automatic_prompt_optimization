"""
generate.py – FinanceBench ProTeGi fork
---------------------------------------
Generates model answers using the single-vector-store (per-filing) retriever.

Original behavior:
  • Reads data from data/dataset_prepared.parquet
  • Uses Chroma per-filing vector stores under vectorstores/<doc_name>
  • Writes results/answers.jsonl
  • Uses default RetrievalQA prompt

This version keeps the original structure but adds CLI flags:
  • --model / --temperature / --max_tokens
  • --top_k retrieval fan-out
  • --limit to cap #examples
  • --answers to choose output file
  • --prompt to inject a custom prompt template ({context}, {question})
"""

import argparse
import json
import pathlib
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader  # kept for parity with original
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings          # imported in original
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ----------------------------------------------------------------------
# DEFAULT PATHS (overridable via CLI for outputs only; inputs kept for parity)
# ----------------------------------------------------------------------
from paths import ROOT
DATASET   = ROOT / "data" / "dataset_prepared.parquet"
VS_DIR    = ROOT / "vectorstores"
RESULTS   = ROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
OUT_PATH  = RESULTS / "answers.jsonl"

# Embeddings: match vectorization pipeline (e5-small-v2)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")


# ----------------------------------------------------------------------
# MODEL FACTORY
# ----------------------------------------------------------------------
def get_model(model_name="gpt-4o-mini", temp=0.01, max_tokens=2048):
    return ChatOpenAI(model_name=model_name, temperature=temp, max_tokens=max_tokens)


# ----------------------------------------------------------------------
# QA CALL (optionally with a custom prompt)
# ----------------------------------------------------------------------
def get_answer(model, question, retriever, prompt_text=None):
    """
    If prompt_text is provided, it must contain {context} and {question}.
    We inject it via chain_type_kwargs={"prompt": PromptTemplate(...)}.
    """
    chain_kwargs = {}
    if prompt_text:
        prompt = PromptTemplate.from_template(prompt_text)
        chain_kwargs["prompt"] = prompt

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",                 # late fusion: concatenate top-K into {context}
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs=chain_kwargs if chain_kwargs else None
    )
    return qa(question)["result"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Generate FinanceBench answers (single-store retriever)")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model name")
    p.add_argument("--temperature", type=float, default=0.0, help="decoder temperature")
    p.add_argument("--max_tokens", type=int, default=2048, help="max tokens for generation")
    p.add_argument("--top_k", type=int, default=3, help="retriever fan-out per question")
    p.add_argument("--limit", type=int, default=None, help="optional cap on #examples")
    p.add_argument("--answers", type=pathlib.Path, default=OUT_PATH,
                   help="output answers JSONL (default: results/answers.jsonl)")
    p.add_argument("--prompt", type=pathlib.Path, default=None,
                   help="path to a custom prompt template with {context} and {question}")
    return p.parse_args()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = get_args()

    if not DATASET.exists():
        sys.exit("Missing dataset_prepared.parquet. Run prepare_data/vectorize first.")

    # Load dataset
    df = pd.read_parquet(DATASET).sort_values("doc_name")
    if args.limit:
        df = df.head(args.limit)

    # Load optional custom prompt text
    prompt_text = None
    if args.prompt is not None:
        if not args.prompt.exists():
            sys.exit(f"Prompt file not found: {args.prompt}")
        prompt_text = args.prompt.read_text(encoding="utf-8")

    # Model
    model = get_model(args.model, args.temperature, args.max_tokens)

    # One retriever per filing (cached)
    cache = {}

    # Ensure output directory exists
    args.answers.parent.mkdir(parents=True, exist_ok=True)

    # Iterate and generate
    with args.answers.open("w", encoding="utf8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
            doc = row.doc_name
            if doc not in cache:
                cache[doc] = Chroma(
                    persist_directory=str(VS_DIR / doc),
                    embedding_function=EMBEDDINGS
                ).as_retriever(search_kwargs={"k": args.top_k})

            answer = get_answer(model, row.question, cache[doc], prompt_text=prompt_text)

            rec = {
                "financebench_id": row.financebench_id,
                "question": row.question,
                "gold_answer": row.answer,
                "model_answer": answer,
                "doc_name": doc
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {args.answers}")

if __name__ == "__main__":
    main()
