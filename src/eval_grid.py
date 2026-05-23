"""
src/eval_grid.py — Parameter grid evaluation with ephemeral ChromaDB collections.

Usage:
    python src/eval_grid.py --kb codon
    python src/eval_grid.py --kb codon --chunk-strategies paragraph fixed \\
                             --top-k-values 3 5 8 --threshold-values 0.5 0.65 0.8

Builds one ephemeral ChromaDB collection per unique chunk_strategy, runs all
top_k × threshold combinations against it, then deletes the collection.
Writes all results to data/eval_log.jsonl.

Ephemeral collection naming:
    {kb_id}__eval__{chunk_strategy}__{chunk_size}__ephemeral
These are never written to kb_registry.json and are deleted in try/finally.

The run_grid() function accepts an optional `emit` callback for SSE streaming
from the FastAPI workbench.  When emit=None (CLI), progress goes to stdout.
"""

import argparse
import json
import sys
import time
import chromadb
from datetime import datetime, timezone
from pathlib import Path
from itertools import product

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import importlib
importlib.reload(cfg)

# ── RAGAS ─────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import RunConfig
from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# ── existing pipeline functions ───────────────────────────────────────────────
from ingest import build_kb
from rag import retrieve_from, generate

JUDGE_MODEL   = "gemma4:31b-cloud"
JUDGE_SLEEP_S = 2

judge_llm = LangchainLLMWrapper(
    ChatOllama(model=JUDGE_MODEL, base_url="http://127.0.0.1:11434")
)
# gemma4:31b-cloud has no /api/embeddings — use local MiniLM (already cached)
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
)
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=3, timeout=300)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ephemeral_name(kb_id: str, chunk_strategy: str, chunk_size) -> str:
    """Stable, recognisable name for an ephemeral collection."""
    return f"{kb_id}__eval__{chunk_strategy}__{chunk_size}__ephemeral"


def delete_ephemeral(collection_name: str):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    try:
        client.delete_collection(collection_name)
        print(f"[grid] deleted ephemeral collection: {collection_name}")
    except Exception as e:
        print(f"[grid] warning: could not delete '{collection_name}': {e}")


def _progress(msg: str, emit=None):
    """Print to stdout or call the SSE emit callback, depending on context."""
    if emit:
        emit({"type": "progress", "msg": msg})
    else:
        print(msg)


# ── Core scoring ──────────────────────────────────────────────────────────────

def _combo_checkpoint(run_id: str, chunk_strategy: str, chunk_size, top_k: int, threshold: float) -> Path:
    """Unique checkpoint file per combination — valid on Windows (no colons)."""
    return Path("data") / f"grid_responses_{run_id}_{chunk_strategy}_{chunk_size}_k{top_k}_t{threshold}.jsonl"


def _scalar(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    return float(np.nanmean(val))


def collect_one_combo(
    collection_name: str,
    test_set: list[dict],
    top_k: int,
    threshold: float,
    checkpoint: Path,
    emit=None,
) -> Path:
    """
    Phase 1 for one combination: retrieve + generate for every question.
    Writes one JSON line per question immediately (crash-safe).
    Skips collection entirely if the checkpoint already exists.
    Returns the checkpoint path.
    """
    if checkpoint.exists():
        _progress(f"[grid]   checkpoint exists — skipping collection: {checkpoint.name}", emit)
        return checkpoint

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    n = len(test_set)
    with open(checkpoint, "w", encoding="utf-8") as f:
        for i, item in enumerate(test_set, 1):
            _progress(f"[grid]     question {i}/{n}: {item['question'][:60]}...", emit)
            if emit:
                emit({"type": "question", "i": i, "n": n, "question": item["question"]})
            chunks = retrieve_from(
                collection_name, item["question"], top_k=top_k, threshold=threshold
            )
            answer = generate(
                cfg.SYSTEM_PROMPT,
                chunks if chunks else ["(no relevant context found)"],
                item["question"],
            )
            row = {
                "question":     item["question"],
                "answer":       answer,
                "contexts":     chunks if chunks else ["(no relevant context found)"],
                "ground_truth": item["ground_truth"],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i < n:
                time.sleep(JUDGE_SLEEP_S)
    return checkpoint


def score_one_combo(checkpoint: Path, emit=None) -> dict:
    """
    Phase 2 for one combination: read checkpoint, run RAGAS, return scores.
    """
    rows = [json.loads(l) for l in checkpoint.read_text(encoding="utf-8").splitlines() if l.strip()]
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            ContextRecall(llm=judge_llm),
        ],
        run_config=RAGAS_RUN_CONFIG,
    )
    return {
        "faithfulness":     round(_scalar(result["faithfulness"]),     4),
        "answer_relevancy": round(_scalar(result["answer_relevancy"]), 4),
        "context_recall":   round(_scalar(result["context_recall"]),   4),
    }


# ── Grid runner (importable by workbench) ─────────────────────────────────────

def run_grid(
    kb_id: str,
    chunk_configs: list[tuple],
    top_k_values: list[int],
    threshold_values: list[float],
    questions_path: str,
    note: str = "",
    emit=None,
) -> list[dict]:
    """
    Run the full parameter grid.

    chunk_configs: list of (chunk_strategy, chunk_size) tuples.
        chunk_size is None for "paragraph", an int for "fixed".

    emit: optional callable that accepts a dict.  When provided (workbench SSE
        mode), progress events are emitted rather than printed.  The CLI sets
        emit=None so progress goes to stdout.

    Returns a list of result records (one per combination), which the caller
    can write to eval_log.jsonl and/or return as an SSE "done" payload.
    """
    with open(questions_path, encoding="utf-8") as f:
        test_set = json.load(f)

    all_records = []
    run_ts      = datetime.now(timezone.utc).isoformat()
    n_combos    = len(chunk_configs) * len(top_k_values) * len(threshold_values)
    combo_n     = 0

    for chunk_strategy, chunk_size in chunk_configs:
        col_name = ephemeral_name(kb_id, chunk_strategy, chunk_size)

        _progress(f"\n[grid] building ephemeral KB: {col_name}", emit)
        if emit:
            emit({"type": "build_start", "collection": col_name})

        try:
            total_chunks = build_kb(
                kb_id=kb_id,
                collection_name=col_name,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
            )
            _progress(f"[grid] build done: {total_chunks} chunks", emit)
            if emit:
                emit({"type": "build_done", "collection": col_name,
                      "chunks": total_chunks})

            for top_k, threshold in product(top_k_values, threshold_values):
                combo_n += 1
                _progress(
                    f"[grid] combo {combo_n}/{n_combos}:"
                    f"  top_k={top_k}  threshold={threshold}",
                    emit,
                )
                if emit:
                    emit({"type": "combo_start", "top_k": top_k,
                          "threshold": threshold,
                          "combo_n": combo_n, "combo_total": n_combos})

                checkpoint = _combo_checkpoint(
                    run_ts, chunk_strategy, chunk_size, top_k, threshold
                )
                collect_one_combo(col_name, test_set, top_k, threshold,
                                  checkpoint, emit=emit)
                scores = score_one_combo(checkpoint, emit=emit)

                record = {
                    "run_id":         run_ts,
                    "kb_id":          kb_id,
                    "embed_model":    cfg.EMBED_MODEL,
                    "chunk_strategy": chunk_strategy,
                    "chunk_size":     chunk_size,
                    "top_k":          top_k,
                    "threshold":      threshold,
                    "ollama_model":   cfg.OLLAMA_MODEL,
                    "judge_model":    JUDGE_MODEL,
                    "prompt_version": cfg.PROMPT_VERSION,
                    "prompt_hash":    cfg.PROMPT_HASH,
                    "n_questions":    len(test_set),
                    "ephemeral":      True,
                    "note":           note,
                    "scores":         scores,
                }
                all_records.append(record)

                _progress(f"         → {scores}", emit)
                if emit:
                    emit({"type": "combo_done", "top_k": top_k,
                          "threshold": threshold, "scores": scores,
                          "record": record})

        finally:
            delete_ephemeral(col_name)
            if emit:
                emit({"type": "build_cleanup", "collection": col_name})

    return all_records


# ── Log + summary helpers ─────────────────────────────────────────────────────

def write_log(records: list[dict], log_path: str = "data/eval_log.jsonl"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[grid] {len(records)} runs logged → {log_path}")


def _mean_score(record: dict) -> float:
    s = record["scores"]
    return (s["faithfulness"] + s["answer_relevancy"] + s["context_recall"]) / 3


def print_summary(records: list[dict]):
    winner = max(records, key=_mean_score)

    print("\n── Grid results ──────────────────────────────────────────────────────")
    print(f"{'strategy':<12} {'size':<6} {'top_k':<6} {'thresh':<8} "
          f"{'faith':<8} {'relev':<8} {'recall':<8} mean")
    print("─" * 74)
    for r in sorted(records, key=_mean_score, reverse=True):
        s      = r["scores"]
        mean   = _mean_score(r)
        marker = " ◀ best" if r is winner else ""
        print(
            f"{r['chunk_strategy']:<12} {str(r['chunk_size']):<6} "
            f"{r['top_k']:<6} {r['threshold']:<8} "
            f"{s['faithfulness']:<8} {s['answer_relevancy']:<8} "
            f"{s['context_recall']:<8} {mean:.4f}{marker}"
        )
    print("─" * 74)
    w = winner
    print(
        f"\nRecommendation: chunk_strategy={w['chunk_strategy']}, "
        f"chunk_size={w['chunk_size']}, top_k={w['top_k']}, "
        f"threshold={w['threshold']}"
    )
    print("Update config.py defaults to these values.\n")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid eval with ephemeral ChromaDB collections"
    )
    parser.add_argument("--kb",               required=True)
    parser.add_argument("--chunk-strategies", nargs="+", default=["paragraph", "fixed"],
                        help="Chunking strategies to sweep (default: paragraph fixed)")
    parser.add_argument("--chunk-size",       type=int, default=200,
                        help="chunk_size for fixed strategy (default: 200)")
    parser.add_argument("--top-k-values",     nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--threshold-values", nargs="+", type=float,
                        default=[0.50, 0.65, 0.80])
    parser.add_argument("--questions",        default="data/test_questions.json")
    parser.add_argument("--note",             default="")
    args = parser.parse_args()

    # Build (strategy, size) tuples — size is None for paragraph
    chunk_configs = [
        (s, args.chunk_size if s == "fixed" else None)
        for s in args.chunk_strategies
    ]

    n_builds = len(chunk_configs)
    n_runs   = len(chunk_configs) * len(args.top_k_values) * len(args.threshold_values)
    print(f"[grid] {n_builds} build(s) · {n_runs} total runs")
    print(f"[grid] strategies={args.chunk_strategies}  "
          f"top_k={args.top_k_values}  threshold={args.threshold_values}")
    print(f"[grid] judge={JUDGE_MODEL}  questions={args.questions}")

    records = run_grid(
        kb_id=args.kb,
        chunk_configs=chunk_configs,
        top_k_values=args.top_k_values,
        threshold_values=args.threshold_values,
        questions_path=args.questions,
        note=args.note,
        emit=None,   # CLI mode: print to stdout
    )

    write_log(records)
    print_summary(records)


if __name__ == "__main__":
    main()
