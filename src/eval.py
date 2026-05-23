"""
src/eval.py — Two-phase RAGAS evaluation runner for codon-rag.

Phase 1 (collect): runs every question through RAG, writes a checkpoint JSONL
                   to disk line-by-line so partial results survive a crash.
Phase 2 (score):   reads the checkpoint, scores with RAGAS + gemma4:31b-cloud.

Because the phases are separate files, you can re-run Phase 2 alone if the
judge crashes or rate-limits — no need to repeat 15 embedding + LLM calls.

Usage:
    # Full run (both phases, checkpoint always written):
    python src/eval.py --kb codon --note "baseline"

    # Phase 1 only — inspect answers before scoring:
    python src/eval.py --kb codon --collect-only

    # Phase 2 only — re-score an existing checkpoint:
    python src/eval.py --score-file data/eval_responses_2026-05-23T14-30-00Z.jsonl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# ── Disable RAGAS anonymous telemetry ─────────────────────────────────────────
# Must be set BEFORE `import ragas`.  RAGAS otherwise POSTs a usage event after
# evaluate() completes; if that call blocks (corporate firewall, slow DNS), the
# whole evaluate() appears to hang at 100% for minutes.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import importlib
importlib.reload(cfg)

# ── RAGAS ─────────────────────────────────────────────────────────────────────
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# ── judge LLM + embeddings ────────────────────────────────────────────────────
# gemma4:31b-cloud: used as judge LLM (chat/thinking).
# HuggingFaceEmbeddings: used for AnswerRelevancy cosine step — gemma4 has no
# /api/embeddings endpoint, so we reuse the same local MiniLM already on disk.
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

JUDGE_MODEL      = "gemma4:31b-cloud"
JUDGE_SLEEP_S    = 2
# timeout=120 / max_retries=2: a stuck judge call now fails fast (~4 min worst
# case per sample) instead of silently chewing 15 min on retries.  If you see
# NaN scores after this change, raise these back up — the judge is just slow,
# not deadlocked.
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=2, timeout=120)

judge_llm = LangchainLLMWrapper(
    ChatOllama(model=JUDGE_MODEL, base_url="http://127.0.0.1:11434")
)
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────
from rag import retrieve_from, generate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _checkpoint_path(run_id: str) -> Path:
    """Checkpoint filename — colons replaced so it's valid on Windows."""
    return Path("data") / f"eval_responses_{run_id}.jsonl"


def load_test_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _scalar(val) -> float:
    """Handle both scalar and per-sample list returns across RAGAS versions."""
    if isinstance(val, (int, float)):
        return float(val)
    return float(np.nanmean(val))


# ── Phase 1: collect ──────────────────────────────────────────────────────────

def phase1_collect(
    kb_id: str,
    top_k: int,
    threshold: float,
    questions_path: str,
    sleep_s: float,
    run_id: str,
    emit=None,
) -> Path:
    """
    Run every question through RAG and write responses to a checkpoint file.

    Writes one JSON line per question immediately after each response — so even
    if the process crashes mid-run, every completed response is on disk.

    Returns the checkpoint Path.
    """
    with open(cfg.KB_REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    if kb_id not in registry:
        print(f"Error: KB '{kb_id}' not found in registry.")
        sys.exit(1)
    collection_name = registry[kb_id].get("collection", kb_id)

    test_set = load_test_set(questions_path)
    n        = len(test_set)

    print(f"\n[eval] Phase 1 — collecting {n} responses")
    print(f"       kb={kb_id}  top_k={top_k}  threshold={threshold}  sleep={sleep_s}s\n")

    if emit:
        emit({"type": "collect_start", "n": n, "kb_id": kb_id,
              "top_k": top_k, "threshold": threshold})

    checkpoint = _checkpoint_path(run_id)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint, "w", encoding="utf-8") as f:
        for i, item in enumerate(test_set, 1):
            print(f"  [{i:>2}/{n}] {item['question'][:70]}...")
            chunks = retrieve_from(
                collection_name, item["question"], top_k=top_k, threshold=threshold
            )
            if not chunks:
                print(f"         ⚠  no chunks passed threshold")
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
            f.flush()   # on-disk immediately — survives a crash
            if emit:
                emit({
                    "type":        "qa",
                    "i":           i,
                    "n":           n,
                    "question":    item["question"],
                    "answer":      answer,
                    "chunks_kept": len(chunks),
                })
            if i < n:
                time.sleep(sleep_s)

    print(f"\n[eval] Phase 1 complete. Checkpoint → {checkpoint}\n")
    if emit:
        emit({"type": "collect_done", "checkpoint": str(checkpoint)})
    return checkpoint


# ── Phase 2: score ────────────────────────────────────────────────────────────

def phase2_score(checkpoint: Path) -> dict:
    """
    Read a Phase 1 checkpoint and score every response with RAGAS.
    Returns the three metric scores as a dict.
    """
    rows = [json.loads(l) for l in checkpoint.read_text(encoding="utf-8").splitlines() if l.strip()]
    n    = len(rows)

    print(f"[eval] Phase 2 — scoring {n} responses with {JUDGE_MODEL}...\n")

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


# ── Log + print ───────────────────────────────────────────────────────────────

def write_log(record: dict, log_path: str = "data/eval_log.jsonl"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[eval] Run logged → {log_path}")


def _print_scores(scores: dict):
    print("\n── RAGAS scores ─────────────────────────")
    for k, v in scores.items():
        print(f"  {k:<22} {v:.4f}")
    print("─────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Two-phase RAGAS eval runner")

    # Phase 1
    parser.add_argument("--kb",           help="KB ID (required unless --score-file)")
    parser.add_argument("--top-k",        type=int,   default=cfg.TOP_K)
    parser.add_argument("--threshold",    type=float, default=cfg.RELEVANCE_THRESHOLD)
    parser.add_argument("--questions",    default="data/test_questions.json")
    parser.add_argument("--sleep",        type=float, default=JUDGE_SLEEP_S)
    parser.add_argument("--collect-only", action="store_true",
                        help="Stop after Phase 1 (write checkpoint, skip scoring)")

    # Phase 2
    parser.add_argument("--score-file",
                        help="Skip Phase 1; score this existing checkpoint instead")

    # Shared
    parser.add_argument("--note", default="",
                        help="Free-text note logged with this run")

    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    # ── Phase 2 only (re-score existing checkpoint) ───────────────────────────
    if args.score_file:
        checkpoint = Path(args.score_file)
        if not checkpoint.exists():
            print(f"Error: checkpoint not found: {checkpoint}")
            sys.exit(1)
        scores = phase2_score(checkpoint)
        n = sum(1 for l in checkpoint.read_text().splitlines() if l.strip())
        write_log(_build_record(run_id, args, scores, n))
        _print_scores(scores)
        return

    # ── Phase 1 required from here ────────────────────────────────────────────
    if not args.kb:
        parser.error("--kb is required (or use --score-file to re-score a checkpoint)")

    print(f"[eval] kb={args.kb}  top_k={args.top_k}  threshold={args.threshold}"
          f"  prompt={cfg.PROMPT_VERSION} ({cfg.PROMPT_HASH})  judge={JUDGE_MODEL}")

    checkpoint = phase1_collect(
        kb_id=args.kb,
        top_k=args.top_k,
        threshold=args.threshold,
        questions_path=args.questions,
        sleep_s=args.sleep,
        run_id=run_id,
    )

    if args.collect_only:
        print("[eval] --collect-only: Phase 1 done. Re-score later with:")
        print(f"       python src/eval.py --score-file {checkpoint} --note \"...\"")
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    scores = phase2_score(checkpoint)
    write_log(_build_record(run_id, args, scores, len(load_test_set(args.questions))))
    _print_scores(scores)


def _build_record(run_id: str, args, scores: dict, n_questions: int) -> dict:
    return {
        "run_id":         run_id,
        "kb_id":          getattr(args, "kb", None),
        "embed_model":    cfg.EMBED_MODEL,
        "chunk_strategy": cfg.CHUNK_STRATEGY,
        "top_k":          getattr(args, "top_k", cfg.TOP_K),
        "threshold":      getattr(args, "threshold", cfg.RELEVANCE_THRESHOLD),
        "ollama_model":   cfg.OLLAMA_MODEL,
        "judge_model":    JUDGE_MODEL,
        "prompt_version": cfg.PROMPT_VERSION,
        "prompt_hash":    cfg.PROMPT_HASH,
        "n_questions":    n_questions,
        "note":           getattr(args, "note", ""),
        "scores":         scores,
    }


if __name__ == "__main__":
    main()
