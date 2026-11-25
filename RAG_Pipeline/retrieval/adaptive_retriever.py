"""Adaptive retrieval and stopping.

Fixed top-k either wastes context or fails the task. Here top-k is a *decision*:

    Query
      -> Rank candidates once
      -> Admit context in growing batches (k = min_k, min_k+step, ...)
      -> Evaluate context (coverage, score distribution, novelty, diminishing returns)
      -> Is context sufficient?
             YES -> STOP
             NO  -> admit more -> evaluate again
      -> Stop on: sufficiency, diminishing returns, or the k budget

Because the retriever returns a ranked list, "retrieving more" means admitting
more of that ranking; the stopping logic is what this technique is about.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

from retrieval.iterative_retriever import (  # noqa: E402
    analyze_coverage,
    is_same_unit,
    normalise_scores,
    query_concepts,
)
from retrieval.semantic_retriever import SemanticRetriever  # noqa: E402

DEFAULT_QUERY = "predict anomaly score from extracted features"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_OUTPUT = PROJECT_ROOT / "adaptive_retrieval_data.json"


# ============================================================
# Signals
# ============================================================

def score_signals(documents: Sequence[Document]) -> Dict[str, float]:
    """Confidence of the ranking prefix: top, mean, margin, relative drop."""
    scores = [document.score or 0.0 for document in documents]
    if not scores:
        return {"top": 0.0, "mean": 0.0, "margin": 0.0, "drop": 1.0}

    ordered = sorted(scores, reverse=True)
    top = ordered[0]
    mean = sum(scores) / len(scores)
    second = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "top": top,
        "mean": mean,
        "margin": top - second,
        "drop": (top - ordered[-1]) / top if top > 0 else 1.0,
    }


def novelty_ratio(
    new_documents: Sequence[Document], admitted: Sequence[Document]
) -> Tuple[float, int]:
    """How much of the latest batch is genuinely new context."""
    if not new_documents:
        return 0.0, 0

    fresh = sum(
        1
        for candidate in new_documents
        if not any(is_same_unit(candidate, kept) for kept in admitted)
    )
    return fresh / len(new_documents), fresh


def evaluate_context(
    admitted: Sequence[Document],
    batch: Sequence[Document],
    concepts: Sequence[str],
    min_score: float,
    coverage_threshold: float,
    min_gain: float,
    min_batch: int,
) -> Dict[str, Any]:
    """Compute the stopping signals for the current context."""
    previous = list(admitted)[: max(len(admitted) - len(batch), 0)]
    covered, missing, coverage = analyze_coverage(concepts, admitted)
    signals = score_signals(admitted)
    gain, fresh = novelty_ratio(batch, previous)

    sufficient = coverage >= coverage_threshold and signals["top"] >= min_score

    decision = "CONTINUE"
    reason = (
        f"coverage {coverage:.2f} < {coverage_threshold:.2f}"
        if coverage < coverage_threshold
        else f"top score {signals['top']:.3f} < {min_score:.3f}"
    )
    if sufficient:
        decision = "STOP"
        reason = (
            f"sufficient: coverage {coverage:.2f} >= {coverage_threshold:.2f} "
            f"and top score {signals['top']:.3f} >= {min_score:.3f}"
        )
    elif gain < min_gain and len(admitted) > min_batch:
        decision = "STOP"
        reason = f"diminishing returns: only {gain:.2f} of the last batch was new"

    return {
        "context_size": len(admitted),
        "coverage": round(coverage, 4),
        "covered_concepts": list(covered),
        "missing_concepts": list(missing),
        "top_score": round(signals["top"], 6),
        "mean_score": round(signals["mean"], 6),
        "score_margin": round(signals["margin"], 6),
        "novelty": round(gain, 4),
        "new_units": fresh,
        "token_estimate": sum(len(d.content) for d in admitted) // 4,
        "sufficient": sufficient,
        "decision": decision,
        "reason": reason,
    }


# ============================================================
# Retriever
# ============================================================

class AdaptiveRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        concepts: Optional[Sequence[str]] = None,
        min_k: int = 2,
        max_k: int = 20,
        step: int = 4,
        min_score: float = 0.3,
        coverage_threshold: float = 0.8,
        min_gain: float = 0.3,
    ) -> None:
        self.semantic = semantic
        self.concepts = list(concepts) if concepts else []
        self.min_k = min_k
        self.max_k = max_k
        self.step = step
        self.min_score = min_score
        self.coverage_threshold = coverage_threshold
        self.min_gain = min_gain
        self.trace: List[Dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.semantic.ready

    def rank(self, query: str) -> List[Document]:
        ranking = normalise_scores(self.semantic.retrieve_documents(query, self.max_k))
        ranking.sort(key=lambda d: d.score or 0.0, reverse=True)
        return ranking

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        max_k = top_k or self.max_k
        ranking = self.rank(query)
        concepts = self.concepts or query_concepts(query)

        admitted: List[Document] = []
        self.trace = []

        k = self.min_k
        while True:
            batch = list(ranking[max(0, k - self.step): k])
            admitted.extend(batch)

            record = evaluate_context(
                admitted=admitted,
                batch=batch,
                concepts=concepts,
                min_score=self.min_score,
                coverage_threshold=self.coverage_threshold,
                min_gain=self.min_gain,
                min_batch=self.min_k,
            )
            record["k"] = k
            record["batch_size"] = len(batch)
            self.trace.append(record)

            if record["decision"] == "STOP":
                break
            if k >= max_k:
                record["decision"] = "STOP"
                record["reason"] = f"k budget reached ({max_k})"
                break
            if not batch:
                record["decision"] = "STOP"
                record["reason"] = "retriever returned no further candidates"
                break
            k += self.step

        return admitted

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="adaptive_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Adaptive retrieval and stopping")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-gain", type=float, default=0.3)
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Adaptive Retrieval / Stopping")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"k schedule: {args.min_k} -> +{args.step} -> max {args.max_k}")
    print(f"stop when coverage >= {args.coverage_threshold} and top score >= {args.min_score}")
    print(f"stop early when novelty < {args.min_gain}")

    semantic = SemanticRetriever(
        embeddings_file=Path(args.embeddings),
        vector_store_path=Path(args.vector_store),
        top_k=args.max_k,
    )
    if not semantic.ready:
        print("Error: missing pipeline artifacts:")
        for path in semantic.missing_artifacts:
            print(f"  {path}")
        return

    ranking = semantic.retrieve_documents(args.query, args.max_k)
    ranking = normalise_scores(ranking)
    ranking.sort(key=lambda d: d.score or 0.0, reverse=True)
    print(f"\nRetriever returned {len(ranking)} ranked candidates")

    concepts = query_concepts(args.query)
    print(f"Query concepts: {concepts}")

    retriever = AdaptiveRetriever(
        semantic=semantic,
        concepts=concepts,
        min_k=args.min_k,
        max_k=args.max_k,
        step=args.step,
        min_score=args.min_score,
        coverage_threshold=args.coverage_threshold,
        min_gain=args.min_gain,
    )
    kept = retriever.retrieve_documents(args.query)
    trace = retriever.trace

    print(f"\n{'=' * 60}")
    print("Decision trace")
    print("=" * 60)
    print(f"{'k':>4} {'size':>5} {'cov':>6} {'top':>7} {'margin':>7} "
          f"{'novel':>6} {'tokens':>7}  decision")
    for record in trace:
        print(f"{record['k']:>4} {record['context_size']:>5} "
              f"{record['coverage']:>6.2f} {record['top_score']:>7.3f} "
              f"{record['score_margin']:>7.3f} {record['novelty']:>6.2f} "
              f"{record['token_estimate']:>7}  {record['decision']}")
        print(f"      reason: {record['reason']}")

    final = trace[-1] if trace else {}
    final_tokens = final.get("token_estimate", 0)
    max_tokens = sum(len(d.content) for d in ranking) // 4
    saved = (1 - final_tokens / max_tokens) * 100 if max_tokens else 0.0

    print(f"\n{'=' * 60}")
    print("Result")
    print("=" * 60)
    print(f"Final k        : {final.get('k', 0)} of {len(ranking)} available candidates")
    print(f"Context units  : {len(kept)}")
    print(f"Token estimate : ~{final_tokens} (~{max_tokens} if k = {args.max_k})")
    print(f"Context saved  : {saved:.1f}%")

    for rank, document in enumerate(kept, start=1):
        metadata = document.metadata or {}
        print(f"\n--- #{rank} score={document.score} ---")
        print(f"{Path(metadata.get('file_path', '?')).name}:"
              f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
        print(document.content[:220])

    payload = {
        "query": args.query,
        "retriever_id": "adaptive_retrieval",
        "documents": [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in kept
        ],
        "adaptive": {
            "decisions": trace,
            "final_k": final.get("k", 0),
        },
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()