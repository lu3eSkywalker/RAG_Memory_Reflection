"""Iterative (multi-round) retrieval.

Single-shot RAG (Query -> Retrieve -> Generate) often retrieves only part of the
context a repository-level task needs. This retriever loops:

    Query
      -> Initial Retrieval
      -> Analyse retrieved context (coverage of query concepts)
      -> Identify missing context
      -> Retrieve again (follow-up query built from the gaps)
      -> Merge (dedupe against earlier rounds)
      -> Hand off to generation

Stopping conditions: coverage threshold reached, no gaps left, a round that adds
nothing new, or the round budget exhausted.

Optionally the gaps are also filled structurally with the AST retriever, so the
second round returns whole functions/classes instead of sliding-window text.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

from retrieval.ast_retriever import ASTBasedRetriever, ASTIndex  # noqa: E402
from retrieval.semantic_retriever import SemanticRetriever  # noqa: E402

DEFAULT_QUERY = "Add authentication to the user API"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_AST_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "iterative_retrieval_data.json"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "add", "implement", "create", "make", "code", "file", "files", "this",
    "that", "it", "is", "are", "be", "should", "must", "can", "will", "how",
    "using", "use", "used", "into", "from", "by", "at", "as", "we", "i",
    "def", "return", "self", "import", "class", "function", "method",
}


# ============================================================
# Concept analysis
# ============================================================

def split_identifier(name: str) -> List[str]:
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)
    return [part.lower() for part in parts] or [name.lower()]


def tokenize(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        tokens.update(split_identifier(raw))
    return tokens - STOPWORDS


def query_concepts(query: str) -> List[str]:
    """Ordered, meaningful concepts the query asks for."""
    concepts: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query or ""):
        if raw.lower() in STOPWORDS or len(raw) < 3:
            continue
        if raw not in concepts:
            concepts.append(raw)
    return concepts


def analyze_coverage(
    concepts: Sequence[str], documents: Sequence[Document]
) -> Tuple[List[str], List[str], float]:
    """Split concepts into covered / missing against the merged context."""
    context_tokens: Set[str] = set()
    for document in documents:
        context_tokens |= tokenize(document.content)

    covered: List[str] = []
    missing: List[str] = []
    for concept in concepts:
        if set(split_identifier(concept)) & context_tokens:
            covered.append(concept)
        else:
            missing.append(concept)

    ratio = len(covered) / len(concepts) if concepts else 1.0
    return covered, missing, ratio


def expansion_terms(documents: Sequence[Document], limit: int = 6) -> List[str]:
    """Frequent identifiers in the current context, used to widen the next query."""
    counter: Counter = Counter()
    for document in documents[:5]:
        for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", document.content):
            counter[raw] += 1
    return [name for name, _ in counter.most_common(limit)]


def build_followup_query(
    original_query: str, missing: Sequence[str], documents: Sequence[Document]
) -> str:
    parts: List[str] = [original_query]
    if missing:
        parts.append(" ".join(missing))
    extra = expansion_terms(documents)
    if extra:
        parts.append(" ".join(extra))
    return " ".join(parts)[:600]


# ============================================================
# Merging helpers
# ============================================================

def line_range(document: Document) -> Optional[Tuple[int, int]]:
    metadata = document.metadata or {}
    start, end = metadata.get("start_line"), metadata.get("end_line")
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None


def is_same_unit(left: Document, right: Document) -> bool:
    left_meta = left.metadata or {}
    right_meta = right.metadata or {}
    if left_meta.get("file_path") != right_meta.get("file_path"):
        return False

    left_range = line_range(left)
    right_range = line_range(right)
    if left_range and right_range:
        return left_range[0] <= right_range[1] and right_range[0] <= left_range[1]
    return left.content.strip() == right.content.strip()


def merge_documents(
    existing: List[Document], new_documents: Sequence[Document]
) -> Tuple[List[Document], int]:
    """Append only unseen units; upgrade the score of already-kept units."""
    added: List[Document] = []
    duplicates = 0

    for candidate in new_documents:
        duplicate = False
        for kept in existing + added:
            if is_same_unit(kept, candidate):
                duplicates += 1
                duplicate = True
                if (candidate.score or 0) > (kept.score or 0):
                    kept.score = candidate.score
                break
        if not duplicate:
            added.append(candidate)

    return added, duplicates


def normalise_scores(documents: Sequence[Document]) -> List[Document]:
    scores = [document.score or 0.0 for document in documents]
    top = max(scores) if scores else 0.0
    if top <= 0:
        return list(documents)
    for document in documents:
        document.score = round((document.score or 0.0) / top, 6)
    return list(documents)


def tag_round(document: Document, round_no: int) -> Document:
    document.metadata = {**(document.metadata or {}), "round_found": round_no}
    return document


# ============================================================
# Retriever
# ============================================================

class IterativeRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        concepts: Optional[Sequence[str]] = None,
        rounds: int = 3,
        top_k: int = 5,
        coverage_threshold: float = 0.8,
        round_decay: float = 0.95,
        structural: Optional[ASTBasedRetriever] = None,
        structural_top_k: int = 3,
    ) -> None:
        self.semantic = semantic
        self.concepts = list(concepts) if concepts else []
        self.rounds = rounds
        self.top_k = top_k
        self.coverage_threshold = coverage_threshold
        self.round_decay = round_decay
        self.structural = structural
        self.structural_top_k = structural_top_k
        self.history: List[Dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.semantic.ready

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        rounds = self.rounds
        top_k = top_k or self.top_k
        concepts = self.concepts or query_concepts(query)

        merged: List[Document] = []
        self.history = []
        missing = list(concepts)
        coverage = 0.0
        stop_reason = "rounds exhausted"

        for round_no in range(1, rounds + 1):
            if round_no == 1:
                round_query = query
            else:
                if not missing:
                    stop_reason = "no missing concepts"
                    break
                round_query = build_followup_query(query, missing, merged)

            candidates = normalise_scores(
                self.semantic.retrieve_documents(round_query, top_k=top_k * round_no)
            )
            for document in candidates:
                document.score = round(
                    (document.score or 0.0) * (self.round_decay ** (round_no - 1)), 6
                )
                tag_round(document, round_no)

            added, duplicates = merge_documents(merged, candidates)
            merged.extend(added)

            if self.structural is not None and self.structural.ready and missing:
                gap_documents = normalise_scores(
                    self.structural.retrieve_documents(" ".join(missing), self.structural_top_k)
                )
                for document in gap_documents:
                    tag_round(document, round_no)
                    document.score = round(
                        (document.score or 0.0) * (self.round_decay ** (round_no - 1)), 6
                    )
                structural_added, structural_duplicates = merge_documents(merged, gap_documents)
                merged.extend(structural_added)
                added = added + structural_added
                duplicates += structural_duplicates

            covered, missing, coverage = analyze_coverage(concepts, merged)
            self.history.append(
                {
                    "round": round_no,
                    "query_used": round_query,
                    "candidates": len(candidates),
                    "duplicates_skipped": duplicates,
                    "new_documents": len(added),
                    "context_size": len(merged),
                    "covered_concepts": list(covered),
                    "missing_concepts": list(missing),
                    "coverage": round(coverage, 4),
                }
            )

            if coverage >= self.coverage_threshold:
                stop_reason = (
                    f"coverage {coverage:.2f} >= threshold {self.coverage_threshold:.2f}"
                )
                break
            if not added:
                stop_reason = "no new code units retrieved"
                break

        if self.history and self.history[-1]["round"] == rounds and coverage < self.coverage_threshold:
            stop_reason = f"rounds exhausted (coverage {coverage:.2f})"
        self.history.append({"stop_reason": stop_reason})

        return sorted(
            merged,
            key=lambda d: (
                -(d.score or 0.0),
                (d.metadata or {}).get("round_found", 1),
            ),
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="iterative_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Iterative (multi-round) retrieval")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--round-decay", type=float, default=0.95)
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--ast-index", default=str(DEFAULT_AST_INDEX))
    parser.add_argument("--use-ast", action="store_true",
                        help="Also fill gaps structurally with the AST retriever")
    parser.add_argument("--ast-top-k", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Iterative Retrieval")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Rounds: {args.rounds}  top-k/round: {args.top_k}  "
          f"coverage threshold: {args.coverage_threshold}")

    semantic = SemanticRetriever(
        embeddings_file=Path(args.embeddings),
        vector_store_path=Path(args.vector_store),
        top_k=args.top_k,
    )
    if not semantic.ready:
        print("Error: missing pipeline artifacts:")
        for path in semantic.missing_artifacts:
            print(f"  {path}")
        print("Run scripts/02_run_embedding.py and scripts/03_run_vector_store.py first.")
        return

    structural = None
    if args.use_ast:
        ast_index_path = Path(args.ast_index)
        if ast_index_path.exists():
            structural = ASTBasedRetriever(ASTIndex.load(ast_index_path), hybrid=False)
        else:
            print(f"Note: no AST index at {ast_index_path}; continuing without it")

    concepts = query_concepts(args.query)
    print(f"Query concepts: {concepts}")

    retriever = IterativeRetriever(
        semantic=semantic,
        concepts=concepts,
        rounds=args.rounds,
        top_k=args.top_k,
        coverage_threshold=args.coverage_threshold,
        round_decay=args.round_decay,
        structural=structural,
        structural_top_k=args.ast_top_k,
    )
    documents = retriever.retrieve_documents(args.query)

    print(f"\n{'=' * 60}")
    print("Round summary")
    print("=" * 60)
    for record in retriever.history:
        if "stop_reason" in record:
            print(f"Stop: {record['stop_reason']}")
            break
        print(f"\nRound {record['round']}")
        print(f"  query      : {record['query_used'][:120]}")
        print(f"  candidates : {record['candidates']} "
              f"(new: {record['new_documents']}, dupes skipped: {record['duplicates_skipped']})")
        print(f"  context    : {record['context_size']} units")
        print(f"  coverage   : {record['coverage']:.2f}")
        print(f"  covered    : {record['covered_concepts']}")
        print(f"  missing    : {record['missing_concepts']}")

    print(f"\n{'=' * 60}")
    print(f"Merged context: {len(documents)} code units")
    print("=" * 60)
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        print(f"\n--- #{rank} round={metadata.get('round_found', 1)} "
              f"score={document.score} ---")
        print(f"{Path(metadata.get('file_path', '?')).name}:"
              f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
        print(document.content[:250])

    payload = {
        "query": args.query,
        "retriever_id": "iterative_retrieval",
        "documents": [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in documents
        ],
        "iterative": retriever.history,
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()