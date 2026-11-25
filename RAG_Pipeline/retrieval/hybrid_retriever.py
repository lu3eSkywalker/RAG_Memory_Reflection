"""Hybrid retrieval (semantic + BM25).

    User Query
        +-------------------+
        |                   |
        v                   v
  Semantic Retriever   BM25 Retriever        (reused, not re-implemented)
        |                   |
        +---------+---------+
                  v
        Merge -> Deduplicate -> Normalise -> Fuse -> Top-K

Design decisions
  * candidate_k = top_k * candidate_multiplier (default 10 * 2 = 20 candidates)
  * dedup on the stable id "file_path:start_line:end_line", keeping every source's
    score
  * min-max normalisation per source (scores live on different scales), then
    weighted fusion (0.6 semantic / 0.4 BM25 by default) or Reciprocal Rank
    Fusion as an alternative
  * fallback: one source failing must not break the pipeline - the remaining
    source(s) carry the result, and weights are renormalised

Optional sources (all reuse existing retrievers):
  --with-ast         AST structural retrieval
  --with-cross-file  cross-file relationship expansion
  --use-multiquery   run every source over expanded queries
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

from retrieval.ast_retriever import ASTBasedRetriever, ASTIndex  # noqa: E402
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.cross_file_retriever import CrossFileRetriever  # noqa: E402
from retrieval.semantic_retriever import SemanticRetriever  # noqa: E402

DEFAULT_QUERY = "Why is validate_jwt_token failing with ExpiredSignatureError?"
DEFAULT_CHUNKS = PROJECT_ROOT / "chunks.json"
DEFAULT_AST_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_OUTPUT = PROJECT_ROOT / "hybrid_retrieval_data.json"


# ============================================================
# Config / results
# ============================================================

@dataclass
class HybridRetrievalConfig:
    top_k: int = 10
    candidate_multiplier: int = 2
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    normalization: str = "minmax"
    fusion_method: str = "weighted_score"
    rrf_k: int = 60
    ast_weight: float = 0.3
    cross_file_weight: float = 0.3


@dataclass
class HybridRetrievalResult:
    document_id: str
    file_path: str
    content: str
    vector_score: Optional[float]
    bm25_score: Optional[float]
    hybrid_score: float
    retrieval_methods: List[str]
    extra_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    matched_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridRetrievalResponse:
    results: List[HybridRetrievalResult]
    source_status: Dict[str, str]
    normalised_weights: Dict[str, float]
    candidate_k: int
    queries: List[Dict[str, str]]


class RetrievalAdapter:
    """Uniform adapter so any retriever can feed the fusion."""

    def __init__(self, name: str, retriever: Any) -> None:
        self.name = name
        self.retriever = retriever

    @property
    def ready(self) -> bool:
        return bool(getattr(self.retriever, "ready", True))

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        return self.retriever.retrieve_documents(query, top_k)


# ============================================================
# Fusion
# ============================================================

class HybridRetriever:
    def __init__(
        self,
        vector_retriever: Optional[Any] = None,
        bm25_retriever: Optional[Any] = None,
        config: Optional[HybridRetrievalConfig] = None,
        extra_retrievers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or HybridRetrievalConfig()
        self.retrievers: Dict[str, RetrievalAdapter] = {}

        if vector_retriever is not None:
            self.retrievers["vector"] = RetrievalAdapter("vector", vector_retriever)
        if bm25_retriever is not None:
            self.retrievers["bm25"] = RetrievalAdapter("bm25", bm25_retriever)
        for name, retriever in (extra_retrievers or {}).items():
            if retriever is not None:
                self.retrievers[name] = RetrievalAdapter(name, retriever)

    # -- helpers -------------------------------------------------------
    @property
    def ready(self) -> bool:
        """True when at least one source retriever can run."""
        return any(adapter.ready for adapter in self.retrievers.values())

    @staticmethod
    def document_id(metadata: Dict[str, Any]) -> str:
        return "{}:{}:{}".format(
            metadata.get("file_path", "?"),
            metadata.get("start_line", "?"),
            metadata.get("end_line", "?"),
        )

    def _normalised_weights(self, active: Sequence[str]) -> Dict[str, float]:
        raw = {
            "vector": self.config.vector_weight,
            "bm25": self.config.bm25_weight,
            "ast": self.config.ast_weight,
            "cross_file": self.config.cross_file_weight,
        }
        weights = {name: raw.get(name, 0.3) for name in active}
        total = sum(weights.values())
        if total <= 0:
            return {name: round(1.0 / len(active), 4) for name in active}
        return {name: round(weight / total, 4) for name, weight in weights.items()}

    # -- retrieval -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        queries: Optional[Sequence[Dict[str, str]]] = None,
    ) -> HybridRetrievalResponse:
        config = self.config
        top_k = top_k or config.top_k
        candidate_k = max(top_k * config.candidate_multiplier, top_k)
        queries = list(queries) if queries else [{"method": "original", "query": query}]

        # Each hit remembers which query produced it.
        source_hits: Dict[str, List[Tuple[str, Document]]] = {}
        source_status: Dict[str, str] = {}

        for name, adapter in self.retrievers.items():
            if not adapter.ready:
                source_status[name] = "unavailable"
                continue
            collected: List[Tuple[str, Document]] = []
            try:
                for entry in queries:
                    for document in adapter.retrieve(entry["query"], candidate_k):
                        collected.append((entry["query"], document))
                source_status[name] = "ok" if collected else "no results"
            except Exception as error:  # one failing source must not break the pipeline
                source_status[name] = f"failed: {error}"
            source_hits[name] = collected

        active = [name for name in self.retrievers if source_hits.get(name)]
        if not active:
            return HybridRetrievalResponse(
                results=[],
                source_status=source_status,
                normalised_weights={},
                candidate_k=candidate_k,
                queries=queries,
            )

        # Deduplicate on the stable document id, keeping every source score.
        buckets: Dict[str, Dict[str, Any]] = {}
        for name in active:
            for query_text, document in source_hits[name]:
                metadata = document.metadata or {}
                key = self.document_id(metadata)
                bucket = buckets.setdefault(
                    key,
                    {
                        "file_path": str(metadata.get("file_path", "?")),
                        "start_line": metadata.get("start_line"),
                        "end_line": metadata.get("end_line"),
                        "content": document.content,
                        "scores": {},
                        "ranks": {},
                        "methods": [],
                        "matched_queries": [],
                        "metadata": metadata,
                    },
                )
                score = float(document.score or 0.0)
                if name not in bucket["scores"] or score > bucket["scores"][name]:
                    bucket["scores"][name] = score
                if query_text not in bucket["matched_queries"]:
                    bucket["matched_queries"].append(query_text)

        # Per-source min-max normalisation.
        normalised: Dict[str, Dict[float, float]] = {name: {} for name in active}
        for name in active:
            values = [bucket["scores"][name] for bucket in buckets.values()
                      if name in bucket["scores"]]
            if config.normalization == "minmax" and len(set(values)) > 1:
                low, high = min(values), max(values)
                for bucket in buckets.values():
                    if name in bucket["scores"]:
                        normalised[name][bucket["scores"][name]] = (
                            (bucket["scores"][name] - low) / (high - low)
                        )
            else:
                for bucket in buckets.values():
                    if name in bucket["scores"]:
                        normalised[name][bucket["scores"][name]] = 1.0

        # Ranks (needed for RRF).
        for name in active:
            ordered = sorted(
                (bucket["scores"][name] for bucket in buckets.values()
                 if name in bucket["scores"]),
                reverse=True,
            )
            rank_of = {score: index + 1 for index, score in enumerate(ordered)}
            for bucket in buckets.values():
                if name in bucket["scores"]:
                    bucket["ranks"][name] = rank_of[bucket["scores"][name]]
                    if name not in bucket["methods"]:
                        bucket["methods"].append(name)

        weights = self._normalised_weights(active)
        results: List[HybridRetrievalResult] = []

        for key, bucket in buckets.items():
            if config.fusion_method == "rrf":
                hybrid = sum(
                    1.0 / (config.rrf_k + bucket["ranks"].get(name, config.rrf_k + top_k))
                    for name in bucket["methods"]
                )
            else:
                hybrid = sum(
                    weights.get(name, 0.0) * normalised[name].get(bucket["scores"][name], 0.0)
                    for name in bucket["methods"]
                )

            results.append(
                HybridRetrievalResult(
                    document_id=key,
                    file_path=bucket["file_path"],
                    content=bucket["content"],
                    vector_score=bucket["scores"].get("vector"),
                    bm25_score=bucket["scores"].get("bm25"),
                    hybrid_score=round(float(hybrid), 6),
                    retrieval_methods=list(bucket["methods"]),
                    extra_scores={
                        name: bucket["scores"].get(name)
                        for name in bucket["methods"]
                        if name not in {"vector", "bm25"}
                    },
                    matched_queries=bucket["matched_queries"],
                    metadata=bucket["metadata"],
                )
            )

        results.sort(key=lambda result: result.hybrid_score, reverse=True)
        return HybridRetrievalResponse(
            results=results[:top_k],
            source_status=source_status,
            normalised_weights=weights,
            candidate_k=candidate_k,
            queries=queries,
        )

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        response = self.retrieve(query, top_k)
        documents = []
        for result in response.results:
            documents.append(
                Document(
                    content=result.content,
                    metadata={
                        "file_path": result.file_path,
                        "document_id": result.document_id,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "extra_scores": result.extra_scores,
                        "hybrid_score": result.hybrid_score,
                        "retrieval_methods": result.retrieval_methods,
                        "matched_queries": result.matched_queries,
                        **{k: v for k, v in result.metadata.items()
                           if k not in {"file_path"}},
                    },
                    score=result.hybrid_score,
                )
            )
        return documents

    def retrieve_result(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="hybrid_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid semantic + BM25 retrieval")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--ast-weight", type=float, default=0.3)
    parser.add_argument("--cross-file-weight", type=float, default=0.3)
    parser.add_argument("--normalization", choices=["minmax", "none"], default="minmax")
    parser.add_argument("--fusion", choices=["weighted_score", "rrf"], default="weighted_score")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--corpus", choices=["chunks", "ast", "both"], default="chunks")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--ast-index", default=str(DEFAULT_AST_INDEX))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--with-ast", action="store_true")
    parser.add_argument("--with-cross-file", action="store_true")
    parser.add_argument("--use-multiquery", action="store_true")
    parser.add_argument("--query-methods", default="keywords,patterns,subq")
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    config = HybridRetrievalConfig(
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        normalization=args.normalization,
        fusion_method=args.fusion,
        rrf_k=args.rrf_k,
        ast_weight=args.ast_weight,
        cross_file_weight=args.cross_file_weight,
    )

    print("=" * 60)
    print("Hybrid Retrieval (semantic + BM25)")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"top_k={config.top_k} candidate_k={config.top_k * config.candidate_multiplier} "
          f"fusion={config.fusion_method} normalization={config.normalization}")

    index = None
    if Path(args.ast_index).exists():
        index = ASTIndex.load(Path(args.ast_index))

    vector = None
    if not args.no_vector:
        candidate = SemanticRetriever(
            embeddings_file=Path(args.embeddings),
            vector_store_path=Path(args.vector_store),
        )
        if candidate.ready:
            vector = candidate
        else:
            print("Note: dense artifacts missing; falling back to keyword retrieval only")

    bm25 = None
    if not args.no_bm25:
        bm25 = BM25Retriever(
            corpus=args.corpus,
            chunks_file=Path(args.chunks),
            ast_index_file=Path(args.ast_index),
            top_k=max(args.top_k, 5),
        )

    extras: Dict[str, Any] = {}
    if args.with_ast and index is not None:
        extras["ast"] = ASTBasedRetriever(index, top_k=args.top_k, hybrid=False)
    if args.with_cross_file and index is not None:
        extras["cross_file"] = CrossFileRetriever(
            index, semantic=vector, hops=1, max_per_file=2
        )

    retriever = HybridRetriever(vector, bm25, config, extra_retrievers=extras)
    if not retriever.retrievers:
        print("Error: no retriever available (need dense artifacts and/or a corpus).")
        return

    queries = None
    if args.use_multiquery:
        from retrieval.multiquery_retriever import build_queries

        queries = build_queries(args.query, args.query_methods.split(","),
                                args.num_queries, index)
        print(f"\nQuery expansion produced {len(queries)} queries:")
        for position, item in enumerate(queries, start=1):
            print(f"  {position:>2}. [{item['method']:<8}] {item['query']}")

    response = retriever.retrieve(args.query, top_k=config.top_k, queries=queries)

    print(f"\n{'=' * 60}")
    print("Retrievers")
    print("=" * 60)
    for name, status in response.source_status.items():
        print(f"  {name:<11} {status}")
    print(f"  weights     {response.normalised_weights}")

    if not response.results:
        print("\nNo results: every retriever failed or returned nothing.")
        return

    print(f"\n{'=' * 60}")
    print(f"Top {len(response.results)} hybrid results")
    print("=" * 60)
    for rank, result in enumerate(response.results, start=1):
        print(f"\n--- #{rank} hybrid={result.hybrid_score:.4f} "
              f"methods={result.retrieval_methods} ---")
        print(result.document_id)
        print(f"vector={result.vector_score} bm25={result.bm25_score} "
              f"extra={result.extra_scores}")
        if result.matched_queries:
            print(f"queries: {result.matched_queries[:3]}")

    payload = {
        "query": args.query,
        "retriever_id": "hybrid_retrieval",
        "documents": [
            {
                "content": result.content,
                "metadata": {
                    "file_path": result.file_path,
                    "document_id": result.document_id,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                    "extra_scores": result.extra_scores,
                    "hybrid_score": result.hybrid_score,
                    "retrieval_methods": result.retrieval_methods,
                    "matched_queries": result.matched_queries,
                    **{k: v for k, v in result.metadata.items() if k != "file_path"},
                },
                "score": result.hybrid_score,
            }
            for result in response.results
        ],
        "hybrid": {
            "config": {
                "top_k": config.top_k,
                "candidate_k": response.candidate_k,
                "fusion_method": config.fusion_method,
                "normalization": config.normalization,
                "vector_weight": config.vector_weight,
                "bm25_weight": config.bm25_weight,
                "ast_weight": config.ast_weight,
                "cross_file_weight": config.cross_file_weight,
                "rrf_k": config.rrf_k,
                "normalised_weights": response.normalised_weights,
                "source_status": response.source_status,
                "queries": response.queries,
            }
        },
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()