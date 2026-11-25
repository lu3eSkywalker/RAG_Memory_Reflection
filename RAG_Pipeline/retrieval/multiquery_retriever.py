"""Multi-query retrieval (query expansion).

One user query rarely contains the exact terminology a repository uses, so a
single embedding lookup wastes recall. This module expands the query into several
searches, retrieves for each, merges, deduplicates and reranks:

    User Query
        -> Query Expansion  (Query 1..N)
        -> Retrieve per query
        -> Merge (Reciprocal Rank Fusion)
        -> Deduplicate (same file + overlapping lines)
        -> Rerank (fusion score + query-term coverage)

Expansion methods:
    keywords  salient terms only
    patterns  task-pattern templates (fix / add / test / optimise ...)
    vocab     repository vocabulary mined from the AST index
    subq      sub-question style phrasings
    gemini    LLM expansion (GEMINI_API_KEY in the environment, never hardcoded)
"""

import argparse
import json
import os
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
from retrieval.iterative_retriever import query_concepts, split_identifier  # noqa: E402
from retrieval.semantic_retriever import SemanticRetriever  # noqa: E402

DEFAULT_QUERY = "Fix the authentication bug in the login API"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_AST_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "multiquery_retrieval_data.json"

RRF_K = 60

EXPANSION_STOPWORDS = {
    "fix", "bug", "bugs", "error", "errors", "exception", "refactor",
    "optimize", "optimise", "improve", "remove", "delete", "debug",
    "please", "need", "want", "issue", "issues", "using", "make", "sure",
}

TASK_PATTERNS = {
    "fix": ["{terms} error handling", "{terms} bug fix", "{terms} failure path"],
    "bug": ["{terms} bug", "{terms} error trace", "{terms} incorrect behaviour"],
    "error": ["{terms} exception", "{terms} traceback", "{terms} error handling"],
    "add": ["{terms} implementation", "{terms} definition", "{terms} usage example"],
    "implement": ["{terms} implementation", "{terms} helper", "{terms} interface"],
    "create": ["{terms} class", "{terms} factory", "{terms} definition"],
    "test": ["{terms} unit test", "{terms} test case", "{terms} assertion"],
    "optimize": ["{terms} performance", "{terms} caching", "{terms} bottleneck"],
    "refactor": ["{terms} refactoring", "{terms} helper", "{terms} structure"],
    "remove": ["{terms} usage", "{terms} dependency", "{terms} cleanup"],
}

SUB_QUESTION_TEMPLATES = [
    "how is {terms} implemented",
    "where is {terms} used",
    "{terms} definition and signature",
    "{terms} related tests",
    "dependencies of {terms}",
]


def unique(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(item.strip())
    return output


# ============================================================
# Expansion
# ============================================================

def salient_terms(query: str, limit: int = 6) -> List[str]:
    concepts = [
        concept
        for concept in query_concepts(query)
        if concept.lower() not in EXPANSION_STOPWORDS
    ]
    return concepts[:limit]


def expansion_keywords(query: str, limit: int) -> List[str]:
    terms = salient_terms(query)
    return [" ".join(terms)] if terms else [query]


def expansion_patterns(query: str, limit: int) -> List[str]:
    terms = salient_terms(query)
    if not terms:
        return [query]

    joined = " ".join(terms)
    queries: List[str] = []
    for word in query.lower().split():
        for template in TASK_PATTERNS.get(word, []):
            queries.append(template.format(terms=joined))
    if not queries:
        queries = [template.format(terms=joined) for template in SUB_QUESTION_TEMPLATES]
    return unique(queries)[:limit]


def expansion_subquestions(query: str, limit: int) -> List[str]:
    terms = salient_terms(query)
    if not terms:
        return [query]
    joined = " ".join(terms)
    return [template.format(terms=joined) for template in SUB_QUESTION_TEMPLATES][:limit]


def repository_vocabulary(index: ASTIndex, query: str, limit: int) -> List[str]:
    """Expand with identifiers that actually exist in the repository."""
    from retrieval.ast_retriever import score_element, tokenize

    query_tokens = tokenize(query)
    scored: List[Tuple[str, float]] = []
    for element in index.elements:
        if element.kind not in {"function", "method", "class", "variable"}:
            continue
        name_tokens = tokenize(element.name)
        if not name_tokens:
            continue
        overlap = len(query_tokens & name_tokens) / max(len(query_tokens), 1)
        if overlap <= 0:
            continue
        scored.append((element.name, overlap + 0.01 * len(element.source) / 1000))

    scored.sort(key=lambda entry: entry[1], reverse=True)
    names = unique([name for name, _ in scored])
    hints = " ".join(salient_terms(query, limit=3))
    return [f"{name} {hints}".strip() for name in names[:limit]]


def expansion_gemini(query: str, limit: int) -> List[str]:
    """LLM expansion; requires GEMINI_API_KEY in the environment."""
    import gemini_client

    if not gemini_client.available():
        print("Note: GEMINI_API_KEY not set; skipping gemini expansion")
        return []

    prompt = (
        "Rewrite the following software task as "
        f"{limit} short code search queries. One per line, no numbering.\n\n"
        f"Task: {query}"
    )
    try:
        text = gemini_client.generate(prompt, temperature=0.4, max_tokens=512)
    except Exception as error:  # network/API problems must not break retrieval
        print(f"Note: gemini expansion failed ({error}); continuing without it")
        return []

    lines = [line.strip("-* ").strip() for line in text.splitlines()]
    return [line for line in lines if line][:limit]


def build_queries(
    query: str,
    methods: Sequence[str],
    limit: int = 4,
    index: Optional[ASTIndex] = None,
) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = [{"method": "original", "query": query}]

    for method in methods:
        if method == "keywords":
            values = expansion_keywords(query, limit)
        elif method == "patterns":
            values = expansion_patterns(query, limit)
        elif method == "subq":
            values = expansion_subquestions(query, limit)
        elif method == "vocab":
            if index is None:
                print("Note: vocab expansion needs an AST index; skipping")
                continue
            values = repository_vocabulary(index, query, limit)
        elif method == "gemini":
            values = expansion_gemini(query, limit)
        else:
            print(f"Note: unknown expansion method '{method}' ignored")
            continue
        queries.extend({"method": method, "query": value} for value in values if value.strip())

    return _dedupe(queries)


def _dedupe(queries: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    output: List[Dict[str, str]] = []
    for item in queries:
        key = item["query"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(item)
    return output


# ============================================================
# Merge / dedup / rerank
# ============================================================

def same_unit(left: Document, right: Document) -> bool:
    left_meta = left.metadata or {}
    right_meta = right.metadata or {}
    if left_meta.get("file_path") != right_meta.get("file_path"):
        return False

    left_start, left_end = left_meta.get("start_line"), left_meta.get("end_line")
    right_start, right_end = right_meta.get("start_line"), right_meta.get("end_line")
    if isinstance(left_start, int) and isinstance(right_start, int):
        return left_start <= right_end and right_start <= left_end
    return left.content.strip() == right.content.strip()


def rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def merge_and_fuse(
    per_query: Sequence[Dict[str, Any]],
    fusion: str = "rrf",
    rrf_k: int = RRF_K,
    coverage_weight: float = 0.25,
    concepts: Optional[Sequence[str]] = None,
) -> Tuple[List[Document], Dict[str, int]]:
    """Fuse multi-query rankings, dedupe overlapping units, rerank by coverage."""
    concepts = list(concepts or [])
    fused: Dict[int, Dict[str, Any]] = {}
    next_key = 0
    duplicates = 0
    references = 0

    for entry in per_query:
        documents: List[Document] = entry["documents"]
        top = max((d.score or 0.0 for d in documents), default=0.0) or 1.0
        for rank, document in enumerate(documents, start=1):
            references += 1
            contribution = rrf(rank, rrf_k) if fusion == "rrf" else (document.score or 0.0) / top

            duplicate = False
            for bucket in fused.values():
                if same_unit(bucket["document"], document):
                    duplicates += 1
                    duplicate = True
                    bucket["fusion"] += contribution
                    bucket["queries"].append(entry["query"])
                    bucket["ranks"].append(rank)
                    bucket["best_rank"] = min(bucket["best_rank"], rank)
                    break
            if not duplicate:
                fused[next_key] = {
                    "document": document,
                    "fusion": contribution,
                    "queries": [entry["query"]],
                    "ranks": [rank],
                    "best_rank": rank,
                }
                next_key += 1

    max_fusion = max((bucket["fusion"] for bucket in fused.values()), default=0.0) or 1.0
    documents: List[Document] = []

    for bucket in fused.values():
        document = bucket["document"]
        fusion_norm = min(1.0, bucket["fusion"] / max_fusion)

        doc_tokens: Set[str] = set()
        for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", document.content):
            doc_tokens.update(split_identifier(raw))
        coverage = (
            sum(
                1 for concept in concepts
                if set(split_identifier(concept)) & doc_tokens
            ) / len(concepts)
            if concepts
            else 0.0
        )

        final = fusion_norm + coverage_weight * coverage
        document.metadata = {
            **(document.metadata or {}),
            "matched_queries": unique(bucket["queries"]),
            "num_queries_matched": len(bucket["queries"]),
            "rank_per_query": bucket["ranks"],
            "best_rank": bucket["best_rank"],
            "fusion": round(bucket["fusion"], 6),
            "coverage": round(coverage, 4),
            "rerank_score": round(final, 6),
        }
        document.score = round(final, 6)
        documents.append(document)

    documents.sort(key=lambda d: (d.metadata or {}).get("rerank_score", 0.0), reverse=True)
    return documents, {
        "references": references,
        "duplicates_merged": duplicates,
        "unique_units": len(documents),
    }


# ============================================================
# Retriever
# ============================================================

class MultiQueryRetriever:
    def __init__(
        self,
        semantic: Optional[SemanticRetriever] = None,
        structural: Optional[ASTBasedRetriever] = None,
        methods: Sequence[str] = ("keywords", "patterns", "subq"),
        num_queries: int = 4,
        top_k: int = 5,
        fusion: str = "rrf",
        rrf_k: int = RRF_K,
        coverage_weight: float = 0.25,
        index: Optional[ASTIndex] = None,
    ) -> None:
        self.semantic = semantic
        self.structural = structural
        self.methods = list(methods)
        self.num_queries = num_queries
        self.top_k = top_k
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.coverage_weight = coverage_weight
        self.index = index
        self.queries: List[Dict[str, str]] = []
        self.merge_stats: Dict[str, int] = {}

    @property
    def ready(self) -> bool:
        return bool(
            (self.semantic is not None and self.semantic.ready)
            or (self.structural is not None and self.structural.ready)
        )

    def _retrieve_one(self, query: str, top_k: int) -> List[Document]:
        documents: List[Document] = []
        if self.semantic is not None and self.semantic.ready:
            documents += self.semantic.retrieve_documents(query, top_k)
        if self.structural is not None and self.structural.ready:
            documents += self.structural.retrieve_documents(query, top_k)
        return documents

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.top_k
        self.queries = build_queries(query, self.methods, self.num_queries, self.index)
        concepts = query_concepts(query)

        per_query = []
        for item in self.queries:
            documents = self._retrieve_one(item["query"], top_k)
            per_query.append({**item, "documents": documents})

        documents, stats = merge_and_fuse(
            per_query=per_query,
            fusion=self.fusion,
            rrf_k=self.rrf_k,
            coverage_weight=self.coverage_weight,
            concepts=concepts,
        )
        self.merge_stats = stats
        return documents[: max(top_k, self.num_queries * 2)]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="multi_query_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-query (expanded) retrieval")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--method", default="keywords,patterns,vocab,subq")
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fusion", choices=["rrf", "max"], default="rrf")
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--ast-index", default=str(DEFAULT_AST_INDEX))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Multi-Query Retrieval / Query Expansion")
    print("=" * 60)
    print(f"Query: {args.query}")

    index = None
    ast_index_path = Path(args.ast_index)
    if ast_index_path.exists():
        index = ASTIndex.load(ast_index_path)

    semantic = None
    if not args.no_dense:
        candidate = SemanticRetriever(
            embeddings_file=Path(args.embeddings),
            vector_store_path=Path(args.vector_store),
        )
        if candidate.ready:
            semantic = candidate
        else:
            print("Note: dense artifacts missing; using AST lexical retrieval")

    structural = None
    if index is not None:
        structural = ASTBasedRetriever(index, top_k=args.top_k, hybrid=False)

    if semantic is None and structural is None:
        print("Error: need either dense artifacts (02/03) or an AST index.")
        return

    retriever = MultiQueryRetriever(
        semantic=semantic,
        structural=structural,
        methods=args.method.split(","),
        num_queries=args.num_queries,
        top_k=args.top_k,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        coverage_weight=args.coverage_weight,
        index=index,
    )

    documents = retriever.retrieve_documents(args.query)

    print(f"\nGenerated {len(retriever.queries)} queries")
    for position, item in enumerate(retriever.queries, start=1):
        print(f"  {position:>2}. [{item['method']:<8}] {item['query']}")

    print(f"\n{'=' * 60}")
    print("Merge / dedup")
    print("=" * 60)
    print(f"query results    : {retriever.merge_stats.get('references', 0)}")
    print(f"duplicates merged: {retriever.merge_stats.get('duplicates_merged', 0)}")
    print(f"unique units     : {retriever.merge_stats.get('unique_units', 0)}")

    print(f"\n{'=' * 60}")
    print(f"Reranked context ({len(documents)} units)")
    print("=" * 60)
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        print(f"\n--- #{rank} score={document.score} "
              f"(fusion {metadata.get('fusion')}, coverage {metadata.get('coverage')}, "
              f"matched {metadata.get('num_queries_matched')} queries) ---")
        print(f"{Path(metadata.get('file_path', '?')).name}:"
              f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
        print(f"best rank: {metadata.get('best_rank')}  ranks: {metadata.get('rank_per_query')}")
        for matched in (metadata.get("matched_queries") or [])[:3]:
            print(f"  via: {matched}")

    payload = {
        "query": args.query,
        "retriever_id": "multi_query_retrieval",
        "documents": [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in documents
        ],
        "multi_query": {
            "queries": retriever.queries,
            "fusion": args.fusion,
            "rrf_k": args.rrf_k,
            "coverage_weight": args.coverage_weight,
            "merge_stats": retriever.merge_stats,
        },
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()