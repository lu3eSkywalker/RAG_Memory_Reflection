"""Cross-file retrieval.

Follows the relationship graph instead of staying inside one file:

    User Query
        -> retrieve_user()                     (seed)
             +--> User model                   (references / call)
             +--> Database repository          (imports)
             +--> Authentication utility       (imports / call)
             +--> API route                    (references)
             +--> test_user.py                 (test for)

Seeds come from the semantic retriever (dense) or from structural matching, then
relationships are followed for `--hops` levels with score decay and a per-file cap
so the context stays multi-file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

from retrieval.ast_retriever import ASTIndex  # noqa: E402
from retrieval.relationship_index import (  # noqa: E402
    CrossFileGraph,
    expand,
    seed_from_dense,
    seed_from_structure,
    to_documents,
)
from retrieval.semantic_retriever import SemanticRetriever  # noqa: E402

DEFAULT_QUERY = "authenticate user request"
DEFAULT_AST_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_OUTPUT = PROJECT_ROOT / "cross_file_retrieval_data.json"


class CrossFileRetriever:
    def __init__(
        self,
        index: ASTIndex,
        semantic: Optional[SemanticRetriever] = None,
        hops: int = 2,
        decay: float = 0.6,
        max_per_file: int = 2,
        include_same_file: bool = False,
        include_imports: bool = False,
        seeds: int = 3,
    ) -> None:
        self.index = index
        self.semantic = semantic
        self.hops = hops
        self.decay = decay
        self.max_per_file = max_per_file
        self.seeds = seeds
        self.include_imports = include_imports
        self.graph = CrossFileGraph(index, include_same_file=include_same_file)

    @property
    def ready(self) -> bool:
        return bool(self.index.elements)

    def find_seeds(self, query: str) -> List[Any]:
        seeds = seed_from_dense(self.semantic, self.index, query, self.seeds)
        if not seeds:
            seeds = seed_from_structure(self.index, query, self.seeds)
        return seeds[: self.seeds]

    def retrieve_documents(
        self, query: str, top_k: Optional[int] = None, max_per_file: Optional[int] = None
    ) -> List[Document]:
        seeds = self.find_seeds(query)
        if not seeds:
            return []

        entries, _ = expand(
            graph=self.graph,
            seeds=seeds,
            hops=self.hops,
            decay=self.decay,
            max_results=top_k or (self.max_per_file * 5),
            max_per_file=max_per_file or self.max_per_file,
            include_imports=self.include_imports,
        )
        return to_documents(entries)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="cross_file_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-file (relational) retrieval")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--decay", type=float, default=0.6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-per-file", type=int, default=2)
    parser.add_argument("--ast-index", default=str(DEFAULT_AST_INDEX))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--include-same-file", action="store_true")
    parser.add_argument("--include-imports", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Cross-File Retrieval")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"hops={args.hops} decay={args.decay} seeds={args.seeds} "
          f"top-k={args.top_k} max-per-file={args.max_per_file}")

    ast_index_path = Path(args.ast_index)
    if not ast_index_path.exists():
        print(f"Error: AST index not found: {ast_index_path}")
        print("Run python -m retrieval.ast_retriever first, or pass --ast-index <path>.")
        return

    index = ASTIndex.load(ast_index_path)
    print(f"AST index: {len(index.elements)} units")

    semantic = None
    if not args.no_dense:
        semantic = SemanticRetriever(
            embeddings_file=Path(args.embeddings),
            vector_store_path=Path(args.vector_store),
        )
        if not semantic.ready:
            print("Note: dense artifacts missing; using structural seeds only")
            semantic = None

    retriever = CrossFileRetriever(
        index,
        semantic=semantic,
        hops=args.hops,
        decay=args.decay,
        max_per_file=args.max_per_file,
        include_same_file=args.include_same_file,
        include_imports=args.include_imports,
        seeds=args.seeds,
    )

    stats = retriever.graph.stats()
    print(f"Graph: {stats['files']} files, edges {stats['edges']}")

    seeds = retriever.find_seeds(args.query)
    source = "dense retrieval" if semantic is not None else "structural match"
    print(f"\nSeeds ({len(seeds)}) from {source}:")
    for element, score, reason in seeds:
        print(f"  {score:.3f}  {element.qualified_name} "
              f"({Path(element.file_path).name}:{element.start_line})")
        print(f"         {reason}")

    if not seeds:
        print("\nNo seed matched the query; nothing to expand.")
        return

    entries, relation_stats = expand(
        graph=retriever.graph,
        seeds=seeds,
        hops=args.hops,
        decay=args.decay,
        max_results=args.top_k,
        max_per_file=args.max_per_file,
        include_imports=args.include_imports,
    )
    documents = to_documents(entries)

    files = sorted({doc.metadata["file_path"] for doc in documents})
    print(f"\n{'=' * 60}")
    print(f"Retrieved {len(documents)} units from {len(files)} files")
    print(f"contributions by relation: {relation_stats}")
    print("=" * 60)

    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        print(f"\n--- #{rank} [{metadata['kind']}] hop={metadata['hop']} "
              f"relation={metadata['relation']} score={metadata['score']} ---")
        print(f"{metadata['qualified_name']}  ({Path(metadata['file_path']).name}:"
              f"{metadata['start_line']}-{metadata['end_line']})")
        print(f"path: {metadata['path']}")
        print(document.content[:220])

    payload = {
        "query": args.query,
        "retriever_id": "cross_file_retrieval",
        "documents": [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in documents
        ],
        "cross_file": {
            "hops": args.hops,
            "decay": args.decay,
            "graph_stats": stats,
            "relation_contributions": relation_stats,
            "files": files,
            "seeds": [
                {"element_id": e.element_id, "score": round(s, 6), "reason": r}
                for e, s, r in seeds
            ],
        },
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()