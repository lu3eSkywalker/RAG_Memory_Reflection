"""Retrieval techniques for the modular RAG pipeline.

Every technique is an importable retriever module:

    semantic_retriever      dense embeddings + FAISS (cosine similarity)
    ast_retriever           structural retrieval over the Python AST
    relationship_index      cross-file relationship graph built from the AST index
    cross_file_retriever    multi-hop, cross-file retrieval over that graph
    iterative_retriever     multi-round retrieval driven by context gaps
    adaptive_retriever      adaptive top-k with explicit stopping decisions
    multiquery_retriever    query expansion + RRF reranking
    bm25_retriever          code-aware BM25 keyword retrieval
    hybrid_retriever        vector + BM25 fusion orchestrator

Every retriever exposes:

    retrieve(query, top_k) -> list[Document]      (raw hits, scored)
    retrieve_result(query) -> RetrievalResult     (rag_core contract)

and can be run directly:

    python -m retrieval.ast_retriever --help
"""

from .semantic_retriever import (
    SemanticRetriever,
    SentenceTransformerEmbedder,
    FaissVectorStore,
)
from .ast_retriever import (
    ASTIndex,
    ASTBasedRetriever,
    CodeElement,
    ElementCollector,
    build_index,
    tokenize,
    split_identifier,
    score_element,
)
from .relationship_index import (
    CrossFileGraph,
    RELATION_WEIGHTS,
    build_relationship_index,
)
from .cross_file_retriever import CrossFileRetriever
from .iterative_retriever import (
    IterativeRetriever,
    analyze_coverage,
    query_concepts,
    tokenize as tokenize_code,
)
from .adaptive_retriever import AdaptiveRetriever
from .multiquery_retriever import MultiQueryRetriever, build_queries
from .bm25_retriever import BM25Index, BM25Retriever, CodeDocument
from .hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalConfig,
    HybridRetrievalResult,
)

__all__ = [
    "SemanticRetriever",
    "SentenceTransformerEmbedder",
    "FaissVectorStore",
    "ASTIndex",
    "ASTBasedRetriever",
    "CodeElement",
    "ElementCollector",
    "build_index",
    "tokenize",
    "split_identifier",
    "score_element",
    "CrossFileGraph",
    "RELATION_WEIGHTS",
    "build_relationship_index",
    "CrossFileRetriever",
    "IterativeRetriever",
    "analyze_coverage",
    "query_concepts",
    "tokenize_code",
    "AdaptiveRetriever",
    "MultiQueryRetriever",
    "build_queries",
    "BM25Index",
    "BM25Retriever",
    "CodeDocument",
    "HybridRetriever",
    "HybridRetrievalConfig",
    "HybridRetrievalResult",
]