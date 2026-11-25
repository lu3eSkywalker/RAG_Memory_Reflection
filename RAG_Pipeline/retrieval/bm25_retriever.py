"""Code-aware BM25 keyword retrieval.

Dense retrieval misses exact identifiers. BM25 is strong when the query names a
symbol, an error class or a file. Pure Python, no extra dependencies.

Implemented:
  Phase 1  BM25 (k1/b) over code documents -> top-k
  Phase 2  Code-aware tokenization: snake_case, camelCase, PascalCase, file
           paths, symbols, Error/Exception names
  Phase 3  Exact-match boosting: identifier / symbol / file-name / error / path
  Phase 4  Result metadata: matched_terms, retrieval_method, document_id, boosts
  Phase 5  Hybrid retrieval: BM25 + semantic fusion via RRF
  Phase 6  Multi-query integration: BM25 over expanded queries
  Phase 7  AST integration: --corpus ast|both adds AST code units to the corpus

Corpus sources:
  chunks   chunks.json     (sliding-window / fixed / semantic chunks)
  ast      ast_index.json  (functions, classes, methods, ...)
  both     union of the two
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

DEFAULT_QUERY = "Why is validate_jwt_token failing with ExpiredSignatureError?"
DEFAULT_CHUNKS = PROJECT_ROOT / "chunks.json"
DEFAULT_AST_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_OUTPUT = PROJECT_ROOT / "bm25_retrieval_data.json"

RRF_K = 60

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")
ERROR_SUFFIXES = ("Error", "Exception", "Warning", "Fault", "Err")

QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "is", "are", "why", "how", "what", "does", "do", "did", "i", "it", "this",
    "that", "be", "can", "should", "would", "there", "here", "my", "me",
    "we", "you", "not", "but", "if", "then", "than", "so", "as", "at", "by",
    "from", "into", "use", "using", "used", "get", "when", "while",
}


# ============================================================
# Phase 2: code-aware tokenisation
# ============================================================

def split_identifier(identifier: str) -> List[str]:
    return [part.lower() for part in CAMEL_RE.findall(identifier)] or [identifier.lower()]


def is_error_name(identifier: str) -> bool:
    return any(identifier.endswith(suffix) for suffix in ERROR_SUFFIXES)


def path_tokens(file_path: str) -> List[str]:
    tokens: List[str] = []
    for part in str(file_path).replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        stem = Path(part).stem
        tokens.append(part.lower())
        tokens.extend(split_identifier(stem))
        if stem != part:
            tokens.append(stem.lower())
    return tokens


def tokenize_code(text: str) -> List[str]:
    """Keep the full identifier *and* its camel/snake parts."""
    tokens: List[str] = []
    for identifier in IDENTIFIER_RE.findall(text or ""):
        lowered = identifier.lower()
        if lowered in QUERY_STOPWORDS:
            continue
        tokens.append(lowered)
        parts = split_identifier(identifier)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def query_terms(query: str) -> List[str]:
    """Verbatim query identifiers (used for exact-match boosting)."""
    terms: List[str] = []
    for identifier in IDENTIFIER_RE.findall(query or ""):
        lowered = identifier.lower()
        if lowered in QUERY_STOPWORDS or len(lowered) < 2:
            continue
        if lowered not in terms:
            terms.append(lowered)
    return terms


def query_tokens(query: str) -> List[str]:
    tokens: List[str] = []
    for term in query_terms(query):
        tokens.append(term)
        tokens.extend(split_identifier(term))
    return tokens


# ============================================================
# Corpus
# ============================================================

def make_document_id(file_path: str, start: Any, end: Any, content: str) -> str:
    digest = hashlib.sha1(
        f"{file_path}:{start}:{end}:{content[:200]}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


class CodeDocument:
    def __init__(self, document_id: str, content: str, metadata: Dict[str, Any]) -> None:
        self.document_id = document_id
        self.content = content
        self.metadata = metadata
        self.file_path = str(metadata.get("file_path", ""))

        self.symbols = sorted({t for t in tokenize_code(content) if "_" in t or len(t) > 3})
        self.errors = sorted({t for t in self.symbols if is_error_name(t)})
        self.path_terms = set(path_tokens(self.file_path))
        self.file_name = Path(self.file_path).name.lower()

        self.tokens = tokenize_code(content) + path_tokens(self.file_path)
        self.length = len(self.tokens) or 1
        self.frequencies: Counter = Counter(self.tokens)

    def to_document(self, score: float, extra: Dict[str, Any]) -> Document:
        return Document(
            content=self.content,
            metadata={
                "file_path": self.file_path,
                "document_id": self.document_id,
                "start_line": self.metadata.get("start_line"),
                "end_line": self.metadata.get("end_line"),
                **{k: v for k, v in self.metadata.items()
                   if k not in {"file_path", "start_line", "end_line"}},
                **extra,
            },
            score=round(score, 6),
        )


def load_chunks_corpus(chunks_file: Path) -> List[CodeDocument]:
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        print(f"Note: {chunks_file.name} not found; skipping chunk corpus")
        return []

    with open(chunks_file, "r") as handle:
        chunks = json.load(handle)

    corpus = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        file_path = str(metadata.get("file_path", ""))
        corpus.append(
            CodeDocument(
                document_id=make_document_id(
                    file_path, metadata.get("start_line"), metadata.get("end_line"), content
                ),
                content=content,
                metadata=metadata,
            )
        )
    return corpus


def load_ast_corpus(ast_index_file: Path) -> List[CodeDocument]:
    from retrieval.ast_retriever import ASTIndex

    ast_index_file = Path(ast_index_file)
    if not ast_index_file.exists():
        print(f"Note: {ast_index_file.name} not found; skipping AST corpus")
        return []

    index = ASTIndex.load(ast_index_file)
    corpus = []
    for element in index.elements:
        corpus.append(
            CodeDocument(
                document_id=make_document_id(
                    element.file_path, element.start_line, element.end_line, element.source
                ),
                content=element.source,
                metadata={
                    "file_path": element.file_path,
                    "start_line": element.start_line,
                    "end_line": element.end_line,
                    "element_id": element.element_id,
                    "kind": element.kind,
                    "qualified_name": element.qualified_name,
                },
            )
        )
    return corpus


def dedupe_corpus(documents: Sequence[CodeDocument]) -> List[CodeDocument]:
    seen: set = set()
    unique_documents = []
    for document in documents:
        if document.document_id in seen:
            continue
        seen.add(document.document_id)
        unique_documents.append(document)
    return unique_documents


def build_corpus(
    corpus: str = "chunks",
    chunks_file: Path = DEFAULT_CHUNKS,
    ast_index_file: Path = DEFAULT_AST_INDEX,
) -> List[CodeDocument]:
    documents: List[CodeDocument] = []
    if corpus in {"chunks", "both"}:
        documents += load_chunks_corpus(chunks_file)
    if corpus in {"ast", "both"}:
        documents += load_ast_corpus(ast_index_file)
    return dedupe_corpus(documents)


# ============================================================
# Phase 1/3/4: BM25 index
# ============================================================

class BM25Index:
    def __init__(self, documents: Sequence[CodeDocument], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = list(documents)
        self.count = len(self.documents)
        self.average_length = (
            sum(document.length for document in self.documents) / self.count
            if self.count
            else 0.0
        )
        self.document_frequency: Counter = Counter()
        for document in self.documents:
            for token in set(document.tokens):
                self.document_frequency[token] += 1

    def idf(self, token: str) -> float:
        df = self.document_frequency.get(token, 0)
        return math.log(1 + (self.count - df + 0.5) / (df + 0.5))

    def bm25(self, document: CodeDocument, terms: Sequence[str]) -> float:
        if not self.average_length:
            return 0.0
        length_norm = self.k1 * (1 - self.b + self.b * (document.length / self.average_length))
        score = 0.0
        for term in terms:
            frequency = document.frequencies.get(term, 0)
            if frequency:
                score += self.idf(term) * (frequency * (self.k1 + 1)) / (frequency + length_norm)
        return score

    def search(
        self,
        query: str,
        top_k: int,
        boost_identifier: float = 1.8,
        boost_symbol: float = 1.5,
        boost_filename: float = 2.5,
        boost_error: float = 2.0,
        boost_path: float = 1.2,
    ) -> List[Tuple[CodeDocument, float, Dict[str, Any]]]:
        terms = query_tokens(query)
        exact_terms = set(query_terms(query))
        results: List[Tuple[CodeDocument, float, Dict[str, Any]]] = []

        for document in self.documents:
            score = self.bm25(document, terms)
            matched_terms = sorted({t for t in terms if document.frequencies.get(t)})
            exact_hits = sorted(exact_terms & set(document.tokens))
            if not score and not exact_hits:
                continue

            boosts: Dict[str, float] = {}
            detail: Dict[str, List[str]] = defaultdict(list)
            for term in exact_hits:
                if term in document.errors:
                    boosts["error_name"] = boost_error
                    detail["error_name"].append(term)
                elif term in document.symbols:
                    boosts["identifier"] = max(boosts.get("identifier", 0), boost_identifier)
                    detail["identifier"].append(term)
                elif term in document.path_terms:
                    boosts["file_path"] = max(boosts.get("file_path", 0), boost_path)
                    detail["file_path"].append(term)

            if document.file_name:
                stem_tokens = set(split_identifier(Path(document.file_name).stem))
                if exact_terms & stem_tokens:
                    boosts["file_name"] = boost_filename
                    detail["file_name"].append(document.file_name)

            base_idf = max((self.idf(t) for t in exact_hits), default=1.0)
            boost_total = sum(boosts.values()) * base_idf

            results.append(
                (
                    document,
                    score + boost_total,
                    {
                        "bm25_score": round(score, 6),
                        "boost_score": round(boost_total, 6),
                        "matched_terms": matched_terms,
                        "exact_matches": exact_hits,
                        "boosts": {k: round(v, 6) for k, v in boosts.items()},
                        "boost_detail": dict(detail),
                        "retrieval_method": "bm25_code",
                    },
                )
            )

        results.sort(key=lambda entry: entry[1], reverse=True)
        return results[:top_k]


# ============================================================
# Phase 5: hybrid fusion
# ============================================================

def rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def match_corpus_document(
    metadata: Dict[str, Any], corpus: Sequence[CodeDocument]
) -> Optional[CodeDocument]:
    """Find the corpus unit that a dense (text chunk) hit refers to."""
    file_path = str(metadata.get("file_path", ""))
    start, end = metadata.get("start_line"), metadata.get("end_line")
    if not file_path:
        return None

    best: Optional[CodeDocument] = None
    best_overlap = 0
    for document in corpus:
        if document.file_path != file_path:
            continue
        document_start = document.metadata.get("start_line")
        document_end = document.metadata.get("end_line")
        if isinstance(start, int) and isinstance(end, int) \
                and isinstance(document_start, int) and isinstance(document_end, int):
            overlap = min(end, document_end) - max(start, document_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = document
        elif best is None:
            best = document
    return best


def hybrid_fuse(
    keyword_ranking: Sequence[Tuple[CodeDocument, float, Dict[str, Any]]],
    semantic_documents: Sequence[Document],
    corpus: Sequence[CodeDocument],
    top_k: int,
) -> List[Tuple[CodeDocument, float, Dict[str, Any]]]:
    buckets: Dict[str, Dict[str, Any]] = {}

    for rank, (document, score, metadata) in enumerate(keyword_ranking, start=1):
        buckets[document.document_id] = {
            "document": document,
            "fusion": rrf(rank),
            "bm25_rank": rank,
            "meta": dict(metadata),
            "semantic_rank": None,
        }

    for rank, document in enumerate(semantic_documents, start=1):
        corpus_document = match_corpus_document(document.metadata or {}, corpus)
        if corpus_document is None:
            continue
        if corpus_document.document_id in buckets:
            bucket = buckets[corpus_document.document_id]
            bucket["fusion"] += rrf(rank)
            bucket["semantic_rank"] = rank
        else:
            buckets[corpus_document.document_id] = {
                "document": corpus_document,
                "fusion": rrf(rank),
                "bm25_rank": None,
                "meta": {
                    "matched_terms": [],
                    "exact_matches": [],
                    "boosts": {},
                    "retrieval_method": "hybrid_semantic_only",
                },
                "semantic_rank": rank,
            }

    ranked = sorted(buckets.values(), key=lambda bucket: bucket["fusion"], reverse=True)
    fused = []
    for bucket in ranked[:top_k]:
        metadata = bucket["meta"]
        metadata["retrieval_method"] = "hybrid_bm25+semantic"
        metadata["fusion_score"] = round(bucket["fusion"], 6)
        metadata["bm25_rank"] = bucket["bm25_rank"]
        metadata["semantic_rank"] = bucket["semantic_rank"]
        fused.append((bucket["document"], bucket["fusion"], metadata))
    return fused


# ============================================================
# Retriever
# ============================================================

class BM25Retriever:
    def __init__(
        self,
        corpus: str = "chunks",
        chunks_file: Path = DEFAULT_CHUNKS,
        ast_index_file: Path = DEFAULT_AST_INDEX,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 5,
        semantic: Optional[Any] = None,
        multiquery: Optional[Any] = None,
    ) -> None:
        self.corpus = corpus
        self.index = BM25Index(build_corpus(corpus, chunks_file, ast_index_file), k1=k1, b=b)
        self.top_k = top_k
        self.semantic = semantic
        self.multiquery = multiquery

    @property
    def ready(self) -> bool:
        return self.index.count > 0

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.top_k

        if self.multiquery is not None and self.multiquery.ready:
            ranking = self._retrieve_multi_query(query, top_k)
        else:
            ranking = self.index.search(query, top_k)

        if self.semantic is not None and getattr(self.semantic, "ready", False):
            ranking = hybrid_fuse(
                ranking,
                self.semantic.retrieve_documents(query, top_k),
                self.index.documents,
                top_k,
            )

        return [
            document.to_document(score, metadata)
            for document, score, metadata in ranking
        ]

    def _retrieve_multi_query(
        self, query: str, top_k: int
    ) -> List[Tuple[CodeDocument, float, Dict[str, Any]]]:
        from retrieval.multiquery_retriever import build_queries, unique

        queries = build_queries(query, self.multiquery.methods,
                               self.multiquery.num_queries, self.multiquery.index)
        buckets: Dict[str, Dict[str, Any]] = {}
        for item in queries:
            for rank, (document, score, metadata) in enumerate(
                self.index.search(item["query"], top_k), start=1
            ):
                bucket = buckets.setdefault(
                    document.document_id,
                    {
                        "document": document,
                        "fusion": 0.0,
                        "queries": [],
                        "ranks": [],
                        "meta": dict(metadata),
                        "best_score": 0.0,
                    },
                )
                bucket["fusion"] += rrf(rank)
                bucket["queries"].append(item["query"])
                bucket["ranks"].append(rank)
                bucket["best_score"] = max(bucket["best_score"], score)
                bucket["meta"]["matched_terms"] = sorted(
                    set(bucket["meta"].get("matched_terms", [])) | set(metadata["matched_terms"])
                )

        ranked = sorted(buckets.values(), key=lambda bucket: bucket["fusion"], reverse=True)
        results = []
        for bucket in ranked[:top_k]:
            metadata = bucket["meta"]
            metadata["retrieval_method"] = "bm25_code_multi_query"
            metadata["fusion_score"] = round(bucket["fusion"], 6)
            metadata["matched_queries"] = unique(bucket["queries"])
            metadata["rank_per_query"] = bucket["ranks"]
            metadata["bm25_score"] = round(bucket["best_score"], 6)
            results.append((bucket["document"], bucket["fusion"], metadata))
        return results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="bm25_keyword",
        )


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="BM25 / code-aware keyword retrieval")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--corpus", choices=["chunks", "ast", "both"], default="chunks")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--ast-index", default=str(DEFAULT_AST_INDEX))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--hybrid", action="store_true",
                        help="Fuse BM25 with semantic retrieval")
    parser.add_argument("--use-multiquery", action="store_true",
                        help="Run BM25 over expanded queries and fuse with RRF")
    parser.add_argument("--query-methods", default="keywords,patterns,subq")
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    print("=" * 60)
    print("BM25 / Keyword Retrieval (code-aware)")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"k1={args.k1} b={args.b} corpus={args.corpus} top-k={args.top_k}")

    semantic = None
    if args.hybrid:
        candidate = SemanticRetriever(
            embeddings_file=Path(args.embeddings),
            vector_store_path=Path(args.vector_store),
            top_k=args.top_k,
        )
        if candidate.ready:
            semantic = candidate
        else:
            print("Note: dense artifacts missing; skipping semantic fusion")

    multiquery = None
    if args.use_multiquery:
        index = None
        if Path(args.ast_index).exists():
            from retrieval.ast_retriever import ASTIndex

            index = ASTIndex.load(Path(args.ast_index))
        multiquery = MultiQueryRetriever(
            semantic=semantic,
            methods=args.query_methods.split(","),
            num_queries=args.num_queries,
            top_k=args.top_k,
            index=index,
        )

    retriever = BM25Retriever(
        corpus=args.corpus,
        chunks_file=Path(args.chunks),
        ast_index_file=Path(args.ast_index),
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
        semantic=semantic,
        multiquery=multiquery,
    )
    if not retriever.ready:
        print("Error: empty corpus. Run scripts/01_run_chunking.py or build the "
              "AST index first.")
        return

    print(f"Corpus: {retriever.index.count} documents, "
          f"{len(retriever.index.document_frequency)} distinct tokens, "
          f"avg length {retriever.index.average_length:.1f}")

    if multiquery is not None:
        print(f"\nExpanded into {len(multiquery.queries) or '?'} queries")

    documents = retriever.retrieve_documents(args.query)

    fusion = "bm25"
    if args.use_multiquery:
        fusion = "rrf_multi_query"
    if args.hybrid and semantic is not None:
        fusion = "rrf_hybrid"

    print(f"\n{'=' * 60}")
    print(f"Top {len(documents)} results (fusion: {fusion})")
    print("=" * 60)
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        print(f"\n--- #{rank} score={document.score:.4f} "
              f"method={metadata.get('retrieval_method')} ---")
        print(f"{metadata.get('file_path')}:{metadata.get('start_line')}-"
              f"{metadata.get('end_line')}  [id {metadata.get('document_id')}]")
        if metadata.get("bm25_score") is not None:
            print(f"bm25={metadata['bm25_score']} boost={metadata.get('boost_score')}")
        if metadata.get("boosts"):
            print(f"boosts: {metadata['boosts']} {metadata.get('boost_detail')}")
        if metadata.get("bm25_rank") is not None:
            print(f"bm25_rank={metadata['bm25_rank']} "
                  f"semantic_rank={metadata.get('semantic_rank')}")
        if metadata.get("rank_per_query"):
            print(f"rank_per_query: {metadata['rank_per_query']}")
        print(f"matched_terms: {metadata.get('matched_terms', [])[:12]}")
        print(document.content[:220])

    payload = {
        "query": args.query,
        "retriever_id": "bm25_keyword",
        "documents": [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in documents
        ],
        "bm25": {
            "fusion": fusion,
            "k1": args.k1,
            "b": args.b,
            "corpus": args.corpus,
            "documents": retriever.index.count,
            "vocabulary": len(retriever.index.document_frequency),
            "queries": multiquery.queries if multiquery is not None else [],
        },
    }
    with open(Path(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()