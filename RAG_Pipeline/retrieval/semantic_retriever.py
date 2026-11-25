"""Semantic (dense vector) retrieval.

Dense retriever used by every other technique that needs vector search:

    query -> SentenceTransformer embedding -> FAISS (normalised inner product)
          -> top-k cosine-similar code chunks

Falls back to a NumPy linear scan when faiss or sentence-transformers are not
installed so the rest of the pipeline still runs (with mock embeddings).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from rag_core.base import Document, RetrievalResult  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"
DEFAULT_DIMENSION = 384

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    FAISS_AVAILABLE = False


class SentenceTransformerEmbedder:
    """Thin embedder wrapper (lazy import, mock fallback).

    Instances are cached per (model, device) so a suite of retrievers loads the
    model once instead of once per retriever.
    """

    _CACHE: Dict[Tuple[str, str], "SentenceTransformerEmbedder"] = {}

    def __new__(
        cls,
        model: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str = "cpu",
        dimension: int = DEFAULT_DIMENSION,
    ) -> "SentenceTransformerEmbedder":
        key = (model, device)
        cached = cls._CACHE.get(key)
        if cached is not None and cached.batch_size == batch_size:
            return cached
        instance = super().__new__(cls)
        cls._CACHE[key] = instance
        return instance

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str = "cpu",
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        if getattr(self, "_ready", False) and (self.model_name, self.device) == (model, device):
            return

        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.dimension = dimension
        self.model = None
        self.available = False

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model, device=device)
            # sentence-transformers >= 5 renamed this method
            for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                getter = getattr(self.model, attribute, None)
                if callable(getter):
                    self.dimension = int(getter())
                    break
            self.available = True
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"Warning: sentence-transformers unavailable ({error}); "
                  "using mock embeddings")

        self._ready = True

    def embed_documents(self, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []

        if not self.available:
            return [
                Document(
                    content=document.content,
                    metadata=dict(document.metadata or {}),
                    embedding=self.mock_embedding(),
                )
                for document in documents
            ]

        embedded: List[Document] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start: start + self.batch_size]
            vectors = self.model.encode(
                [document.content for document in batch],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for document, vector in zip(batch, vectors):
                embedded.append(
                    Document(
                        content=document.content,
                        metadata=dict(document.metadata or {}),
                        embedding=vector.tolist(),
                    )
                )
        return embedded

    def embed_query(self, query: str) -> List[float]:
        if not self.available:
            return self.mock_embedding()
        vector = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    def mock_embedding(self) -> List[float]:
        return np.random.randn(self.dimension).tolist()


class FaissVectorStore:
    """FAISS IndexFlatIP over L2-normalised vectors (== cosine similarity)."""

    def __init__(self, documents: Optional[Sequence[Document]] = None) -> None:
        self.documents: List[Document] = list(documents or [])
        self.index = None
        self.dimension: Optional[int] = None
        self.use_faiss = FAISS_AVAILABLE

    @property
    def count(self) -> int:
        return len(self.documents)

    def add_documents(self, documents: Sequence[Document]) -> None:
        self.documents.extend(documents)
        embeddings = [
            document.embedding
            for document in self.documents
            if document.embedding is not None
        ]
        if not embeddings:
            return

        array = np.array(embeddings, dtype=np.float32)
        self.dimension = array.shape[1]
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(array)
            self.index.add(array)

    def load(self, index_path: Path, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        if self.use_faiss and Path(index_path).exists():
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d

    def save(self, index_path: Path) -> None:
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(index_path))
            with open(f"{index_path}.docs", "wb") as handle:
                pickle.dump(self.documents, handle)

    def search(self, query_embedding: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        if not self.documents:
            return []

        if self.use_faiss and self.index is not None:
            query = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, min(top_k, self.count))
            return [
                (int(index), float(score))
                for score, index in zip(scores[0], indices[0])
                if 0 <= index < self.count
            ]

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query) or 1.0
        scored: List[Tuple[int, float]] = []
        for position, document in enumerate(self.documents):
            if not document.embedding:
                continue
            vector = np.array(document.embedding, dtype=np.float32)
            similarity = float(np.dot(query, vector) / (norm * (np.linalg.norm(vector) or 1.0)))
            scored.append((position, similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def load_documents(embeddings_file: Path) -> List[Document]:
    with open(embeddings_file, "r") as handle:
        payload = json.load(handle)
    return [
        Document(
            content=item["content"],
            metadata=item.get("metadata", {}),
            embedding=item.get("embedding"),
        )
        for item in payload
    ]


class SemanticRetriever:
    """Dense retrieval over embeddings.json + vector_store.faiss."""

    def __init__(
        self,
        embeddings_file: Path = DEFAULT_EMBEDDINGS,
        vector_store_path: Path = DEFAULT_VECTOR_STORE,
        top_k: int = 10,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        filter_future_context: bool = False,
        repo: Optional[str] = None,
    ) -> None:
        self.embeddings_file = Path(embeddings_file)
        self.vector_store_path = Path(vector_store_path)
        self.top_k = top_k
        self.filter_future_context = filter_future_context
        self.repo = repo
        self.missing_artifacts = [
            str(path)
            for path in (self.embeddings_file, self.vector_store_path)
            if not Path(path).exists()
        ]
        self.embedder = SentenceTransformerEmbedder(model=model, device=device)
        self.store = FaissVectorStore()
        self._loaded = False

    @property
    def ready(self) -> bool:
        if not self.missing_artifacts:
            self._ensure_loaded()
        return not self.missing_artifacts

    @property
    def documents(self) -> List[Document]:
        return self.store.documents

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        documents = load_documents(self.embeddings_file)

        # Optional repository scope. Applied *before* indexing so the ranking
        # itself is restricted: a purely semantic query can otherwise fill the
        # whole top-k with lookalike code from another repository, leaving a
        # post-hoc per-repo filter with nothing to keep.
        if self.repo:
            scoped = [
                document
                for document in documents
                if (document.metadata or {}).get("repo") == self.repo
            ]
            if scoped:
                documents = scoped
            else:
                print(f"Warning: no documents for repo {self.repo!r}; "
                      "falling back to the whole corpus")

        self.store.load(self.vector_store_path, documents)
        if self.store.index is None:
            self.store.add_documents(documents)
            self.store.save(self.vector_store_path)
        self._loaded = True

    # -- retrieval -----------------------------------------------------
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        if not self.ready:
            return []

        self._ensure_loaded()
        top_k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        query_metadata = {}

        hits: List[Document] = []
        for position, score in self.store.search(query_embedding, top_k):
            source = self.store.documents[position]
            metadata = dict(source.metadata or {})
            if self.filter_future_context and self._is_future_context(metadata, query_metadata):
                continue
            hits.append(
                Document(
                    content=source.content,
                    metadata=metadata,
                    embedding=source.embedding,
                    score=float(score),
                )
            )
        return hits

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="semantic",
        )

    @staticmethod
    def _is_future_context(metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        if not metadata or not query_metadata:
            return False
        if metadata.get("file_path") != query_metadata.get("file_path"):
            return False
        return metadata.get("end_line", 0) > query_metadata.get("context_start_lineno", 0)


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Semantic (dense) retrieval")
    parser.add_argument("--query", default="def fibonacci_iterative(n):")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    retriever = SemanticRetriever(
        embeddings_file=Path(args.embeddings),
        vector_store_path=Path(args.vector_store),
        top_k=args.top_k,
        model=args.model,
        device=args.device,
    )
    if not retriever.ready:
        print("Error: missing artifacts:", ", ".join(retriever.missing_artifacts))
        return

    print("=" * 60)
    print("Semantic Retrieval")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Documents in index: {len(retriever.documents)}")

    result = retriever.retrieve(args.query)
    print(f"\nRetrieved {len(result.documents)} documents")
    for rank, document in enumerate(result.documents, start=1):
        metadata = document.metadata or {}
        print(f"\n--- #{rank} score={document.score:.4f} ---")
        print(f"{Path(metadata.get('file_path', '?')).name}:"
              f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
        print(document.content[:250])


if __name__ == "__main__":
    main()