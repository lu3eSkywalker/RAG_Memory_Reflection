#!/usr/bin/env python3
"""
02_run_embedding.py — embed the chunked corpus.

The corpus location and the filter rules come from `code_corpus.py` (shared with
01_run_chunking.py), so this stage always embeds exactly what the chunker
produced from:

    /home/sooraj/Downloads/data_for_rag/sample_testing_new/data_to_embed
    *.py only, minus tests/ minus docs/ minus .github/ minus CI/config files

By default it embeds `chunks.json` (written by 01_run_chunking.py). Pass
`--from-source` to chunk and embed in one go without running step 01.

Writes `embeddings.json` (schema unchanged, consumed by 03/04).

Usage:
    python scripts/02_run_embedding.py
    python scripts/02_run_embedding.py --from-source
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, Embedder  # noqa: E402
from typing import List, Dict, Any  # noqa: E402

import code_corpus  # noqa: E402


class SentenceTransformerEmbedder(Embedder):
    """Sentence Transformers embeddings (all-MiniLM-L6-v2, 384 dims)."""

    def __init__(self, component_id: str, config: Dict[str, Any]):
        super().__init__(component_id, config)
        try:
            from sentence_transformers import SentenceTransformer
            self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self.batch_size = self.config.get("batch_size", 32)
            self.device = self.config.get("device", "cpu")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # sentence-transformers >= 5 renamed this method
            for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                getter = getattr(self.model, attribute, None)
                if callable(getter):
                    self.dimension = int(getter())
                    break
            else:
                self.dimension = 384
            self._available = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Using mock embeddings.")
            self._available = False
            self.dimension = 384
        except Exception as e:
            print(f"Warning: Failed to load model: {e}. Using mock embeddings.")
            self._available = False
            self.dimension = 384

    def _mock(self):
        import numpy as np
        return np.random.randn(self.dimension).tolist()

    def embed(self, documents: List[Document]) -> List[Document]:
        embedded = []

        if not self._available:
            for doc in documents:
                embedded.append(Document(
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=self._mock()
                ))
            return embedded

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            texts = [doc.content for doc in batch]

            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                for doc, embedding in zip(batch, embeddings):
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=embedding.tolist()
                    ))
            except Exception as e:
                print(f"Embedding failed: {e}")
                for doc in batch:
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=self._mock()
                    ))

        return embedded

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query for retrieval."""
        if not self._available:
            return self._mock()

        try:
            embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Query embedding failed: {e}")
            return self._mock()


def load_chunks_from_json(file_path: Path) -> List[Document]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Document(content=item["content"], metadata=item.get("metadata", {})) for item in data]


def save_embeddings_to_file(documents: List[Document], output_path: Path):
    with open(output_path, "w") as f:
        json.dump(
            [
                {"content": d.content, "metadata": d.metadata, "embedding": d.embedding}
                for d in documents
            ],
            f,
            indent=2,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Embed the RAG corpus")
    parser.add_argument("--root", default=str(code_corpus.DEFAULT_EMBED_ROOT),
                        help="Corpus root (used with --from-source and for reporting)")
    parser.add_argument("--chunks", default="chunks.json",
                        help="Chunks produced by 01_run_chunking.py")
    parser.add_argument("--from-source", action="store_true",
                        help="Chunk and embed in one pass, skipping chunks.json")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="embeddings.json")
    args = parser.parse_args(argv)

    chunks_path = Path(args.chunks)

    if args.from_source:
        chunks, stats = code_corpus.chunk_corpus(root=Path(args.root).expanduser())
        if not chunks:
            print(f"Error: no chunks produced from {args.root}")
            return 1
        documents = [Document(content=c["content"], metadata=c["metadata"]) for c in chunks]
        print(f"Chunked {stats['total_files']} files from {stats['root']} "
              f"-> {stats['total_chunks']} chunks (inline)")
    else:
        if not chunks_path.exists():
            print(f"Error: {chunks_path} not found. Run 01_run_chunking.py first, "
                  "or pass --from-source.")
            return 1
        documents = load_chunks_from_json(chunks_path)
        print(f"Loading chunks from {chunks_path}...")

    if not documents:
        print("Error: corpus is empty")
        return 1

    repos: dict = {}
    for document in documents:
        repos.setdefault((document.metadata or {}).get("repo", "?"), 0)
        repos[(document.metadata or {}).get("repo", "?")] += 1

    print("=" * 70)
    print("STEP 2  EMBEDDING")
    print("=" * 70)
    print(f"Corpus root   : {args.root}")
    print(f"Repositories  : {', '.join(sorted(repos))}")
    print(f"Chunks        : {len(documents)}")
    for repo, count in sorted(repos.items()):
        print(f"   {repo:<28}{count:>6} chunks")

    embedder = SentenceTransformerEmbedder(
        "sentence_transformer_embedder",
        {
            "model": args.model,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    print(f"Model         : {args.model if embedder._available else 'mock (random)'}"
          f"  dim={embedder.dimension}")

    embedded_docs = embedder.embed(documents)

    print()
    print("First 3 embedded chunks:")
    for i, doc in enumerate(embedded_docs[:3], start=1):
        metadata = doc.metadata or {}
        print(f"  {i}. [{metadata.get('kind', 'chunk')}] "
              f"{metadata.get('qualified_name', metadata.get('file_path'))} "
              f"({metadata.get('file_path')}) dim={len(doc.embedding)}")

    output_path = Path(args.output)
    save_embeddings_to_file(embedded_docs, output_path)
    print()
    print(f"Saved -> {output_path}")

    # Query embedding sanity check (used by 04_run_retrieval.py)
    query = "def fibonacci_iterative(n):"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding: {query!r} dim={len(query_embedding)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())