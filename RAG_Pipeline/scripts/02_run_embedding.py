#!/usr/bin/env python3
"""
Run embedding component independently.
Usage: python scripts/02_run_embedding.py
"""

import sys
import os
import json
import numpy as np
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Document, Embedder
from typing import List, Dict, Any


class SentenceTransformerEmbedder(Embedder):
    """Sentence Transformers embeddings (all-MiniLM-L6-v2)."""
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        super().__init__(component_id, config)
        try:
            from sentence_transformers import SentenceTransformer
            self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self.batch_size = self.config.get("batch_size", 32)
            self.device = self.config.get("device", "cpu")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._available = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Using mock embeddings.")
            self._available = False
        except Exception as e:
            print(f"Warning: Failed to load model: {e}. Using mock embeddings.")
            self._available = False
    
    def embed(self, documents: List[Document]) -> List[Document]:
        embedded = []
        
        if not self._available:
            for doc in documents:
                mock_embedding = np.random.randn(384).tolist()
                embedded.append(Document(
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=mock_embedding
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
                    mock_embedding = np.random.randn(384).tolist()
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=mock_embedding
                    ))
        
        return embedded
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query for retrieval."""
        if not self._available:
            return np.random.randn(384).tolist()
        
        try:
            embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Query embedding failed: {e}")
            return np.random.randn(384).tolist()


def load_chunks_from_json(file_path: str) -> List[Document]:
    """Load chunks from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    documents = []
    for item in data:
        documents.append(Document(
            content=item["content"],
            metadata=item["metadata"]
        ))
    return documents


def save_embeddings_to_file(documents: List[Document], output_path: str):
    """Save embeddings to JSON file."""
    data = []
    for doc in documents:
        data.append({
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": doc.embedding
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    chunks_file = "chunks.json"
    output_file = "embeddings.json"
    
    if not os.path.exists(chunks_file):
        print(f"Error: {chunks_file} not found. Run 01_run_chunking.py first.")
        return
    
    print(f"Loading chunks from {chunks_file}...")
    documents = load_chunks_from_json(chunks_file)
    print(f"Loaded {len(documents)} chunks")
    
    print("Initializing SentenceTransformer embedder...")
    embedder = SentenceTransformerEmbedder(
        "sentence_transformer_embedder",
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
            "device": "cpu"
        }
    )
    
    print(f"\nEmbedding {len(documents)} documents...")
    embedded_docs = embedder.embed(documents)
    
    print(f"\n{'='*60}")
    print("Embedding Results")
    print(f"{'='*60}")
    
    for i, doc in enumerate(embedded_docs):
        print(f"\n--- Document {i+1} ---")
        print(f"File: {doc.metadata.get('file_path', 'unknown')}")
        print(f"Embedding dimension: {len(doc.embedding) if doc.embedding else 'None'}")
        print(f"First 5 values: {doc.embedding[:5] if doc.embedding else 'None'}")
        print(f"Content preview: {doc.content[:80]}...")
    
    save_embeddings_to_file(embedded_docs, output_file)
    print(f"\nEmbeddings saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("Query Embedding Test")
    print(f"{'='*60}")
    
    query = "def fibonacci_iterative(n):"
    query_embedding = embedder.embed_query(query)
    print(f"Query: {query}")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")


if __name__ == "__main__":
    main()