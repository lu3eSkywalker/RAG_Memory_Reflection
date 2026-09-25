#!/usr/bin/env python3
"""
Run vector store (FAISS) component independently.
Usage: python scripts/03_run_vector_store.py
"""

import sys
import os
import json
import importlib.util
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Document, VectorStore
from rag_core.context import PipelineContext
import numpy as np
from typing import List, Tuple, Any, Dict


class MockFAISSVectorStore(VectorStore):
    """Mock FAISS vector store for testing without FAISS installed."""
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        super().__init__(component_id, config)
        self._documents = []
        self._embeddings = []
        self._dimension = None
        self._faiss_available = False
        
        try:
            import faiss
            self.faiss = faiss
            self._faiss_available = True
            self._index = None
        except ImportError:
            print("Warning: faiss-cpu not installed. Using mock vector store.")
    
    def add_documents(self, documents: List[Document], context: PipelineContext) -> Any:
        if not documents:
            return self._index if self._faiss_available else self._embeddings
        
        embeddings = []
        valid_docs = []
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                valid_docs.append(doc)
        
        if not embeddings:
            print("Warning: No documents with embeddings to add")
            return self._index if self._faiss_available else self._embeddings
        
        self._documents.extend(valid_docs)
        context.documents = self._documents  # Store in context for retrieval
        
        if self._faiss_available:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if self._index is None:
                self._dimension = embeddings_array.shape[1]
                self._index = self.faiss.IndexFlatIP(self._dimension)
            
            self.faiss.normalize_L2(embeddings_array)
            self._index.add(embeddings_array)
            
            context.vector_store = self._index
            return self._index
        else:
            self._embeddings.extend(embeddings)
            return self._embeddings
    
    def search(self, query_embedding: List[float], top_k: int, context: PipelineContext) -> List[Tuple[Document, float]]:
        if not self._documents:
            return []
        
        if self._faiss_available and self._index is not None:
            query_array = np.array([query_embedding], dtype=np.float32)
            self.faiss.normalize_L2(query_array)
            
            scores, indices = self._index.search(query_array, min(top_k, len(self._documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._documents):
                    doc = self._documents[idx]
                    doc.score = float(score)
                    results.append((doc, float(score)))
            return results
        else:
            query_array = np.array(query_embedding)
            scored_docs = []
            
            for i, doc in enumerate(self._documents):
                if i < len(self._embeddings):
                    doc_embedding = np.array(self._embeddings[i])
                    sim = np.dot(query_array, doc_embedding) / (np.linalg.norm(query_array) * np.linalg.norm(doc_embedding))
                    doc.score = float(sim)
                    scored_docs.append((doc, float(sim)))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
    
    def save(self, path: str) -> None:
        if self._faiss_available and self._index is not None:
            self.faiss.write_index(self._index, path)
            import pickle
            with open(path + ".docs", "wb") as f:
                pickle.dump(self._documents, f)
    
    def load(self, path: str) -> None:
        if self._faiss_available:
            self._index = self.faiss.read_index(path)
            import pickle
            with open(path + ".docs", "rb") as f:
                self._documents = pickle.load(f)
            self._dimension = self._index.d


def load_embeddings_from_json(file_path: str) -> List[Document]:
    """Load embedded documents from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    documents = []
    for item in data:
        documents.append(Document(
            content=item["content"],
            metadata=item["metadata"],
            embedding=item["embedding"]
        ))
    return documents


def main():
    embeddings_file = "embeddings.json"
    vector_store_path = "vector_store.faiss"
    
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found. Run 02_run_embedding.py first.")
        return
    
    print(f"Loading embeddings from {embeddings_file}...")
    documents = load_embeddings_from_json(embeddings_file)
    print(f"Loaded {len(documents)} embedded documents")
    
    print("Initializing FAISS vector store...")
    vector_store = MockFAISSVectorStore("faiss_store", {"index_type": "flat_ip", "normalize": True})
    
    context = PipelineContext()
    
    print("Adding documents to vector store...")
    index = vector_store.add_documents(documents, context)
    
    print(f"\n{'='*60}")
    print("Vector Store Results")
    print(f"{'='*60}")
    print(f"Index created: {index is not None}")
    print(f"Number of documents indexed: {len(vector_store._documents)}")
    print(f"Index dimension: {vector_store._dimension}")
    if vector_store._faiss_available and vector_store._index:
        print(f"Total vectors in index: {vector_store._index.ntotal}")
    
    # Save vector store
    vector_store.save(vector_store_path)
    print(f"\nVector store saved to {vector_store_path}")
    
    print(f"\n{'='*60}")
    print("Search Test")
    print(f"{'='*60}")
    
    # Load embedding module for query embedding
    scripts_dir = "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/scripts"
    spec = importlib.util.spec_from_file_location("embedding", f"{scripts_dir}/02_run_embedding.py")
    embedding_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embedding_mod)
    SentenceTransformerEmbedder = embedding_mod.SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(
        "sentence_transformer_embedder",
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
            "device": "cpu"
        }
    )
    
    query = "def fibonacci_iterative(n):"
    query_embedding = embedder.embed_query(query)
    print(f"Query: {query}")
    
    results = vector_store.search(query_embedding, top_k=3, context=context)
    
    print(f"\nTop {len(results)} results:")
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
        print(f"File: {doc.metadata.get('file_path', 'unknown')}")
        print(f"Content: {doc.content[:100]}...")


if __name__ == "__main__":
    main()