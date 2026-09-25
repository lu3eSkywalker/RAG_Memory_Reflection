#!/usr/bin/env python3
"""
Run retrieval component independently.
Usage: python scripts/04_run_retrieval.py
"""

import sys
import os
import json
import importlib.util
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Document, Retriever, RetrievalResult
from rag_core.context import PipelineContext
import numpy as np
import scipy.spatial.distance
from typing import List, Any, Dict


QUERY = "def fibonacci_iterative(n):"


# Load dependent modules
def load_module(name, filename):
    scripts_dir = "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/scripts"
    spec = importlib.util.spec_from_file_location(name, f"{scripts_dir}/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

embedding_mod = load_module("embedding", "02_run_embedding.py")
vector_store_mod = load_module("vector_store", "03_run_vector_store.py")

SentenceTransformerEmbedder = embedding_mod.SentenceTransformerEmbedder
MockFAISSVectorStore = vector_store_mod.MockFAISSVectorStore


class CosineSimilarityRetriever(Retriever):
    """Cosine similarity retriever using FAISS vector store or linear scan."""
    
    def retrieve(self, query: str, index: List[Document], context: PipelineContext) -> RetrievalResult:
        top_k = self.config.get("top_k", 20)
        filter_future = self.config.get("filter_future_context", True)
        
        # Get embedder from context to embed query
        embedder = context.get("embedder_instance")
        if embedder and hasattr(embedder, 'embed_query'):
            query_embedding = embedder.embed_query(query)
        else:
            dim = 768
            query_embedding = np.random.randn(dim).tolist()
        
        # Use FAISS vector store if available
        vector_store = context.vector_store
        if vector_store is not None and hasattr(vector_store, 'search'):
            query_array = np.array([query_embedding], dtype=np.float32)
            # Try to normalize if faiss is available
            try:
                import faiss
                faiss.normalize_L2(query_array)
            except:
                pass
            
            scores, indices = vector_store.search(query_array, top_k)
            
            scored_docs = []
            query_metadata = context.metadata
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(context.documents):
                    doc = context.documents[idx]
                    if filter_future and self._is_future_context(doc, query_metadata):
                        continue
                    doc.score = float(score)
                    scored_docs.append(doc)
        else:
            scored_docs = []
            query_metadata = context.metadata
            
            for doc in index:
                if filter_future and self._is_future_context(doc, query_metadata):
                    continue
                
                doc_embedding = doc.embedding
                if doc_embedding is None:
                    continue
                
                sim = 1 - scipy.spatial.distance.cosine(query_embedding, doc_embedding)
                
                scored_docs.append(Document(
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=sim
                ))
            
            scored_docs.sort(key=lambda d: d.score or 0, reverse=True)
            scored_docs = scored_docs[:top_k]
        
        return RetrievalResult(
            query=query,
            documents=scored_docs,
            retriever_id=self.component_id
        )
    
    def _is_future_context(self, doc: Document, query_metadata: Dict) -> bool:
        doc_meta = doc.metadata
        if not doc_meta or not query_metadata:
            return False
        if doc_meta.get("file_path") != query_metadata.get("file_path"):
            return False
        doc_end = doc_meta.get("end_line", 0)
        hole_start = query_metadata.get("context_start_lineno", 0)
        return doc_end > hole_start


def load_vector_store(vector_store_path: str, embeddings_file: str) -> tuple:
    """Load vector store from file and return vector_store and context."""
    if not os.path.exists(vector_store_path):
        print(f"Error: {vector_store_path} not found. Run 03_run_vector_store.py first.")
        return None, None
    
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found.")
        return None, None
    
    # Load embeddings to get documents
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    documents = []
    for item in embeddings_data:
        documents.append(Document(
            content=item["content"],
            metadata=item["metadata"],
            embedding=item["embedding"]
        ))
    
    vector_store = MockFAISSVectorStore("faiss_store", {"index_type": "flat_ip", "normalize": True})
    vector_store.load(vector_store_path)
    
    context = PipelineContext()
    context.documents = documents
    context.vector_store = vector_store._index
    
    return vector_store, context


def save_retrieval_data(result: RetrievalResult, output_path: str):
    """Save retrieval results to JSON file."""
    data = {
        "query": result.query,
        "retriever_id": result.retriever_id,
        "documents": []
    }
    for doc in result.documents:
        data["documents"].append({
            "content": doc.content,
            "metadata": doc.metadata,
            "score": doc.score
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    vector_store_path = "vector_store.faiss"
    embeddings_file = "embeddings.json"
    output_file = "retrieval_data.json"
    
    print("Loading vector store...")
    vector_store, context = load_vector_store(vector_store_path, embeddings_file)
    if vector_store is None:
        return
    
    print(f"Loaded {len(context.documents)} documents from vector store")
    
    print("Initializing embedder...")
    embedder = SentenceTransformerEmbedder(
        "sentence_transformer_embedder",
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
            "device": "cpu"
        }
    )
    
    print("Initializing retriever...")
    retriever = CosineSimilarityRetriever("cosine_similarity", {"top_k": 5, "filter_future_context": False})
    
    context.set("embedder_instance", embedder)
    
    print(f"\n{'='*60}")
    print(f"Query: {QUERY}")
    print(f"{'='*60}")
    
    result = retriever.retrieve(QUERY, context.documents, context)
    
    print(f"Retrieved {len(result.documents)} documents:")
    for i, doc in enumerate(result.documents):
        print(f"\n--- Result {i+1} (score: {doc.score:.4f}) ---")
        print(f"File: {doc.metadata.get('file_path', 'unknown')}")
        print(f"Lines: {doc.metadata.get('start_line', '?')}-{doc.metadata.get('end_line', '?')}")
        print(f"Content:\n{doc.content}")
    
    save_retrieval_data(result, output_file)
    print(f"\nRetrieval data saved to {output_file}")


if __name__ == "__main__":
    main()