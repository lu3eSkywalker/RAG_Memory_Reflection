#!/usr/bin/env python3
"""
Run full RAG pipeline with manual step-by-step execution.
Usage: python scripts/08_run_pipeline.py
"""

import sys
import os
import numpy as np
import importlib.util
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core import ComponentRegistry, PipelineRegistry, PipelineConfig
from rag_core.pipeline import PipelineRunner
from rag_core.base import Document
from rag_core.context import PipelineContext
from rag_core.pipeline import RAGPipeline

# Import component implementations from our scripts using importlib
scripts_dir = "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/scripts"

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, f"{scripts_dir}/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

chunking_mod = load_module("chunking", "01_run_chunking.py")
embedding_mod = load_module("embedding", "02_run_embedding.py")
vector_store_mod = load_module("vector_store", "03_run_vector_store.py")
retrieval_mod = load_module("retrieval", "04_run_retrieval.py")
prompt_mod = load_module("prompt", "05_run_prompt.py")
generation_mod = load_module("generation", "06_run_generation.py")
evaluation_mod = load_module("evaluation", "07_run_evaluation.py")

SlidingWindowChunker = chunking_mod.SlidingWindowChunker
SentenceTransformerEmbedder = embedding_mod.SentenceTransformerEmbedder
MockFAISSVectorStore = vector_store_mod.MockFAISSVectorStore
CosineSimilarityRetriever = retrieval_mod.CosineSimilarityRetriever
RepoCoderPromptBuilder = prompt_mod.RepoCoderPromptBuilder
StandardRAGPromptBuilder = prompt_mod.StandardRAGPromptBuilder
MockGenerator = generation_mod.MockGenerator
ExactMatchEvaluator = evaluation_mod.ExactMatchEvaluator
EditSimilarityEvaluator = evaluation_mod.EditSimilarityEvaluator


def create_mock_index():
    """Create a mock repository index for testing."""
    mock_docs = [
        Document(
            content="def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            metadata={"file_path": "math/fib.py", "start_line": 0, "end_line": 3},
            embedding=np.random.randn(384).tolist()
        ),
        Document(
            content="def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
            metadata={"file_path": "math/utils.py", "start_line": 0, "end_line": 2},
            embedding=np.random.randn(384).tolist()
        ),
        Document(
            content="def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2",
            metadata={"file_path": "algorithms/search.py", "start_line": 0, "end_line": 3},
            embedding=np.random.randn(384).tolist()
        ),
        Document(
            content="class DataProcessor:\n    def __init__(self):\n        self.data = []\n    def add_item(self, item):\n        self.data.append(item)\n    def process_all(self):\n        return [self._process_item(x) for x in self.data]",
            metadata={"file_path": "processors/data.py", "start_line": 0, "end_line": 6},
            embedding=np.random.randn(384).tolist()
        ),
    ]
    return mock_docs


def main():
    print("=" * 60)
    print("Full RAG Pipeline - Step by Step (Manual)")
    print("=" * 60)
    
    # Query and ground truth
    query = """def fibonacci_iterative(n: int) -> int:
    \"\"\"Return the nth Fibonacci number iteratively.\"\"\"
"""
    ground_truth = """    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a"""
    
    mock_index = create_mock_index()
    metadata = {"file_path": "math/fib.py", "context_start_lineno": 4}
    
    print(f"\n[1/7] Chunking query...")
    chunker = SlidingWindowChunker("sliding_window", {"window_size": 20, "slice_size": 2})
    chunks = chunker.chunk(query, metadata)
    print(f"  Created {len(chunks)} chunks")
    
    print(f"\n[2/7] Embedding chunks...")
    embedder = SentenceTransformerEmbedder("sentence_transformer_embedder", {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "cpu"
    })
    embedded_chunks = embedder.embed(chunks)
    print(f"  Embedded {len(embedded_chunks)} chunks")
    
    print(f"\n[3/7] Building vector index...")
    vector_store = MockFAISSVectorStore("faiss_store", {"index_type": "flat_ip", "normalize": True})
    context = PipelineContext()
    vector_store.add_documents(embedded_chunks, context)
    # Also add the mock index documents for retrieval
    vector_store.add_documents(mock_index, context)
    print(f"  Indexed {len(context.documents)} documents")
    
    print(f"\n[4/7] Retrieving relevant chunks...")
    retriever = CosineSimilarityRetriever("cosine_similarity", {"top_k": 3, "filter_future_context": False})
    context.set("embedder_instance", embedder)
    retrieval_result = retriever.retrieve(query, embedded_chunks, context)
    print(f"  Retrieved {len(retrieval_result.documents)} documents")
    for i, doc in enumerate(retrieval_result.documents):
        print(f"    {i+1}. {doc.metadata.get('file_path', 'unknown')} (score: {doc.score:.4f})")
    
    print(f"\n[5/7] Building prompt...")
    prompt_builder = RepoCoderPromptBuilder("repo_coder_style", {
        "max_examples": 10,
        "max_retrieval_tokens": 2000,
        "include_file_paths": True,
        "comment_out_code": True,
        "extended_context_mode": False
    })
    context.original_query = query
    context.metadata = metadata
    prompt = prompt_builder.build_prompt(query, retrieval_result, context)
    print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    
    print(f"\n[6/7] Generating completion...")
    generator = MockGenerator("gemini_generator", {"model": "mock-gemini"})
    generation_result = generator.generate(prompt, context)
    print(f"  Generated {len(generation_result.completions)} completion(s)")
    
    print(f"\n[7/7] Evaluating...")
    context.predictions = generation_result.completions
    context.ground_truth = ground_truth
    
    evaluators = [
        ExactMatchEvaluator("exact_match", {}),
        EditSimilarityEvaluator("edit_similarity", {}),
    ]
    
    for evaluator in evaluators:
        result = evaluator.evaluate(generation_result.completions, [ground_truth], context)
        print(f"  {result.evaluator_id}: {result.metrics}")
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    
    print(f"\n--- Final Prediction ---")
    print(generation_result.completions[0] if generation_result.completions else "(empty)")


if __name__ == "__main__":
    main()