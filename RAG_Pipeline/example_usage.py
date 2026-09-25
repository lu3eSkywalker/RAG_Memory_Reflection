"""
Example usage of the modular RAG pipeline system.
Demonstrates how to configure and run different RAG pipelines.
"""

import yaml
from pathlib import Path

from rag_core import (
    ComponentRegistry, PipelineRegistry, PipelineConfig, PipelineRunner
)
from rag_components import *  # Registers all components


def load_spec_from_yaml():
    """Load pipeline specification from YAML file."""
    spec_path = Path(__file__).parent / "rag_pipeline_spec.yaml"
    
    component_registry = ComponentRegistry()
    component_registry.load_from_yaml(str(spec_path))
    
    # Also register our Python implementations
    component_registry.load_extensions({
        "chunker": "rag_components",
        "embedder": "rag_components", 
        "retriever": "rag_components",
        "prompt_builder": "rag_components",
        "generator": "rag_components",
        "iteration_strategy": "rag_components",
        "evaluator": "rag_components",
        "vector_store": "rag_components",
    })
    
    pipeline_registry = PipelineRegistry(component_registry)
    pipeline_registry.load_from_yaml(str(spec_path))
    
    return component_registry, pipeline_registry


def run_standard_rag_example():
    """Run standard RAG pipeline."""
    print("=" * 60)
    print("Running Standard RAG Pipeline")
    print("=" * 60)
    
    component_registry, pipeline_registry = load_spec_from_yaml()
    runner = PipelineRunner(component_registry, pipeline_registry)
    
    # Example query (code completion)
    query = """
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n: int) -> int:
    \"\"\"Return the nth Fibonacci number iteratively.\"\"\"
"""
    
    ground_truth = """    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a"""
    
    # Mock repository index (in real use, this would be built from actual repo)
    # Using numpy arrays for embeddings (768 dims for Gemini)
    import numpy as np
    mock_index = [
        type('Doc', (), {
            'content': 'def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)',
            'metadata': {'file_path': 'math/utils.py', 'end_line': 3},
            'embedding': np.random.randn(768).tolist()
        })(),
        type('Doc', (), {
            'content': 'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2',
            'metadata': {'file_path': 'algorithms/search.py', 'end_line': 4},
            'embedding': np.random.randn(768).tolist()
        })(),
    ]
    
    context = runner.run(
        pipeline_id="standard_rag",
        query=query,
        ground_truth=ground_truth,
        metadata={"file_path": "math/fib.py", "context_start_lineno": 6},
        index=mock_index
    )
    
    print(f"Run ID: {context.run_id}")
    print(f"Total time: {context.get_total_time():.3f}s")
    print(f"Step timings: {context.step_timings}")
    print(f"Predictions: {context.predictions}")
    print(f"Evaluations: {context.evaluation_results}")
    print()


def run_repocoder_example():
    """Run RepoCoder iterative pipeline."""
    print("=" * 60)
    print("Running RepoCoder Iterative Pipeline")
    print("=" * 60)
    
    component_registry, pipeline_registry = load_spec_from_yaml()
    runner = PipelineRunner(component_registry, pipeline_registry)
    
    query = """
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
    
    def process_all(self):
        \"\"\"Process all items and return results.\"\"\"
"""
    
    ground_truth = """        results = []
        for item in self.data:
            results.append(self._process_item(item))
        return results
    
    def _process_item(self, item):
        return item * 2"""
    
    import numpy as np
    mock_index = [
        type('Doc', (), {
            'content': 'class BatchProcessor:\n    def process_batch(self, items):\n        return [self.transform(x) for x in items]\n    def transform(self, x):\n        return x.upper()',
            'metadata': {'file_path': 'processors/batch.py', 'end_line': 4},
            'embedding': np.random.randn(768).tolist()
        })(),
    ]
    
    context = runner.run(
        pipeline_id="repocoder_full",
        query=query,
        ground_truth=ground_truth,
        metadata={"file_path": "processors/data.py", "context_start_lineno": 7},
        index=mock_index
    )
    
    print(f"Run ID: {context.run_id}")
    print(f"Iterations: {context.iteration}")
    print(f"Iteration history: {context.iteration_history}")
    print(f"Total time: {context.get_total_time():.3f}s")
    print(f"Predictions: {context.predictions}")
    print()


def run_comparison_example():
    """Compare multiple pipelines on same queries."""
    print("=" * 60)
    print("Comparing Pipelines")
    print("=" * 60)
    
    component_registry, pipeline_registry = load_spec_from_yaml()
    runner = PipelineRunner(component_registry, pipeline_registry)
    
    queries = [
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):",
        "class Cache:\n    def __init__(self):\n        self.store = {}\n    def get(self, key):",
    ]
    
    ground_truths = [
        "    return a * b",
        "        return self.store.get(key)",
    ]
    
    import numpy as np
    mock_index = [
        type('Doc', (), {
            'content': 'def subtract(a, b):\n    return a - b',
            'metadata': {'file_path': 'math/ops.py', 'end_line': 2},
            'embedding': np.random.randn(768).tolist()
        })(),
    ]
    
    pipeline_ids = ["standard_rag", "repocoder_rg1", "repocoder_full"]
    
    results = runner.compare_pipelines(
        pipeline_ids=pipeline_ids,
        queries=queries,
        ground_truths=ground_truths,
        metadatas=[{"file_path": "test.py", "context_start_lineno": 3}] * len(queries)
    )
    
    for pid, contexts in results.items():
        print(f"\n{pid}:")
        for i, ctx in enumerate(contexts):
            print(f"  Query {i+1}: {len(ctx.predictions)} predictions, "
                  f"time: {ctx.get_total_time():.3f}s")
            if ctx.evaluation_results:
                for eval_id, eval_result in ctx.evaluation_results.items():
                    print(f"    {eval_id}: {eval_result.metrics}")


def create_custom_pipeline():
    """Example of creating a custom pipeline programmatically."""
    print("=" * 60)
    print("Creating Custom Pipeline")
    print("=" * 60)
    
    from rag_core import PipelineSpec, PipelineStep
    from rag_core.registry import ComponentRegistry, PipelineRegistry
    from rag_core.pipeline import RAGPipeline, PipelineConfig
    
    # Create registries
    component_registry = ComponentRegistry()
    component_registry.load_extensions({
        "chunker": "rag_components",
        "embedder": "rag_components",
        "retriever": "rag_components", 
        "prompt_builder": "rag_components",
        "generator": "rag_components",
        "iteration_strategy": "rag_components",
        "evaluator": "rag_components",
        "vector_store": "rag_components",
    })
    
    pipeline_registry = PipelineRegistry(component_registry)
    
    # Define custom pipeline: Sliding window + Gemini embedder + FAISS + Self-refine
    custom_spec = PipelineSpec(
        pipeline_id="custom_gemini_self_refine",
        name="Gemini + FAISS + Self-Refine",
        description="Custom pipeline with Gemini embeddings and self-refinement",
        steps=[
            PipelineStep("chunker", "sliding_window", {"window_size": 30, "slice_size": 3}),
            PipelineStep("embedder", "gemini_embedder", {"model": "models/text-embedding-004"}),
            PipelineStep("vector_store", "faiss_store"),
            PipelineStep("retriever", "cosine_similarity", {"top_k": 15}),
            PipelineStep("prompt_builder", "cot_rag", {"include_reasoning": True}),
            PipelineStep("generator", "gemini_generator", {"model": "gemini-1.5-pro"}),
            PipelineStep("iteration_strategy", "self_refine", {"max_iterations": 3}),
            PipelineStep("evaluator", "pass_at_k", {"k_values": [1, 3, 5]}),
        ]
    )
    
    pipeline_registry.register(custom_spec)
    
    # Validate
    errors = pipeline_registry.validate_pipeline("custom_gemini_self_refine")
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Custom pipeline registered and validated successfully!")
        print(f"Steps: {[f'{s.component_type}.{s.component_id}' for s in custom_spec.steps]}")


def main():
    """Run all examples."""
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    run_standard_rag_example()
    run_repocoder_example()
    run_comparison_example()
    create_custom_pipeline()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()