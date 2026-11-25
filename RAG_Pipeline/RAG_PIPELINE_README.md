# Modular RAG Pipeline System

A composable, extensible RAG pipeline framework inspired by **RepoCoder's** iterative retrieval-generation architecture.

## Features

- **Pluggable Components**: Swap chunkers, embedders, retrievers, prompt builders, generators, iteration strategies, evaluators
- **YAML Configuration**: Define pipelines declaratively in `rag_pipeline_spec.yaml`
- **Multiple Iteration Strategies**: Single-pass, RepoCoder iterative, Self-Refine, Active RAG
- **RepoCoder Built-in**: Sliding window chunking, BoW/Jaccard retrieval, two-round iterative RG
- **Extensible**: Add custom components via Python modules or YAML
- **Pipeline Comparison**: Run multiple pipelines on same queries for benchmarking

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Chunker    │──▶│  Embedder   │──▶│  Retriever  │──▶│ PromptBuilder│
│ (sliding    │   │ (BoW,       │   │ (Jaccard,   │   │ (RepoCoder,  │
│  window,    │   │  Ada002,    │   │  Cosine,    │   │  Standard,   │
│  semantic)  │   │  ST)        │   │  Hybrid)    │   │  CoT)        │
└─────────────┘   └─────────────┘   └─────────────┘   └──────┬──────┘
                                                              │
                                                              ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Evaluator  │◀──│  Generator  │◀──│ Iteration   │◀──│  (Prompt)   │
│ (EM, Edit,  │   │ (OpenAI,    │   │ Strategy    │   │             │
│  Pass@k)    │   │  HF, vLLM)  │   │ (Single,    │   │             │
└─────────────┘   └─────────────┘   │  RepoCoder, │   │             │
                                    │  SelfRefine)│   │             │
                                    └─────────────┘   └─────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install pyyaml numpy scipy

# Run examples
python example_usage.py
```

## Configuration

All pipelines defined in [`rag_pipeline_spec.yaml`](rag_pipeline_spec.yaml):

```yaml
pipelines:
  - id: "repocoder_full"
    name: "RepoCoder Iterative"
    steps:
      - chunker: "sliding_window"
      - embedder: "bag_of_words"
      - retriever: "jaccard_similarity"
      - prompt_builder: "repo_coder_style"
      - generator: "openai_codex"
      - iteration: "repocoder_iterative"
      - evaluator: ["exact_match", "edit_similarity"]
```

## Built-in Pipelines

| Pipeline ID | Description |
|-------------|-------------|
| `standard_rag` | Single-pass retrieve-then-generate |
| `repocoder_rg1` | RepoCoder round 1 (vanilla RAG) |
| `repocoder_full` | RepoCoder 2-round iterative |
| `repocoder_ada002` | RepoCoder with OpenAI embeddings |
| `hybrid_self_refine` | Hybrid retrieval + self-refinement |
| `active_rag` | Multi-hop retrieval with confidence stopping |

## Adding Custom Components

### 1. Create Python module (`my_components/chunkers.py`)

```python
from rag_core.base import Chunker, Document

class MyCustomChunker(Chunker):
    def chunk(self, text: str, metadata: dict = None) -> List[Document]:
        # Your chunking logic
        return [Document(content=chunk, metadata=metadata) for chunk in chunks]
```

### 2. Register in `my_components/__init__.py`

```python
REGISTER_COMPONENTS = [
    {
        "component_id": "my_custom_chunker",
        "class_path": "my_components.chunkers.MyCustomChunker",
        "config": {"param": "value"},
        "description": "My custom chunker"
    }
]
```

### 3. Load in code

```python
component_registry.load_extensions({
    "chunker": "my_components",
})
```

## Component Types

| Type | Base Class | Examples |
|------|------------|----------|
| Chunker | `Chunker` | SlidingWindow, FixedSize, Semantic |
| Embedder | `Embedder` | BagOfWords, OpenAI, SentenceTransformer |
| Retriever | `Retriever` | Jaccard, Cosine, Hybrid BM25+Vector |
| PromptBuilder | `PromptBuilder` | RepoCoder, Standard, CoT |
| Generator | `Generator` | OpenAI, HuggingFace, vLLM |
| IterationStrategy | `IterationStrategy` | SinglePass, RepoCoderIterative, SelfRefine, ActiveRAG |
| Evaluator | `Evaluator` | ExactMatch, EditSimilarity, PassAtK |

## RepoCoder-Specific Components

The system includes faithful implementations of RepoCoder's techniques:

- **SlidingWindowChunker**: `window_size=20`, `slice_size=2` (10-line stride)
- **BagOfWordsEmbedder**: 1-gram tokenization for code
- **JaccardRetriever**: With future-context filtering (excludes chunks after hole)
- **RepoCoderPromptBuilder**: File paths + commented code blocks, token budget
- **RepoCoderIterativeStrategy**: Round 1 on query, Round 2 on predictions

## Programmatic Pipeline Creation

```python
from rag_core import PipelineSpec, PipelineStep, ComponentRegistry, PipelineRegistry, RAGPipeline

# Build pipeline in code
spec = PipelineSpec(
    pipeline_id="my_pipeline",
    name="Custom Pipeline",
    steps=[
        PipelineStep("chunker", "sliding_window", {"window_size": 30}),
        PipelineStep("embedder", "sentence_transformer"),
        PipelineStep("retriever", "cosine_similarity", {"top_k": 10}),
        PipelineStep("prompt_builder", "standard_rag"),
        PipelineStep("generator", "vllm_local"),
        PipelineStep("iteration_strategy", "single_pass"),
        PipelineStep("evaluator", "pass_at_k", {"k_values": [1, 5]}),
    ]
)

pipeline_registry.register(spec)
pipeline = RAGPipeline(spec, component_registry)
context = pipeline.run(query="def foo():", ground_truth="    pass")
```

## Directory Structure

```
rag_core/           # Core framework (registry, pipeline, context, base classes)
rag_components/     # Reference implementations (RepoCoder + standard RAG)
rag_pipeline_spec.yaml  # Declarative pipeline definitions
example_usage.py    # Usage examples
```

## Extending for Your Use Case

1. **Add new chunker**: Implement `Chunker` interface
2. **Add new embedder**: Implement `Embedder` interface  
3. **Add new retriever**: Implement `Retriever` interface
4. **Add new iteration strategy**: Implement `IterationStrategy` for custom multi-round logic
5. **Register via YAML or Python** and use in any pipeline

## License

MIT License - Microsoft Research