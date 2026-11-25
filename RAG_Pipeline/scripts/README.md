# Modular RAG Pipeline Scripts

This directory contains individual scripts to run each component of the RAG pipeline independently. This allows you to test and debug each stage separately.

## Retrieval techniques

Every retrieval technique is implemented in the `retrieval/` package and can be
run directly with `python -m retrieval.<module>`. The numbered scripts below are
thin wrappers around that package, so both forms behave identically.

| Module (`retrieval/`) | Script | Technique |
|-----------|--------|-----------|
| `semantic_retriever.py` | `04_run_retrieval.py` | Dense embeddings + FAISS cosine retrieval |
| `ast_retriever.py` | `04_techique_1_AST_based_Retrieval.py` | AST / structural retrieval |
| `relationship_index.py` | — | Cross-file relationship graph (calls, extends, imports, tests) |
| `cross_file_retriever.py` | `04_techinque_4_Cross_file_Retrieval.py` | Multi-hop cross-file retrieval |
| `iterative_retriever.py` | `04_techinque_2_Iterative.py` | Iterative (multi-round) retrieval |
| `adaptive_retriever.py` | `04_techinque_3_Adaptive_Retrieval.py` | Adaptive top-k + stopping |
| `multiquery_retriever.py` | `04_techinque_5_MultiQuery.py` | Query expansion + RRF rerank |
| `bm25_retriever.py` | `04_techinque_6_BM25_Keyword_Retrieval.py` | Code-aware BM25 |
| `hybrid_retriever.py` | `04_techinque_7_Hybrid_Retrieval.py` | Semantic + BM25 fusion |

Every retriever exposes the same interface:

```python
from retrieval import HybridRetriever, SemanticRetriever

retriever = SemanticRetriever()          # or HybridRetriever(vector, bm25, config)
hits = retriever.retrieve_documents("validate_jwt_token", top_k=5)
result = retriever.retrieve_result("validate_jwt_token")   # rag_core RetrievalResult
```

## Scripts

| Script | Description |
|--------|-------------|
| `01_run_chunking.py` | Test document chunking strategies (sliding window, fixed size, semantic) |
| `02_run_embedding.py` | Generate embeddings for documents/queries using SentenceTransformers |
| `03_run_vector_store.py` | Build and search FAISS vector index (with mock fallback) |
| `04_run_retrieval.py` | Dense retrieval baseline (wrapper around `retrieval.semantic_retriever`) |
| `04_techique_*` | One script per advanced retrieval technique (see table above) |
| `05_run_prompt.py` | Build prompts from retrieved documents (RepoCoder, Standard RAG, CoT) |
| `06_run_generation.py` | Generate completions using LLM (Gemini or mock) |
| `07_run_evaluation.py` | Evaluate predictions against ground truth (exact match, edit similarity, pass@k) |
| `08_run_pipeline.py` | Run full pipeline end-to-end with manual step-by-step execution |
| `09_reflection.py` | Reflection + memory extraction over a completed generation |
| `10.full_evaluation.py` | Edit-similarity evaluation of generated code vs reference code (two files in, one score out) |

## Usage

Run any script directly using the project's virtual environment:

```bash
# From project root
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/01_run_chunking.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/02_run_embedding.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/03_run_vector_store.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/04_run_retrieval.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/05_run_prompt.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/06_run_generation.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/07_run_evaluation.py
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/python scripts/08_run_pipeline.py
```

Or run a retrieval technique straight from the package:

```bash
venv/bin/python -m retrieval.ast_retriever --repo /path/to/repo --query "add auth" 
venv/bin/python -m retrieval.hybrid_retriever --query "add auth" --fusion rrf --with-ast
```

## Prerequisites

Install required packages (using the venv):

```bash
/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/venv/bin/pip install sentence-transformers faiss-cpu google-generativeai numpy scipy
```

Set environment variable for Gemini (optional - uses mock generator if not set):

```bash
export GEMINI_API_KEY="your-api-key"
```

## Pipeline Flow

```
01_run_chunking.py → 02_run_embedding.py → 03_run_vector_store.py
                                                    ↓
07_run_evaluation.py ← 06_run_generation.py ← 05_run_prompt.py ← 04_run_retrieval.py
```

Or run the complete pipeline:
```
08_run_pipeline.py
```

## Notes

- All scripts are self-contained and don't modify the core code in `rag_core/` or `rag_components/`
- Scripts use mock implementations when dependencies (sentence-transformers, faiss, google-generativeai) are not available
- The original `rag_components/__init__.py` has import issues - these scripts work around them by implementing the component logic directly
- Core pipeline infrastructure (`rag_core/`) is used for base classes, context, and registry