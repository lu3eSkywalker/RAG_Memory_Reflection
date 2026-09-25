# Modular RAG Pipeline Scripts

This directory contains individual scripts to run each component of the RAG pipeline independently. This allows you to test and debug each stage separately.

## Scripts

| Script | Description |
|--------|-------------|
| `01_run_chunking.py` | Test document chunking strategies (sliding window, fixed size, semantic) |
| `02_run_embedding.py` | Generate embeddings for documents/queries using SentenceTransformers |
| `03_run_vector_store.py` | Build and search FAISS vector index (with mock fallback) |
| `04_run_retrieval.py` | Retrieve relevant chunks for a query using cosine similarity |
| `05_run_prompt.py` | Build prompts from retrieved documents (RepoCoder, Standard RAG, CoT) |
| `06_run_generation.py` | Generate completions using LLM (Gemini or mock) |
| `07_run_evaluation.py` | Evaluate predictions against ground truth (exact match, edit similarity, pass@k) |
| `08_run_pipeline.py` | Run full pipeline end-to-end with manual step-by-step execution |

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