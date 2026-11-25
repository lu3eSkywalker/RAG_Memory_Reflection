#!/usr/bin/env python3
"""
Technique 6: BM25 / Keyword Retrieval (code-aware).

Thin CLI wrapper - the implementation lives in `retrieval/bm25_retriever.py`.

Usage:
    python scripts/04_techinque_6_BM25_Keyword_Retrieval.py
    python scripts/04_techinque_6_BM25_Keyword_Retrieval.py \
        --query "Why is validate_jwt_token failing?" --corpus both --top-k 5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()