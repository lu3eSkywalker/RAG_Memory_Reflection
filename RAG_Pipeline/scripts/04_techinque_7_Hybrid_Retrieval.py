#!/usr/bin/env python3
"""
Technique 7: Hybrid Retrieval (semantic + BM25).

Thin CLI wrapper - the implementation lives in `retrieval/hybrid_retriever.py`.

Usage:
    python scripts/04_techinque_7_Hybrid_Retrieval.py
    python scripts/04_techinque_7_Hybrid_Retrieval.py \
        --query "Why is validate_jwt_token failing?" --top-k 10 \
        --vector-weight 0.5 --bm25-weight 0.5 --fusion rrf
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()