#!/usr/bin/env python3
"""
Technique 5: Multi-Query Retrieval (Query Expansion).

Thin CLI wrapper - the implementation lives in `retrieval/multiquery_retriever.py`.

Usage:
    python scripts/04_techinque_5_MultiQuery.py
    python scripts/04_techinque_5_MultiQuery.py \
        --query "Fix the authentication bug in the login API" \
        --method keywords,patterns,vocab,subq --top-k 5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.multiquery_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()