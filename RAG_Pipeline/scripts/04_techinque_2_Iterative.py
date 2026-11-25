#!/usr/bin/env python3
"""
Technique 2: Iterative Retrieval (multi-round, gap driven).

Thin CLI wrapper - the implementation lives in `retrieval/iterative_retriever.py`.

Usage:
    python scripts/04_techinque_2_Iterative.py
    python scripts/04_techinque_2_Iterative.py \
        --query "Add authentication to the user API" --rounds 3 --top-k 5 --use-ast
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.iterative_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()