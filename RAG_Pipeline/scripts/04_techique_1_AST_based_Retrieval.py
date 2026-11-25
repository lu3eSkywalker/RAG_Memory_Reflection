#!/usr/bin/env python3
"""
Technique 1: AST-Based (structural) Retrieval.

Thin CLI wrapper - the implementation lives in `retrieval/ast_retriever.py`.

Usage:
    python scripts/04_techique_1_AST_based_Retrieval.py
    python scripts/04_techique_1_AST_based_Retrieval.py \
        --repo /path/to/repo --query "Add authentication middleware" --top-k 8
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.ast_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()