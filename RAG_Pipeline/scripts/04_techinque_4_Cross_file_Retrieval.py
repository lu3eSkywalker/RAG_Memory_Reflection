#!/usr/bin/env python3
"""
Technique 4: Cross-File Retrieval (relational, multi-hop).

Thin CLI wrapper - the implementation lives in
`retrieval/cross_file_retriever.py` with `retrieval/relationship_index.py`.

Usage:
    python scripts/04_techinque_4_Cross_file_Retrieval.py
    python scripts/04_techinque_4_Cross_file_Retrieval.py \
        --query "authenticate user request" --hops 2 --max-per-file 2
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.cross_file_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()