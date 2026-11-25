#!/usr/bin/env python3
"""
Technique 3: Adaptive Retrieval / Adaptive Stopping.

Thin CLI wrapper - the implementation lives in `retrieval/adaptive_retriever.py`.

Usage:
    python scripts/04_techinque_3_Adaptive_Retrieval.py
    python scripts/04_techinque_3_Adaptive_Retrieval.py \
        --query "predict anomaly score" --min-k 2 --max-k 20 --step 4
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.adaptive_retriever import main  # noqa: E402

if __name__ == "__main__":
    main()