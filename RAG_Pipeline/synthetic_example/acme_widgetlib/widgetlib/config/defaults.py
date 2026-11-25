"""Non-standard configuration constants for acme_widgetlib.

These names are intentionally bespoke so a model cannot rely on
pretraining knowledge of a popular framework.
"""

MAX_RETRY = 7
BACKOFF_MS = 125
SKU_PREFIX = "WX-"
FUZZY_THRESHOLD = 0.83
BATCH_CEILING = 64
"""Maximum items accepted by process_batch in one call."""