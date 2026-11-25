"""Core processing engine for acme_widgetlib."""

from widgetlib.config.defaults import MAX_RETRY, BATCH_CEILING
from widgetlib.utils.text import normalize_sku


class WidgetResult:
    def __init__(self, sku, score, ok):
        self.sku = sku
        self.score = score
        self.ok = ok


class WidgetEngine:
    def __init__(self, mode="strict", max_retry=None):
        self.mode = mode
        self.max_retry = max_retry if max_retry is not None else MAX_RETRY

    def process_batch(self, items, mode=None):
        """Process a list of raw item dicts into WidgetResult objects.

        items: iterable of dicts with keys 'raw_sku' and 'score'.
        mode: optional override for self.mode.
        Returns list of WidgetResult, capped at BATCH_CEILING items.
        """
        active_mode = mode or self.mode
        capped = list(items)[:BATCH_CEILING]
        results = []
        for item in capped:
            sku = normalize_sku(item.get("raw_sku", ""))
            score = float(item.get("score", 0.0))
            ok = score >= 0.5 if active_mode == "strict" else True
            results.append(WidgetResult(sku=sku, score=score, ok=ok))
        return results

    def retry_failed(self, results):
        """Return only results whose ok is False, up to max_retry count."""
        failed = [r for r in results if not r.ok]
        return failed[:self.max_retry]