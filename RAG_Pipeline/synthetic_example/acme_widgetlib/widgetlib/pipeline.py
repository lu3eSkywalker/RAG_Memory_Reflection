"""High-level pipeline orchestration for acme_widgetlib."""

from widgetlib.core.engine import WidgetEngine
from widgetlib.utils.text import slugify_label


def run_pipeline(raw_items, mode="strict", label=None):
    engine = WidgetEngine(mode=mode)
    results = engine.process_batch(raw_items, mode=mode)
    # HOLE: aggregate ok results into a summary dict with keys
    # 'total', 'passed', 'failed', 'pass_rate', 'label'.
    # Use slugify_label on label when provided.
    pass


def filter_and_retry(raw_items, mode="strict"):
    engine = WidgetEngine(mode=mode)
    results = engine.process_batch(raw_items, mode=mode)
    # HOLE: return (passed_results, retry_queue) where retry_queue comes from
    # engine.retry_failed(results).
    pass