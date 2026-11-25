# Synthetic (non-memorizable) benchmark example

This folder shows the *shape* of data that differentiates RAG techniques.

## Why this is harder for the LLM
- Repo name `acme_widgetlib` and APIs (`WidgetEngine`, `normalize_sku`,
  `BACKOFF_MS`, `slugify_label`) are bespoke — not in pretraining.
- Completions are 5-12 lines and depend on repo-specific helpers/constants
  defined in OTHER files (cross-file dependency).
- The prompt never contains the helper definition, so the model must retrieve
  it; without retrieval the answer is under-determined.

## How to plug into your pipeline
1. Copy `acme_widgetlib/` into your embed root
   (`data_to_embed/acme_widgetlib`).
2. Re-run chunking/embedding/vector store scripts (01,02,03).
3. Put `tasks.jsonl` into your questions folder.
4. Run retrieval per technique (script 11) and generation (12).

## Task design checklist (use when authoring your own)
- Ground truth 8-40 lines, not 1-3.
- Completion calls at least one helper/constant defined in a *different* file.
- Names are repo-specific (no `load_data`, `compute`, `assertEqual`).
- Avoid popular frameworks; prefer internal conventions.
- Include cases that exercise each retriever's strength:
  * exact identifier -> BM25
  * paraphrased concept -> semantic
  * import/call-graph hop -> cross_file / relationship
  * ambiguous query -> multiquery / iterative
  * mixed -> hybrid / adaptive
- Add a no-retrieval baseline and retrieval metrics (Recall@k, MRR, nDCG).
