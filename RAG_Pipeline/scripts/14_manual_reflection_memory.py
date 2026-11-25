#!/usr/bin/env python3
"""
14_manual_reflection_memory.py — manual reflection (no LLM).

`09_reflection.py` needs the Gemini API to turn an episode into a STORE /
NO_MEMORY decision. When the API is unavailable, this script performs the same
two stages by hand:

    1. a hand-written MemoryCandidate (episodic or bug) distilled from an
       evaluation run,
    2. optional de-duplication against the existing store (cosine similarity,
       same threshold as Reflector: 0.92),
    3. persistence through MemoryManager -> BasicMemoryExtractor ->
       VectorMemoryStore -> the shared memory_store.pkl.

No component writes the pickle directly and no path is hardcoded here: the
store path comes from memory.config.DEFAULT_STORE_PATH.

Usage:
    python scripts/14_manual_reflection_memory.py --dry-run
    python scripts/14_manual_reflection_memory.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PIPELINE_DIR = Path.home() / "Downloads" / "memory pipeline"
if str(MEMORY_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_PIPELINE_DIR))

from memory.manager import MemoryManager  # noqa: E402
from memory.config import DEFAULT_STORE_PATH  # noqa: E402

SIMILARITY_THRESHOLD = 0.92

# ---------------------------------------------------------------------------
# Memories distilled from the 30-case, 8-technique evaluation
# (reports_by_RAG_Techniques/*/evaluation_report_{raw,fair}.json).
#
# episodic -> task/context/action/result
# bug      -> bug(result)/cause/fix + context
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "memory_type": "episodic",
        "task": "Find the hard cases of function-level code completion on trlx and tracr",
        "context": (
            "30 cases x 8 retrieval techniques. Only 9 cases ever failed. "
            "Mean edit similarity: cross_file 1.000, relationship 0.989, bm25 0.977, "
            "multiquery 0.941, hybrid 0.934, iterative 0.917, semantic 0.901, adaptive 0.857."
        ),
        "action": (
            "Retrieve the exact enclosing file and its sibling definitions; keep semantic-only "
            "retrieval as a fallback, not the primary strategy, when a function has divergent variants."
        ),
        "result": (
            "22/30 cases were solved exactly by every technique; the remaining errors are "
            "repository-specific variants, not reasoning failures."
        ),
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx BaseTrainer.setup_scheduler",
        "context": "trlx/trainer/accelerate_base_trainer.py; adaptive and semantic scored 0.978.",
        "bug": "setup_scheduler was completed with scheduler_class(self.optimizer, ...).",
        "cause": (
            "trlx BaseTrainer binds the optimizer as self.opt; prompts without a same-file "
            "retrieval hint invite the generic self.optimizer name."
        ),
        "fix": "scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)",
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx ILQLHeads.sync_target_q_heads",
        "context": (
            "trlx/models/modeling_ilql.py; adaptive, hybrid, iterative and multiquery scored 0.074."
        ),
        "bug": "sync_target_q_heads was completed as a bare self._sync_target_q_heads(self.alpha).",
        "cause": (
            "The repository version also handles DeepSpeed ZeRO-3, where the q-head parameters "
            "are sharded and must be gathered before copying."
        ),
        "fix": (
            "If DEEPSPEED_ZERO_STAGE == '3', chain q_heads and target_q_heads parameters, gather "
            "them with deepspeed.zero.GatheredParameters(params, modifier_rank=0) and sync on rank 0; "
            "otherwise call self._sync_target_q_heads(self.alpha)."
        ),
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx batched_index_select",
        "context": "trlx/models/modeling_ilql.py; five techniques scored 0.29-0.66.",
        "bug": "batched_index_select was completed with a docstring and a generic expand(-1, -1, last) gather.",
        "cause": (
            "The repository version has no docstring and expands the index explicitly to "
            "idxs.shape[0], idxs.shape[1], x.shape[-1] before gathering."
        ),
        "fix": (
            "idxs = idxs.unsqueeze(-1).expand(idxs.shape[0], idxs.shape[1], x.shape[-1]); "
            "return x.gather(dim=dim, index=idxs)"
        ),
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx get_global_statistics",
        "context": "trlx/utils/modeling.py; bm25 0.320 and iterative 0.348.",
        "bug": "get_global_statistics was completed with a torch.stack stats tensor or per-rank torch.mean/var.",
        "cause": (
            "The repository version reduces raw sums (x and x squared) across processes and then "
            "derives the mean and variance from them."
        ),
        "fix": (
            "sum_and_count = tensor([xs.sum(), xs.numel()]); dist.all_reduce(sum_and_count, SUM); "
            "global_mean = sum / count; sum_var = torch.sum((xs - global_mean) ** 2); all_reduce; "
            "global_var = sum_var / count; return global_mean, global_var, count."
        ),
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx BaseTrainer.save_pretrained",
        "context": "trlx/trainer/accelerate_base_trainer.py; adaptive 0.180 and semantic 0.337.",
        "bug": "save_pretrained was completed as a three-line unwrap-and-save.",
        "cause": (
            "The repository version documents its arguments, defaults to a hf_model subdirectory, "
            "waits for every process, and saves the tokenizer on the main process only."
        ),
        "fix": (
            "directory = directory or f'{checkpoint_dir}/hf_model'; accelerator.wait_for_everyone(); "
            "unwrap_model(self.model).save_pretrained(directory, **kwargs); if is_main_process: "
            "self.tokenizer.save_pretrained(directory)."
        ),
    },
    {
        "memory_type": "bug",
        "task": "Complete trlx ShuffledCyclicSequence.__init__",
        "context": "trlx/trainer/nemo_ilql_trainer.py; four techniques scored 0.47-0.62.",
        "bug": "ShuffledCyclicSequence.__init__ was completed with a numpy RNG.",
        "cause": (
            "The repository version builds a torch.Generator seeded with the given seed and uses "
            "torch.randperm(new_length), not numpy."
        ),
        "fix": (
            "self.data = data; self.new_length = new_length; rng = torch.Generator().manual_seed(seed); "
            "self.perm = torch.randperm(new_length, generator=rng, device='cpu')"
        ),
    },
]


def as_text(candidate: dict) -> str:
    if candidate["memory_type"] == "bug":
        return (
            f"Bug: {candidate['bug']}\nCause: {candidate['cause']}\n"
            f"Fix: {candidate['fix']}\nContext: {candidate['context']}"
        )
    return (
        f"Task: {candidate['task']}\nContext: {candidate['context']}\n"
        f"Action: {candidate['action']}\nResult: {candidate['result']}"
    )


def remember_kwargs(candidate: dict) -> dict:
    if candidate["memory_type"] == "bug":
        return {
            "task": candidate["task"],
            "context": candidate["context"],
            "action": candidate["fix"],
            "result": candidate["bug"],
            "test_result": "FAIL",
            "reflection": {"cause": candidate["cause"], "fix": candidate["fix"]},
        }
    return {
        "task": candidate["task"],
        "context": candidate["context"],
        "action": candidate["action"],
        "result": candidate["result"],
        "test_result": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show decisions, write nothing")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    manager = MemoryManager(use_default_store=True, persist=not args.dry_run)
    print(f"store : {DEFAULT_STORE_PATH}")
    print(f"before: {manager.store.count()} memories\n")

    stored = skipped = 0
    for candidate in CANDIDATES:
        text = as_text(candidate)
        duplicate = next(
            (
                (score, memory)
                for score, memory in manager.store.similarities(text)
                if memory.memory_type == candidate["memory_type"] and score >= args.threshold
            ),
            None,
        )
        label = f"{candidate['memory_type']:<8} {candidate.get('task', '')[:60]}"
        if duplicate:
            score, memory = duplicate
            print(f"SKIP  {label}  (duplicate {memory.id[:8]}, similarity {score:.3f})")
            skipped += 1
            continue

        if args.dry_run:
            print(f"WOULD STORE  {label}")
            stored += 1
            continue

        memories = manager.remember(**remember_kwargs(candidate))
        for memory in memories:
            print(f"STORE {label}  -> {memory.id} ({memory.memory_type})")
        stored += len(memories)

    print(f"\nstored: {stored}   skipped: {skipped}   total now: {manager.store.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
