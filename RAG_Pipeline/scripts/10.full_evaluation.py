#!/usr/bin/env python3
"""
10.full_evaluation.py - Edit-similarity evaluation of generated code vs reference code.

Takes two files and reports how close the generated code is to the correct code.

    file_A  predictions - what the RAG pipeline generated
    file_B  ground truth - the correct code

Primary metric is normalized Levenshtein edit similarity; exact match is reported
as a secondary column.

    edit_similarity = 1 - levenshtein(pred, gold) / max(len(pred), len(gold))
    exact_match     = pred.strip() == gold.strip()

Edit similarity is the primary metric because it degrades smoothly. Exact match
collapses "truncated at 60%" and "returned nothing" into the same 0.0, which makes
a partially working pipeline indistinguishable from a broken one. A similarity
metric is not a correctness metric: a confidently wrong answer (renamed
identifier, wrong subfolder) scores ~0.95 and that is the intended behaviour -
it means "structurally almost right", not "semantically correct".

Deterministic, free, reproducible run to run.

Supported input shapes
----------------------
    *.jsonl   one JSON object per line; text read from a field path
    *.json    a list of strings, a list of objects, or {key: [...]}
    *.txt     the whole file is one prediction / one gold answer
    directory all *.py (or --glob) files below it, paired by relative path

Field paths are sniffed automatically; override with --pred-field / --gold-field.

Usage
-----
    # predictions first, ground truth second
    python scripts/10.full_evaluation.py generated.py correct.py

    # gold first, or explicit roles
    python scripts/10.full_evaluation.py correct.jsonl generated.jsonl --swap
    python scripts/10.full_evaluation.py --predictions a.jsonl --gold b.jsonl

    # the RepoCoder question set against a predictions jsonl
    python scripts/10.full_evaluation.py \
        --gold  /home/sooraj/Downloads/data_for_rag/questions/line_level_completion_2k_context_codex.test.jsonl \
        --pred path/to/predictions.jsonl --pred-field prediction --output report.json

    # whole-file comparison of two code trees
    python scripts/10.full_evaluation.py --pred generated_dir/ --gold correct_dir/

Options
-------
    --limit N        evaluate only the first N pairs
    --min-score S    only list pairs scoring below S in the table
    --rows N         how many table rows to print (default 20)
    --no-json        do not write the JSON report
    --include-text   embed the raw prediction and gold text in the JSON report
    --no-fence-strip score the raw text, fences and all
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------


def levenshtein_reference(a: str, b: str) -> int:
    """Classic O(len(a) * len(b)) Levenshtein distance. Reference implementation."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _levenshtein_myers(a: str, b: str) -> int:
    """Myers bit-parallel Levenshtein distance.

    Bit-vector algorithm: one pass over `a` with O(len(b) / 64) word operations,
    so a 1000 x 1000 character pair is milliseconds instead of seconds.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)

    m = len(b)
    # peq[c] has bit j set when b[j] == c
    peq: Dict[str, int] = {}
    for j, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << j)

    mask = (1 << m) - 1
    top = 1 << (m - 1)
    pv = mask
    mv = 0
    score = m

    for ch in a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | (~(xh | pv) & mask)) & mask
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | (~(xv | ph) & mask)) & mask
        mv = (ph & xv) & mask

    return score


_BACKEND_OK: Optional[bool] = None


def _verify_backend() -> bool:
    """Check the fast backend against the reference on a fixed battery, once."""
    global _BACKEND_OK
    if _BACKEND_OK is not None:
        return _BACKEND_OK
    battery = [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
        ("kitten", "sitting"),
        ("sitting", "kitten"),
        ("    unet = UNet2DConditionModel.from_pretrained(\n", "    vae = UNet2DConditionModel.from_pretrained("),
        ("StableDiffusionInpaintPipelineLegacy,", "StableDiffusionInpaintPipelineXLegacy,"),
        ("return b", "return   b"),
        ("def f():\n    pass", "def g():\n    pass"),
        ("\u00e9\u00e8\u00ea", "\u00e9\u00e8"),
        ("a" * 200, "a" * 199 + "b"),
        ("abcabcabc", "bcabcabca"),
    ]
    try:
        _BACKEND_OK = all(
            _levenshtein_myers(a, b) == levenshtein_reference(a, b) for a, b in battery
        )
    except Exception:  # pragma: no cover - defensive
        _BACKEND_OK = False
    return _BACKEND_OK


def levenshtein(a: str, b: str) -> int:
    """Levenshtein distance, with common prefix/suffix trimming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # A shared prefix or suffix can never be part of an optimal edit script,
    # so trimming it is exact, not an approximation.
    limit = min(len(a), len(b))
    start = 0
    while start < limit and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a, b = a[start:end_a], b[start:end_b]

    if not _verify_backend():
        return levenshtein_reference(a, b)
    return _levenshtein_myers(a, b)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def edit_similarity(prediction: str, gold: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]. 1.0 for two empty strings."""
    pred = prediction.strip()
    truth = gold.strip()
    longest = max(len(pred), len(truth))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(pred, truth) / longest


def exact_match(prediction: str, gold: str) -> bool:
    """True when the two agree after stripping outer whitespace."""
    return prediction.strip() == gold.strip()


def line_match(prediction: str, gold: str) -> float:
    """Fraction of non-empty gold lines that appear in order inside the prediction.

    Multi-line golds (api level averages 2.1 lines, function level 9.9) get a
    gradient that single-number exact match cannot express.
    """
    gold_lines = [line.strip() for line in gold.strip().splitlines() if line.strip()]
    if not gold_lines:
        return 1.0
    pred_lines = [line.strip() for line in prediction.strip().splitlines()]
    matched = 0
    cursor = 0
    for line in pred_lines:
        if cursor < len(gold_lines) and line == gold_lines[cursor]:
            matched += 1
            cursor += 1
    return matched / len(gold_lines)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)

GOLD_FIELDS = (
    "metadata.ground_truth",
    "ground_truth",
    "gold",
    "target",
    "reference",
    "expected",
    "answer",
    "solution",
)
PRED_FIELDS = (
    "prediction",
    "completion",
    "generated_code",
    "generated",
    "output",
    "pred",
    "response",
    "text",
    "code",
    "answer",
)
ID_FIELDS = ("task_id", "id", "metadata.task_id", "name")


def get_field(obj: Any, path: str) -> Optional[str]:
    """Read a dotted path out of nested dicts. Returns None if absent or not a string."""
    current: Any = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, str) else None


def find_path(obj: Any, name: str, depth: int = 0) -> Optional[str]:
    """Locate a string-valued key called `name` anywhere in a nested dict.

    Records are not always flat - the same logical field shows up as
    `ground_truth` in one dataset and `metadata.ground_truth` in another. This
    resolves the key by name so both work, and returns the dotted path so the
    rest of the script keeps using exact lookups.
    """
    if not isinstance(obj, dict) or depth > 5:
        return None
    if isinstance(obj.get(name), str):
        return name
    for key, value in obj.items():
        if isinstance(value, dict):
            found = find_path(value, name, depth + 1)
            if found:
                return f"{key}.{found}"
    return None


def resolve_field(
    records: Sequence[Any],
    wanted: Optional[str],
    candidates: Sequence[str],
) -> Optional[str]:
    """Resolve the dotted path holding the text, from an override or by sniffing.

    Returns "" for files that are already plain strings.
    """
    if records and isinstance(records[0], str):
        return ""

    sample = [record for record in records[:50] if isinstance(record, dict)]
    if not sample:
        return wanted

    if wanted is not None:
        for record in sample:
            if get_field(record, wanted) is not None:
                return wanted
        found = find_path(sample[0], wanted.rsplit(".", 1)[-1])
        if found:
            return found
        raise SystemExit(
            f"field '{wanted}' not found. Keys seen: "
            f"{sorted({key for record in sample[:5] for key in record})}"
        )

    for candidate in candidates:
        path = find_path(sample[0], candidate.rsplit(".", 1)[-1])
        if path is None:
            continue
        hits = sum(1 for record in sample if get_field(record, path) is not None)
        if hits >= max(1, len(sample) // 2):
            return path
    return None


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced code block, or the text unchanged."""
    match = FENCE_RE.search(text)
    return match.group(1) if match else text


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class FileData:
    """Records loaded from one input file, plus how the text was located."""

    def __init__(
        self,
        path: Path,
        records: List[Any],
        field: Optional[str],
        kind: str,
    ) -> None:
        self.path = path
        self.records = records
        self.field = field
        self.kind = kind
        self._contents: Optional[List[str]] = None
        self._ids: Optional[List[str]] = None

    @property
    def label(self) -> str:
        if self.kind == "directory":
            return f"{len(self.records)} files, pattern matched"
        if self.field == "":
            return "plain strings"
        if self.field:
            return f"json field '{self.field}'"
        return self.kind

    def keys(self) -> List[str]:
        """Pairing keys: relative file paths for directories, '' otherwise."""
        if self.kind == "directory":
            return [str(record) for record in self.records]
        return [""] * len(self.records)

    def texts(self) -> List[str]:
        if self.kind == "directory":
            if self._contents is None:
                self._contents = [
                    (self.path / record).read_text(encoding="utf-8", errors="replace")
                    for record in self.records
                ]
            return self._contents
        if self.field == "":
            return [
                record if isinstance(record, str) else str(record)
                for record in self.records
            ]
        return [get_field(record, self.field) or "" for record in self.records]

    def resolve(self, name: str) -> Optional[str]:
        """Resolve any key of this file by name, ignoring nesting differences."""
        if self.field == "":
            return None
        sample = next(
            (record for record in self.records[:50] if isinstance(record, dict)), None
        )
        if sample is None:
            return None
        if get_field(sample, name) is not None:
            return name
        return find_path(sample, name.rsplit(".", 1)[-1])

    def ids(self) -> List[str]:
        if self._ids is None:
            if self.kind == "directory":
                self._ids = list(self.keys())
            else:
                resolved = [self.resolve(name) for name in ID_FIELDS]
                resolved = [path for path in resolved if path]
                if resolved:
                    self._ids = [
                        (get_field(record, resolved[0]) or "") for record in self.records
                    ]
                else:
                    self._ids = [""] * len(self.records)
        return self._ids


def load_jsonl(path: Path, field_override: Optional[str], candidates: Sequence[str]) -> FileData:
    records: List[Any] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({error})")
    field = resolve_field(records, field_override, candidates)
    if field is None:
        raise SystemExit(
            f"{path}: no known text field found. Pass the field name explicitly, "
            f"e.g. --pred-field metadata.ground_truth. Top-level keys seen: "
            f"{sorted({key for record in records[:5] if isinstance(record, dict) for key in record})}"
        )
    return FileData(path, records, field, "jsonl")


def load_json(path: Path, field_override: Optional[str], candidates: Sequence[str]) -> FileData:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    records: List[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []
        for key in ("data", "records", "items", "results", "rows", "answers", "predictions"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            raise SystemExit(
                f"{path}: JSON object has no list of records. Keys seen: {sorted(payload)}"
            )
    else:
        raise SystemExit(f"{path}: unsupported JSON top level ({type(payload).__name__})")
    field = resolve_field(records, field_override, candidates)
    if field is None:
        raise SystemExit(f"{path}: no known text field found in the records.")
    return FileData(path, records, field, "json")


def load_text(path: Path, field_override: Optional[str], split_lines: bool) -> FileData:
    text = path.read_text(encoding="utf-8", errors="replace")
    if split_lines:
        records = [
            line for line in (raw.rstrip("\n") for raw in text.splitlines()) if line.strip()
        ]
        return FileData(path, records, "", "lines")
    return FileData(path, [text], field_override if field_override is not None else "", "text")


def load_directory(path: Path, pattern: str) -> FileData:
    files = sorted(p for p in path.rglob(pattern) if p.is_file())
    if not files:
        raise SystemExit(f"{path}: no files matching {pattern}")
    return FileData(path, [str(p.relative_to(path)) for p in files], "", "directory")


def load_file(
    path_str: str,
    field_override: Optional[str],
    candidates: Sequence[str],
    pattern: str,
    split_lines: bool,
) -> FileData:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{path}: not found")
    if path.is_dir():
        return load_directory(path, pattern)
    suffix = path.suffix.lower()
    if suffix == ".jsonl" or suffix == ".ndjson":
        return load_jsonl(path, field_override, candidates)
    if suffix == ".json":
        return load_json(path, field_override, candidates)
    return load_text(path, field_override, split_lines)


def pair_data(
    predictions: FileData,
    gold: FileData,
    join_key: Optional[str],
) -> List[Tuple[str, str, str]]:
    """Return (identifier, prediction, gold) triples.

    Directory inputs pair by relative file path. Everything else pairs by index,
    or by `join_key` when both sides carry that field.
    """
    pred_texts = predictions.texts()
    gold_texts = gold.texts()

    if predictions.kind == "directory" and gold.kind == "directory":
        pred_map = dict(zip(predictions.keys(), pred_texts))
        gold_map = dict(zip(gold.keys(), gold_texts))
        shared = sorted(key for key in gold_map if key in pred_map)
        for key in pred_map:
            if key not in gold_map:
                print(f"warning: {key} has no ground-truth counterpart, skipped",
                      file=sys.stderr)
        for key in gold_map:
            if key not in pred_map:
                print(f"warning: {key} has no generated counterpart, skipped",
                      file=sys.stderr)
        return [(key, pred_map[key], gold_map[key]) for key in shared]

    if join_key:
        pred_path = predictions.resolve(join_key)
        gold_path = gold.resolve(join_key)
        if not pred_path or not gold_path:
            raise SystemExit(
                f"--join-key {join_key}: not present on "
                f"{'both' if pred_path and gold_path else 'the predictions' if pred_path else 'the ground truth'} side"
            )
        pred_ids = [get_field(record, pred_path) or "" for record in predictions.records]
        gold_ids = [get_field(record, gold_path) or "" for record in gold.records]
        gold_by_id = {key: text for key, text in zip(gold_ids, gold_texts)}
        pred_by_id = {key: text for key, text in zip(pred_ids, pred_texts)}
        shared = [key for key in pred_by_id if key in gold_by_id]
        for key in gold_by_id:
            if key not in pred_by_id:
                print(f"warning: {join_key}={key} missing from predictions", file=sys.stderr)
        return [(key, pred_by_id[key], gold_by_id[key]) for key in shared]

    count = min(len(pred_texts), len(gold_texts))
    if len(pred_texts) != len(gold_texts):
        print(
            f"warning: {len(pred_texts)} predictions vs {len(gold_texts)} ground truths; "
            f"comparing the first {count}",
            file=sys.stderr,
        )
    pred_ids = predictions.ids()
    rows = []
    for index in range(count):
        rows.append((pred_ids[index] if index < len(pred_ids) else str(index),
                     pred_texts[index], gold_texts[index]))
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_pair(
    identifier: str,
    prediction: str,
    gold: str,
    index: int,
    strip_fences: bool,
) -> Dict[str, Any]:
    if strip_fences:
        prediction = strip_code_fences(prediction)
    pred_text = prediction.strip()
    gold_text = gold.strip()
    longest = max(len(pred_text), len(gold_text))
    return {
        "index": index,
        "task_id": identifier,
        "edit_similarity": edit_similarity(pred_text, gold_text),
        "exact_match": exact_match(pred_text, gold_text),
        "line_match": line_match(pred_text, gold_text),
        "levenshtein": levenshtein(pred_text, gold_text),
        "prediction_len": len(pred_text),
        "gold_len": len(gold_text),
        "prediction": pred_text,
        "gold": gold_text,
    }


def aggregate(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [item["edit_similarity"] for item in items]
    buckets = {
        "exact (1.0)": sum(1 for s in scores if s >= 1.0),
        "near (>=0.8)": sum(1 for s in scores if 0.8 <= s < 1.0),
        "partial (>=0.5)": sum(1 for s in scores if 0.5 <= s < 0.8),
        "poor (<0.5)": sum(1 for s in scores if 0.0 < s < 0.5),
        "nothing (0.0)": sum(1 for s in scores if s <= 0.0),
    }
    return {
        "count": len(items),
        "mean_edit_similarity": statistics.fmean(scores) if scores else 0.0,
        "median_edit_similarity": statistics.median(scores) if scores else 0.0,
        "min_edit_similarity": min(scores) if scores else 0.0,
        "max_edit_similarity": max(scores) if scores else 0.0,
        "exact_match_rate": (
            sum(1 for item in items if item["exact_match"]) / len(items) if items else 0.0
        ),
        "line_match_rate": statistics.fmean(
            [item["line_match"] for item in items]
        ) if items else 0.0,
        "zero_length_predictions": sum(
            1 for item in items if item["prediction_len"] == 0
        ),
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    items: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    predictions: FileData,
    gold: FileData,
    strip_fences: bool,
    rows: int,
    min_score: Optional[float],
    elapsed: float,
) -> None:
    line = "=" * 78
    print(line)
    print("FULL EVALUATION - generated code vs reference code")
    print(line)
    print(f"predictions : {predictions.path}  [{predictions.label}]")
    print(f"ground truth : {gold.path}  [{gold.label}]")
    print(f"fence strip  : {'on' if strip_fences else 'off'}      pairs: {summary['count']}")
    print(f"elapsed      : {elapsed:.3f}s")
    print()

    print(f"{'idx':>6}  {'task_id':<34} {'edit_sim':>9}  {'EM':<5} {'line':>6}  {'len p/g':>11}")
    print("-" * 78)
    visible = [item for item in items if min_score is None or item["edit_similarity"] < min_score]
    if not visible:
        visible = list(items[:rows])
        if min_score is not None:
            print("(nothing below the threshold; showing the first rows instead)\n")
    elif min_score is not None:
        visible = visible[:rows]
    shown = visible[:rows]
    for item in shown:
        task_id = item["task_id"] or "-"
        if len(task_id) > 34:
            task_id = "..." + task_id[-31:]
        print(
            f"{item['index']:>6}  {task_id:<34} {item['edit_similarity']:>9.3f}  "
            f"{'yes' if item['exact_match'] else 'no':<5} {item['line_match']:>6.2f}  "
            f"{item['prediction_len']:>4}/{item['gold_len']:<4}"
        )
    if len(visible) > len(shown):
        print(f"... {len(visible) - len(shown)} more")
    print()

    print(line)
    print("AGGREGATE")
    print(line)
    print(f"pairs                      : {summary['count']}")
    print(f"mean edit similarity       : {summary['mean_edit_similarity']:.4f}")
    print(f"median edit similarity     : {summary['median_edit_similarity']:.4f}")
    print(f"min / max                  : {summary['min_edit_similarity']:.4f} / "
          f"{summary['max_edit_similarity']:.4f}")
    print(f"exact match rate           : {summary['exact_match_rate']:.4f}")
    print(f"line match rate            : {summary['line_match_rate']:.4f}")
    print(f"empty predictions          : {summary['zero_length_predictions']}")
    print()
    for label, count in summary["buckets"].items():
        share = count / summary["count"] if summary["count"] else 0.0
        bar = "#" * int(round(share * 40))
        print(f"  {label:<18} {count:>6}  {share:>6.1%}  {bar}")
    print(line)
    print(f"headline edit similarity   : {summary['mean_edit_similarity']:.4f}")
    print(f"headline exact match       : {summary['exact_match_rate']:.4f}")
    print(line)


def write_report(
    path: Path,
    items: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    predictions: FileData,
    gold: FileData,
    strip_fences: bool,
    elapsed: float,
    include_text: bool,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 4),
        "predictions_file": str(predictions.path),
        "predictions_source": predictions.label,
        "gold_file": str(gold.path),
        "gold_source": gold.label,
        "fence_strip": strip_fences,
        "primary_metric": "edit_similarity",
        "secondary_metric": "exact_match",
        "aggregate": summary,
        "items": [
            item
            if include_text
            else {key: value for key, value in item.items() if key not in ("prediction", "gold")}
            for item in items
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"json report: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit-similarity evaluation of generated code vs reference code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_a", nargs="?", help="predictions (jsonl / json / txt / directory)")
    parser.add_argument("file_b", nargs="?", help="ground truth (same formats)")
    parser.add_argument("--predictions", "--pred", dest="predictions",
                        help="explicit predictions file")
    parser.add_argument("--gold", "--ground-truth", dest="gold",
                        help="explicit ground truth file")
    parser.add_argument("--swap", action="store_true",
                        help="treat file_a as ground truth and file_b as predictions")
    parser.add_argument("--pred-field", help="dotted field path for predictions, e.g. metadata.prediction")
    parser.add_argument("--gold-field", help="dotted field path for ground truth, e.g. metadata.ground_truth")
    parser.add_argument("--join-key", help="pair records by this field (e.g. task_id) instead of by index")
    parser.add_argument("--glob", default="*.py", help="file pattern when an input is a directory")
    parser.add_argument("--split-lines", action="store_true",
                        help="treat each non-empty line of a text file as one record")
    parser.add_argument("--limit", type=int, help="evaluate only the first N pairs")
    parser.add_argument("--rows", type=int, default=20, help="table rows to print (default 20)")
    parser.add_argument("--min-score", type=float,
                        help="list only pairs scoring below this value")
    parser.add_argument("--output", "-o", default="evaluation_report.json",
                        help="JSON report path (default evaluation_report.json)")
    parser.add_argument("--no-json", action="store_true", help="do not write a JSON report")
    parser.add_argument("--include-text", action="store_true",
                        help="embed the raw prediction and gold text in the JSON report")
    parser.add_argument("--no-fence-strip", action="store_true",
                        help="score raw text instead of stripping ``` fences")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    pred_path = args.predictions or args.file_a
    gold_path = args.gold or args.file_b
    if not pred_path or not gold_path:
        print(
            "error: give two files (predictions first, ground truth second) or "
            "--predictions/--gold\n"
            "  python scripts/10.full_evaluation.py generated.py correct.py",
            file=sys.stderr,
        )
        return 2
    if args.swap:
        pred_path, gold_path = gold_path, pred_path

    strip_fences = not args.no_fence_strip

    predictions = load_file(pred_path, args.pred_field, PRED_FIELDS, args.glob, args.split_lines)
    gold = load_file(gold_path, args.gold_field, GOLD_FIELDS, args.glob, args.split_lines)

    rows = pair_data(predictions, gold, args.join_key)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("error: nothing to compare", file=sys.stderr)
        return 1

    started = time.time()
    items = [
        score_pair(identifier, prediction, truth, index, strip_fences)
        for index, (identifier, prediction, truth) in enumerate(rows)
    ]
    elapsed = time.time() - started

    summary = aggregate(items)
    print_report(items, summary, predictions, gold, strip_fences, args.rows,
                 args.min_score, elapsed)

    if not args.no_json:
        write_report(Path(args.output).expanduser(), items, summary,
                     predictions, gold, strip_fences, elapsed, args.include_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())