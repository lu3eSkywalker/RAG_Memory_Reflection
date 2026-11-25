#!/usr/bin/env python3
"""
13_answers_txt_to_predictions.py - turn answer_from_llm/*.txt into prediction jsonl.

Input : answer_from_llm/answers_<technique>.txt   (blocks of "case_NN" + code)
Gold  : Data_to_test_folder/1/answers/function_level_completion_2k_context_codex.test.jsonl
        (row i corresponds to case_(i+1), verified by task_id suffix)

Two prediction variants are produced per technique:

  raw   the answer text exactly as written
  fair  the same text with its common leading indentation removed and re-indented
        to the indentation implied by the prompt context (the last line of the
        "Existing code" block, plus 4 spaces if that line opens a block). This is
        derived from the prompt only, never from the ground truth, so it removes
        indentation-only artefacts without leaking the answer.

Usage:
    python scripts/13_answers_txt_to_predictions.py
    python scripts/13_answers_txt_to_predictions.py --techniques adaptive bm25
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSWER_DIR = PROJECT_ROOT / "answer_from_llm"
PROMPT_ROOT = PROJECT_ROOT / "offline_prompts_by_RAG_techniques"
GOLD = Path(
    "/home/sooraj/Downloads/Data_to_test_folder/1/answers/"
    "function_level_completion_2k_context_codex.test.jsonl"
)
N_CASES = 30

# answer-file stem -> prompt folder suffix (the answer file is pluralised)
PROMPT_TECHNIQUE = {
    "relationships": "relationship",
}

CASE_RE = re.compile(r"^case_(\d{2})$")


def parse_answers(path: Path):
    """Return {case_number: code} for a answers_<technique>.txt file."""
    answers, current, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CASE_RE.match(line.strip())
        if match:
            if current is not None:
                answers[current] = "\n".join(buf).strip("\n")
            current, buf = int(match.group(1)), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        answers[current] = "\n".join(buf).strip("\n")
    return answers


def prompt_base_indent(case_no: int, technique: str) -> int:
    """Indentation the missing code should start at, from the prompt context only."""
    prompt_technique = PROMPT_TECHNIQUE.get(technique, technique)
    folder = PROMPT_ROOT / f"offline_case_prompts_{prompt_technique}"
    files = sorted(folder.glob(f"case_{case_no:02d}_*.txt"))
    if not files:
        return 0
    text = files[0].read_text(encoding="utf-8", errors="replace")
    start = text.rfind("# Existing code")
    block = text[start:]
    stop = block.find("# Missing code:")
    if stop >= 0:
        block = block[:stop]
    code = [line for line in block.splitlines() if line.strip()]
    if not code:
        return 0
    last = code[-1]
    indent = len(last) - len(last.lstrip())
    if last.rstrip().endswith(":"):
        indent += 4
    return indent


def normalise(code: str, indent: int) -> str:
    """Dedent by the common leading whitespace, then re-indent to `indent`."""
    lines = code.split("\n")
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return code
    common = min(len(line) - len(line.lstrip()) for line in non_empty)
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        else:
            out.append(" " * indent + line[common:])
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--techniques", nargs="*", default=None)
    parser.add_argument("--gold", default=str(GOLD))
    args = parser.parse_args()

    gold_rows = [json.loads(line) for line in open(args.gold, encoding="utf-8") if line.strip()]
    if len(gold_rows) < N_CASES:
        raise SystemExit(f"{args.gold}: only {len(gold_rows)} rows, need {N_CASES}")

    answers = sorted(ANSWER_DIR.glob("answers_*.txt"))
    techniques = args.techniques or [p.stem[len("answers_"):] for p in answers]
    if not techniques:
        raise SystemExit(f"no answers_*.txt files in {ANSWER_DIR}")

    for technique in techniques:
        ans_path = ANSWER_DIR / f"answers_{technique}.txt"
        if not ans_path.exists():
            raise SystemExit(f"{ans_path}: not found")
        parsed = parse_answers(ans_path)

        raw_rows, fair_rows = [], []
        for index in range(N_CASES):
            case_no = index + 1
            if case_no not in parsed or not parsed[case_no].strip():
                raise SystemExit(f"{ans_path}: case_{case_no:02d} is missing or empty")
            task_id = gold_rows[index]["metadata"]["task_id"]
            code = parsed[case_no].strip("\n")
            raw_rows.append({"metadata": {"task_id": task_id, "case": f"case_{case_no:02d}",
                                           "technique": technique, "prediction": code}})
            fair_code = normalise(code, prompt_base_indent(case_no, technique))
            fair_rows.append({"metadata": {"task_id": task_id, "case": f"case_{case_no:02d}",
                                           "technique": technique, "prediction": fair_code}})

        for variant, rows in (("raw", raw_rows), ("fair", fair_rows)):
            out = PROJECT_ROOT / f"predictions_{technique}_{variant}.jsonl"
            with out.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            print(f"{out.name}: {len(rows)} predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
