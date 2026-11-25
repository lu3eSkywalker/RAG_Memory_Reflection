#!/usr/bin/env python3
"""
12_generate_answers.py — send every retrieved question to the Google Gemini API.

Reads the retrieval output from 11_retrieve_questions.py (retrieval_all.jsonl),
makes one Gemini call per case, extracts the missing code lines, and writes
predictions.jsonl in the shape 10.full_evaluation.py expects:

    {"metadata": {"task_id": ..., "prediction": "..."}}

The ground truth in the retrieval file is never included in the prompt.

Usage:
    python scripts/12_generate_answers.py --input retrieval_all.jsonl \
        --out predictions.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gemini_client  # noqa: E402

logger = logging.getLogger("generate_answers")

DEFAULT_INPUT = PROJECT_ROOT / "retrieval_all.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.jsonl"

CODE_HINT = re.compile(r"^(\s{2,}|from\s|import\s|def\s|class\s|return\b|@|#|\)|\]|\})")


def strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_completion(text: str) -> str:
    """Pull just the generated code out of a chatty response.

    1. if a fenced block exists, take the first one
    2. otherwise drop leading prose lines until something that looks like code
    3. drop a trailing '# Missing code:' style marker if the model echoed it
    """
    cleaned = strip_fences(text)
    if not cleaned:
        return ""

    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text or "", re.DOTALL)
    if blocks:
        return strip_fences(blocks[0])

    lines = cleaned.splitlines()
    for position, line in enumerate(lines):
        if CODE_HINT.match(line):
            lines = lines[position:]
            break
    cleaned = "\n".join(lines).strip()

    # The model sometimes restates the instruction block; keep the last part.
    cleaned = re.sub(r"\n#\s*Missing code:\s*$", "", cleaned).strip()
    return cleaned


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate answers with Gemini")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="retrieval_all.jsonl from 11_retrieve_questions.py")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not call the API; write empty predictions "
                             "(useful to verify the plumbing)")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to wait between calls (rate limiting)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 11_retrieve_questions.py first.")
        return 1

    cases = load_cases(input_path)
    if args.limit:
        cases = cases[: args.limit]

    if not args.dry_run and not gemini_client.available():
        print("Error: GEMINI_API_KEY is not set.")
        print("  export GEMINI_API_KEY=\"your-key\"")
        print("  (or use --dry-run to verify the plumbing without the API)")
        return 1

    print("=" * 70)
    print("GENERATION  (Gemini)")
    print("=" * 70)
    print(f"Input      : {input_path} ({len(cases)} cases)")
    print(f"Model      : {'dry-run (no API call)' if args.dry_run else args.model}")
    print(f"Output     : {args.out}")

    started = time.time()
    written = 0
    failures = 0

    with open(args.out, "w") as handle:
        for position, case in enumerate(cases, start=1):
            metadata = case.get("metadata", {})
            task_id = metadata.get("task_id", f"case_{position}")
            prompt = metadata.get("prompt", "")

            raw = ""
            error = ""
            if args.dry_run:
                prediction = ""
            else:
                try:
                    raw = gemini_client.generate(
                        prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    prediction = extract_completion(raw)
                except Exception as failure:      # one bad case must not stop the run
                    error = str(failure)
                    prediction = ""
                    failures += 1
                    print(f"  case {position:>2}: API FAILED ({failure})", file=sys.stderr)

            record: Dict[str, Any] = {
                "metadata": {
                    "task_id": task_id,
                    "repo": metadata.get("repo"),
                    "fpath_tuple": metadata.get("fpath_tuple"),
                    "line_no": metadata.get("line_no"),
                    "prediction": prediction,
                    "retrieved_chunks": len(metadata.get("retrieved", []) or []),
                }
            }
            if error:
                record["metadata"]["error"] = error
            if args.dry_run or "--include-raw" in sys.argv:
                record["metadata"]["raw"] = raw

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

            preview = prediction.replace("\n", "\\n")[:70]
            print(f"  case {position:>2}/{len(cases)}  {task_id:<34} "
                  f"pred={len(prediction):>5} chars  {preview}")

            if args.sleep and position < len(cases):
                time.sleep(args.sleep)

    elapsed = time.time() - started
    print()
    print(f"Generated {written} predictions in {elapsed:.2f}s"
          + (f" ({failures} failed)" if failures else ""))
    print(f"Saved -> {args.out}")
    print()
    print("Next:")
    print("  python scripts/10.full_evaluation.py \\")
    print("      --predictions " + str(args.out) + " --pred-field metadata.prediction \\")
    print("      --gold <answers.jsonl> --gold-field metadata.ground_truth \\")
    print("      --join-key metadata.task_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())