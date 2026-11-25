#!/usr/bin/env bash
# run_pipeline.sh
#
# One command for the full loop:
#   01 chunking -> 02 embedding -> 03 vector store -> 04 retrieval
#   -> 05 prompt -> 06 generation -> 09 reflection & memory extraction
#
# Usage:
#   ./run_pipeline.sh
#   PY=../RAG_Basic_Repocoder_Inspired_EXP/venv/bin/python ./run_pipeline.sh
#   TASK="add rate limiting to the login API" ./run_pipeline.sh
#
# Environment:
#   PY             interpreter to use           (default: venv/bin/python)
#   TASK           task passed to reflection     (default: see below)
#   REPORT         reflection report filename   (default: reflection_report.txt)
#   GEMINI_API_KEY required for real generation and reflection

set -uo pipefail

cd "$(dirname "$0")" || exit 1

PY="${PY:-venv/bin/python}"
TASK="${TASK:-Fix the authentication logic in the repository}"
REPORT="${REPORT:-reflection_report.txt}"
FAILED=0

echo "============================================================"
echo " RAG PIPELINE  (chunk -> embed -> store -> retrieve -> prompt -> generate -> reflect)"
echo "============================================================"
echo "interpreter : $("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")"
echo "task        : $TASK"
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "warning     : GEMINI_API_KEY is not set — generation falls back to the mock"
    echo "              generator and reflection will return NO_MEMORY."
fi

run_step() {
    local file="$1"
    echo
    echo "############################################################"
    echo "# ${file}"
    echo "############################################################"
    "$PY" "scripts/${file}"
    local rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "--- ${file}: OK"
    else
        echo "--- ${file}: FAILED (exit ${rc})" >&2
        FAILED=1
    fi
}

# Steps 1-6
for step in \
    01_run_chunking.py \
    02_run_embedding.py \
    03_run_vector_store.py \
    04_run_retrieval.py \
    05_run_prompt.py \
    06_run_generation.py
do
    run_step "${step}"
done

# Step 7: reflection & memory extraction
echo
echo "############################################################"
echo "# 09_reflection.py"
echo "############################################################"
if [ -f llm_response.txt ]; then
    "$PY" scripts/09_reflection.py \
        --task "${TASK}" \
        --generation-file llm_response.txt \
        --context-file retrieval_data.json \
        --output "${REPORT}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "--- 09_reflection.py: OK (report: ${REPORT})"
    else
        echo "--- 09_reflection.py: FAILED (exit ${rc})" >&2
        FAILED=1
    fi
else
    echo "skipped: llm_response.txt not found (generation did not complete)"
    FAILED=1
fi

echo
echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo " Pipeline finished. Artifacts: chunks.json embeddings.json"
    echo " vector_store.faiss retrieval_data.json final_prompt.txt"
    echo " llm_response.txt ${REPORT}"
else
    echo " Pipeline finished with errors (see above)."
fi
echo "============================================================"
exit "${FAILED}"