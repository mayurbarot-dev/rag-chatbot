#!/usr/bin/env bash
# Phase 2 ablation: switch one stage off at a time and re-run the question set.
# Query-time switches reuse the full index in eval/.work; index-time switches
# build their own index (embeddings and chart transcriptions come from cache).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 EMBEDDING_PROVIDER=local
run() { local label=$1 dir=$2; shift 2; echo "=== $label"; "$PY" eval/run_eval.py --label "$label" --work-dir "$dir" --set EMBEDDING_PROVIDER=local "$@" | grep -E "^\| \*\*All|^\[FAIL\]"; }
run no-rerank     eval/.work           --set RERANK=off
run no-hybrid     eval/.work           --set HYBRID=off
run no-scoping    eval/.work           --set QUERY_SCOPING=off
run no-vision     eval/.work-novision  --set VISION_MODE=off
run no-tables     eval/.work-notables  --set TABLES=off
run no-headers    eval/.work-noheaders --set CONTEXT_HEADERS=off
echo "=== ablations done"
