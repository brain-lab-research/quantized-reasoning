#!/usr/bin/env bash
# Gate the next experiment on complete W4 artifacts, not merely terminal jobs.
set -euo pipefail
repo=$1
root="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
gpqa="/home/data/$USER/projects/low-bit-reasoning/exps/1029-gpqa-w4-full/artifacts/summary.json"
export PYTHONPATH="$repo"
"/home/data/$USER/envs/low-bit-vllm/bin/python" - "$root" "$gpqa" <<'PY'
import json,sys
from pathlib import Path
from h200.paper_matrix import task_complete
root=Path(sys.argv[1]); tasks=json.loads((root/'matrix.json').read_text())
if len(tasks)!=8 or any(t['size']!=32 or t['precision']!='w4' for t in tasks):
    raise SystemExit('INVALID_W4_MATRIX')
missing=[t['benchmark'] for t in tasks if not task_complete(t)]
gpqa=json.loads(Path(sys.argv[2]).read_text())
if not gpqa.get('complete') or gpqa.get('completed')!=198:
    missing.append('gpqa_diamond')
if missing:
    raise SystemExit('W4_INCOMPLETE '+', '.join(missing))
print('W4_ALL_NINE_COMPLETE',flush=True)
PY
