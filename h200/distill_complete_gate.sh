#!/usr/bin/env bash
# Do not resume W4 until all nine FP16 Distill benchmark artifacts are complete.
set -euo pipefail
repo=$1; root=$2
export PYTHONPATH="$repo"
"/home/data/$USER/envs/low-bit-vllm/bin/python" - "$root" <<'PY'
import json,sys
from pathlib import Path
from h200.paper_matrix import task_complete
root=Path(sys.argv[1]); tasks=json.loads((root/'matrix-fp16.json').read_text())
if len(tasks)!=9 or any(t['size']!=8 or t['precision']!='fp16' for t in tasks):
    raise SystemExit('INVALID_DISTILL_MATRIX')
missing=[t['benchmark'] for t in tasks if not task_complete(t)]
if missing:
    raise SystemExit('DISTILL_INCOMPLETE '+', '.join(missing))
print('DISTILL_ALL_NINE_COMPLETE',flush=True)
PY
