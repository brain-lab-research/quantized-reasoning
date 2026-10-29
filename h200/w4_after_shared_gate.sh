#!/usr/bin/env bash
# Do not resume Qwen W4 until the entire coordinated Distill W2 queue is done.
set -euo pipefail
set -a; . "$HOME/.s3-cod.env"; set +a
"/home/data/$USER/envs/low-bit-vllm/bin/python" - <<'PY'
import json,boto3
s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu')
key='shveikinda/h200/low-bit/1029-distill-shared/complete.json'
result=json.loads(s.get_object(Bucket='iii-storage2',Key=key)['Body'].read())
if not result.get('complete') or result.get('expected')!=9387 or result.get('done')!=9387:
    raise SystemExit('DISTILL_SHARED_NOT_COMPLETE')
print('DISTILL_SHARED_COMPLETE',flush=True)
PY
