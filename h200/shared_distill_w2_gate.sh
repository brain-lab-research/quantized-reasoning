#!/usr/bin/env bash
set -euo pipefail
repo=$1
w4="/home/data/$USER/projects/low-bit-reasoning/exps/1029-w4-32b-full"
gpqa="/home/data/$USER/projects/low-bit-reasoning/exps/1029-gpqa-w4-full/artifacts/summary.json"
export PYTHONPATH="$repo"
python="/home/data/$USER/envs/low-bit-vllm/bin/python"
if [[ "${2:-after-w4}" != priority ]]; then
"$python" - "$w4" "$gpqa" <<'PY'
import json,sys
from pathlib import Path
from h200.paper_matrix import task_complete
tasks=json.loads((Path(sys.argv[1])/'matrix.json').read_text())
gpqa=json.loads(Path(sys.argv[2]).read_text())
if len(tasks)!=8 or any(not task_complete(t) for t in tasks) or not gpqa.get('complete') or gpqa.get('completed')!=198:
    raise SystemExit('W4_INCOMPLETE')
print('W4_COMPLETE',flush=True)
PY
fi
set -a; . "$HOME/.s3-cod.env"; set +a
"$python" - <<'PY'
import json,time,boto3
from botocore.exceptions import ClientError
s=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu'); bucket='iii-storage2'
model_ready='shveikinda/h200/low-bit/1029-distill-w2-temporary-model/ready.json'
deadline=time.monotonic()+7*3600
while time.monotonic()<deadline:
    try:
        s.head_object(Bucket=bucket,Key=model_ready)
        break
    except ClientError as exc:
        if exc.response.get('Error',{}).get('Code') not in ('404','NoSuchKey','NotFound'):raise
        print('WAIT_MODEL_UPLOAD',flush=True);time.sleep(30)
else:raise SystemExit('MODEL_NOT_READY')
request='shveikinda/h200/low-bit/1029-distill-shared/switch-request.json'
payload=json.dumps({'fingerprint':'distill-w2-c8c207ac-seed42-native-v2','reason':'w4_complete'}).encode()
try:s.put_object(Bucket=bucket,Key=request,Body=payload,IfNoneMatch='*')
except ClientError as exc:
    if exc.response.get('Error',{}).get('Code') not in ('412','PreconditionFailed'):raise
    if json.loads(s.get_object(Bucket=bucket,Key=request)['Body'].read()).get('fingerprint')!='distill-w2-c8c207ac-seed42-native-v2':raise ValueError('STALE_SWITCH_REQUEST')
print('VV_HANDOFF_REQUESTED',flush=True)
keys=['shveikinda/h200/low-bit/1029-distill-shared/ready.json',
      model_ready]
while time.monotonic()<deadline:
    try:
        for key in keys:s.head_object(Bucket=bucket,Key=key)
        ready=json.loads(s.get_object(Bucket=bucket,Key=keys[0])['Body'].read())
        if ready.get('fingerprint')!='distill-w2-c8c207ac-seed42-native-v2':raise ValueError('SHARED_QUEUE_FINGERPRINT_MISMATCH')
        print('SHARED_QUEUE_AND_MODEL_READY',flush=True);break
    except ClientError as exc:
        if exc.response.get('Error',{}).get('Code') not in ('404','NoSuchKey','NotFound'):raise
        print('WAIT_SHARED_QUEUE_OR_MODEL',flush=True);time.sleep(30)
else:raise SystemExit('SHARED_QUEUE_NOT_READY')
PY
