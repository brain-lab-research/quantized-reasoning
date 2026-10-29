"""Verify that every expected shared-queue item is done, including in-flight claims."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import boto3
    import yaml
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.runner import _load_benchmark_examples
    from scripts.shared_queue import SharedQueue
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--prefix',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--wait-seconds',type=int,default=0); a=p.parse_args()
    client=boto3.client('s3',endpoint_url='https://s3.cod.phystech.edu'); bucket='iii-storage2'
    fingerprint='distill-w2-c8c207ac-seed42-native-v2'; q=SharedQueue(client,bucket,a.prefix,'audit',fingerprint)
    expected=set()
    for task in json.loads(a.manifest.read_text()):
        config=ExperimentConfig.model_validate(yaml.safe_load(Path(task['config']).read_text()))
        examples=_load_benchmark_examples(BenchmarkRegistry.build(config.benchmark.name),config.benchmark)
        for e in examples:expected.add(q.key(config.benchmark.name,e.example_id))
    if len(expected)!=9387:raise ValueError(f'Unexpected unique example count: {len(expected)}')
    deadline=time.monotonic()+a.wait_seconds
    while True:
        actual={x['Key'] for pg in client.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=a.prefix.rstrip('/')+'/items/') for x in pg.get('Contents',[])}
        unexpected=actual-expected
        if unexpected:raise ValueError(f'Unexpected queue item count: {len(unexpected)}')
        done=claimed=None
        if actual==expected:
            def state(key):
                meta=client.head_object(Bucket=bucket,Key=key).get('Metadata',{})
                if meta.get('fingerprint') not in (None,fingerprint):raise ValueError('Queue fingerprint mismatch')
                if meta.get('state') not in ('done','claimed'):raise ValueError('Unknown queue state')
                return meta['state']
            with ThreadPoolExecutor(max_workers=16) as pool:states=list(pool.map(state,expected))
            done=states.count('done'); claimed=states.count('claimed')
        summary={'expected':len(expected),'registered':len(actual),'missing':len(expected-actual),'done':done,'claimed':claimed,'complete':done==len(expected)}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2))
        client.put_object(Bucket=bucket,Key=a.prefix.rstrip('/')+'/audit.json',Body=json.dumps(summary).encode())
        print('QUEUE_AUDIT',json.dumps(summary),flush=True)
        if summary['complete']:
            client.put_object(Bucket=bucket,Key=a.prefix.rstrip('/')+'/complete.json',Body=json.dumps(summary).encode())
            return
        if time.monotonic()>=deadline:raise SystemExit('QUEUE_INCOMPLETE')
        time.sleep(180)


if __name__=='__main__':main()
