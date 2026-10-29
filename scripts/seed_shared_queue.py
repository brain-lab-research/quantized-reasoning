"""Publish previously completed traces to a shared queue after the old writer stops."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import boto3
    from quantlab.artifacts.store import ArtifactStore
    from scripts.shared_queue import SharedQueue
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, action='append', required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--model-sha256', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--fingerprint', required=True)
    p.add_argument('--ready-key', help='Publish this key after a final, stable seed pass')
    args = p.parse_args()
    h = hashlib.sha256()
    with (args.model_path / 'model.safetensors').open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != args.model_sha256:
        raise ValueError('W2 checkpoint checksum mismatch')
    client = boto3.client('s3', endpoint_url='https://s3.cod.phystech.edu')
    queue = SharedQueue(client, 'iii-storage2', args.prefix, 'seed', args.fingerprint)
    total = 0
    for source in args.source:
        for cfg_path in sorted(source.glob('*/artifacts/*/config.json')):
            if not cfg_path.parent.parent.parent.name.endswith('-w2'):
                continue
            config = json.loads(cfg_path.read_text())
            actor = config['actors'][0]
            params = actor['generation_params']
            quant = actor.get('quantization_config', {})
            if actor.get('quantization') != 'gptq' or quant.get('bits') != 2 or quant.get('group_size') != 32:
                raise ValueError(f'Unexpected quantization: {cfg_path}')
            expected = {'max_new_tokens': 32768, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                        'repetition_penalty': 1.0, 'stop_sequences': ['<|im_end|>', '<|endoftext|>'], 'seed': 42}
            if any(params.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Sampling mismatch: {cfg_path}')
            if 'native_chat_explicit_token_ids_v2' not in config.get('description', ''):
                raise ValueError(f'Unverified prompt recipe: {cfg_path}')
            name = config['benchmark']['name']
            run_id = cfg_path.parent.name
            store = ArtifactStore(str(cfg_path.parent.parent))
            traces = {t.example_id: t for t in store.load_traces(run_id)}
            judgements = {r['example_id']: r for r in store.load_judgements(run_id)}
            for eid, judgement in judgements.items():
                trace = traces.get(eid)
                if trace is None:
                    raise ValueError(f'Missing trace {eid} in {cfg_path}')
                etag = queue.reserve(name, eid, trace.prompt)
                if etag:
                    queue.finish(name, eid, trace.prompt, etag,
                                 {'trace': trace.to_dict(), 'judgement': judgement, 'source': str(cfg_path)})
                    total += 1
            print('SEEDED', name, len(judgements), flush=True)
    if args.ready_key:
        client.put_object(Bucket='iii-storage2', Key=args.ready_key,
                          Body=json.dumps({'fingerprint': args.fingerprint, 'seeded': total}).encode(), IfNoneMatch='*')
        print('BOOTSTRAP_READY', total, flush=True)
    else:
        print('PRESEED_COMPLETE', total, flush=True)


if __name__ == '__main__':
    main()
