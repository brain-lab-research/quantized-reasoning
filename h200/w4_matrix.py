#!/usr/bin/env python3
"""Prepare and consume a shared full W4 matrix; workers lock individual runs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from h200.paper_matrix import BENCHMARKS, prepare, task_complete

ORDER = ['strategyqa', 'aime2026', 'arc_challenge', 'winogrande', 'piqa',
         'arc_easy', 'gpqa_diamond', 'gsm8k', 'math500']


def prepare_w4(root, sizes=(8, 32)):
    tasks = prepare(root / 'source', full_benchmarks=True)
    selected = []
    for source in tasks:
        if source['precision'] != 'w2' or source['size'] not in sizes or (source['size'], source['benchmark']) == (32, 'gpqa_diamond'):
            continue
        cfg = yaml.safe_load(Path(source['config']).read_text())
        size, bench = source['size'], source['benchmark']
        name = f'1029-qwen{size}b-{bench}-w4'
        path = root / name / 'cfg.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = cfg['actors'][0]
        actor['actor_id'] = f'qwen3_{size}b_w4'
        actor['model_id'] = f'kaitchup/Qwen3-{size}B-autoround-4bit-gptq'
        actor['quantization_config'] = {'bits': 4}
        cfg['pipeline'][0]['actor_id'] = actor['actor_id']
        cfg['experiment_name'] = name
        cfg['output']['base_dir'] = str(path.parent / 'artifacts')
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            raise ValueError(f'Existing W4 configuration differs: {path}')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        selected.append({**source, 'config': str(path), 'precision': 'w4'})
    selected.sort(key=lambda t: (ORDER.index(t['benchmark']), t['size']))
    (root / 'matrix.json').write_text(json.dumps(selected, indent=2))
    return selected


def run(root, worker, stage):
    import fcntl
    from huggingface_hub import HfApi, snapshot_download
    tasks = json.loads((root / 'matrix.json').read_text())
    tasks.sort(key=lambda t: (t['size'] != (8 if worker == 'a' else 32), ORDER.index(t['benchmark'])))
    staged = {}
    for task in tasks:
        path = Path(task['config'])
        with (path.parent / '.claim').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if task_complete(task):
                print('SKIP_COMPLETE', path.parent.name, flush=True)
                continue
            size = task['size']
            if size not in staged:
                model = f'kaitchup/Qwen3-{size}B-autoround-4bit-gptq'
                revfile = root / f'model-{size}b-revision.json'
                revision = json.loads(revfile.read_text())['revision'] if revfile.exists() else HfApi().model_info(model).sha
                if not revfile.exists():
                    revfile.write_text(json.dumps({'model_id': model, 'revision': revision}))
                destination = stage / f'model-{size}b'
                print('DOWNLOAD_START', model, revision, flush=True)
                snapshot_download(model, revision=revision, local_dir=str(destination), max_workers=4,
                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
                staged[size] = str(destination)
                print('DOWNLOAD_DONE', model, flush=True)
                smoke = root / f'smoke-{worker}-{size}b'
                if not (smoke / 'summary.json').exists():
                    subprocess.run([sys.executable, str(REPO / 'scripts/profile_baseline_latency.py'),
                        '--config', str(path), '--output', str(smoke), '--examples', '1',
                        '--model-path', str(destination)], cwd=REPO, check=True)
                summary = json.loads((smoke / 'summary.json').read_text())
                if summary.get('completed') != 1 or not summary.get('total_output_tokens'):
                    raise RuntimeError(f'W4 smoke failed: {smoke}')
                print('SMOKE_OK', worker, size, flush=True)
            cmd = [sys.executable, str(REPO / 'scripts/collect_baseline_traces.py'), str(path),
                   '--output', str(path.parent / 'artifacts'), '--max-examples', str(task['count']),
                   '--max-num-seqs', '32' if size == 8 else '16', '--kv-cache-gib', '80' if size == 8 else '48',
                   '--memory-utilization', '0.9', '--open-thinking', '--model-path', staged[size]]
            (path.parent / 'cmd.txt').write_text(subprocess.list2cmdline(cmd) + '\n')
            print('TASK_START', path.parent.name, flush=True)
            with (path.parent / 'run.log').open('a') as log:
                subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not task_complete(task):
                raise RuntimeError(f'Incomplete task: {path}')
            print('TASK_COMPLETE', path.parent.name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', choices=['a', 'b'])
    p.add_argument('--sizes', nargs='+', type=int, choices=[8, 32], default=[8, 32])
    p.add_argument('--stage', type=Path)
    a = p.parse_args()
    if a.prepare:
        print('PREPARED_TASKS', len(prepare_w4(a.root, a.sizes)))
    else:
        if not a.worker or not a.stage:
            p.error('worker and stage are required')
        run(a.root, a.worker, a.stage)


if __name__ == '__main__':
    main()
