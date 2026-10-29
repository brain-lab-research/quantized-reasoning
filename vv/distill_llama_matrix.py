#!/usr/bin/env python3
"""Full FP16 DeepSeek-R1-Distill-Llama-8B benchmark collection."""
import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from h200.paper_matrix import prepare, task_complete

MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
ORDER = ['strategyqa', 'aime2026', 'arc_challenge', 'winogrande', 'piqa',
         'arc_easy', 'gpqa_diamond', 'gsm8k', 'math500']


def prepare_tasks(root, precision='fp16'):
    tasks = prepare(root / 'source', full_benchmarks=True)
    selected = []
    for task in tasks:
        if task['size'] != 8 or task['precision'] != 'fp16':
            continue
        raw = yaml.safe_load(Path(task['config']).read_text())
        name = '1029-r1-distill-llama-8b-' + task['benchmark'] + '-' + precision
        path = root / name / 'cfg.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        raw['experiment_name'] = name
        raw['actors'][0]['actor_id'] = 'r1_distill_llama_8b_' + precision
        raw['actors'][0]['model_id'] = MODEL if precision == 'fp16' else str(root / 'model-gptq2-g32')
        if precision == 'w2':
            raw['actors'][0]['quantization'] = 'gptq'
            raw['actors'][0]['quantization_config'] = {'bits': 2, 'group_size': 32}
        raw['pipeline'][0]['actor_id'] = raw['actors'][0]['actor_id']
        raw['output']['base_dir'] = str(path.parent / 'artifacts')
        if path.exists() and yaml.safe_load(path.read_text()) != raw:
            raise ValueError(f'Existing config differs: {path}')
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        selected.append({**task, 'config': str(path), 'precision': precision})
    selected.sort(key=lambda t: ORDER.index(t['benchmark']))
    (root / ('matrix-' + precision + '.json')).write_text(json.dumps(selected, indent=2))
    return selected


def run(root, model_path, precision, shared=False):
    tasks = json.loads((root / ('matrix-' + precision + '.json')).read_text())
    for task in tasks:
        cfg = Path(task['config'])
        if shared:
            import fcntl
            lock = (cfg.parent / '.claim').open('w')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
        else:
            lock = contextlib.nullcontext()
        with lock:
            run_one(task, cfg, model_path)


def run_one(task, cfg, model_path):
        if task_complete(task):
            print('TASK_SKIP_COMPLETE', cfg.parent.name, flush=True)
            return
        cmd = [sys.executable, str(REPO / 'scripts/collect_baseline_traces.py'), str(cfg),
               '--output', str(cfg.parent / 'artifacts'), '--max-examples', str(task['count']),
               '--max-num-seqs', '32', '--kv-cache-gib', '80', '--memory-utilization', '0.85',
               '--native-chat', '--model-path', str(model_path)]
        (cfg.parent / 'cmd.txt').write_text(subprocess.list2cmdline(cmd) + '\n')
        print('TASK_START', cfg.parent.name, flush=True)
        with (cfg.parent / 'run.log').open('a') as log:
            subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not task_complete(task):
            raise RuntimeError(f'Incomplete task: {cfg}')
        print('TASK_COMPLETE', cfg.parent.name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--precision', choices=['fp16', 'w2'], default='fp16')
    p.add_argument('--shared', action='store_true', help='Claim benchmark tasks across workers')
    args = p.parse_args()
    if args.prepare:
        print('PREPARED_TASKS', len(prepare_tasks(args.root, args.precision)))
    else:
        if not args.model_path:
            p.error('--model-path is required')
        run(args.root, args.model_path, args.precision, shared=args.shared)


if __name__ == '__main__':
    main()
