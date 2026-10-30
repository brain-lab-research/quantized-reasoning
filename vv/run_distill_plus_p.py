#!/usr/bin/env python3
"""Run W2 using a complete collection of FP16 plans, one benchmark at a time."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.collect_distill_plans import ORDER
from scripts.collect_multi_bench_traces import repair_trailing_jsonl

# Order by total generated W2 tokens in the completed seed-42 baselines.
W2_ORDER = ('aime2026', 'gpqa_diamond', 'strategyqa', 'arc_challenge', 'math500',
            'arc_easy', 'gsm8k', 'piqa', 'winogrande')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-path', required=True)
    args = p.parse_args()
    tasks = sorted(json.loads(args.manifest.read_text()), key=lambda t: W2_ORDER.index(t['benchmark']))
    if len(tasks) != 9 or {t['benchmark'] for t in tasks} != set(ORDER):
        raise ValueError('Expected all nine benchmarks')
    for task in tasks:
        path = args.root / 'plans' / (task['benchmark'] + '.jsonl')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != task['count'] or len({r['example_id'] for r in rows}) != task['count']:
            raise ValueError('Plans incomplete: ' + str(path))
    print('ALL_PLANS_VERIFIED', flush=True)
    for task in tasks:
        benchmark = task['benchmark']
        path = args.root / 'w2' / benchmark
        path.mkdir(parents=True, exist_ok=True)
        cfg = yaml.safe_load(Path(task['config']).read_text())
        cfg['experiment_name'] = '1030-distill-plus-p-' + benchmark
        cfg['description'] = 'DeepSeek-R1-Distill-Llama-8B FP16 plan followed by GPTQ W2; native chat adaptation.'
        cfg['actors'][0]['model_id'] = args.model_path
        cfg['actors'][0]['generation_params']['seed'] = 42
        cfg['output']['base_dir'] = str(path / 'artifacts')
        cfg_path = path / 'cfg.yaml'
        text = yaml.safe_dump(cfg, sort_keys=False)
        if cfg_path.exists() and cfg_path.read_text() != text:
            raise ValueError('Existing config changed: ' + str(cfg_path))
        cfg_path.write_text(text)
        plan_file = args.root / 'plans' / (benchmark + '.jsonl')
        hybrid_path = REPO / 'configs' / benchmark / 'qwen32b_gptq2bit' / 'hybrid_fp16_gptq2bit.yaml'
        instruction = yaml.safe_load(hybrid_path.read_text())['pipeline'][1]['stage_prompt']
        for artifact in (path / 'artifacts').glob('*/*.jsonl'):
            repair_trailing_jsonl(artifact)
        query = subprocess.check_output(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
              '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        free_mib = int(query.strip())
        kv, seq, utilization = (80, 32, '0.75') if free_mib >= 100000 else (32, 16, '0.27')
        if free_mib < 47000:
            raise RuntimeError(f'Insufficient GPU memory to start W2 safely: {free_mib} MiB')
        cmd = [sys.executable, str(REPO / 'scripts/collect_baseline_traces.py'), str(cfg_path),
               '--output', str(path / 'artifacts'), '--max-examples', str(task['count']),
               '--max-tokens', '30720', '--max-num-seqs', str(seq), '--kv-cache-gib', str(kv),
               '--memory-utilization', utilization, '--native-chat', '--model-path', args.model_path,
               '--plan-file', str(plan_file), '--plan-instruction', instruction]
        (path / 'cmd.json').write_text(json.dumps(cmd, indent=2))
        print('W2_START', benchmark, 'free_mib', free_mib, 'kv_gib', kv, 'seqs', seq, flush=True)
        with (path / 'run.log').open('a') as log:
            subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        pointer = path / 'artifacts' / 'run-id.txt'
        summary = json.loads((path / 'artifacts' / pointer.read_text().strip() / 'summary.json').read_text())
        if summary.get('n_judged') != task['count'] or summary.get('n_errors') != 0:
            raise RuntimeError('Incomplete benchmark ' + benchmark)
        print('W2_COMPLETE', benchmark, summary.get('n_judged'), flush=True)
    print('ALL_PLUS_P_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
