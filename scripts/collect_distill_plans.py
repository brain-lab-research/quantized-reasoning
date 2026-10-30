#!/usr/bin/env python3
"""Collect resumable FP16 plans for the DeepSeek distill +P experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYSTEM = ("You are a careful reasoning assistant. Produce only a concise solution plan.\n"
          "Do not solve the problem fully. Write only a concise plan with 3-6 short bullet points or numbered steps.\n"
          "Do not give the final answer, do not use \\boxed{}, and stop immediately after the plan.")
SUFFIX = "Return only the plan. No final answer."
ORDER = ('aime2026', 'gpqa_diamond', 'strategyqa', 'math500', 'arc_challenge',
         'winogrande', 'gsm8k', 'piqa', 'arc_easy')


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def plan_prompt(tokenizer, problem: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM},
         {'role': 'user', 'content': 'Problem:\n' + problem + '\n' + SUFFIX}],
        tokenize=False, add_generation_prompt=True)
    if not prompt.rstrip().endswith('<think>'):
        raise ValueError('Unexpected model chat template: no open think')
    return prompt + '\n</think>\n\n'


def main() -> None:
    import yaml
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from quantlab.actors.backends.vllm_backend import VLLMBackend
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.core.trace import extract_chat_user_content
    from quantlab.core.types import PrecisionMode, QuantizationMethod
    from quantlab.runner import _load_benchmark_examples
    from scripts.collect_baseline_traces import completed_outputs, engine_prompt
    from scripts.collect_multi_bench_traces import repair_trailing_jsonl

    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--kv-cache-gib', type=float, default=8)
    p.add_argument('--memory-utilization', type=float, default=0.25)
    p.add_argument('--max-num-seqs', type=int, default=32)
    args = p.parse_args()
    tasks = sorted(json.loads(args.manifest.read_text()), key=lambda t: ORDER.index(t['benchmark']))
    if len(tasks) != 9 or {t['benchmark'] for t in tasks} != set(ORDER):
        raise ValueError('Expected all nine distinct full benchmarks')
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    backend = VLLMBackend(args.model_path, precision=PrecisionMode.FP16,
        quantization=QuantizationMethod.NONE, tensor_parallel_size=1,
        gpu_memory_utilization=args.memory_utilization, max_model_len=4096,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=8192,
        kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
        enable_prefix_caching=True, enforce_eager=False)
    backend.load()
    engine = backend._llm.llm_engine
    params = SamplingParams(max_tokens=1024, temperature=0.6, top_p=1.0, top_k=-1,
        seed=42, stop=['[PLAN_FINISH]', 'PLAN_FINISH'], output_kind=RequestOutputKind.FINAL_ONLY)
    for task in tasks:
        raw = yaml.safe_load(Path(task['config']).read_text())
        raw['benchmark'].update(max_examples=task['count'], start_index=0, seed=42)
        cfg = ExperimentConfig.model_validate(raw)
        adapter = BenchmarkRegistry.build(cfg.benchmark.name)
        examples = _load_benchmark_examples(adapter, cfg.benchmark)
        if len(examples) != task['count'] or len({e.example_id for e in examples}) != len(examples):
            raise ValueError('Unexpected dataset count or duplicate IDs: ' + task['benchmark'])
        path = args.output / (task['benchmark'] + '.jsonl')
        repair_trailing_jsonl(path)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        saved = {r['example_id']: r for r in rows}
        if len(saved) != len(rows):
            raise ValueError('Duplicate saved plans: ' + str(path))
        expected = {e.example_id: e for e in examples}
        if any(eid not in expected or row['base_prompt_sha256'] != digest(expected[eid].prompt)
               for eid, row in saved.items()):
            raise ValueError('Saved plan does not match current benchmark prompt')
        pending = {e.example_id: e for e in examples if e.example_id not in saved}
        print('PLAN_START', task['benchmark'], len(saved), '/', len(examples), flush=True)
        for eid, example in pending.items():
            prompt = plan_prompt(tokenizer, extract_chat_user_content(example.prompt))
            engine.add_request(eid, engine_prompt(prompt, tokenizer, True), params)
        with path.open('a', encoding='utf-8') as out:
            for output in completed_outputs(engine):
                ex = pending[output.request_id]
                generated = output.outputs[0]
                if not generated.text.strip():
                    raise ValueError('Empty plan: ' + ex.example_id)
                row = {'example_id': ex.example_id, 'benchmark': task['benchmark'],
                       'base_prompt_sha256': digest(ex.prompt), 'ground_truth': ex.ground_truth,
                       'plan': generated.text, 'token_ids': list(generated.token_ids),
                       'finish_reason': generated.finish_reason,
                       'prompt_token_ids': list(output.prompt_token_ids),
                       'temperature': 0.6, 'top_p': 1.0, 'top_k': -1, 'seed': 42}
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                saved[ex.example_id] = row
                print('PLAN_SAVED', task['benchmark'], len(saved), '/', len(examples), flush=True)
        print('PLAN_COMPLETE', task['benchmark'], len(saved), flush=True)
    backend.unload()
    print('ALL_PLANS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
