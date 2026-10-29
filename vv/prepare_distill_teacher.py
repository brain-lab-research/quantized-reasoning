#!/usr/bin/env python3
"""Generate and validate diverse synthetic reasoning windows for GPTQ calibration."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quantlab.benchmarks.boxed import extract_last_boxed
from quantlab.benchmarks.math500 import _math_equal, _strip_string


def is_correct(row, text):
    value = extract_last_boxed(text)
    if value is None:
        return False
    if row['answer_kind'] == 'choice':
        return value.strip().upper() == str(row['ground_truth']).strip().upper()
    try:
        return _math_equal(_strip_string(value), _strip_string(row['ground_truth']))
    except (TypeError, ValueError):
        return False


def main():
    from vllm import LLM, SamplingParams
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    args = p.parse_args()
    rows = [json.loads(x) for x in (args.root / 'training_candidates.jsonl').read_bytes().splitlines()]
    families = collections.defaultdict(list)
    for row in rows:
        families[int(row['family'])].append(row)
    if set(families) != set(range(16)):
        raise ValueError('Expected all 16 independent synthetic training families')
    model = LLM(model=str(args.root / 'model'), dtype='float16', tensor_parallel_size=1,
                max_model_len=12288, max_num_seqs=16, max_num_batched_tokens=4096,
                kv_cache_memory_bytes=24 * 1024**3, gpu_memory_utilization=0.70,
                enforce_eager=False, enable_prefix_caching=True)
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=42, max_tokens=8192)
    out = args.root / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    selected, counts = [], collections.Counter()
    target_per_family = max(1, args.count // 16)
    traces = out / 'synthetic-teacher.jsonl'
    with traces.open('w', encoding='utf8', buffering=1) as trace_file:
        for index in range(max(map(len, families.values()))):
            candidates = [family[index] for key, family in sorted(families.items())
                          if index < len(family) and counts[key] < target_per_family]
            if not candidates:
                break
            prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['user_prompt']}],
                       tokenize=False, add_generation_prompt=True) for r in candidates]
            prompts = [s if s.rstrip().endswith('<think>') else s + '<think>\n' for s in prompts]
            outputs = model.generate(prompts, sampling, use_tqdm=False)
            for row, prompt, result in zip(candidates, prompts, outputs):
                answer = result.outputs[0]
                ok = bool(answer.finish_reason == 'stop' and '</think>' in answer.text and is_correct(row, answer.text))
                record = {'example_id': row['example_id'], 'family': row['family'],
                          'prompt': prompt, 'text': answer.text, 'output_tokens': len(answer.token_ids),
                          'finish_reason': answer.finish_reason, 'verified_correct': ok}
                trace_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                if ok and len(selected) < args.count:
                    ids = list(result.prompt_token_ids) + list(answer.token_ids)
                    if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
                        ids.append(tokenizer.eos_token_id)
                    selected.append({'input_ids': ids, 'attention_mask': [1] * len(ids),
                                     'example_id': row['example_id'], 'family': row['family']})
                    counts[row['family']] += 1
            print('TEACHER_PROGRESS', len(selected), 'of', args.count, flush=True)
            if len(selected) >= args.count:
                break
    if len(selected) != args.count:
        raise RuntimeError(f'Only {len(selected)} verified complete synthetic answers; need {args.count}')
    path = out / 'teacher-windows.jsonl'
    path.write_text(''.join(json.dumps(x) + '\n' for x in selected))
    (out / 'teacher-metadata.json').write_text(json.dumps({
        'teacher': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'sampling': {'temperature': 0.6,
        'top_p': 0.95, 'top_k': 20, 'seed': 42, 'max_new_tokens': 8192},
        'source_sha256': hashlib.sha256((args.root / 'training_candidates.jsonl').read_bytes()).hexdigest(),
        'selected_per_family': dict(counts), 'benchmark_examples_used': False}, indent=2))
    print('TEACHER_READY', len(selected), flush=True)
    del model


if __name__ == '__main__':
    main()
