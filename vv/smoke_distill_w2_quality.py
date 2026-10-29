#!/usr/bin/env python3
"""Qualitative sanity check of the freshly quantized reasoning model."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quantlab.benchmarks.boxed import extract_last_boxed

CASES = [
    ('add', 'Compute 2 + 3.', '5'),
    ('subtract', 'Compute 13 - 5.', '8'),
    ('multiply', 'Compute 7 * 6.', '42'),
    ('divide', 'Compute 24 / 4.', '6'),
]


def main():
    from vllm import LLM, SamplingParams
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    model = LLM(model=str(args.root / 'model-gptq2-g32'), quantization='gptq', dtype='float16',
                max_model_len=8192, max_num_seqs=4, kv_cache_memory_bytes=8 * 1024**3,
                gpu_memory_utilization=0.5, enforce_eager=False)
    tok = model.get_tokenizer()
    prompts = []
    for _, question, _ in CASES:
        prompt = tok.apply_chat_template([{'role': 'user', 'content': question +
            ' Think carefully and put the final numeric answer in \\boxed{}.'}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(prompt if prompt.rstrip().endswith('<think>') else prompt + '<think>\n')
    outputs = model.generate(prompts, SamplingParams(temperature=0, max_tokens=4096), use_tqdm=False)
    records = []
    for (name, question, expected), result in zip(CASES, outputs):
        answer = result.outputs[0]
        parsed = extract_last_boxed(answer.text)
        records.append({'case': name, 'question': question, 'expected': expected, 'parsed': parsed,
                        'correct': parsed is not None and parsed.strip() == expected,
                        'text': answer.text, 'token_ids': list(answer.token_ids),
                        'finish_reason': answer.finish_reason})
    out = args.root / 'w2-quality-smoke.json'
    out.write_text(json.dumps({'cases': records, 'correct': sum(x['correct'] for x in records)},
                              ensure_ascii=False, indent=2), encoding='utf8')
    print('W2_QUALITY_SMOKE', sum(x['correct'] for x in records), '/', len(records), flush=True)
    if sum(x['correct'] for x in records) < 2:
        raise RuntimeError('W2 passed fewer than two simple checks; inspect outputs before full evaluation')
    del model


if __name__ == '__main__':
    main()
