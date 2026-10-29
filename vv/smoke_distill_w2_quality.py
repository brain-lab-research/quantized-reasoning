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


def validate_tokenizers(fp16, w2):
    if fp16.get_vocab() != w2.get_vocab() or fp16.eos_token_id != w2.eos_token_id:
        raise ValueError('FP16 and W2 tokenizer vocabulary or EOS differ')
    for _, question, _ in CASES:
        messages = [{'role': 'user', 'content': question}]
        a = fp16.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        b = w2.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if a != b:
            raise ValueError(f'FP16 and W2 token IDs differ for {question}')


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--precision', choices=['fp16', 'w2'], default='w2')
    args = p.parse_args()
    original = AutoTokenizer.from_pretrained(args.root / 'model', local_files_only=True)
    model_path = args.root / ('model-gptq2-g32' if args.precision == 'w2' else 'model')
    if args.precision == 'w2':
        quantized = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        validate_tokenizers(original, quantized)
        print('TOKENIZER_PARITY_OK', flush=True)
    kwargs = {'quantization': 'gptq'} if args.precision == 'w2' else {}
    model = LLM(model=str(model_path), dtype='float16', max_model_len=8192,
                max_num_seqs=4, kv_cache_memory_bytes=8 * 1024**3,
                gpu_memory_utilization=0.5, enforce_eager=False, **kwargs)
    tok = model.get_tokenizer()
    prompts = []
    for _, question, _ in CASES:
        prompt = tok.apply_chat_template([{'role': 'user', 'content': question +
            ' Think carefully and put the final numeric answer in \\boxed{}.'}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(prompt if prompt.rstrip().endswith('<think>') else prompt + '<think>\n')
    token_prompts = [{'prompt_token_ids': tok.encode(s, add_special_tokens=False)} for s in prompts]
    outputs = model.generate(token_prompts, SamplingParams(temperature=0, max_tokens=4096), use_tqdm=False)
    records = []
    for (name, question, expected), result in zip(CASES, outputs):
        answer = result.outputs[0]
        parsed = extract_last_boxed(answer.text)
        records.append({'case': name, 'question': question, 'expected': expected, 'parsed': parsed,
                        'correct': parsed is not None and parsed.strip() == expected,
                        'prompt_token_ids': list(result.prompt_token_ids),
                        'text': answer.text, 'token_ids': list(answer.token_ids),
                        'finish_reason': answer.finish_reason})
    out = args.root / ('quality-smoke-v2-' + args.precision + '.json')
    out.write_text(json.dumps({'cases': records, 'correct': sum(x['correct'] for x in records)},
                              ensure_ascii=False, indent=2), encoding='utf8')
    print(args.precision.upper() + '_QUALITY_SMOKE', sum(x['correct'] for x in records), '/', len(records), flush=True)
    if sum(x['correct'] for x in records) < 2:
        raise RuntimeError('W2 passed fewer than two simple checks; inspect outputs before full evaluation')
    del model


if __name__ == '__main__':
    main()
