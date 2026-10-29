#!/usr/bin/env python3
"""Check the source FP16 model in GPTQ's Transformers stack on identical IDs."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    sample = json.loads((args.root / 'quality-smoke-v2-fp16.json').read_text())['cases'][0]
    ids = sample['prompt_token_ids']
    model = AutoModelForCausalLM.from_pretrained(str(args.root / 'model'), dtype=torch.float16,
                                                device_map='cuda:0', local_files_only=True)
    model.eval()
    inputs = torch.tensor([ids], device='cuda:0')
    with torch.inference_mode():
        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                   max_new_tokens=64, do_sample=False)
    output = generated[0, len(ids):].tolist()
    reference = sample['token_ids'][:len(output)]
    record = {'prompt_token_ids': ids, 'output_token_ids': output,
              'vllm_fp16_output_token_ids': reference,
              'first_token_matches': bool(output and reference and output[0] == reference[0]),
              'matched_positions': sum(a == b for a, b in zip(output, reference))}
    out = args.root / 'diagnostics' / 'transformers-fp16.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding='utf8')
    print('TRANSFORMERS_FP16_CHECK', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
