#!/usr/bin/env python3
"""Compare deterministic tokens from vLLM and GPTQModel on identical token IDs."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--backend', choices=['vllm', 'gptq'], required=True)
    args = p.parse_args()
    out = args.root / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    model_path = args.root / 'model-gptq2-g32'
    if args.backend == 'vllm':
        from vllm import LLM, SamplingParams
        model = LLM(model=str(model_path), quantization='gptq', dtype='float16', max_model_len=4096,
                    max_num_seqs=1, kv_cache_memory_bytes=4 * 1024**3,
                    gpu_memory_utilization=0.4, enforce_eager=False)
        tokenizer = model.get_tokenizer()
        messages = [{'role': 'user', 'content': 'Compute 2 + 3. Put the final answer in \\boxed{}.'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not prompt.rstrip().endswith('<think>'):
            prompt += '<think>\n'
        result = model.generate([prompt], SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)[0]
        record = {'prompt': prompt, 'prompt_token_ids': list(result.prompt_token_ids),
                  'output_token_ids': list(result.outputs[0].token_ids), 'text': result.outputs[0].text}
        (out / 'vllm.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        print('VLLM_DIAGNOSTIC_DONE', len(record['output_token_ids']), flush=True)
        del model
    else:
        import torch
        from gptqmodel import BACKEND, GPTQModel
        source = json.loads((out / 'vllm.json').read_text(encoding='utf8'))
        model = GPTQModel.load(str(model_path), device='cuda:0', backend=BACKEND.GPTQ_TORCH)
        tokens = torch.tensor([source['prompt_token_ids']], device='cuda:0')
        print('GPTQ_TORCH_MODEL_LOADED', flush=True)
        with torch.inference_mode():
            result = model.generate(input_ids=tokens, max_new_tokens=64, do_sample=False)
        ids = result[0, tokens.shape[1]:].tolist()
        common = sum(1 for a, b in zip(ids, source['output_token_ids']) if a == b)
        record = {'prompt_token_ids': source['prompt_token_ids'], 'output_token_ids': ids,
                  'matched_positions': common, 'first_token_matches': bool(ids and ids[0] == source['output_token_ids'][0])}
        (out / 'gptq-torch.json').write_text(json.dumps(record, indent=2), encoding='utf8')
        print('GPTQ_TORCH_DIAGNOSTIC_DONE', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
