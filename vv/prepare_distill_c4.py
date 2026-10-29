#!/usr/bin/env python3
"""Prepare deterministic 2048-token C4 windows without benchmark examples."""
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--count', type=int, default=192)
    p.add_argument('--length', type=int, default=2048)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.root / 'model', local_files_only=True)
    stream = load_dataset('allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                          split='train', streaming=True).shuffle(seed=42, buffer_size=1000)
    out = args.root / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'c4-windows.jsonl'
    tmp = target.with_suffix('.tmp')
    count, pool = 0, []
    with tmp.open('w', encoding='utf8') as f:
        for row in stream:
            text = row.get('text', '')
            if not text:
                continue
            pool.extend(tokenizer.encode(text, add_special_tokens=False))
            if tokenizer.eos_token_id is not None:
                pool.append(tokenizer.eos_token_id)
            while len(pool) >= args.length and count < args.count:
                ids, pool = pool[:args.length], pool[args.length:]
                f.write(json.dumps({'input_ids': ids, 'attention_mask': [1] * len(ids)}) + '\n')
                count += 1
            if count == args.count:
                break
    if count != args.count:
        raise RuntimeError(f'C4 stream produced only {count} windows')
    tmp.replace(target)
    print('C4_READY', count, args.length, hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
