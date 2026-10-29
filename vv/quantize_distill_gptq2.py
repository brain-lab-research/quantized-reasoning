#!/usr/bin/env python3
"""Quantize the local Distill-Llama with audited C4 and synthetic windows."""
import argparse
import hashlib
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def main():
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    calib = args.root / 'calibration'
    c4_path, teacher_path = calib / 'c4-windows.jsonl', calib / 'teacher-windows.jsonl'
    c4, teacher = read_rows(c4_path), read_rows(teacher_path)
    if len(c4) != 192 or len(teacher) != 64:
        raise ValueError(f'Expected 192 C4 + 64 teacher windows; got {len(c4)} + {len(teacher)}')
    source_meta = json.loads((calib / 'teacher-metadata.json').read_text())
    if source_meta.get('benchmark_examples_used') is not False:
        raise ValueError('Synthetic calibration provenance is missing')
    windows = []
    for row in c4 + teacher:
        ids = row['input_ids']
        if len(ids) < 10 or len(ids) > 12288 or len(row['attention_mask']) != len(ids):
            raise ValueError('Invalid calibration window')
        windows.append({'input_ids': ids, 'attention_mask': row['attention_mask']})
    output = args.root / 'model-gptq2-g32'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing quantized model: {output}')
    staging = args.root / 'model-gptq2-g32-building'
    if staging.exists():
        raise FileExistsError(f'Previous incomplete quantization requires inspection: {staging}')
    config = QuantizeConfig(bits=2, group_size=32, desc_act=False, sym=True, offload_to_disk=False)
    print('GPTQ_LOAD', flush=True)
    model = GPTQModel.load(str(args.root / 'model'), quantize_config=config, device='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.root / 'model', local_files_only=True)
    print('GPTQ_START', len(windows), flush=True)
    model.quantize(calibration=windows, batch_size=1, tokenizer=tokenizer)
    model.save(str(staging))
    tokenizer.save_pretrained(staging)
    (staging / 'calibration-provenance.json').write_text(json.dumps({
        'bits': 2, 'group_size': 32, 'desc_act': False, 'sym': True,
        'c4_windows': len(c4), 'teacher_windows': len(teacher),
        'c4_sha256': hashlib.sha256(c4_path.read_bytes()).hexdigest(),
        'teacher_sha256': hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        'teacher': source_meta['teacher'], 'benchmark_examples_used': False,
        'source_model': json.loads((args.root / 'model-revision.json').read_text())}, indent=2))
    staging.rename(output)
    print('GPTQ_DONE', output, flush=True)


if __name__ == '__main__':
    main()
