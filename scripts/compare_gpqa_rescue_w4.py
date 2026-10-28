#!/usr/bin/env python3
"""Compare complete, paired GPQA W4 and online W2+FP16 rescue measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.measure_gpqa_rescue import paper_gpqa_choice


def read_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs in {path}")
    return {r["example_id"]: r for r in rows}


def percentile95(values):
    return statistics.quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]


def compare(w4, rescue, expected=198):
    if len(w4) != expected or len(rescue) != expected or set(w4) != set(rescue):
        raise ValueError("Both complete runs must have exactly matching IDs")
    w4_time, rescue_time = [], []
    w4_correct = rescue_correct = both_correct = rescue_only = w4_only = 0
    fallback = commits = 0
    w4_tokens = w2_tokens = fp16_tokens = 0
    paired = []
    for eid in sorted(w4):
        wr, rr = w4[eid], rescue[eid]
        prompt_hash = hashlib.sha256(rr['prompt'].encode('utf8')).hexdigest()
        if wr.get('prompt_sha256') != prompt_hash or wr.get('ground_truth') != rr['ground_truth']:
            raise ValueError(f"W4 prompt or ground truth missing/different: {eid}")
        predicted = paper_gpqa_choice(wr["text"])
        wc = predicted is not None and predicted == rr["ground_truth"]
        rc = bool(rr["is_correct"])
        if rc != (rr["predicted"] is not None and rr["predicted"] == rr["ground_truth"]):
            raise ValueError(f"Rescue judgement mismatch: {eid}")
        branch = rr["branch"]
        if branch not in ("normal", "commit", "fallback"):
            raise ValueError(f"Unknown rescue branch: {eid}")
        if branch != "fallback" and rr["fp16_tokens"] != 0:
            raise ValueError(f"FP16 tokens in non-fallback branch: {eid}")
        if branch == "fallback" and rr["fp16_tokens"] == 0:
            raise ValueError(f"Empty FP16 fallback: {eid}")
        wt, rt = float(wr["wall_seconds"]), float(rr["end_to_end_seconds"])
        if wt <= 0 or rt <= 0 or rt < float(rr["w2_seconds"]) + float(rr["fp16_seconds"]):
            raise ValueError(f"Invalid timing: {eid}")
        w4_time.append(wt); rescue_time.append(rt)
        w4_correct += wc; rescue_correct += rc
        both_correct += wc and rc; rescue_only += rc and not wc; w4_only += wc and not rc
        fallback += branch == "fallback"; commits += branch == "commit"
        w4_tokens += wr["output_tokens"]; w2_tokens += rr["w2_tokens"]; fp16_tokens += rr["fp16_tokens"]
        paired.append({"example_id": eid, "ground_truth": rr["ground_truth"],
                       "w4_prediction": predicted, "w4_correct": wc,
                       "rescue_prediction": rr["predicted"], "rescue_correct": rc,
                       "rescue_branch": branch, "w4_seconds": wt, "rescue_seconds": rt})
    result = {"examples": expected,
        "w4": {"correct": w4_correct, "accuracy": w4_correct / expected,
               "total_seconds": sum(w4_time), "mean_seconds": statistics.mean(w4_time),
               "p50_seconds": statistics.median(w4_time), "p95_seconds": percentile95(w4_time),
               "output_tokens": w4_tokens},
        "rescue": {"correct": rescue_correct, "accuracy": rescue_correct / expected,
                   "fallbacks": fallback, "fallback_fraction": fallback / expected,
                   "commits": commits, "commit_fraction": commits / expected,
                   "total_seconds": sum(rescue_time), "mean_seconds": statistics.mean(rescue_time),
                   "p50_seconds": statistics.median(rescue_time), "p95_seconds": percentile95(rescue_time),
                   "w2_output_tokens": w2_tokens, "fp16_output_tokens": fp16_tokens},
        "paired": {"both_correct": both_correct, "w4_only": w4_only, "rescue_only": rescue_only},
        "timing": "Warm batch-1 inference; rescue includes W2 generation, detection, routing and FP16 rerun with both models resident. One-time model loading and warmup are separate in each runtime.json.",
        "parser": "Original case-sensitive GPQAAdapter letter parser applied consistently to W4 and rescue."}
    return result, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--w4', type=Path, required=True, help='W4 artifacts directory')
    p.add_argument('--rescue', type=Path, required=True, help='online rescue artifacts directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    wc = json.loads((args.w4 / 'config.json').read_text())
    rc = json.loads((args.rescue / 'config.json').read_text())
    if wc['benchmark'] != rc['benchmark'] or wc['actors'][0]['generation_params'] != rc['sampling']:
        raise ValueError('Benchmark or sampling parameters differ')
    if wc['actors'][0]['model_id'] != 'kaitchup/Qwen3-32B-autoround-4bit-gptq' or rc['w2_model'] != 'kaitchup/Qwen3-32B-autoround-2bit-gptq' or rc['fp16_model'] != 'Qwen/Qwen3-32B':
        raise ValueError('Unexpected model identity')
    wr = json.loads((args.w4 / 'runtime.json').read_text())
    rr = json.loads((args.rescue / 'runtime.json').read_text())
    if wr.get('vllm') != rc['vllm'] or wr.get('torch') != rc['torch'] or not wr.get('gpu_name') or wr['gpu_name'] != rr.get('gpu_name'):
        raise ValueError('Runtime version or GPU model missing/different')
    w4_summary = json.loads((args.w4 / 'summary.json').read_text())
    rescue_summary = json.loads((args.rescue / 'summary.json').read_text())
    if not w4_summary.get('complete') or not rescue_summary.get('complete'):
        raise ValueError('Both runs must be complete')
    result, paired = compare(read_rows(args.w4 / 'results.jsonl'), read_rows(args.rescue / 'results.jsonl'))
    result['runtime'] = {'w4_gpu_name': wr['gpu_name'], 'rescue_gpu_name': rr['gpu_name'],
                         'vllm': rc['vllm'], 'torch': rc['torch'],
                         'w4_model_load_seconds': wr['model_load_seconds'],
                         'rescue_w2_load_seconds': rr['w2_load_seconds'],
                         'rescue_fp16_load_seconds': rr['fp16_load_seconds'],
                         'w4_warmup_seconds': wr['warmup_seconds'],
                         'rescue_warmup_seconds': rr['warmup_seconds']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    (args.output / 'paired.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in paired), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
