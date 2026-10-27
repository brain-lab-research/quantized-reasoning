#!/usr/bin/env python3
"""Evaluate count-matched routing policies on the full historical cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from quantlab.benchmarks.math500 import MATH500Adapter
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    policies = json.loads((args.root/"policies.json").read_text())
    config = json.loads((args.root/"config.json").read_text())
    if config["input_sha256"] != hashlib.sha256((args.root/"requests.jsonl").read_bytes()).hexdigest():
        raise ValueError("Routing input hash mismatch")
    requests = {r["example_id"]:r for r in rows(args.root/"requests.jsonl")}
    raw = rows(args.root/"results.jsonl")
    if len({r["example_id"] for r in raw}) != len(raw) or {r["example_id"] for r in raw} != set(requests):
        raise ValueError("Routing generations are incomplete or duplicated")
    adapter = MATH500Adapter()
    gold = {r["example_id"]:r["ground_truth"] for r in rows(args.source/"judgements.jsonl")}
    base = {r["example_id"]: bool(adapter.is_correct(
        adapter.extract_answer("".join(s["text"] for s in r["segments"])), gold[r["example_id"]]))
        for r in rows(args.source/"traces.jsonl")}
    graded = {}
    for row in raw:
        request = requests[row["example_id"]]
        if row["request_id"] != request["request_id"] or row["ground_truth"] != request["ground_truth"]:
            raise ValueError("Routing result/request mismatch")
        pred = adapter.extract_answer(row["text"])
        graded[row["example_id"]] = dict(is_correct=bool(adapter.is_correct(pred, request["ground_truth"])),
                                         output_tokens=row["token_count"], prompt_tokens=row["prompt_tokens"])
    summary = dict(source_cases=len(base), baseline_correct=sum(base.values()),
        baseline_accuracy=sum(base.values())/len(base), unique_generations=len(raw),
        fp16_budget=16384, policies={})
    for name in ("targeted", "random"):
        selected = set(policies[name+"_ids"])
        correct = sum(graded[eid]["is_correct"] if eid in selected else value for eid,value in base.items())
        summary["policies"][name] = dict(calls=len(selected), fallback_fraction=len(selected)/len(base),
            correct=correct, accuracy=correct/len(base),
            fp16_output_tokens=sum(graded[eid]["output_tokens"] for eid in selected),
            fp16_prompt_tokens=sum(graded[eid]["prompt_tokens"] for eid in selected))
    summary["cost_scope"] = "FP16 token costs only; no combined W2+FP16 latency claim."
    summary["grading"] = "Both W2 baseline and FP16 outputs regraded with the same MATH500Adapter."
    (args.root/"summary.json").write_text(json.dumps(summary, indent=2))
    (args.root/"judgements.json").write_text(json.dumps(graded, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
