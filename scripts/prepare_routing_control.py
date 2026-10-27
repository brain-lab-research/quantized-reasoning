#!/usr/bin/env python3
"""Match random and loop-selected FP16 call counts without using answer correctness."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def prepare(traces: list[dict], judgements: list[dict], targeted: list[dict], seed: int):
    source = {row["example_id"]: row for row in traces}
    gold = {row["example_id"]: row["ground_truth"] for row in judgements}
    chosen = [r for r in targeted if r["arm"] == "restart"]
    targeted_ids = sorted(r["example_id"] for r in chosen)
    if not targeted_ids or len(set(targeted_ids)) != len(targeted_ids):
        raise ValueError("Expected unique targeted restart cases")
    for row in chosen:
        if row["prompt"] != source[row["example_id"]]["prompt"]:
            raise ValueError("Targeted prompt differs from source cohort")
    random_ids = sorted(random.Random(seed).sample(sorted(source), len(targeted_ids)))
    selected = sorted(set(targeted_ids) | set(random_ids))
    requests = [dict(request_id=eid+":routing", example_id=eid, arm="routing",
        prompt=source[eid]["prompt"], ground_truth=gold[eid], max_new_tokens=16384,
        seed=42) for eid in selected]
    return requests, dict(targeted_ids=targeted_ids, random_ids=random_ids,
        selection_seed=seed, source_cases=len(source), calls_per_policy=len(chosen),
        fallback_fraction=len(chosen)/len(source), unique_generations=len(selected),
        definition="Loop-selected versus random FP16 routing; gold unused in selection; 16K FP16 budget.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--targeted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    requests, policies = prepare(rows(args.source/"traces.jsonl"), rows(args.source/"judgements.jsonl"),
                                 rows(args.targeted), args.seed)
    policies["source_sha256"] = hashlib.sha256((args.source/"traces.jsonl").read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in requests))
    (args.output/"policies.json").write_text(json.dumps(policies, indent=2))
    print(json.dumps({k:v for k,v in policies.items() if not k.endswith("_ids")}), flush=True)


if __name__ == "__main__":
    main()
