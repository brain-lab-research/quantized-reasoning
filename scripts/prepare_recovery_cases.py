#!/usr/bin/env python3
"""Create causal, gold-independent restart/continue cases from saved baseline traces."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def first_loop(ids, n=20, repeats=4, window=1024):
    seen = defaultdict(deque)
    for end in range(n, len(ids) + 1):
        start = end - n
        key = tuple(ids[start:end])
        occurrences = seen[key]
        while occurrences and occurrences[0] < end - window:
            occurrences.popleft()
        occurrences.append(start)
        if len(occurrences) >= repeats:
            return {"onset": occurrences[0], "trigger": end}
    return None


def last_complete_box(text):
    import re
    value = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for pos in range(match.end(), len(text)):
            depth += (text[pos] == "{") - (text[pos] == "}")
            if depth == 0:
                candidate = text[match.end():pos].strip()
                if candidate:
                    value = candidate
                break
    return value


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-cases", type=int, default=100)
    p.add_argument("--fp16-budget", type=int, default=16384)
    p.add_argument("--context", type=int, default=36864)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    traces = rows(args.run / "traces.jsonl")
    judgements = {r["example_id"]: r for r in rows(args.run / "judgements.jsonl")}
    counts = {"total": len(traces), "loop": 0, "commit": 0, "fallback": 0,
              "context_excluded": 0}
    requests, events = [], []
    for trace in traces:
        text = "".join(s["text"] for s in trace["segments"])
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        event = first_loop(encoded["input_ids"])
        if event is None:
            continue
        counts["loop"] += 1
        onset = encoded["offset_mapping"][event["onset"]][0]
        trigger = encoded["offset_mapping"][event["trigger"] - 1][1]
        prefix = text[:trigger]
        candidate = last_complete_box(prefix)
        kind = "commit" if candidate is not None else "fallback"
        counts[kind] += 1
        events.append({"example_id": trace["example_id"], "kind": kind,
                       "onset_tokens": event["onset"], "trigger_tokens": event["trigger"],
                       "onset_char": onset, "trigger_char": trigger, "candidate": candidate,
                       "source_final_correct": judgements.get(trace["example_id"], {}).get("is_correct")})
        if kind != "fallback" or len(requests) >= args.max_cases * 3:
            continue
        prompt = trace["prompt"]
        continued = prompt + prefix
        if len(tokenizer.encode(continued, add_special_tokens=False)) + args.fp16_budget > args.context:
            counts["context_excluded"] += 1
            continue
        gt = judgements[trace["example_id"]]["ground_truth"]
        for arm, full_prompt in (("restart", prompt), ("continue", continued),
                                 ("trim", prompt + text[:onset])):
            requests.append({"request_id": trace["example_id"] + ":" + arm,
                "example_id": trace["example_id"], "arm": arm, "prompt": full_prompt,
                "ground_truth": gt, "max_new_tokens": args.fp16_budget, "seed": 42,
                "source_trigger_tokens": event["trigger"], "source_onset_tokens": event["onset"],
                "retokenized_source": True})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (args.output / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    summary = {**counts, "selected_cases": len(requests) // 3, "requests": len(requests),
        "detector": {"n": 20, "repeats": 4, "window": 1024},
        "fp16_budget_per_arm": args.fp16_budget, "context": args.context,
        "selection": "first eligible pre-answer loop cases; gold unused in selection",
        "source_trace_sha256": hashlib.sha256((args.run / "traces.jsonl").read_bytes()).hexdigest()}
    (args.output / "case-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
