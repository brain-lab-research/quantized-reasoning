#!/usr/bin/env python3
"""Validate and grade matched recovery arms without treating unfinished arms as errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ARMS = ("restart", "continue", "trim")


def read_rows(path: Path, allow_partial_tail: bool = False) -> list[dict]:
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_partial_tail and i == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return rows


def grade(requests: list[dict], results: list[dict], source_prompts: dict, adapter):
    expected = {r["request_id"]: r for r in requests}
    if len(expected) != len(requests):
        raise ValueError("Duplicate request IDs")
    cases = {}
    for request in requests:
        case = cases.setdefault(request["example_id"], {})
        if request["arm"] not in ARMS or request["arm"] in case:
            raise ValueError("Invalid or duplicate arm")
        case[request["arm"]] = request["request_id"]
    if any(set(case) != set(ARMS) for case in cases.values()):
        raise ValueError("Each case must have all three requested arms")
    graded = {}
    for result in results:
        rid = result["request_id"]
        if rid in graded or rid not in expected:
            raise ValueError("Duplicate or unknown result ID: " + rid)
        request = expected[rid]
        for key in ("example_id", "arm", "ground_truth", "max_new_tokens", "seed"):
            if result[key] != request[key]:
                raise ValueError("Result metadata differs from request: " + key)
        count = result["token_count"]
        if count > request["max_new_tokens"] or count != len(result["token_ids"]):
            raise ValueError("Invalid output token count")
        original = source_prompts[request["example_id"]]
        if not request["prompt"].startswith(original):
            raise ValueError("Recovery request does not preserve source prompt")
        prefix = request["prompt"][len(original):]
        if request["arm"] == "restart" and prefix:
            raise ValueError("Restart must not contain source completion")
        # A continuation can close an answer expression opened in its prefix.
        assistant_text = prefix + result["text"]
        predicted = adapter.extract_answer(assistant_text)
        graded[rid] = dict(request_id=rid, example_id=request["example_id"],
            arm=request["arm"], predicted=predicted,
            is_correct=bool(adapter.is_correct(predicted, request["ground_truth"])),
            output_tokens=count, prompt_tokens=result["prompt_tokens"],
            finish_reason=result["finish_reason"])
    paired = [case for case in cases.values() if all(rid in graded for rid in case.values())]
    metrics = {}
    for arm in ARMS:
        rows = [graded[case[arm]] for case in paired]
        metrics[arm] = dict(n=len(rows), correct=sum(r["is_correct"] for r in rows),
            accuracy=sum(r["is_correct"] for r in rows) / len(rows) if rows else None,
            mean_output_tokens=sum(r["output_tokens"] for r in rows) / len(rows) if rows else None,
            mean_prompt_tokens=sum(r["prompt_tokens"] for r in rows) / len(rows) if rows else None,
            length_finishes=sum(r["finish_reason"] == "length" for r in rows))
    contrasts = {}
    for left, right in (("restart", "continue"), ("restart", "trim"), ("trim", "continue")):
        wins = losses = 0
        for case in paired:
            a, b = graded[case[left]]["is_correct"], graded[case[right]]["is_correct"]
            wins += a and not b
            losses += b and not a
        contrasts[left + "_vs_" + right] = dict(wins=wins, losses=losses, ties=len(paired)-wins-losses)
    summary = dict(expected_cases=len(cases), expected_requests=len(expected),
        completed_requests=len(graded), paired_cases=len(paired),
        complete=len(graded) == len(expected), metrics=metrics, contrasts=contrasts,
        interpretation="Compare only after completion; early finishers are a selected subset.",
        timing="Prompt and output tokens are separate; shared batch completion time is not request latency.")
    return summary, list(graded.values())


def main():
    from quantlab.benchmarks.math500 import MATH500Adapter
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--source-traces", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    digest = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    if digest != config["input_sha256"]:
        raise ValueError("Request file hash differs from generation config")
    prompts = {r["example_id"]: r["prompt"] for r in read_rows(args.source_traces)}
    summary, rows = grade(read_rows(args.requests), read_rows(args.results, True), prompts, MATH500Adapter())
    summary["requests_sha256"] = digest
    summary["grading"] = "Existing MATH500Adapter on complete assistant prefix plus continuation."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "judgements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
