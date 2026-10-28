#!/usr/bin/env python3
"""Online batch-1 GPQA loop rescue with resident W2 and FP16 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LoopDetector:
    """Causal n-gram trigger, active only before the first think-close token."""

    def __init__(self, close_id: int, n: int = 20, repeats: int = 4, window: int = 1024):
        self.close_id, self.n, self.repeats, self.window = close_id, n, repeats, window
        self.ids = []
        self.seen = defaultdict(deque)
        self.closed = False

    def feed(self, token: int):
        self.ids.append(token)
        if token == self.close_id:
            self.closed = True
        end = len(self.ids)
        if self.closed or end < self.n:
            return None
        start = end - self.n
        occurrences = self.seen[tuple(self.ids[start:end])]
        while occurrences and occurrences[0] < end - self.window:
            occurrences.popleft()
        occurrences.append(start)
        if len(occurrences) >= self.repeats:
            return {"onset": occurrences[0], "trigger": end}
        return None


def paper_gpqa_choice(text: str):
    """Case-sensitive parser in the original repository's GPQAAdapter."""
    line = text.strip().split("\n")[-1].strip()
    match = re.match(r"^([ABCD])[).\s]?$", line)
    if match:
        return match.group(1)
    matches = list(re.finditer(r"\b([ABCD])\b", text))
    return matches[-1].group(1) if matches else None


def stream_one(engine, request_id, prompt, params, detector=None):
    """Run one request, aborting immediately after a detected loop."""
    from vllm.sampling_params import RequestOutputKind

    assert params.output_kind == RequestOutputKind.DELTA
    ids = []
    event = None
    finish_reason = None
    engine.add_request(request_id, prompt, params)
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.request_id != request_id:
                raise RuntimeError("Unexpected request in batch-1 engine")
            generated = out.outputs[0]
            for token in generated.token_ids:
                ids.append(token)
                if detector is not None:
                    event = detector.feed(token)
                    if event:
                        engine.abort_request([request_id])
                        return ids, event, "loop_abort"
            if out.finished:
                finish_reason = generated.finish_reason
                return ids, None, finish_reason
    raise RuntimeError(f"Request disappeared without completion: {request_id}")


def read_completed(path: Path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # Only an interrupted, unterminated final line may be discarded on resume.
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        with path.open("r+b") as f:
            f.truncate(end)
        data = data[:end]
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate completed GPQA IDs")
    return rows


def write_summary(path: Path, rows, expected):
    times = [r["end_to_end_seconds"] for r in rows]
    n = len(rows)
    data = {"expected": expected, "completed": n, "complete": n == expected,
            "correct": sum(r["is_correct"] for r in rows),
            "fallbacks": sum(r["branch"] == "fallback" for r in rows),
            "commits": sum(r["branch"] == "commit" for r in rows),
            "w2_output_tokens": sum(r["w2_tokens"] for r in rows),
            "fp16_output_tokens": sum(r["fp16_tokens"] for r in rows),
            "mean_end_to_end_seconds": sum(times) / n if n else None,
            "p50_end_to_end_seconds": statistics.median(times) if n else None,
            "timing": "warm batch-1, W2 and FP16 coresident; includes detection, routing and FP16 rerun; excludes one-time model load and warmup"}
    path.write_text(json.dumps(data, indent=2))


def main():
    import atexit
    import yaml
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.actors.backends.vllm_backend import _reap_vllm_subprocesses
    from quantlab.config.schema import ExperimentConfig
    from quantlab.runner import _load_benchmark_examples

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--w2-path", required=True)
    p.add_argument("--fp16-path", required=True)
    p.add_argument("--fp16-revision", required=True)
    p.add_argument("--examples", type=int, default=198)
    p.add_argument("--smoke", action="store_true", help="Also force one short FP16 path")
    args = p.parse_args()
    atexit.register(_reap_vllm_subprocesses)
    raw = yaml.safe_load(args.config.read_text())
    raw["benchmark"]["max_examples"] = args.examples
    config = ExperimentConfig.model_validate(raw)
    examples = _load_benchmark_examples(BenchmarkRegistry.build("gpqa_diamond"), config.benchmark)
    if len(examples) != args.examples:
        raise ValueError("Incomplete GPQA dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256("\n".join(e.example_id + "\0" + e.prompt for e in examples).encode()).hexdigest()
    identity = {"benchmark": config.benchmark.model_dump(), "w2_model": config.actors[0].model_id,
                "w2_revision": Path(args.w2_path).name, "fp16_model": "Qwen/Qwen3-32B",
                "fp16_revision": args.fp16_revision, "sampling": config.actors[0].generation_params.model_dump(),
                "dataset_prompt_sha256": prompt_hash, "parser": "original GPQAAdapter case-sensitive letter",
                "trigger": {"ngram": 20, "repeats": 4, "window": 1024, "active": "before </think>",
                            "candidate_prefix": "before first occurrence of repeated ngram"},
                "vllm": vllm.__version__, "torch": torch.__version__}
    cfg_path = args.output / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != identity:
        raise ValueError("Resume configuration or model revision differs")
    cfg_path.write_text(json.dumps(identity, indent=2))
    rows_path = args.output / "results.jsonl"
    rows = read_completed(rows_path)
    done = {r["example_id"] for r in rows}
    expected_ids = {e.example_id for e in examples}
    if not done <= expected_ids:
        raise ValueError("Stored result IDs differ from the current dataset")
    if len(done) == len(examples):
        write_summary(args.output / "summary.json", rows, len(examples))
        print("RESCUE_ALREADY_COMPLETE", flush=True)
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    tokenizer = AutoTokenizer.from_pretrained(args.w2_path, local_files_only=True)
    fp16_tokenizer = AutoTokenizer.from_pretrained(args.fp16_path, local_files_only=True)
    if tokenizer.get_vocab() != fp16_tokenizer.get_vocab():
        raise ValueError("W2 and FP16 tokenizer vocabularies differ")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if close_id is None or close_id == tokenizer.unk_token_id:
        raise ValueError("Missing dedicated </think> token")
    common = dict(dtype="float16", tensor_parallel_size=1, enforce_eager=False,
                  max_model_len=36864, kv_cache_memory_bytes=12 * 1024**3,
                  max_num_seqs=1, max_num_batched_tokens=4096, enable_prefix_caching=True)
    init = time.perf_counter()
    w2 = LLM(model=args.w2_path, quantization="gptq", gpu_memory_utilization=0.35, **common)
    w2_load = time.perf_counter() - init
    free_after_w2, total_gpu = torch.cuda.mem_get_info()
    fp16_utilization = min(0.70, 0.95 * free_after_w2 / total_gpu)
    if fp16_utilization < 0.60:
        raise RuntimeError("Insufficient free GPU memory to load the resident FP16 engine")
    init = time.perf_counter()
    fp16 = LLM(model=args.fp16_path, gpu_memory_utilization=fp16_utilization, **common)
    fp16_load = time.perf_counter() - init
    warmup = "<|im_start|>user\nCompute 2+3.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    warmup_started = time.perf_counter()
    for model in (w2, fp16):
        model.generate([warmup], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
    warmup_seconds = time.perf_counter() - warmup_started
    if args.smoke:
        # Exercise a real fallback engine, independently of whether the first case loops.
        ids, _, _ = stream_one(fp16.llm_engine, "smoke-fp16", examples[0].prompt,
            SamplingParams(temperature=0, max_tokens=16, output_kind=RequestOutputKind.DELTA))
        if not ids:
            raise RuntimeError("FP16 fallback smoke produced no tokens")
    (args.output / "runtime.json").write_text(json.dumps({"w2_load_seconds": w2_load,
        "fp16_load_seconds": fp16_load, "warmup_seconds": warmup_seconds,
        "w2_gpu_memory_utilization": 0.35, "fp16_gpu_memory_utilization": fp16_utilization,
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "gpu_free_total_bytes_after_load": list(torch.cuda.mem_get_info())}, indent=2))
    gen = config.actors[0].generation_params
    def params():
        return SamplingParams(max_tokens=gen.max_new_tokens, temperature=gen.temperature,
            top_p=gen.top_p, top_k=gen.top_k, seed=gen.seed, stop=gen.stop_sequences or None,
            output_kind=RequestOutputKind.DELTA)
    with rows_path.open("a", encoding="utf8", buffering=1) as f:
        for example in examples:
            if example.example_id in done:
                continue
            started = time.perf_counter()
            w2_ids, event, finish = stream_one(w2.llm_engine, "w2-" + example.example_id,
                                               example.prompt, params(), LoopDetector(close_id))
            w2_seconds = time.perf_counter() - started
            candidate = paper_gpqa_choice(tokenizer.decode(w2_ids[:event["onset"]],
                skip_special_tokens=False, clean_up_tokenization_spaces=False)) if event else None
            fallback_ids = []
            fp16_seconds = 0.0
            if event and candidate is None:
                branch = "fallback"
                stage_start = time.perf_counter()
                fallback_ids, _, fp16_finish = stream_one(fp16.llm_engine, "fp16-" + example.example_id,
                                                         example.prompt, params())
                fp16_seconds = time.perf_counter() - stage_start
                response = tokenizer.decode(fallback_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                predicted = paper_gpqa_choice(response)
            elif event:
                branch, fp16_finish, predicted = "commit", None, candidate
            else:
                branch, fp16_finish = "normal", None
                predicted = paper_gpqa_choice(tokenizer.decode(w2_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            elapsed = time.perf_counter() - started
            record = {"example_id": example.example_id, "prompt": example.prompt,
                      "ground_truth": example.ground_truth, "predicted": predicted,
                      "is_correct": predicted == example.ground_truth, "branch": branch,
                      "candidate": candidate, "loop": event, "w2_tokens": len(w2_ids),
                      "fp16_tokens": len(fallback_ids), "w2_token_ids": w2_ids,
                      "fp16_token_ids": fallback_ids, "w2_finish_reason": finish,
                      "fp16_finish_reason": fp16_finish, "w2_seconds": w2_seconds,
                      "fp16_seconds": fp16_seconds, "end_to_end_seconds": elapsed}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
            write_summary(args.output / "summary.json", rows, len(examples))
            print("RESCUE_SAVED", example.example_id, branch, elapsed, len(rows), flush=True)
    print("RESCUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
