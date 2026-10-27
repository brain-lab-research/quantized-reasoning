#!/usr/bin/env python3
"""Single-stage baseline collection with continuous admission and per-result saves.

All pending prompts enter vLLM's scheduler together. Finished requests are saved
immediately; a slow request never holds up admission of the next prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def completed_outputs(engine):
    """Yield each finished request once, regardless of completion order."""
    seen = set()
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished and output.request_id not in seen:
                seen.add(output.request_id)
                yield output


def format_chat_prompts(examples, tokenizer, benchmark):
    """Use the evaluated model's own special tokens for cross-family baselines."""
    instruction = ("Solve step by step. End with the numeric answer after '#### '."
                   if benchmark == "gsm8k" else
                   "Solve step by step. Put your final answer inside \\boxed{}.")
    for example in examples:
        problem = example.raw.get("problem", example.raw.get("question"))
        if problem is None:
            raise ValueError("Chat prompt recipe requires a raw problem or question")
        example.prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem + "\n\n" + instruction}],
            tokenize=False, add_generation_prompt=True,
        )


def reuse_judgements(root, config, examples):
    from quantlab.artifacts.store import ArtifactStore
    expected = {e.example_id: e.prompt for e in examples}
    reused = {}
    actor = config.actors[0]
    for cfg_path in sorted(root.glob("*/artifacts/*/config.json")):
        raw = json.loads(cfg_path.read_text())
        old_actors = raw.get("actors", [])
        if len(old_actors) != 1 or len(raw.get("pipeline", [])) != 1:
            continue
        old = old_actors[0]
        if (old.get("model_id") != actor.model_id
                or raw.get("benchmark", {}).get("name") != config.benchmark.name
                or old.get("generation_params") != actor.generation_params.model_dump()):
            continue
        store = ArtifactStore(str(cfg_path.parent.parent))
        run_id = cfg_path.parent.name
        prompts = {t.example_id: t.prompt for t in store.load_traces(run_id)}
        for row in store.load_judgements(run_id):
            eid = row["example_id"]
            if eid in expected and prompts.get(eid) == expected[eid]:
                reused[eid] = {**row, "source": str(cfg_path.parent)}
    return reused


def main():
    import yaml
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from quantlab.actors.backends.vllm_backend import VLLMBackend
    from quantlab.artifacts.store import ArtifactStore
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.core.trace import Trace, TraceSegment
    from quantlab.core.types import PrecisionMode, QuantizationMethod, SegmentRole
    from quantlab.metrics.registry import MetricRegistry
    from quantlab.runner import _load_benchmark_examples, _record_example

    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-root", type=Path)
    p.add_argument("--max-examples", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--kv-cache-gib", type=float, default=80)
    p.add_argument("--chat-template", action="store_true")
    args = p.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    if len(raw["actors"]) != 1 or len(raw["pipeline"]) != 1:
        raise ValueError("Only single-stage baselines are supported")
    if raw["pipeline"][0].get("exit_conditions"):
        raise ValueError("Baseline collection must not contain early-exit conditions")
    raw["benchmark"].update(max_examples=args.max_examples, start_index=0)
    raw["actors"][0]["generation_params"]["max_new_tokens"] = args.max_tokens
    raw["pipeline"][0]["max_new_tokens"] = args.max_tokens
    kwargs = raw["actors"][0]["backend_kwargs"]
    kwargs.pop("cuda_visible_devices", None)
    kwargs.update(max_num_seqs=args.max_num_seqs, gpu_memory_utilization=0.8,
                  kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
                  max_num_batched_tokens=8192)
    raw["output"].update(base_dir=str(args.output), staged_wave_checkpoints=False)
    raw["staged_execution"] = False
    raw["wandb"] = {"enabled": False}
    if args.chat_template:
        raw["description"] = raw.get("description", "") + " Prompt recipe: tokenizer_chat."
    config = ExperimentConfig.model_validate(raw)
    adapter = BenchmarkRegistry.build(config.benchmark.name)
    examples = _load_benchmark_examples(adapter, config.benchmark)
    if args.chat_template:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.actors[0].model_id)
        format_chat_prompts(examples, tokenizer, config.benchmark.name)
    reused = reuse_judgements(args.reuse_root, config, examples) if args.reuse_root else {}
    store = ArtifactStore(str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    pointer = args.output / "run-id.txt"
    if pointer.exists():
        run_id = pointer.read_text().strip()
        saved = json.loads((args.output / run_id / "config.json").read_text())
        if saved != config.model_dump():
            raise ValueError("Resume config mismatch")
    else:
        run_id = store.new_run(config.experiment_name, config.model_dump())
        pointer.write_text(run_id)
    for row in store.load_judgements(run_id):
        reused[row["example_id"]] = row
    pending = {e.example_id: e for e in examples if e.example_id not in reused}
    (args.output / "reused.json").write_text(json.dumps(reused, indent=2))
    print(f"CONTINUOUS total={len(examples)} saved={len(reused)} pending={len(pending)} "
          f"active_limit={args.max_num_seqs} kv_gib={args.kv_cache_gib}", flush=True)
    if not pending:
        print("ALREADY_COMPLETE", flush=True)
        return
    actor = config.actors[0]
    backend = VLLMBackend(actor.model_id, precision=PrecisionMode(actor.precision),
                          quantization=QuantizationMethod(actor.quantization),
                          **actor.backend_kwargs)
    backend.load()
    engine = backend._llm.llm_engine
    gen = actor.generation_params
    params = SamplingParams(max_tokens=gen.max_new_tokens, temperature=gen.temperature,
                            top_p=gen.top_p, top_k=gen.top_k, seed=gen.seed,
                            stop=gen.stop_sequences or None,
                            output_kind=RequestOutputKind.FINAL_ONLY)
    for eid, example in pending.items():
        engine.add_request(eid, example.prompt, params)
    metrics = [MetricRegistry.build(m.name, **m.kwargs) for m in config.metrics]
    started = time.monotonic()
    tokens = 0
    for output in completed_outputs(engine):
        example = pending[output.request_id]
        generated = output.outputs[0]
        trace = Trace(example.example_id, example.prompt)
        trace.append_segment(TraceSegment(
            actor.actor_id, generated.text, len(generated.token_ids), 0,
            role=SegmentRole.ANSWER, llm_prompt_full=example.prompt,
            metadata={"token_ids": list(generated.token_ids),
                      "finish_reason": generated.finish_reason,
                      "serving_mode": "continuous"},
        ))
        trace.finished_at = time.time()
        _record_example(store=store, run_id=run_id, config=config, metrics=metrics,
                        adapter=adapter, example=example, trace=trace, verbose=True)
        tokens += len(generated.token_ids)
        own_rows = store.load_judgements(run_id)
        combined = dict(reused)
        combined.update({r["example_id"]: r for r in own_rows})
        summary = {"n_examples": len(examples), "n_judged": len(combined),
                   "n_new": len(own_rows), "n_errors": 0,
                   "accuracy": sum(r["is_correct"] for r in combined.values()) / len(combined),
                   "generated_tokens": tokens, "wall_seconds": time.monotonic() - started}
        store.save_summary(run_id, summary)
        print(f"SAVED {example.example_id} total={len(combined)}/{len(examples)} "
              f"generated_tokens={tokens} wall={summary['wall_seconds']:.1f}", flush=True)
    backend.unload()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
