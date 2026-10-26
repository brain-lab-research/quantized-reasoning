#!/usr/bin/env python3
"""GPU smoke: both precisions, actual arithmetic, and the benchmark prompt."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    from quantlab.actors.backends.vllm_backend import VLLMBackend
    from quantlab.benchmarks.math500 import MATH500Adapter
    from quantlab.core.types import GenerationParams, PrecisionMode, QuantizationMethod
    from h200.prepare import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache-gib", type=float, default=None)
    args = parser.parse_args()
    backend = VLLMBackend(
        model_id=MODELS[args.precision], precision=PrecisionMode.FP16,
        quantization=QuantizationMethod.GPTQ if args.precision == "w2" else QuantizationMethod.NONE,
        tensor_parallel_size=1, gpu_memory_utilization=args.memory_utilization,
        max_model_len=36864, max_num_seqs=args.batch_size, enforce_eager=True,
        **({"kv_cache_memory_bytes": int(args.kv_cache_gib * 1024**3)}
           if args.kv_cache_gib is not None else {}),
    )
    started = time.monotonic()
    backend.load()
    adapter = MATH500Adapter()
    rows = []
    for problem, answer in (("Compute 2+3.", "5"), ("Compute 8/2.", "4")):
        # Explicitly end thinking only in this arithmetic health check.
        prompt = adapter.build_prompt({"problem": problem}) + "\n</think>\n\n"
        text, n, timing = backend.generate(prompt, GenerationParams(
            max_new_tokens=128, temperature=0.0,
            stop_sequences=["<|im_end|>", "<|endoftext|>"],
        ))
        correct = adapter.is_correct(adapter.extract_answer(text), answer)
        rows.append({"prompt": prompt, "text": text, "tokens": n, "correct": correct})
    # Exercise the real open-think prompt and long-context engine configuration.
    text, n, timing = backend.generate(
        adapter.build_prompt({"problem": "Compute 2+3. Put the answer in \\boxed{}."}),
        GenerationParams(max_new_tokens=256, temperature=0.6, top_p=0.95, top_k=20, seed=42),
    )
    rows.append({"kind": "open_think", "text": text, "tokens": n})
    result = {"precision": args.precision, "model": MODELS[args.precision],
              "elapsed": time.monotonic() - started, "rows": rows,
              "passed": all(r["correct"] for r in rows[:2]) and n > 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    backend.unload()
    if not result["passed"]:
        raise RuntimeError("Arithmetic/trace smoke failed; full collection blocked")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
