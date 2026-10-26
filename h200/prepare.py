#!/usr/bin/env python3
"""Prepare workspace-only configs for paired FP16/W2 trace collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = {
    "fp16": "Qwen/Qwen3-8B",
    "w2": "kaitchup/Qwen3-8B-autoround-2bit-gptq",
}


def make_config(root: Path, benchmark: str, precision: str, start: int, count: int,
                memory_utilization: float = 0.45, batch_size: int = 8,
                kv_cache_gib: float | None = None) -> dict:
    name = f"{root.name[:4]}-{benchmark}-{precision}-{start:04d}"
    return {
        "experiment_name": name,
        "description": "Uninterrupted baseline traces; no planner, rescue, or loop stopping.",
        "benchmark": {"name": benchmark, "split": "test", "seed": 42,
                      "max_examples": count, "start_index": start},
        "staged_execution": True,
        "staged_unload_between_waves": False,
        "staged_batch_size": batch_size,
        "pipeline_max_total_tokens": 32768,
        "actors": [{
            "actor_id": precision,
            "model_id": MODELS[precision], "backend": "vllm", "precision": "fp16",
            "quantization": "gptq" if precision == "w2" else "none",
            "quantization_config": {"bits": 2} if precision == "w2" else {},
            # Respect Slurm's CUDA_VISIBLE_DEVICES; never overwrite physical GPU IDs.
            "backend_kwargs": {
                "tensor_parallel_size": 1, "gpu_memory_utilization": memory_utilization,
                "max_model_len": 36864, "max_num_seqs": batch_size,
                "enable_prefix_caching": True, "enforce_eager": True,
                **({"kv_cache_memory_bytes": int(kv_cache_gib * 1024**3)}
                   if kv_cache_gib is not None else {}),
            },
            "generation_params": {
                "max_new_tokens": 32768, "temperature": 0.6,
                "top_p": 0.95, "top_k": 20, "seed": 42,
                "stop_sequences": ["<|im_end|>", "<|endoftext|>"],
            },
        }],
        "pipeline": [{"actor_id": precision, "role": "answer",
                      "handoff_mode": "full_prefill", "max_new_tokens": 32768,
                      "stage_prompt": "", "exit_conditions": []}],
        "metrics": [{"name": n} for n in (
            "accuracy", "parse_rate", "reasoning_length", "think_closed",
            "total_generation_ms", "tokens_per_second",
        )],
        "output": {"base_dir": str(root / name / "artifacts"),
                   "save_traces": True, "save_raw_generations": True,
                   "save_timing": True, "save_errors": True,
                   "parquet_summary": False, "trace_include_llm_prompt": True,
                   "staged_wave_checkpoints": True},
        "wandb": {"enabled": False},
    }


def main() -> None:
    import yaml
    from quantlab.config.schema import ExperimentConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--memory-utilization", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache-gib", type=float, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    # First matched pair is available before extending coverage. Additional tasks
    # fill the requested overnight budget, without duplicating seeds or examples.
    for benchmark, starts in (("math500", range(0, 500, 100)),
                              ("gsm8k", range(0, 1300, 100))):
        for start in starts:
            for precision in ("fp16", "w2"):
                cfg = make_config(root, benchmark, precision, start, 100,
                                  args.memory_utilization, args.batch_size, args.kv_cache_gib)
                validated = ExperimentConfig.model_validate(cfg)
                assert validated.pipeline_max_total_tokens == 32768
                exp = root / cfg["experiment_name"]
                exp.mkdir(exist_ok=True)
                path = exp / "cfg.yaml"
                if path.exists() and yaml.safe_load(path.read_text()) != cfg:
                    raise RuntimeError(f"Refusing to change existing config: {path}")
                path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                tasks.append(str(path))
    (root / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (root / "recipe.json").write_text(json.dumps({
        "models": MODELS, "dataset_seed": 42, "sampling_seed": 42,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "generation_budget": 32768, "max_model_len": 36864,
        "note": "Explicit paper decoding recipe; repository configs omit top_p/top_k.",
    }, indent=2), encoding="utf-8")
    print(f"PREPARED {len(tasks)} tasks in {root}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
