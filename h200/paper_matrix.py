#!/usr/bin/env python3
"""Reproduce the paper's 17 Qwen model/benchmark pairs at FP16 and W2."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BENCHMARKS = ("math500", "gpqa_diamond", "gsm8k", "arc_easy", "arc_challenge",
              "piqa", "strategyqa", "winogrande", "aime2026")
ADAPTERS = {"aime2026": "aime26", "gsm8k": "gsm8k_paper", "winogrande": "winogrande_paper"}
FULL_COUNTS = dict(zip(BENCHMARKS, (500, 198, 1319, 2376, 1172, 1838, 687, 1267, 30)))


def task_complete(task):
    summaries = list((Path(task["config"]).parent / "artifacts").glob("*/summary.json"))
    if len(summaries) != 1:
        return False
    try:
        saved = json.loads(summaries[0].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return saved.get("n_judged") == task["count"] and saved.get("n_errors") == 0


def prepare(root, full_benchmarks=False):
    from quantlab.config.schema import ExperimentConfig
    tasks = []
    for benchmark in BENCHMARKS:
        for size in (8, 32):
            if size == 8 and benchmark == "aime2026" and not full_benchmarks:
                continue  # Not a model/benchmark pair in Table 2.
            count = FULL_COUNTS[benchmark] if full_benchmarks else (30 if benchmark == "aime2026" else 100)
            for precision in ("fp16", "w2"):
                group = "qwen32b_fp16" if precision == "fp16" else "qwen32b_gptq2bit"
                filename = "single_fp16.yaml" if precision == "fp16" else "single_gptq2bit.yaml"
                cfg = yaml.safe_load((REPO / "configs" / benchmark / group / filename).read_text())
                name = f"1027-paper-qwen{size}b-{benchmark}-{precision}" + ("-all" if full_benchmarks else "")
                cfg["experiment_name"] = name
                cfg["description"] = f"Paper baseline: Qwen3-{size}B {precision}, {benchmark}, seed 42, full 32768-token budget."
                cfg["benchmark"].update(name=ADAPTERS.get(benchmark, benchmark), max_examples=count,
                                         start_index=0, seed=42)
                if benchmark != "winogrande":
                    cfg["benchmark"].pop("subset", None)
                cfg["benchmark"]["split"] = ("validation" if benchmark in ("piqa", "winogrande")
                                              else "train" if benchmark == "aime2026" else "test")
                actor = cfg["actors"][0]
                actor["actor_id"] = f"qwen3_{size}b_{precision}"
                cfg["pipeline"][0]["actor_id"] = actor["actor_id"]
                actor["model_id"] = (f"Qwen/Qwen3-{size}B" if precision == "fp16" else
                                     f"kaitchup/Qwen3-{size}B-autoround-2bit-gptq")
                actor["generation_params"].update(max_new_tokens=32768, temperature=0.6,
                                                 top_p=0.95, top_k=20, seed=42)
                actor["backend_kwargs"].pop("cuda_visible_devices", None)
                if full_benchmarks:
                    actor["backend_kwargs"].update(enforce_eager=False, enable_chunked_prefill=True,
                                                    enable_prefix_caching=True)
                cfg["pipeline"][0]["max_new_tokens"] = 32768
                cfg["pipeline_max_total_tokens"] = 32768
                cfg["output"].update(base_dir=str(root / name / "artifacts"),
                                     trace_include_llm_prompt=True)
                cfg["wandb"] = {"enabled": False}
                ExperimentConfig.model_validate(cfg)
                path = root / name / "cfg.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and yaml.safe_load(path.read_text()) != cfg:
                    if list((path.parent / "artifacts").glob("*/config.json")):
                        raise ValueError(f"Config changed for an existing run: {path}")
                    path.with_suffix(".previous.yaml").write_text(path.read_text())
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                tasks.append({"config": str(path), "count": count,
                              "size": size, "benchmark": benchmark, "precision": precision})
    # Cover every paper setting first. Only then extend the two compact core
    # benchmarks to their full sizes, reusing the first-pass traces by ID/prompt.
    for task in ([] if full_benchmarks else list(tasks)):
        if task["benchmark"] not in ("math500", "gpqa_diamond"):
            continue
        cfg = yaml.safe_load(Path(task["config"]).read_text())
        count = 500 if task["benchmark"] == "math500" else 198
        name = cfg["experiment_name"] + "-full"
        cfg["experiment_name"] = name
        cfg["benchmark"]["max_examples"] = count
        cfg["output"]["base_dir"] = str(root / name / "artifacts")
        path = root / name / "cfg.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            if list((path.parent / "artifacts").glob("*/config.json")):
                raise ValueError(f"Config changed for an existing run: {path}")
            path.with_suffix(".previous.yaml").write_text(path.read_text())
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tasks.append({**task, "config": str(path), "count": count})
    (root / "paper-matrix.json").write_text(json.dumps(tasks, indent=2))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--full-benchmarks", action="store_true", help="All tasks in all nine benchmark splits, including 8B AIME")
    p.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    p.add_argument("--reuse-root", type=Path, action="append", default=[])
    p.add_argument("--data-smoke", action="store_true")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 32])
    p.add_argument("--precisions", nargs="+", choices=["fp16", "w2"], default=["fp16", "w2"])
    p.add_argument("--full-first", action="store_true")
    p.add_argument("--kv-cache-gib", type=float, default=None)
    p.add_argument("--kv-cache-gib-quant", type=float, default=None)
    p.add_argument("--seqs-8", type=int, default=16)
    p.add_argument("--memory-utilization", type=float, default=0.8)
    p.add_argument("--exclude", action="append", default=[], help="Exclude worker pair, e.g. 8:w2")
    p.add_argument("--seqs-32", type=int, default=8, help="Active request limit for 32B workers")
    args = p.parse_args()
    tasks = prepare(args.root, full_benchmarks=args.full_benchmarks)
    tasks = [t for t in tasks if t["size"] in args.sizes and t["precision"] in args.precisions]
    tasks = [t for t in tasks if t["benchmark"] in args.benchmarks]
    tasks = [t for t in tasks if f"{t['size']}:{t['precision']}" not in args.exclude]
    if args.full_first:
        tasks.sort(key=lambda t: (0 if t["benchmark"] == "math500" and t["count"] == 500 else
                                 1 if t["benchmark"] == "gpqa_diamond" and t["count"] == 198 else 2))
    if args.data_smoke:
        from quantlab.benchmarks.registry import BenchmarkRegistry
        checked = set()
        for task in tasks:
            cfg = yaml.safe_load(Path(task["config"]).read_text())
            bm = cfg["benchmark"]
            if bm["name"] in checked:
                continue
            rows = BenchmarkRegistry.build(bm["name"]).load(split=bm["split"],
                subset=bm.get("subset"), max_examples=2, seed=42)
            assert len(rows) == 2
            print("DATA_SMOKE_OK", bm["name"], rows[0].example_id, flush=True)
            checked.add(bm["name"])
    print(f"PAPER_MATRIX {len(tasks)} tasks", flush=True)
    if args.prepare_only:
        return
    failed = []
    for task in tasks:
        cfg = Path(task["config"])
        if task_complete(task):
            print("MATRIX_SKIP_COMPLETE", cfg.parent.name, flush=True)
            continue
        cmd = [sys.executable, str(REPO / "scripts/collect_baseline_traces.py"), str(cfg),
               "--output", str(cfg.parent / "artifacts"),
               "--max-examples", str(task["count"]), "--max-num-seqs", str(args.seqs_8 if task["size"] == 8 else args.seqs_32),
               "--kv-cache-gib", "80" if task["size"] == 8 else "48", "--open-thinking"]
        if args.kv_cache_gib is not None:
            cmd[cmd.index("--kv-cache-gib") + 1] = str(args.kv_cache_gib)
        if task["precision"] == "w2" and args.kv_cache_gib_quant is not None:
            cmd[cmd.index("--kv-cache-gib") + 1] = str(args.kv_cache_gib_quant)
        cmd += ["--memory-utilization", str(args.memory_utilization)]
        for reuse_root in [args.root, *args.reuse_root]:
            cmd += ["--reuse-root", str(reuse_root)]
        (cfg.parent / "cmd.txt").write_text(subprocess.list2cmdline(cmd) + "\n")
        for attempt in range(3):
            print("MATRIX_START", cfg.parent.name, "attempt", attempt + 1, flush=True)
            with (cfg.parent / "run.log").open("a") as log:
                result = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if attempt < 2:
                time.sleep(30)
        if result.returncode:
            failed.append(task)
            (args.root / "failed-paper-tasks.json").write_text(json.dumps(failed, indent=2))
            print("MATRIX_FAILED_CONTINUING", cfg.parent.name, flush=True)
        else:
            print("MATRIX_DONE", cfg.parent.name, flush=True)
    (args.root / "failed-paper-tasks.json").write_text(json.dumps(failed, indent=2))
    if failed:
        raise RuntimeError(f"Matrix finished with {len(failed)} failed tasks; see failed-paper-tasks.json")


if __name__ == "__main__":
    main()
