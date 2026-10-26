#!/usr/bin/env python3
"""Run the prepared queue, failing closed on errors and resuming saved shards."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.6)
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = time.monotonic() + args.hours * 3600
    repo = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "tasks.json").read_text())
    for cfg_path in map(Path, tasks):
        if time.monotonic() >= deadline:
            print("TIME_BUDGET_REACHED; remaining tasks preserved", flush=True)
            break
        cfg = yaml.safe_load(cfg_path.read_text())
        exp = cfg_path.parent
        runs = sorted((exp / "artifacts").glob("*/config.json"))
        if len(runs) > 1:
            raise RuntimeError(f"Ambiguous resume directory: {exp}")
        cmd = [sys.executable, str(repo / "scripts/run_experiment.py"), str(cfg_path), "-v"]
        if runs:
            saved = json.loads(runs[0].read_text())
            from quantlab.config.schema import ExperimentConfig
            if saved != ExperimentConfig.model_validate(cfg).model_dump():
                raise RuntimeError(f"Resume config mismatch: {exp}")
            summary = runs[0].parent / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                if s.get("n_judged") == cfg["benchmark"]["max_examples"] and not s.get("n_errors"):
                    print(f"SKIP_COMPLETE {exp.name}", flush=True)
                    continue
            cmd += ["--resume-run-id", runs[0].parent.name]
        (exp / "cmd.txt").write_text(subprocess.list2cmdline(cmd) + "\n")
        print(f"START {exp.name}", flush=True)
        with (exp / "run.log").open("a") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=repo)
        if completed.returncode:
            raise RuntimeError(f"Task failed: {exp}; inspect run.log")
        runs = list((exp / "artifacts").glob("*/summary.json"))
        if len(runs) != 1:
            raise RuntimeError(f"Missing/ambiguous summary: {exp}")
        s = json.loads(runs[0].read_text())
        if s.get("n_errors") or s.get("n_judged") != cfg["benchmark"]["max_examples"]:
            raise RuntimeError(f"Incomplete shard: {s}")
        if cfg["actors"][0]["quantization"] == "none" and s.get("accuracy", 0) == 0:
            raise RuntimeError("Zero-correct FP16 gate failed; stop queue")
        print(f"DONE {exp.name}: {json.dumps(s)}", flush=True)
    print("COLLECTION_STOPPED", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
