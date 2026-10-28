#!/usr/bin/env python3
"""Import complete VV results into the existing FPMI reuse cache, never a live run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def validate_rows(traces, judgements, expected):
    trace_map = {r["example_id"]: r for r in traces}
    judge_map = {r["example_id"]: r for r in judgements}
    if len(trace_map) != len(traces) or len(judge_map) != len(judgements):
        raise ValueError("Duplicate IDs in imported artifacts")
    if set(trace_map) != set(expected) or set(judge_map) != set(expected):
        raise ValueError("Incomplete or different benchmark IDs")
    for eid, (prompt, ground_truth) in expected.items():
        if trace_map[eid]["prompt"] != prompt or judge_map[eid]["ground_truth"] != ground_truth:
            raise ValueError("Prompt or ground truth differs: " + eid)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def benchmark_matches(source, target, tail):
    source, target = dict(source), dict(target)
    if tail:
        if source.get("start_index") != 800 or source.get("max_examples") != 467:
            return False
        for key in ("start_index", "max_examples"):
            source.pop(key, None)
            target.pop(key, None)
    return source == target


def main():
    import boto3
    import yaml
    from botocore.exceptions import ClientError
    from quantlab.benchmarks.registry import BenchmarkRegistry
    from quantlab.config.schema import ExperimentConfig
    from quantlab.runner import _load_benchmark_examples
    from h200.model_fingerprint import fingerprint

    p = argparse.ArgumentParser()
    p.add_argument("--target-root", type=Path, required=True)
    p.add_argument("--reuse-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--prefix", default="shveikinda/vv/low-bit/1028-fpmi-offload/")
    p.add_argument("--tail", action="store_true", help="Import only the fixed WinoGrande tail shard")
    p.add_argument("--fingerprint-prefix", default=None)
    p.add_argument("--poll", type=int, default=90)
    args = p.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", endpoint_url="https://s3.cod.phystech.edu")
    bucket = "iii-storage2"
    local_fingerprint = fingerprint(os.environ["HF_HOME"])
    print("LOCAL_MODEL_FINGERPRINT_READY", local_fingerprint["revision"], flush=True)
    expected = {}
    configs = {}
    for bench in (("winogrande",) if args.tail else ("strategyqa", "winogrande")):
        folder = args.target_root/f"1027-paper-qwen8b-{bench}-w2-all"
        config = ExperimentConfig.model_validate(yaml.safe_load((folder/"cfg.yaml").read_text()))
        configs[bench] = config
        adapter = BenchmarkRegistry.build(config.benchmark.name)
        examples = _load_benchmark_examples(adapter, config.benchmark)
        if len(examples) != config.benchmark.max_examples:
            raise ValueError("Unexpected local dataset size: " + bench)
        if args.tail:
            examples = examples[800:1267]
        for e in examples:
            if e.prompt.endswith("<|im_start|>assistant\n"):
                e.prompt += "<think>\n"
        expected[bench] = {e.example_id:(e.prompt, e.ground_truth) for e in examples}
        if len(expected[bench]) != (467 if args.tail else config.benchmark.max_examples):
            raise ValueError("Unexpected local dataset size: " + bench)
    done = set()
    while len(done) < len(expected):
        for bench, examples in expected.items():
            if bench in done:
                continue
            if bench == "winogrande":
                local = args.reuse_root/"1027-paper-qwen8b-winogrande-w2-all"/"artifacts"
                for summary_path in local.glob("*/summary.json"):
                    try:
                        local_summary = json.loads(summary_path.read_text())
                    except (OSError, ValueError):
                        continue
                    if local_summary.get("n_judged") == configs[bench].benchmark.max_examples and local_summary.get("n_errors") == 0:
                        print("LOCAL_FPMI_RESULT_READY", bench, flush=True)
                        done.add(bench)
                        break
                if bench in done:
                    continue
            cache = args.reuse_root/f"1028-offload-{bench}{'-tail' if args.tail else ''}"/"artifacts"
            if list(cache.glob("*/config.json")):
                print("ALREADY_IMPORTED", bench, flush=True)
                done.add(bench)
                continue
            task = args.target_root/f"1027-paper-qwen8b-{bench}-w2-all"
            if not args.tail and (task/"cmd.txt").exists():
                print("TOO_LATE_FPMI_ALREADY_STARTED", bench, flush=True)
                done.add(bench)
                continue
            folder_name = "1028-wino-tail" if args.tail else f"1027-paper-qwen8b-{bench}-w2-all"
            prefix = args.prefix.rstrip("/")+f"/{folder_name}/artifacts/"
            objects = []
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                objects.extend(page.get("Contents", []))
            summaries = sorted((o for o in objects if o["Key"].endswith("/summary.json")),
                               key=lambda o:o["LastModified"], reverse=True)
            imported = False
            for item in summaries:
                run_prefix = item["Key"].rsplit("/", 1)[0] + "/"
                summary = json.loads(client.get_object(Bucket=bucket, Key=item["Key"])["Body"].read())
                if summary.get("n_judged") != len(examples) or summary.get("n_new") != len(examples) or summary.get("n_errors") != 0:
                    continue
                names = ["config.json", "summary.json", "traces.jsonl", "judgements.jsonl", "metrics.jsonl", "timing.jsonl"]
                try:
                    fp_prefix = args.fingerprint_prefix or args.prefix
                    remote_fingerprint = json.loads(client.get_object(
                        Bucket=bucket, Key=fp_prefix.rstrip("/")+"/model-fingerprint.json")["Body"].read())
                    if remote_fingerprint != local_fingerprint:
                        raise ValueError("VV and FPMI checkpoint/tokenizer fingerprints differ")
                    heads = {name:client.head_object(Bucket=bucket, Key=run_prefix+name) for name in names}
                    used_kib = int(subprocess.check_output(["du", "-sk", f"/home/data/{os.environ['USER']}"], text=True).split()[0])
                    needed_kib = sum(h["ContentLength"] for h in heads.values())//1024 + 1024*1024
                    if used_kib + needed_kib >= 390*1024*1024:
                        raise RuntimeError("FPMI temporary 390 GiB ceiling would be exceeded")
                    tmp = args.work_root/(bench+"-"+uuid.uuid4().hex)
                    tmp.mkdir()
                    for name, head in heads.items():
                        path = tmp/name
                        client.download_file(bucket, run_prefix+name, str(path))
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        if digest != head.get("Metadata", {}).get("sha256"):
                            raise ValueError("S3 snapshot checksum missing or changed: " + name)
                    source = ExperimentConfig.model_validate(json.loads((tmp/"config.json").read_text()))
                    target = configs[bench]
                    a, b = source.actors[0], target.actors[0]
                    if (not benchmark_matches(source.benchmark.model_dump(), target.benchmark.model_dump(), args.tail)
                        or a.model_id != b.model_id or a.precision != b.precision
                        or a.quantization != b.quantization or a.quantization_config != b.quantization_config
                        or a.generation_params.model_dump() != b.generation_params.model_dump()):
                        raise ValueError("Offloaded model, benchmark or sampling mismatch")
                    traces, judgements = read_rows(tmp/"traces.jsonl"), read_rows(tmp/"judgements.jsonl")
                    validate_rows(traces, judgements, examples)
                    for filename in ("metrics.jsonl", "timing.jsonl"):
                        entries = read_rows(tmp/filename)
                        ids = [r["example_id"] for r in entries]
                        if len(ids) != len(examples) or set(ids) != set(examples):
                            raise ValueError("Incomplete " + filename)
                    manifest = yaml.safe_load(client.get_object(
                        Bucket=bucket, Key=fp_prefix.rstrip("/")+"/cfg.yaml")["Body"].read())
                    (tmp/"offload-provenance.json").write_text(json.dumps({"source": "s3://"+bucket+"/"+run_prefix,
                        "host":"vv", "gpu_uuid":manifest["gpu_uuid"],
                        "per_request_timing":"Not measured by this continuous-batch collector."}, indent=2))
                    cache.mkdir(parents=True, exist_ok=True)
                    destination = cache/run_prefix.rstrip("/").rsplit("/", 1)[-1]
                    if not args.tail and (task/"cmd.txt").exists():
                        print("TOO_LATE_FPMI_ALREADY_STARTED", bench, flush=True)
                        done.add(bench)
                        break
                    os.rename(tmp, destination)
                    print("IMPORTED", bench, len(examples), str(destination), flush=True)
                    done.add(bench)
                    imported = True
                    break
                except ClientError as exc:
                    if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                        raise
                    print("SNAPSHOT_NOT_READY", bench, "artifact not uploaded yet", flush=True)
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    print("SNAPSHOT_NOT_READY", bench, str(exc), flush=True)
            if not imported and bench not in done:
                print("WAIT_COMPLETE", bench, flush=True)
        if len(done) < len(expected):
            time.sleep(args.poll)
    print("IMPORT_MONITOR_FINISHED", flush=True)


if __name__ == "__main__":
    main()
