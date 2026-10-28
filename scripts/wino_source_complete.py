#!/usr/bin/env python3
"""Check whether the FPMI WinoGrande source has already finished."""
import argparse
import json
from pathlib import Path
import sys


def matches(summary, config, manifest):
    actor = config["actors"][0]
    benchmark = config["benchmark"]
    return (summary.get("n_judged") == manifest["expected_examples"]
        and summary.get("n_errors") == 0 and actor["model_id"] == manifest["model_id"]
        and actor["precision"] == "fp16" and actor["quantization"] == "gptq"
        and actor["generation_params"] == manifest["generation_parameters"]
        and benchmark["name"] == "winogrande_paper" and benchmark["split"] == "validation"
        and benchmark["seed"] == 42 and benchmark["start_index"] == 0
        and benchmark["max_examples"] == manifest["expected_examples"])


def main():
    import boto3
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for local in Path(manifest["local_artifacts"]).glob("*/summary.json"):
        try:
            if matches(json.loads(local.read_text()), json.loads((local.parent/"config.json").read_text()), manifest):
                print("LOCAL_WINOGRANDE_COMPLETE", flush=True)
                return 0
        except (OSError, ValueError, KeyError):
            continue
    s = boto3.client("s3", endpoint_url="https://s3.cod.phystech.edu")
    bucket, prefix = manifest["bucket"], manifest["s3_prefix"]
    for page in s.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/summary.json"):
                continue
            data = json.loads(s.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            cfg_key = obj["Key"].rsplit("/", 1)[0]+"/config.json"
            config = json.loads(s.get_object(Bucket=bucket, Key=cfg_key)["Body"].read())
            if matches(data, config, manifest):
                print("FPMI_WINOGRANDE_COMPLETE", flush=True)
                return 0
    print("FPMI_WINOGRANDE_INCOMPLETE", flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FPMI_WINOGRANDE_STATUS_UNAVAILABLE", type(exc).__name__, flush=True)
        sys.exit(1)
