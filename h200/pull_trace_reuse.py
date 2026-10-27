#!/usr/bin/env python3
"""Download only the matching saved W2 baseline records."""
import argparse
from pathlib import Path
import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    relative = "1026-math500-w2-0000/artifacts/20261026_215321_e9b9de"
    prefix = "shveikinda/vv/low-bit/1026-rebuttal-traces/" + relative
    client = boto3.client("s3", endpoint_url="https://s3.cod.phystech.edu")
    dest = args.root / relative
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "traces.jsonl", "judgements.jsonl"):
        client.download_file("iii-storage2", prefix + "/" + name, str(dest / name))
    print("REUSE_READY", dest, flush=True)


if __name__ == "__main__":
    main()
