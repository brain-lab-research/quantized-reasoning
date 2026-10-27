#!/usr/bin/env python3
"""Upload only this experiment series; credentials come from the job environment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    import boto3
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    client = boto3.client("s3", endpoint_url="https://s3.cod.phystech.edu")
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        size = path.stat().st_size
        total += size
        # Trace output is small compared with checkpoints; enforce a strict cap.
        if total > 2 * 1024**3:
            raise RuntimeError("Series exceeds 2 GiB backup cap")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        key = args.prefix.rstrip("/") + "/" + relative
        try:
            previous = client.head_object(Bucket="iii-storage2", Key=key)
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            if (previous["ContentLength"] == size
                    and previous.get("Metadata", {}).get("sha256") == checksum):
                continue
        client.upload_file(str(path), "iii-storage2", key,
                           ExtraArgs={"Metadata": {"sha256": checksum}})
    print(f"BACKUP_OK bytes={total}", flush=True)


if __name__ == "__main__":
    main()
