#!/usr/bin/env python3
"""Check whether the FPMI WinoGrande source has already finished."""
import json
import sys


def main():
    import boto3
    s = boto3.client("s3", endpoint_url="https://s3.cod.phystech.edu")
    prefix = ("shveikinda/h200/low-bit/1027-fpmi-w2-traces/"
              "1027-paper-qwen8b-winogrande-w2-all/artifacts/")
    for page in s.get_paginator("list_objects_v2").paginate(Bucket="iii-storage2", Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/summary.json"):
                continue
            data = json.loads(s.get_object(Bucket="iii-storage2", Key=obj["Key"])["Body"].read())
            if data.get("n_judged") == 1267 and data.get("n_errors") == 0:
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
