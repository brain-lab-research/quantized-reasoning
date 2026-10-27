#!/usr/bin/env python3
"""Cache public model weights/configs without executing repository code."""
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    for model in args.models:
        path = snapshot_download(model, allow_patterns=[
            "*.safetensors", "*.json", "*.model", "tokenizer*", "*.tiktoken",
        ], max_workers=4)
        print(f"MODEL_READY {model} {path}", flush=True)


if __name__ == "__main__":
    main()
