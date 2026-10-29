#!/usr/bin/env python3
"""Download the explicitly versioned FP16 Distill-Llama checkpoint for a run."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    record = args.root / 'model-revision.json'
    revision = json.loads(record.read_text())['revision'] if record.exists() else HfApi().model_info(MODEL).sha
    record.write_text(json.dumps({'model_id': MODEL, 'revision': revision}, indent=2))
    print('DOWNLOAD_START', MODEL, revision, flush=True)
    snapshot_download(MODEL, revision=revision, local_dir=str(args.root / 'model'), max_workers=4,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    (args.root / 'download.done').write_text(revision + '\n')
    print('DOWNLOAD_DONE', revision, flush=True)


if __name__ == '__main__':
    main()
