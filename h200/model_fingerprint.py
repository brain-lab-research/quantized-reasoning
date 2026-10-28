#!/usr/bin/env python3
"""Content fingerprints of the cached quantized checkpoint and tokenizer files."""
import hashlib
import json
import os
from pathlib import Path
import re


def fingerprint(hf_home):
    cache = Path(hf_home)/"hub"/"models--kaitchup--Qwen3-8B-autoround-2bit-gptq"
    revision = (cache/"refs"/"main").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid Hugging Face snapshot revision")
    snapshot = cache/"snapshots"/revision
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("No cached model weights")
    names = {p.name for p in weights}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "config.json",
                 "generation_config.json", "special_tokens_map.json", "added_tokens.json", "quantize_config.json"):
        if (snapshot/name).is_file():
            names.add(name)
    if "tokenizer.json" not in names or "config.json" not in names:
        raise ValueError("Missing tokenizer or model configuration")
    files = {}
    for name in sorted(names):
        digest = hashlib.sha256()
        with (snapshot/name).open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024), b""):
                digest.update(chunk)
        files[name] = digest.hexdigest()
    return {"model_id":"kaitchup/Qwen3-8B-autoround-2bit-gptq", "revision":revision, "sha256":files}


if __name__ == "__main__":
    print(json.dumps(fingerprint(os.environ["HF_HOME"]), sort_keys=True))
