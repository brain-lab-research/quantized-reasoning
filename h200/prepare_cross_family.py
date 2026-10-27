#!/usr/bin/env python3
"""Prepare model-native Llama-family baseline configs in the experiment workspace."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    models = {
        "fp16": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "w2": "irish-quant/deepseek-ai-DeepSeek-R1-Distill-Llama-8B-2bit",
    }
    for precision, model in models.items():
        cfg = yaml.safe_load((args.root / f"1026-math500-{precision}-0000/cfg.yaml").read_text())
        name = f"1027-llama8b-{precision}"
        cfg["experiment_name"] = name
        cfg["description"] = "Llama-family full MATH-500 baseline; model-native chat template."
        cfg["actors"][0]["model_id"] = model
        cfg["actors"][0]["generation_params"]["stop_sequences"] = []
        cfg["pipeline"][0]["stop_sequences"] = []
        cfg["output"]["base_dir"] = str(args.root / name / "artifacts")
        out = args.root / name / "cfg.yaml"
        out.parent.mkdir(exist_ok=True)
        if out.exists() and yaml.safe_load(out.read_text()) != cfg:
            raise ValueError(f"Refusing to change existing config: {out}")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"PREPARED {out}", flush=True)


if __name__ == "__main__":
    main()
