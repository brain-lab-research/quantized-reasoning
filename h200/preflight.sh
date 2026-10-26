#!/usr/bin/env bash
# Read-only login-node checks; no GPU allocation and no downloads.
set -uo pipefail
for env in /home/data/"$USER"/envs/*; do
    [[ -x "$env/bin/python" ]] || continue
    printf 'ENV %s\n' "$env"
    "$env/bin/python" -c 'from importlib.metadata import version,PackageNotFoundError
for name in ("torch","vllm","transformers","datasets","boto3","pydantic","PyYAML","click"):
 try: print(name,version(name))
 except PackageNotFoundError: print(name,"MISSING")'
done
du -h --max-depth=4 "/home/data/$USER/projects/latent-sft" "/home/data/$USER/projects/latent-grpo" 2>/dev/null | sort -hr | head -35
squeue -u "$USER" -o '%.8i %.9T %.20j %.20b %.30R'
