# FPMI H200 trace collection

This launcher collects uninterrupted Qwen3-8B FP16 / GPTQ-W2A16 traces using
the existing QuantLab runner. It does not enable planning or loop rescue.

Use an existing environment with PyTorch, vLLM, transformers, datasets,
pydantic, PyYAML and click. `boto3` must be importable by `QL_SYNC_PYTHON`
(defaults to the inference Python). Credentials remain in `~/.s3-cod.env`;
they are sourced inside the job and never copied into the workspace.

```bash
bash h200/submit.sh /path/to/existing/env/bin/python
```

The submitted job requests one GPU, 8 CPUs, 64 GiB host RAM and eight hours.
vLLM is limited to 45% GPU memory (about 63 GiB on H200 NVL), one model at a
time, at most eight sequences. Physical GPU IDs are left to Slurm.
Submission refuses when `/home/data/$USER` exceeds the 200 GiB soft limit.
No files are deleted by these scripts.

Workspace: `/home/data/$USER/projects/low-bit-reasoning/exps/<mmdd>-rebuttal-traces/`.
Override `QL_SERIES` to resume a previously created series, retaining its date.
Configs, commands, logs, traces and smoke outputs stay in that workspace.

Order: paired first 100 MATH-500 examples, then the remaining disjoint 100-example
MATH shards, followed by GSM8K shards. This extends coverage within the requested
overnight compute budget rather than repeating seeds. The 36-task queue is an
ordered work backlog, not a promise that every task fits eight hours. The job
stops starting tasks after 7.5 hours of collection; Slurm enforces the total
eight-hour limit, including model downloads and GPU smokes.

Both models must pass an arithmetic and open-think smoke before collection.
Completed shards must contain every requested judgement and no runner errors;
zero-correct FP16 shards stop the queue. GPTQ accuracy may legitimately be low.
Existing partial shards resume by run ID with a config-equality check. Saved
traces/checkpoints are flushed after each microbatch; at timeout an unfinished
microbatch may need repeating. Batched timings are not batch-1 latency claims.

The job syncs only its experiment series to
`s3://iii-storage2/shveikinda/h200/low-bit/<series>/` every three minutes,
with a 2 GiB series cap. It does not upload models, environments or credentials.

Decoding is explicit: temperature 0.6, top-p 0.95, top-k 20, seed 42, generation
budget 32768 and engine context 36864. This matches the paper's stated sampling
settings, while some published YAMLs omit top-p/top-k. The benchmark seed is
also 42 and shard offsets preserve IDs. Exact run provenance is in `cfg.yaml`,
`recipe.json`, `config.json` and `code-revision.txt`.

The GSM8K prompt is the repository adapter's existing prompt; MATH-500 uses its
existing Qwen open-think prompt. Comparing prompt recipes is a separate study.
