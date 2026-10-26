from pathlib import Path

from h200.prepare import make_config
from quantlab.config.schema import ExperimentConfig


def test_paired_configs_keep_the_same_sampling_and_full_budget():
    root = Path("/workspace/exps/1026-rebuttal-traces")
    configs = [make_config(root, "math500", p, 100, 100) for p in ("fp16", "w2")]
    for raw in configs:
        cfg = ExperimentConfig.model_validate(raw)
        assert cfg.pipeline_max_total_tokens == 32768
        assert cfg.pipeline[0].max_new_tokens == 32768
        assert cfg.pipeline[0].exit_conditions == []
        assert cfg.benchmark.start_index == 100
        assert cfg.output.save_traces and cfg.output.trace_include_llm_prompt
        assert "cuda_visible_devices" not in cfg.actors[0].backend_kwargs
        assert cfg.actors[0].backend_kwargs["gpu_memory_utilization"] == 0.45
        assert not cfg.wandb.enabled
    assert configs[0]["benchmark"] == configs[1]["benchmark"]
    assert configs[0]["actors"][0]["generation_params"] == configs[1]["actors"][0]["generation_params"]
    assert configs[1]["actors"][0]["quantization_config"]["bits"] == 2
