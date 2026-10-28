from scripts.wino_source_complete import matches


def test_only_complete_compatible_full_source_can_skip_tail():
    gen = {"seed": 42, "max_new_tokens": 32768, "temperature": 0.6}
    manifest = {"expected_examples":1267, "model_id":"model", "generation_parameters":gen}
    cfg = {"actors":[{"model_id":"model", "precision":"fp16", "quantization":"gptq", "generation_params":gen}],
           "benchmark":{"name":"winogrande_paper", "split":"validation", "seed":42, "start_index":0, "max_examples":1267}}
    assert matches({"n_judged":1267,"n_errors":0}, cfg, manifest)
    assert not matches({"n_judged":1266,"n_errors":0}, cfg, manifest)
    assert not matches({"n_judged":1267,"n_errors":1}, cfg, manifest)
    changed = {**cfg, "benchmark":{**cfg["benchmark"], "seed":43}}
    assert not matches({"n_judged":1267,"n_errors":0}, changed, manifest)
