import pytest
from h200.import_vv_offload import benchmark_matches, validate_rows


def test_import_requires_exact_complete_prompts_and_ground_truth():
    expected = {"a": ("prompt a", "yes"), "b": ("prompt b", "no")}
    traces = [{"example_id":"a", "prompt":"prompt a"}, {"example_id":"b", "prompt":"prompt b"}]
    judges = [{"example_id":"a", "ground_truth":"yes"}, {"example_id":"b", "ground_truth":"no"}]
    validate_rows(traces, judges, expected)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_rows(traces[:1], judges, expected)
    with pytest.raises(ValueError, match="Prompt or ground truth"):
        validate_rows(traces, [judges[0], {**judges[1], "ground_truth":"yes"}], expected)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(traces+[traces[0]], judges, expected)


def test_tail_import_requires_fixed_bounds_and_same_dataset_protocol():
    full = {"name":"winogrande_paper", "split":"validation", "seed":42,
            "start_index":0, "max_examples":1267}
    tail = {**full, "start_index":800, "max_examples":467}
    assert benchmark_matches(tail, full, True)
    assert not benchmark_matches({**tail, "start_index":799}, full, True)
    assert not benchmark_matches({**tail, "seed":43}, full, True)
    assert not benchmark_matches(tail, full, False)
