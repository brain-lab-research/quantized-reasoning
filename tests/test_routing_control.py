from scripts.prepare_routing_control import prepare


def test_routing_is_count_matched_gold_independent_and_deduplicates_overlap():
    traces = [dict(example_id=str(i), prompt="question "+str(i)) for i in range(30)]
    gold = [dict(example_id=str(i), ground_truth=str(i), is_correct=True) for i in range(30)]
    targets = [dict(example_id=str(i), prompt="question "+str(i), arm="restart") for i in range(10)]
    requests, policies = prepare(traces, gold, targets, 42)
    changed = [{**r, "ground_truth": "different", "is_correct": False} for r in gold]
    _, second = prepare(list(reversed(traces)), changed, targets, 42)
    assert policies == second
    assert len(policies["targeted_ids"]) == len(policies["random_ids"]) == 10
    assert len(requests) == len(set(policies["targeted_ids"]) | set(policies["random_ids"]))
    assert len({r["request_id"] for r in requests}) == len(requests)
