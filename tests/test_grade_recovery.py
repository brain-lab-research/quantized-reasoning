import json
import pytest
from scripts.grade_fp16_recovery import grade, read_rows


class Adapter:
    def extract_answer(self, text):
        return "5" if "<answer>5</answer>" in text else None

    def is_correct(self, predicted, gold):
        return predicted == gold


def fixtures():
    requests, results = [], []
    for arm in ("restart", "continue", "trim"):
        request = dict(request_id="x:"+arm, example_id="x", arm=arm,
                       ground_truth="5", max_new_tokens=10, seed=42,
                       prompt="PROMPT" + ("<answer>" if arm != "restart" else ""))
        requests.append(request)
        results.append({**request, "text": "<answer>5</answer>" if arm == "restart" else "5</answer>",
                        "token_count": 2, "token_ids": [1, 2], "prompt_tokens": 3, "finish_reason": "stop"})
    return requests, results


def test_continuation_grading_preserves_open_answer_in_source_prefix():
    requests, results = fixtures()
    summary, _ = grade(requests, results, {"x": "PROMPT"}, Adapter())
    assert summary["complete"]
    assert all(m["correct"] == 1 for m in summary["metrics"].values())


def test_incomplete_triplet_is_not_counted_as_wrong_or_used_for_paired_comparison():
    requests, results = fixtures()
    summary, _ = grade(requests, results[:2], {"x": "PROMPT"}, Adapter())
    assert summary["completed_requests"] == 2
    assert summary["paired_cases"] == 0
    assert summary["metrics"]["restart"]["accuracy"] is None
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        grade(requests, results + results[:1], {"x": "PROMPT"}, Adapter())


def test_live_snapshot_only_tolerates_a_truncated_final_record(tmp_path):
    p = tmp_path / "live.jsonl"
    p.write_text(json.dumps({"a": 1})+'\n{"a":')
    assert read_rows(p, True) == [{"a": 1}]
    p.write_text('{"a":\n'+json.dumps({"a": 1})+'\n')
    with pytest.raises(json.JSONDecodeError):
        read_rows(p, True)
