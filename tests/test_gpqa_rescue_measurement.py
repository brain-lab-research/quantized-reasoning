import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load_module():
    path = Path(__file__).parents[1] / 'scripts' / 'measure_gpqa_rescue.py'
    spec = importlib.util.spec_from_file_location('gpqa_rescue_measurement', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_detector_matches_saved_trace_detector():
    from scripts.prepare_recovery_cases import first_loop
    module = load_module()
    tokens = list(range(20)) * 4 + [12345]
    detector = module.LoopDetector(close_id=99999)
    event = next(result for token in tokens if (result := detector.feed(token)))
    assert event == first_loop(tokens) == {'onset': 0, 'trigger': 80}


def test_detector_ignores_repetition_after_think_close():
    module = load_module()
    detector = module.LoopDetector(close_id=99999)
    assert all(detector.feed(token) is None for token in [99999] + list(range(20)) * 5)


def test_legacy_choice_is_case_sensitive_and_uses_last_uppercase_letter():
    module = load_module()
    assert module.paper_gpqa_choice('a proton has a mass') is None
    assert module.paper_gpqa_choice('Compare A with B; the answer is C.') == 'C'
    assert module.paper_gpqa_choice('Reasoning\nD') == 'D'


def test_stream_aborts_only_target_request_on_trigger(monkeypatch):
    module = load_module()
    sampling = ModuleType('vllm.sampling_params')
    sampling.RequestOutputKind = SimpleNamespace(DELTA='delta')
    package = ModuleType('vllm')
    monkeypatch.setitem(sys.modules, 'vllm', package)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', sampling)

    class Engine:
        def __init__(self):
            self.tokens = iter(list(range(20)) * 4)
            self.aborted = []
        def add_request(self, rid, prompt, params):
            assert rid == 'ours'
        def has_unfinished_requests(self):
            return not self.aborted
        def step(self):
            out = SimpleNamespace(token_ids=[next(self.tokens)], finish_reason=None)
            return [SimpleNamespace(request_id='ours', outputs=[out], finished=False)]
        def abort_request(self, ids):
            self.aborted.extend(ids)

    engine = Engine()
    ids, event, finish = module.stream_one(engine, 'ours', 'p', SimpleNamespace(output_kind='delta'), module.LoopDetector(99999))
    assert len(ids) == 80 and event == {'onset': 0, 'trigger': 80}
    assert finish == 'loop_abort' and engine.aborted == ['ours']


def test_resume_drops_only_interrupted_final_record(tmp_path):
    module = load_module()
    path = tmp_path / 'results.jsonl'
    path.write_bytes(b'{"example_id":"one"}\n{"example_id":"partial"')
    assert module.read_completed(path) == [{'example_id': 'one'}]
    assert path.read_bytes() == b'{"example_id":"one"}\n'


def test_summary_counts_both_stages_and_measured_end_to_end(tmp_path):
    module = load_module()
    rows = [dict(branch='fallback', is_correct=True, w2_tokens=100, fp16_tokens=200,
                 end_to_end_seconds=3.5),
            dict(branch='commit', is_correct=False, w2_tokens=90, fp16_tokens=0,
                 end_to_end_seconds=1.5)]
    path = tmp_path / 'summary.json'
    module.write_summary(path, rows, 2)
    import json
    summary = json.loads(path.read_text())
    assert summary['complete'] and summary['fallbacks'] == 1 and summary['commits'] == 1
    assert summary['w2_output_tokens'] == 190 and summary['fp16_output_tokens'] == 200
    assert summary['mean_end_to_end_seconds'] == 2.5
