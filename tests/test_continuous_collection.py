import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_continuous_loop_saves_out_of_order_without_waiting_for_slowest():
    path = Path(__file__).parents[1] / "scripts/collect_baseline_traces.py"
    spec = importlib.util.spec_from_file_location("continuous_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Engine:
        def __init__(self):
            self.calls = 0

        def has_unfinished_requests(self):
            return self.calls < 3

        def step(self):
            self.calls += 1
            ids = [("fast", True), ("slow", False)] if self.calls == 1 else (
                [("fast", True)] if self.calls == 2 else [("slow", True)])
            return [SimpleNamespace(request_id=i, finished=f) for i, f in ids]

    engine = Engine()
    results = module.completed_outputs(engine)
    assert next(results).request_id == "fast"
    assert engine.calls == 1
    assert next(results).request_id == "slow"
    assert list(results) == []
