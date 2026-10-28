import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_collector():
    path = Path(__file__).parents[1] / "scripts/collect_baseline_traces.py"
    spec = importlib.util.spec_from_file_location("continuous_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuous_loop_saves_out_of_order_without_waiting_for_slowest():
    module = load_collector()

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


def test_chat_recipe_uses_model_template_instead_of_qwen_markers():
    module = load_collector()
    examples = [SimpleNamespace(prompt="old-qwen-prompt", raw={"problem": "2+3"})]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages[0]["role"] == "user"
            return "MODEL_TEMPLATE " + messages[0]["content"]

    module.format_chat_prompts(examples, Tokenizer(), "math500")
    assert examples[0].prompt.startswith("MODEL_TEMPLATE 2+3")
    assert "\\boxed{}" in examples[0].prompt


def test_requested_shards_cover_winogrande_without_overlap():
    module = load_collector()
    from quantlab.runner import _load_benchmark_examples

    class Adapter:
        def load(self, split, subset, max_examples, seed):
            return list(range(max_examples))

    shards = []
    for start, count in ((0, 800), (800, 467)):
        raw = {"benchmark": {}}
        module.set_requested_range(raw, count, start)
        cfg = SimpleNamespace(**raw["benchmark"], split="validation", subset=None, seed=42)
        shards.append(_load_benchmark_examples(Adapter(), cfg))
    assert shards[0] == list(range(800))
    assert shards[1] == list(range(800, 1267))
