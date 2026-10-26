from quantlab.config.schema import BenchmarkConfig
from quantlab.runner import _load_benchmark_examples


class Adapter:
    def load(self, *, split, subset, max_examples, seed):
        examples = list(range(250))
        return examples if max_examples is None else examples[:max_examples]


def test_shards_are_disjoint_and_preserve_example_ids():
    adapter = Adapter()
    first = _load_benchmark_examples(adapter, BenchmarkConfig(name="math500", max_examples=100))
    second = _load_benchmark_examples(
        adapter, BenchmarkConfig(name="math500", max_examples=100, start_index=100)
    )
    assert first == list(range(100))
    assert second == list(range(100, 200))
    assert not set(first) & set(second)


def test_last_shard_and_unbounded_offset():
    assert _load_benchmark_examples(
        Adapter(), BenchmarkConfig(name="math500", max_examples=100, start_index=200)
    ) == list(range(200, 250))
    assert _load_benchmark_examples(
        Adapter(), BenchmarkConfig(name="math500", start_index=240)
    ) == list(range(240, 250))
