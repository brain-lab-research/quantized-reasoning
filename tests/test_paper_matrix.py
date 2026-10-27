from h200.paper_matrix import prepare
from quantlab.benchmarks.paper_adapters import PaperGSM8KAdapter
from quantlab.benchmarks.registry import BenchmarkRegistry


def test_paper_matrix_covers_all_pairs_before_full_extensions(tmp_path):
    tasks = prepare(tmp_path)
    first_pass = tasks[:34]
    assert len(first_pass) == 34
    assert len({(t["size"], t["benchmark"]) for t in first_pass}) == 17
    assert {t["precision"] for t in first_pass} == {"fp16", "w2"}
    assert all(t["count"] == (30 if t["benchmark"] == "aime2026" else 100) for t in first_pass)
    assert all(t["benchmark"] in ("math500", "gpqa_diamond") for t in tasks[34:])
    assert all("llama" not in t["config"].lower() for t in tasks)


def test_restored_paper_adapters_and_boxed_gsm_answer():
    for name in ("arc_easy", "arc_challenge", "piqa", "strategyqa", "gsm8k_paper", "winogrande_paper"):
        assert BenchmarkRegistry.build(name)
    adapter = PaperGSM8KAdapter()
    assert adapter.extract_answer(r"Result: \boxed{42}") == "42"
    assert adapter.is_correct("42", "42")
    assert adapter.build_prompt({"question": "2+3?"}).endswith("<|im_start|>assistant\n")
