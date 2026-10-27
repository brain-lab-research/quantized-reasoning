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


def test_full_matrix_contains_all_36_full_splits(tmp_path):
    from h200.paper_matrix import FULL_COUNTS
    tasks = prepare(tmp_path, full_benchmarks=True)
    assert len(tasks) == 36
    assert sum(t["count"] for t in tasks) == 37548
    assert len({(t["size"], t["precision"], t["benchmark"]) for t in tasks}) == 36
    assert all(t["count"] == FULL_COUNTS[t["benchmark"]] for t in tasks)


def test_fpmi_worker_filter_selects_only_eight_b_w2(tmp_path, monkeypatch, capsys):
    from h200 import paper_matrix
    monkeypatch.setattr("sys.argv", ["paper_matrix", "--root", str(tmp_path),
        "--prepare-only", "--sizes", "8", "--precisions", "w2", "--full-first"])
    paper_matrix.main()
    assert "PAPER_MATRIX 10 tasks" in capsys.readouterr().out


def test_completed_tasks_can_be_skipped_before_applying_new_resource_limits(tmp_path):
    import json
    from h200.paper_matrix import task_complete
    task = {"config": str(tmp_path / "cfg.yaml"), "count": 100}
    run = tmp_path / "artifacts" / "run"
    run.mkdir(parents=True)
    summary = run / "summary.json"
    summary.write_text(json.dumps({"n_judged": 99, "n_errors": 0}))
    assert not task_complete(task)
    summary.write_text(json.dumps({"n_judged": 100, "n_errors": 0}))
    assert task_complete(task)
    summary.write_text(json.dumps({"n_judged": 100, "n_errors": 1}))
    assert not task_complete(task)
