from vv.distill_llama_matrix import prepare_tasks
from vv.prepare_distill_teacher import is_correct


def test_all_nine_full_benchmarks_are_assigned_to_the_correct_model(tmp_path):
    tasks = prepare_tasks(tmp_path)
    assert len(tasks) == 9
    assert len({t['benchmark'] for t in tasks}) == 9
    assert all(t['size'] == 8 and t['precision'] == 'fp16' and t['count'] > 0 for t in tasks)


def test_w2_matrix_is_distinct_and_complete(tmp_path):
    tasks = prepare_tasks(tmp_path, precision='w2')
    assert len(tasks) == 9
    assert all(t['precision'] == 'w2' for t in tasks)
    assert all(t['config'].endswith('-w2/cfg.yaml') or t['config'].endswith('-w2\\cfg.yaml') for t in tasks)


def test_teacher_selection_checks_synthetic_answers():
    assert is_correct({'answer_kind': 'math', 'ground_truth': '35'}, r'Work. \boxed{35}')
    assert not is_correct({'answer_kind': 'math', 'ground_truth': '35'}, r'Work. \boxed{36}')
    assert is_correct({'answer_kind': 'choice', 'ground_truth': 'C'}, r'Final \boxed{C}')
    assert not is_correct({'answer_kind': 'choice', 'ground_truth': 'C'}, 'I think C')
