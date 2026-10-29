from vv.distill_llama_matrix import prepare_tasks


def test_all_nine_full_benchmarks_are_assigned_to_the_correct_model(tmp_path):
    tasks = prepare_tasks(tmp_path)
    assert len(tasks) == 9
    assert len({t['benchmark'] for t in tasks}) == 9
    assert all(t['size'] == 8 and t['precision'] == 'fp16' and t['count'] > 0 for t in tasks)
