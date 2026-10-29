from scripts.prepare_shared_manifest import build


def test_full_ordered_manifest_is_repeatable(tmp_path):
    tasks = build(tmp_path)
    assert len(tasks) == 9
    assert sum(t['count'] for t in tasks) == 9387
    assert all(t['start'] == 0 for t in tasks)
    assert 'gpqa_diamond' in tasks[0]['config']
    assert 'winogrande' in tasks[-1]['config']
    assert build(tmp_path) == tasks
