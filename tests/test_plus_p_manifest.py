from scripts.prepare_plus_p_manifest import build


def test_plus_p_manifest_full_order_and_resume(tmp_path):
    tasks = build(tmp_path)
    assert len(tasks) == 9
    assert sum(t['count'] for t in tasks) == 9387
    assert tasks[0]['benchmark'] == 'aime2026'
    assert tasks[-1]['benchmark'] == 'winogrande'
    assert build(tmp_path) == tasks
