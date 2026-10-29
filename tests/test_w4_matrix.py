from h200.w4_matrix import prepare_w4


def test_full_matrix_excludes_existing_gpqa_and_is_idempotent(tmp_path):
    first = prepare_w4(tmp_path)
    second = prepare_w4(tmp_path)
    assert first == second
    assert len(first) == 17
    assert len({(t['size'], t['benchmark']) for t in first}) == 17
    assert (32, 'gpqa_diamond') not in {(t['size'], t['benchmark']) for t in first}
    assert all(t['precision'] == 'w4' for t in first)
    assert all(t['count'] > 0 for t in first)
