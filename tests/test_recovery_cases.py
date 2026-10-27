from scripts.prepare_recovery_cases import first_loop, last_complete_box


def test_loop_trigger_is_causal_and_respects_window():
    ids = [1, 2, 3] * 4
    assert first_loop(ids, n=3, repeats=4, window=12) == {"onset": 0, "trigger": 12}
    assert first_loop(ids[:-1], n=3, repeats=4, window=12) is None
    assert first_loop(ids, n=3, repeats=4, window=8) is None


def test_complete_box_requires_closing_brace_and_handles_nested_values():
    assert last_complete_box(r"x=4 and \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_complete_box(r"\boxed{1} ... \boxed{2") == "1"
    assert last_complete_box("an equation 2+3=5") is None
