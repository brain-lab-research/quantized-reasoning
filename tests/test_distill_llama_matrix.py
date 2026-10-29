import os

from vv.distill_llama_matrix import prepare_tasks
from vv.prepare_distill_teacher import is_correct
from vv.smoke_distill_w2_quality import validate_tokenizers
import pytest


def test_all_nine_full_benchmarks_are_assigned_to_the_correct_model(tmp_path):
    tasks = prepare_tasks(tmp_path)
    assert len(tasks) == 9
    assert len({t['benchmark'] for t in tasks}) == 9
    assert all(t['size'] == 8 and t['precision'] == 'fp16' and t['count'] > 0 for t in tasks)


@pytest.mark.skipif(os.name != 'posix', reason='advisory worker locks require POSIX')
def test_shared_worker_skips_claimed_benchmark(tmp_path, monkeypatch):
    import fcntl
    import vv.distill_llama_matrix as matrix
    tasks = prepare_tasks(tmp_path)
    claimed = tmp_path / ('1029-r1-distill-llama-8b-' + tasks[0]['benchmark'] + '-fp16') / '.claim'
    visited = []
    monkeypatch.setattr(matrix, 'run_one', lambda task, cfg, path: visited.append(task['benchmark']))
    with claimed.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        matrix.run(tmp_path, tmp_path / 'model', 'fp16', shared=True)
    assert tasks[0]['benchmark'] not in visited
    assert len(visited) == 8


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
    assert is_correct({'answer_kind': 'math', 'ground_truth': '48/91'}, r'\boxed{\dfrac{48}{91}}')
    assert is_correct({'answer_kind': 'math', 'ground_truth': '-3/5'}, r'\boxed{-\frac{3}{5}}')
    assert not is_correct({'answer_kind': 'math', 'ground_truth': '48/91'}, r'\boxed{\frac{47}{91}}')


def test_tokenizer_check_rejects_different_token_ids():
    class FakeTokenizer:
        eos_token_id = 2
        def __init__(self, token): self.token = token
        def get_vocab(self): return {'same': 1}
        def apply_chat_template(self, messages, **kwargs): return [1, self.token]
    validate_tokenizers(FakeTokenizer(3), FakeTokenizer(3))
    with pytest.raises(ValueError, match='token IDs differ'):
        validate_tokenizers(FakeTokenizer(3), FakeTokenizer(4))
