import hashlib

import pytest

from scripts.compare_gpqa_rescue_w4 import compare


def pair():
    prompt = 'one GPQA prompt'
    h = hashlib.sha256(prompt.encode()).hexdigest()
    w4 = {'a': dict(text='Reasoning\nA', ground_truth='A', prompt_sha256=h,
                    is_correct=False, wall_seconds=3.0, output_tokens=15)}
    rescue = {'a': dict(prompt=prompt, ground_truth='A', predicted='A', is_correct=True,
                        branch='fallback', w2_tokens=10, fp16_tokens=20,
                        w2_seconds=1.0, fp16_seconds=2.0, end_to_end_seconds=3.5)}
    return w4, rescue


def test_compares_full_cost_and_regrades_w4_with_same_parser():
    w4, rescue = pair()
    result, rows = compare(w4, rescue, expected=1)
    assert result['w4']['correct'] == 1  # Ignores the different stored parser verdict.
    assert result['rescue']['fallbacks'] == 1
    assert result['rescue']['total_seconds'] == 3.5
    assert result['rescue']['w2_output_tokens'] == 10
    assert result['rescue']['fp16_output_tokens'] == 20
    assert rows[0]['w4_correct'] and rows[0]['rescue_correct']


def test_rejects_different_or_unrecorded_prompt():
    w4, rescue = pair()
    w4['a']['prompt_sha256'] = 'different'
    with pytest.raises(ValueError, match='prompt'):
        compare(w4, rescue, expected=1)


def test_rejects_latency_that_excludes_a_stage():
    w4, rescue = pair()
    rescue['a']['end_to_end_seconds'] = 2.0
    with pytest.raises(ValueError, match='timing'):
        compare(w4, rescue, expected=1)
