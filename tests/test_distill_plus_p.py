from types import SimpleNamespace

import pytest

from scripts.collect_baseline_traces import format_native_prompts
from scripts.collect_distill_plans import digest, plan_prompt


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'native<think>\n'


def example():
    return SimpleNamespace(example_id='id', ground_truth='B',
        prompt='<|im_start|>system\nInstruction<|im_end|>\n<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n')


def test_plan_uses_closed_native_think():
    tokenizer = Tokenizer()
    prompt = plan_prompt(tokenizer, 'Question')
    assert prompt.endswith('<think>\n\n</think>\n\n')
    assert 'Question' in tokenizer.messages[1]['content']


def test_native_w2_contains_plan_and_rejects_wrong_source():
    tokenizer = Tokenizer()
    row = example()
    plans = {'id': {'base_prompt_sha256': digest(row.prompt), 'ground_truth': 'B', 'plan': 'step'}}
    format_native_prompts([row], tokenizer, plans, 'Answer B')
    assert 'Plan:\nstep\n\nAnswer B' in tokenizer.messages[0]['content']
    row = example()
    plans['id']['ground_truth'] = 'A'
    with pytest.raises(ValueError, match='mismatches'):
        format_native_prompts([row], tokenizer, plans, 'Answer B')
