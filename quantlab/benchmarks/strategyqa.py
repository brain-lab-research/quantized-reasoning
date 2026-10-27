import re
from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample


class StrategyQAAdapter(BenchmarkAdapter):
    name = "strategyqa"

    def load(self, split="test", subset=None, max_examples=None, seed=42):
        from datasets import load_dataset
        if subset:
            raise ValueError("StrategyQA has no subsets")
        ds = load_dataset("ChilleD/StrategyQA", split=split).shuffle(seed=seed)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        return [BenchmarkExample(
            f"strategyqa_{row['qid']}", self.build_prompt(row),
            "yes" if row["answer"] else "no", raw=dict(row),
        ) for row in ds]

    def build_prompt(self, raw):
        return ("<|im_start|>system\nYou are a helpful assistant. Think step by step, "
                "then answer yes or no on the last line.<|im_end|>\n"
                f"<|im_start|>user\n{raw['question']}<|im_end|>\n"
                "<|im_start|>assistant\n")

    def extract_answer(self, text):
        matches = re.findall(r"\b(yes|no)\b", text, flags=re.I)
        return matches[-1].lower() if matches else None

    def is_correct(self, predicted, ground_truth):
        return predicted is not None and predicted.lower() == ground_truth.lower()
