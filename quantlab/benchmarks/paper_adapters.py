"""Original-paper benchmark variants; existing adapter behavior stays unchanged."""
from pathlib import Path
from quantlab.benchmarks.arc_easy import ARCEasyAdapter
from quantlab.benchmarks.gsm8k import GSM8KAdapter
from quantlab.benchmarks.boxed import extract_last_boxed
from quantlab.benchmarks.math500 import _strip_string, _math_equal


class ARCChallengeAdapter(ARCEasyAdapter):
    name = "arc_challenge"

    def __init__(self):
        super().__init__(Path(__file__).resolve().parents[2] / "data/arc_challenge")


class PaperGSM8KAdapter(GSM8KAdapter):
    name = "gsm8k_paper"

    def build_prompt(self, raw):
        # Restored from FastReasonig's legacy prompt used by configs/final/gsm8k.
        user = raw["question"] + "\n\nSolve this problem and give your final answer in \\boxed{}."
        return ("<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

    def extract_answer(self, generated_text):
        boxed = extract_last_boxed(generated_text)
        return _strip_string(boxed) if boxed is not None else super().extract_answer(generated_text)

    def is_correct(self, predicted, ground_truth):
        return predicted is not None and _math_equal(predicted, ground_truth)
