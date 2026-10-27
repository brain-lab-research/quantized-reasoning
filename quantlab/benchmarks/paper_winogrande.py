from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DIR = _REPO_ROOT / "data" / "winogrande"
_HF_DATASET = "allenai/winogrande"
_DEFAULT_SUBSET = "winogrande_xl"

# lm-evaluation-harness winogrande/preprocess_winogrande.py (doc_to_choice)
def _choice_sentences(raw: dict[str, Any]) -> tuple[str, str]:
    sentence = raw["sentence"]
    idx = sentence.index("_")
    suffix = sentence[idx + 1 :]
    prefix = sentence[:idx]
    return (
        prefix + raw["option1"] + suffix,
        prefix + raw["option2"] + suffix,
    )


class WinoGrandeAdapter(BenchmarkAdapter):
    """WinoGrande XL — validation accuracy, lm-eval choice layout."""

    name = "winogrande"

    def load(
        self,
        split: str = "validation",
        subset: Optional[str] = "winogrande_xl",
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import Dataset, load_dataset

        subset = subset or _DEFAULT_SUBSET
        local_path = _LOCAL_DIR / subset / f"{split}.parquet"
        if local_path.is_file():
            ds = Dataset.from_parquet(str(local_path))
        else:
            ds = load_dataset(_HF_DATASET, subset, split=split)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    example_id=f"winogrande_{split}_{i}",
                    prompt=self.build_prompt(row),
                    ground_truth=str(row["answer"]),
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        opt1, opt2 = _choice_sentences(raw)
        user_content = (
            f"Option 1: {opt1}\n"
            f"Option 2: {opt2}\n\n"
            "Which option is correct? Answer with only 1 or 2."
        )
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        text = generated_text.strip()
        last_line = text.split("\n")[-1].strip()
        m = re.match(r"^([12])[).\s]?$", last_line)
        if m:
            return m.group(1)
        matches = list(re.finditer(r"\b([12])\b", text))
        if matches:
            return matches[-1].group(1)
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        return predicted.strip() == ground_truth.strip()
