from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DIR = _REPO_ROOT / "data" / "piqa"
_HF_DATASET = "baber/piqa"


def _label_to_choice(label: Any) -> str:
    """PIQA label 0/1 (sol1/sol2) → generative answer '1'/'2'."""
    return str(int(label) + 1)


class PIQAAdapter(BenchmarkAdapter):
    """PIQA — validation accuracy, lm-evaluation-harness layout (baber/piqa)."""

    name = "piqa"

    def load(
        self,
        split: str = "validation",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import Dataset, load_dataset

        if subset:
            raise ValueError(f"piqa has no subsets; got subset={subset!r}")
        if split == "test":
            raise ValueError(
                "PIQA test split has no public labels; use split='validation' "
                "(standard lm-evaluation-harness eval)."
            )

        local_path = _LOCAL_DIR / f"{split}.parquet"
        if local_path.is_file():
            ds = Dataset.from_parquet(str(local_path))
        else:
            ds = load_dataset(_HF_DATASET, split=split)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            gt = _label_to_choice(row["label"])
            examples.append(
                BenchmarkExample(
                    example_id=f"piqa_{split}_{i}",
                    prompt=self.build_prompt(row),
                    ground_truth=gt,
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        # lm_eval/tasks/piqa/piqa.yaml: doc_to_text + doc_to_choice (generative 1/2)
        user_content = (
            f"Question: {raw['goal']}\n"
            f"Option 1: {raw['sol1']}\n"
            f"Option 2: {raw['sol2']}\n\n"
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
