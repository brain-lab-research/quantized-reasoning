from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "arc_easy"

_SYSTEM_PROMPT = (
    "You are a helpful science assistant. Answer the multiple-choice question. "
    "Think step by step, then write your final answer as a single letter "
    "(A, B, C, D, or E) on its own line at the end."
)

_LETTER_RE = re.compile(r"\b([A-E])\b(?!.*\b[A-E]\b)", re.DOTALL)


def _default_data_path(split: str) -> Path:
    return _DEFAULT_DATA_DIR / f"{split}.json"


class ARCEasyAdapter(BenchmarkAdapter):
    """Adapter for ARC-Easy (AI2 Reasoning Challenge, multiple-choice).

    Data: vendored JSON under ``data/arc_easy/`` (from ``allenai/ai2_arc``).
    Grading: exact match on ``answerKey`` letter (standard ``acc`` for generative runs).
    """

    name = "arc_easy"

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        import random

        if subset:
            raise ValueError(f"arc_easy has no subsets; got subset={subset!r}")

        path = self._data_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"ARC-Easy split {split!r} not found at {path}. "
                "Run scripts/prepare_arc_easy_data.py to vendor the dataset."
            )

        rows = json.loads(path.read_text(encoding="utf-8"))
        random.Random(seed).shuffle(rows)
        if max_examples:
            rows = rows[: min(max_examples, len(rows))]

        examples = []
        for i, row in enumerate(rows):
            answer_key = str(row["answerKey"]).strip().upper()
            examples.append(
                BenchmarkExample(
                    example_id=f"{self.name}_{row['id']}",
                    prompt=self.build_prompt(row),
                    ground_truth=answer_key,
                    raw=row,
                )
            )
        return examples

    @staticmethod
    def _wrap_chat_prompt(user_content: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def build_prompt(self, raw: dict[str, Any]) -> str:
        question = raw["question"]
        labels = raw["choices"]["label"]
        texts = raw["choices"]["text"]
        choices_lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        user_content = (
            f"Question: {question}\n\n{choices_lines}\n\n"
            "Solve the problem and end with a single line containing only "
            "the letter of the correct choice (A, B, C, D, or E)."
        )
        return self._wrap_chat_prompt(user_content)

    def extract_answer(self, generated_text: str) -> Optional[str]:
        last_line = generated_text.strip().split("\n")[-1].strip()
        m = re.match(r"^([A-E])[).\s]?$", last_line, re.I)
        if m:
            return m.group(1).upper()
        m = _LETTER_RE.search(generated_text)
        if m:
            return m.group(1).upper()
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        return predicted.strip().upper() == ground_truth.strip().upper()
