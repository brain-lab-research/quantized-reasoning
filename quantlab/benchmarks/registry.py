from __future__ import annotations

from typing import Type

from quantlab.benchmarks.base import BenchmarkAdapter


class BenchmarkRegistry:
    _registry: dict[str, Type[BenchmarkAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_cls: Type[BenchmarkAdapter]) -> Type[BenchmarkAdapter]:
            adapter_cls.name = name
            cls._registry[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def build(cls, name: str) -> BenchmarkAdapter:
        if name not in cls._registry:
            raise KeyError(f"Unknown benchmark {name!r}. Available: {list(cls._registry)}")
        return cls._registry[name]()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


def _register_builtins() -> None:
    from quantlab.benchmarks.math500 import MATH500Adapter
    from quantlab.benchmarks.aime26 import AIME26Adapter
    from quantlab.benchmarks.gsm8k import GSM8KAdapter
    from quantlab.benchmarks.gpqa import GPQAAdapter
    from quantlab.benchmarks.gpqa_diamond import GPQADiamondAdapter
    from quantlab.benchmarks.winogrande import WinoGrandeAdapter
    from quantlab.benchmarks.arc_easy import ARCEasyAdapter
    from quantlab.benchmarks.piqa import PIQAAdapter
    from quantlab.benchmarks.paper_adapters import ARCChallengeAdapter, PaperGSM8KAdapter
    from quantlab.benchmarks.paper_winogrande import WinoGrandeAdapter as PaperWinoGrandeAdapter
    from quantlab.benchmarks.strategyqa import StrategyQAAdapter

    mapping = {
        "math500": MATH500Adapter,
        "aime26": AIME26Adapter,
        "gsm8k": GSM8KAdapter,
        "gpqa": GPQAAdapter,
        "gpqa_diamond": GPQADiamondAdapter,
        "winogrande": WinoGrandeAdapter,
        "arc_easy": ARCEasyAdapter,
        "arc_challenge": ARCChallengeAdapter,
        "piqa": PIQAAdapter,
        "strategyqa": StrategyQAAdapter,
        "gsm8k_paper": PaperGSM8KAdapter,
        "winogrande_paper": PaperWinoGrandeAdapter,
    }
    for name, cls in mapping.items():
        BenchmarkRegistry._registry.setdefault(name, cls)


_register_builtins()
