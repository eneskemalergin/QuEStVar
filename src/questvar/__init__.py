from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "QuestVar":
        from questvar._api import QuestVar

        return QuestVar
    if name == "TestResults":
        from questvar._api import TestResults

        return TestResults
    if name == "PowerResults":
        from questvar._api import PowerResults

        return PowerResults
    if name == "test":
        from questvar.test import test

        return test
    if name == "run_power_analysis":
        from questvar.power.run import run_power_analysis

        return run_power_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QuestVar",
    "TestResults",
    "PowerResults",
    "test",
    "run_power_analysis",
]
