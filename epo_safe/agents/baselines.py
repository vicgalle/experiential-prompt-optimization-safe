"""Baseline agent configurations.

- static: Fixed generic prompt, no reflection
- cot: Fixed CoT prompt, no reflection
- reward_only: Full EPO loop but no danger warnings in feedback
- epo_safe: Full EPO loop with danger warnings (level1)
- epo_safe_l0: Full EPO loop with danger warnings (level0)
"""

from epo_safe.prompts.system_prompts import INITIAL_SPECIFICATION, COT_SPECIFICATION


def get_initial_specification(method: str) -> str:
    """Return the initial specification for a given method."""
    if method == "cot":
        return COT_SPECIFICATION
    return INITIAL_SPECIFICATION


def uses_reflection(method: str) -> bool:
    """Whether this method updates the specification via reflection."""
    return method in ("reward_only", "epo_safe", "epo_safe_l0")


def uses_danger_warnings(method: str) -> bool:
    """Whether this method includes danger warnings in the feedback."""
    return method in ("epo_safe", "epo_safe_l0")


def get_feedback_level(method: str) -> str:
    """Return feedback level for danger warnings."""
    if method == "epo_safe_l0":
        return "level0"
    return "level1"
