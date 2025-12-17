from collections.abc import Sequence
from functools import reduce
from typing import Any

from pydantic_settings import BaseSettings


def _prefix_settings_key(input: tuple[BaseSettings | dict[str, Any], str]) -> dict[str, Any]:
    setting, prefix = input
    if isinstance(setting, BaseSettings):
        setting_dict = setting.model_dump()
    else:
        setting_dict = setting
    return {f"{prefix}_{k}": v for k, v in setting_dict.items()}


def merge_settings(settings_prefixes: Sequence[tuple[BaseSettings | dict[str, Any], str]]) -> dict[str, Any]:
    """Merge multiple pydantic settings into a single dict with prefixed keys for MLflow logging.

    Each settings object's fields are flattened and prefixed with the given string,
    producing a dict suitable for passing as ``model_params`` to
    [`evaluate_testset`][ragpill.mlflow_helper.evaluate_testset].

    Args:
        settings_prefixes: Sequence of ``(settings_object, prefix)`` tuples. Each
            settings object (a ``BaseSettings`` instance or plain dict) is flattened
            and its keys are prefixed with ``prefix_``.

    Returns:
        A single merged dictionary with prefixed keys from all settings objects.

    Example:
        ```python
        from ragpill import merge_settings

        params = merge_settings([
            (settings, "mlflow"),
            (agent_settings, "agent"),
            (llm_settings, "llm"),
        ])
        ```
    """
    return reduce(lambda x, y: x | y, map(_prefix_settings_key, settings_prefixes))
