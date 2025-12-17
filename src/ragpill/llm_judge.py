"""LLM-as-a-judge implementation.

Replaces ``pydantic_evals.evaluators.llm_as_a_judge`` with a local
``pydantic_ai.Agent``-based implementation. The system prompts and prompt
structure are copied verbatim from pydantic_evals v1 for behavioral parity.
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UserContent, models
from pydantic_ai.messages import MULTI_MODAL_CONTENT_TYPES
from pydantic_ai.settings import ModelSettings
from pydantic_core import to_json


class GradingOutput(BaseModel, populate_by_name=True):
    """Structured output from the LLM judge.

    Attributes:
        reason: Human-readable explanation of the verdict.
        pass_: Whether the output passes the rubric. Aliased to ``"pass"`` for
            JSON compatibility.
        score: Continuous score in ``[0.0, 1.0]``.
    """

    reason: str
    pass_: bool = Field(validation_alias="pass", serialization_alias="pass")
    score: float


JUDGE_OUTPUT_SYSTEM_PROMPT = dedent(
    """
    You are grading output according to a user-specified rubric. If the statement in the rubric is true, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

    Examples:

    <Output>Hello world</Output>
    <Rubric>Content contains a greeting</Rubric>
    {"reason": "the content contains the word 'Hello'", "pass": true, "score": 1.0}

    <Output>Avast ye swabs, repel the invaders!</Output>
    <Rubric>Does not speak like a pirate</Rubric>
    {"reason": "'avast ye' is a common pirate term", "pass": false, "score": 0.0}
    """
)


JUDGE_INPUT_OUTPUT_SYSTEM_PROMPT = dedent(
    """
    You are grading output according to a user-specified rubric. If the statement in the rubric is true for the provided input and output, then the output passes the test. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}

    Examples:

    <Input>Hello world</Input>
    <Output>Hello</Output>
    <Rubric>Content contains a greeting word which is present in the input</Rubric>
    {"reason": "the content contains the word 'Hello'", "pass": true, "score": 1.0}

    <Input>Pirate</Input>
    <Output>Avast ye swabs, repel the invaders!</Output>
    <Rubric>Does not speak in the style described by the input</Rubric>
    {"reason": "'avast ye' is a common pirate term", "pass": false, "score": 0.0}
    """
)


_judge_output_agent: Agent[None, GradingOutput] = Agent(
    name="judge_output",
    system_prompt=JUDGE_OUTPUT_SYSTEM_PROMPT,
    output_type=GradingOutput,
)


_judge_input_output_agent: Agent[None, GradingOutput] = Agent(
    name="judge_input_output",
    system_prompt=JUDGE_INPUT_OUTPUT_SYSTEM_PROMPT,
    output_type=GradingOutput,
)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return to_json(value).decode()
    except Exception:
        return repr(value)


def _make_section(content: Any, tag: str) -> list[str | UserContent]:
    sections: list[str | UserContent] = []
    items: Sequence[str | UserContent] = (  # pyright: ignore[reportUnknownVariableType]
        content if isinstance(content, Sequence) and not isinstance(content, str) else [content]
    )
    sections.append(f"<{tag}>")
    for item in items:
        sections.append(item if isinstance(item, (str, *MULTI_MODAL_CONTENT_TYPES)) else _stringify(item))
    sections.append(f"</{tag}>")
    return sections


def _build_prompt(
    output: Any,
    rubric: str,
    inputs: Any | None = None,
) -> str | Sequence[str | UserContent]:
    sections: list[str | UserContent] = []
    if inputs is not None:
        sections.extend(_make_section(inputs, "Input"))
    sections.extend(_make_section(output, "Output"))
    sections.extend(_make_section(rubric, "Rubric"))
    if all(isinstance(section, str) for section in sections):
        return "\n".join(sections)  # type: ignore[arg-type]
    return sections


async def judge_output(
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge a task output against a rubric.

    Args:
        output: The task output to grade.
        rubric: Rubric describing what "passing" means for this output.
        model: The pydantic-ai model to use for grading.
        model_settings: Optional pydantic-ai ``ModelSettings``.

    Returns:
        A ``GradingOutput`` with ``reason``, ``pass_``, and ``score``.

    Example:
        ```python
        from pydantic_ai.models.test import TestModel
        from ragpill.llm_judge import judge_output

        result = await judge_output("Hello world", "contains a greeting", TestModel())
        print(result.pass_, result.reason)
        ```
    """
    user_prompt = _build_prompt(output=output, rubric=rubric)
    return (await _judge_output_agent.run(user_prompt, model=model, model_settings=model_settings)).output


async def judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | str,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge a task output against a rubric given the inputs.

    Args:
        inputs: The task inputs included alongside the output in the prompt.
        output: The task output to grade.
        rubric: Rubric describing what "passing" means for this output.
        model: The pydantic-ai model to use for grading.
        model_settings: Optional pydantic-ai ``ModelSettings``.

    Returns:
        A ``GradingOutput`` with ``reason``, ``pass_``, and ``score``.

    Example:
        ```python
        from pydantic_ai.models.test import TestModel
        from ragpill.llm_judge import judge_input_output

        result = await judge_input_output(
            inputs="Pirate",
            output="Avast ye!",
            rubric="does speak like a pirate",
            model=TestModel(),
        )
        ```
    """
    user_prompt = _build_prompt(inputs=inputs, output=output, rubric=rubric)
    return (await _judge_input_output_agent.run(user_prompt, model=model, model_settings=model_settings)).output
