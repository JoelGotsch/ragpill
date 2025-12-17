"""Test LLMJudge initialization via environment variables and judge helpers."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from ragpill.evaluators import LLMJudge
from ragpill.llm_judge import GradingOutput, judge_input_output, judge_output
from ragpill.settings import (
    LLMJudgeSettings,
    configure_llm_judge,
    get_llm_judge_settings,
    reset_llm_judge_settings,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a fresh singleton."""
    reset_llm_judge_settings()
    yield
    reset_llm_judge_settings()


FAKE_ENV = {
    "RAGPILL_LLMJUDGE_API_KEY": "test-api-key",
    "RAGPILL_LLMJUDGE_BASE_URL": "https://my-base-url.com/v1",
    "RAGPILL_LLMJUDGE_MODEL_NAME": "fake-test-model",
}


def test_init_with_env_vars_sets_rubric():
    """LLMJudge(rubric=...) stores the rubric correctly when env vars are set."""
    with patch.dict(os.environ, FAKE_ENV):
        judge = LLMJudge(rubric="whatever")
    assert judge.rubric == "whatever"
    assert isinstance(judge.model, OpenAIChatModel)
    assert judge.model.model_name == FAKE_ENV["RAGPILL_LLMJUDGE_MODEL_NAME"]
    assert judge.model.base_url.strip("/") == FAKE_ENV["RAGPILL_LLMJUDGE_BASE_URL"].strip("/")


def test_singleton_returns_same_instance():
    """get_llm_judge_settings returns the same object on repeated calls."""
    with patch.dict(os.environ, FAKE_ENV):
        a = get_llm_judge_settings()
        b = get_llm_judge_settings()
    assert a is b


def test_configure_replaces_singleton():
    """configure_llm_judge replaces the global singleton."""
    with patch.dict(os.environ, FAKE_ENV):
        original = get_llm_judge_settings()
        new = configure_llm_judge(model_name="custom-model", api_key="k", base_url="https://x")
    assert new is not original
    assert new.model_name == "custom-model"
    assert get_llm_judge_settings() is new


def test_configure_with_settings_instance():
    """configure_llm_judge accepts a pre-built LLMJudgeSettings."""
    settings = LLMJudgeSettings(model_name="injected", api_key="k", base_url="https://x")  # pyright: ignore[reportCallIssue]
    result = configure_llm_judge(settings=settings)
    assert result is settings
    assert get_llm_judge_settings() is settings


def test_llm_model_property_caches():
    """The llm_model property returns the same object on repeated access."""
    with patch.dict(os.environ, FAKE_ENV):
        settings = get_llm_judge_settings()
        model_a = settings.llm_model
        model_b = settings.llm_model
    assert model_a is model_b


def test_set_model_overrides_cached():
    """set_model injects a custom model that is returned by llm_model."""
    with patch.dict(os.environ, FAKE_ENV):
        settings = get_llm_judge_settings()
        original_model = settings.llm_model
        fake = MagicMock()
        settings.set_model(fake)
        assert settings.llm_model is fake
        assert settings.llm_model is not original_model


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_judge_output_returns_grading_output():
    """judge_output returns a GradingOutput with reason/pass_/score using TestModel."""
    model = TestModel()
    result = await judge_output("some output", "the output should be non-empty", model)
    assert isinstance(result, GradingOutput)
    assert isinstance(result.reason, str)
    assert isinstance(result.pass_, bool)
    assert isinstance(result.score, float)


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_judge_input_output_returns_grading_output():
    """judge_input_output returns a GradingOutput via TestModel."""
    model = TestModel()
    result = await judge_input_output(
        inputs="PIRATE",
        output="Avast ye!",
        rubric="matches the style in the input",
        model=model,
    )
    assert isinstance(result, GradingOutput)


def test_judge_input_output_prompt_includes_input_section():
    """``_build_prompt`` wraps the inputs in ``<Input>`` tags when provided."""
    from ragpill.llm_judge import _build_prompt

    prompt = _build_prompt(
        output="the answer",
        rubric="must mention X",
        inputs="the question",
    )
    assert isinstance(prompt, str)
    assert "<Input>" in prompt and "</Input>" in prompt
    assert "the question" in prompt
    assert "the answer" in prompt


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_run_includes_rubric_verbatim_in_reason():
    """``LLMJudge.run`` prepends the rubric to the judge's reasoning in EvaluationResult.reason."""
    from typing import Any

    from ragpill.base import EvaluatorMetadata
    from ragpill.eval_types import EvaluatorContext

    rubric = "Output must mention Paris as the capital of France."
    judge = LLMJudge(rubric=rubric, model=TestModel())
    ctx: EvaluatorContext[Any, Any, EvaluatorMetadata] = EvaluatorContext(
        name="t",
        inputs="What is the capital?",
        metadata=EvaluatorMetadata(expected=True),
        expected_output=None,
        output="Paris is the capital.",
        duration=0.0,
    )
    result = await judge.run(ctx)
    assert rubric in (result.reason or ""), f"Rubric verbatim missing from reason: {result.reason!r}"
    # And the judge's own reasoning is still there (TestModel produces a non-empty reason).
    assert "Rubric:" in (result.reason or "")


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_run_includes_rubric_even_on_pass():
    """Rubric is embedded whether the verdict is pass or fail (unconditional embedding)."""
    from typing import Any

    from ragpill.base import EvaluatorMetadata
    from ragpill.eval_types import EvaluatorContext

    rubric = "Output should be non-empty."
    judge = LLMJudge(rubric=rubric, model=TestModel())
    ctx: EvaluatorContext[Any, Any, EvaluatorMetadata] = EvaluatorContext(
        name="t",
        inputs=None,
        metadata=EvaluatorMetadata(expected=True),
        expected_output=None,
        output="Something",
        duration=0.0,
    )
    result = await judge.run(ctx)
    assert rubric in (result.reason or "")


def test_evaluation_output_roundtrip_preserves_rubric_in_reason():
    """to_json / from_json keep the rubric-plus-judge-reasoning string intact."""
    import pandas as pd

    from ragpill.base import TestCaseMetadata
    from ragpill.eval_types import EvaluationResult, EvaluatorSource
    from ragpill.types import (
        AggregatedResult,
        CaseResult,
        EvaluationOutput,
        RunResult,
    )

    rubric = "Output must mention Paris as the capital of France."
    combined_reason = f"Rubric: {rubric}\nVerdict: the output mentions Paris."
    rr = RunResult(
        run_index=0,
        input_key="k0",
        run_span_id="",
        output="Paris is the capital.",
        duration=0.0,
        assertions={
            "LLMJudge": EvaluationResult(
                name="LLMJudge",
                value=True,
                reason=combined_reason,
                source=EvaluatorSource(name="LLMJudge", arguments={}),
            )
        },
    )
    cr = CaseResult(
        case_name="c",
        inputs="What is the capital?",
        metadata=TestCaseMetadata(),
        base_input_key="c0",
        trace_id="",
        run_results=[rr],
        aggregated=AggregatedResult(
            passed=True, pass_rate=1.0, threshold=0.8, summary="ok", per_evaluator_pass_rates={"LLMJudge": 1.0}
        ),
    )
    eo = EvaluationOutput(runs=pd.DataFrame(), cases=pd.DataFrame(), case_results=[cr])
    restored = EvaluationOutput.from_json(eo.to_json())
    restored_reason = restored.case_results[0].run_results[0].assertions["LLMJudge"].reason
    assert restored_reason == combined_reason
    assert rubric in (restored_reason or "")


def test_judge_output_prompt_has_no_input_section():
    """``_build_prompt`` omits the ``<Input>`` section when inputs is None."""
    from ragpill.llm_judge import _build_prompt

    prompt = _build_prompt(output="the answer", rubric="must mention X")
    assert isinstance(prompt, str)
    assert "<Input>" not in prompt
    assert "<Output>" in prompt
    assert "<Rubric>" in prompt


def test_ssl_ca_cert_field():
    """ssl_ca_cert is exposed and defaults to None."""
    settings = LLMJudgeSettings(api_key="k", base_url="https://x")  # pyright: ignore[reportCallIssue]
    assert settings.ssl_ca_cert is None
    assert settings.ssl_verify is True

    settings2 = LLMJudgeSettings(  # pyright: ignore[reportCallIssue]
        api_key="k", base_url="https://x", ssl_ca_cert="/path/to/cert.pem", ssl_verify=False
    )
    assert settings2.ssl_ca_cert == "/path/to/cert.pem"
    assert settings2.ssl_verify is False
