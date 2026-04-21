"""Test LLMJudge initialization via environment variables."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from ragpill.evaluators import LLMJudge
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
