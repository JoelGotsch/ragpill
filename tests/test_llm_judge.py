"""Test LLMJudge initialization via environment variables."""

import os
from unittest.mock import patch

from pydantic_ai.models.openai import OpenAIChatModel

from ragpill.evaluators import LLMJudge


def test_init_with_env_vars_sets_rubric():
    """LLMJudge(rubric=...) stores the rubric correctly when env vars are set."""

    FAKE_ENV = {
        "RAGPILL_LLMJUDGE_API_KEY": "test-api-key",
        "RAGPILL_LLMJUDGE_BASE_URL": "https://my-base-url.com/v1",
        "RAGPILL_LLMJUDGE_MODEL_NAME": "fake-test-model",
    }

    with patch.dict(os.environ, FAKE_ENV):
        judge = LLMJudge(rubric="whatever")
    assert judge.rubric == "whatever"
    assert isinstance(judge.model, OpenAIChatModel)
    assert judge.model.model_name == FAKE_ENV["RAGPILL_LLMJUDGE_MODEL_NAME"]
    assert judge.model.base_url.strip("/") == FAKE_ENV["RAGPILL_LLMJUDGE_BASE_URL"].strip("/")
