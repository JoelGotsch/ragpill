from typing import Any

from pydantic import Field, PrivateAttr, SecretStr
from pydantic_ai import models
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLFlowSettings(BaseSettings):
    """MLflow connection and evaluation settings.

    Controls where evaluation results are logged and the default repeat/threshold
    behaviour for multi-run evaluations. All fields can be set via environment
    variables with the ``MLFLOW_`` prefix (e.g. ``MLFLOW_RAGPILL_TRACKING_URI``).

    Example:
        ```python
        from ragpill.settings import MLFlowSettings

        settings = MLFlowSettings(
            ragpill_tracking_uri="http://mlflow.internal:5000",
            ragpill_experiment_name="my_evaluation",
        )
        ```
    """

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    ragpill_tracking_uri: str = Field("http://localhost:5000", description="MLFlow tracking server URI.")
    ragpill_experiment_name: str = Field("ragpill_experiment", description="MLFlow experiment name.")
    tracking_username: str | None = Field(
        None,
        description="Optional for dev, but should be used in prod. Username for MLFlow authentication, the env variable needs to be MLFLOW_TRACKING_USERNAME because mlflow expects this env variable directly.",
    )
    tracking_password: SecretStr | None = Field(
        None,
        description="Optional for dev, but should be used in prod. Password for MLFlow authentication, the env variable needs to be MLFLOW_TRACKING_PASSWORD because mlflow expects this env variable directly.",
    )
    ragpill_run_description: str = Field("RAGPill Evaluation Run", description="Description for the MLFlow run.")
    ragpill_repeat: int = Field(
        default=1,
        ge=1,
        description="Default number of times to run each test case. Per-case overrides via TestCaseMetadata.repeat take precedence. Env: MLFLOW_RAGPILL_REPEAT.",
    )
    ragpill_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Default minimum fraction of runs that must pass for a case to be considered passing. Per-case overrides via TestCaseMetadata.threshold take precedence. Env: MLFLOW_RAGPILL_THRESHOLD.",
    )


class LLMJudgeSettings(BaseSettings):
    """Configuration for the LLMJudge evaluator's backing LLM.

    Controls which model, temperature, and API endpoint the
    [`LLMJudge`][ragpill.evaluators.LLMJudge] evaluator uses. All fields can be
    set via environment variables with the ``RAGPILL_LLMJUDGE_`` prefix.

    This class supports a singleton pattern via
    [`get_llm_judge_settings`][ragpill.settings.get_llm_judge_settings] and
    [`configure_llm_judge`][ragpill.settings.configure_llm_judge]. The singleton
    caches a ``pydantic_ai.models.Model`` instance so that custom ``httpx`` / SSL
    configuration (e.g. corporate CA bundles) only needs to be set up once.

    Example:
        ```python
        from ragpill.settings import LLMJudgeSettings, configure_llm_judge

        settings = LLMJudgeSettings(
            model_name="gpt-4o",
            base_url="https://my-proxy.example.com/v1",
            api_key="sk-...",
        )

        # Or configure the singleton with custom SSL handling
        configure_llm_judge(
            api_key="sk-...",
            base_url="https://my-proxy/v1",
            model_name="gpt-4o",
            ssl_ca_cert="/path/to/custom-ca-bundle.pem",
        )
        ```
    """

    model_config = SettingsConfigDict(env_prefix="RAGPILL_LLMJUDGE_")

    model_name: str = Field("gpt-4o", description="Model name for LLMJudge evaluator.")
    temperature: float = Field(0.0, description="Temperature setting for LLMJudge model.")
    base_url: str | None = Field(
        None,
        description="Base URL for the LLM API. If None, the default openai base-url is used (OPENAI_BASE_URL if set, otherwise https://api.openai.com/v1).",
    )
    api_key: SecretStr | None = Field(
        None,
        description="API key for the LLM service. If None, no API key is used. If None, the default of openai is used, usually env variable OPENAI_API_KEY",
    )
    ssl_ca_cert: str | None = Field(
        None,
        description="Path to a custom CA certificate bundle for SSL verification. Passed as httpx 'verify' parameter.",
    )
    ssl_verify: bool = Field(
        True,
        description="Whether to verify SSL certificates. Ignored when ssl_ca_cert is set.",
    )

    _cached_model: Any = PrivateAttr(default=None)

    @property
    def llm_model(self) -> models.Model:
        """Lazily build and cache a ``pydantic_ai.models.Model`` from the current settings.

        The model is created on first access and reused on subsequent calls.
        Use :meth:`set_model` to inject a fully custom model instance instead.
        """
        if self._cached_model is None:
            from ragpill.utils import _get_pydantic_ai_llm_model  # pyright: ignore[reportPrivateUsage]

            if not self.api_key or not self.base_url or not self.model_name:
                raise ValueError(
                    "LLMJudgeSettings must have api_key, base_url, and model_name set to get default LLM model. "
                    "Set them via environment variables RAGPILL_LLMJUDGE_API_KEY, RAGPILL_LLMJUDGE_BASE_URL, "
                    "and RAGPILL_LLMJUDGE_MODEL_NAME respectively."
                )
            self._cached_model = _get_pydantic_ai_llm_model(
                base_url=self.base_url,
                api_key=self.api_key.get_secret_value(),
                model_name=self.model_name,
                temperature=self.temperature,
                ssl_ca_cert=self.ssl_ca_cert,
                ssl_verify=self.ssl_verify,
            )
        return self._cached_model

    def set_model(self, model: models.Model) -> None:
        """Override the cached model with a fully custom instance.

        Useful when you need full control over the ``httpx`` client or provider
        (e.g. custom middleware, mTLS, retry policies).
        """
        self._cached_model = model


_llm_judge_settings: LLMJudgeSettings | None = None


def get_llm_judge_settings() -> LLMJudgeSettings:
    """Return the global :class:`LLMJudgeSettings` singleton.

    Creates a default instance (reading from env vars) on first call.
    Use :func:`configure_llm_judge` to set up custom values before first use.
    """
    global _llm_judge_settings
    if _llm_judge_settings is None:
        _llm_judge_settings = LLMJudgeSettings()  # pyright: ignore[reportCallIssue]
    return _llm_judge_settings


def configure_llm_judge(
    settings: LLMJudgeSettings | None = None,
    **kwargs: Any,
) -> LLMJudgeSettings:
    """Create or replace the global :class:`LLMJudgeSettings` singleton.

    Pass either a pre-built ``settings`` instance or keyword arguments that will
    be forwarded to the ``LLMJudgeSettings`` constructor.

    Returns the newly configured singleton.
    """
    global _llm_judge_settings
    if settings is not None:
        _llm_judge_settings = settings
    else:
        _llm_judge_settings = LLMJudgeSettings(**kwargs)
    return _llm_judge_settings


def reset_llm_judge_settings() -> None:
    """Reset the global singleton (mainly useful for testing)."""
    global _llm_judge_settings
    _llm_judge_settings = None
