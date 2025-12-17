from typing import Any

from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_ai import models
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_pydantic_ai_llm_model(
    base_url: str | None,
    api_key: str | None,
    model_name: str,
    temperature: float = 0.0,
    ssl_ca_cert: str | None = None,
    ssl_verify: bool = True,
) -> models.Model:
    """Build a pydantic-ai model from the given connection settings.

    Args:
        ssl_ca_cert: Path to a custom CA certificate bundle. When set, this is
            passed as the ``verify`` parameter to ``httpx.AsyncClient``.
        ssl_verify: Whether to verify SSL certificates. Ignored when
            *ssl_ca_cert* is provided.
    """
    verify: str | bool = ssl_ca_cert if ssl_ca_cert else ssl_verify
    http_client = AsyncClient(verify=verify)
    openai_client = AsyncOpenAI(max_retries=3, base_url=base_url, api_key=api_key, http_client=http_client)
    return OpenAIChatModel(
        model_name, provider=OpenAIProvider(openai_client=openai_client), settings={"temperature": temperature}
    )


class RagpillTraceSettings(BaseSettings):
    """Trace-ingestion settings for :func:`ragpill.trace.parse_otel`.

    Controls how externally-produced OTLP traces are normalised into the
    vendor-neutral model. Does not affect the live ``execute_dataset`` capture
    path, which is MLflow-native via ``ragpill.trace.from_mlflow_trace``. All
    fields are settable via environment variables with the ``RAGPILL_TRACE_``
    prefix.

    Example:
        ```python
        from ragpill.settings import RagpillTraceSettings

        settings = RagpillTraceSettings(dialect="openinference")
        ```
    """

    model_config = SettingsConfigDict(env_prefix="RAGPILL_TRACE_", env_file=".env", extra="ignore")

    dialect: str = Field(
        "auto",
        description="Dialect for parse_otel: 'auto' to detect per span, or an adapter name "
        "('mlflow', 'openinference', 'gen_ai'). Env: RAGPILL_TRACE_DIALECT.",
    )
    fallback_dialect: str = Field(
        "gen_ai",
        description="Adapter used when 'auto' detection matches nothing for a span. "
        "Env: RAGPILL_TRACE_FALLBACK_DIALECT.",
    )


class TrackingSettings(BaseSettings):
    """Backend-neutral tracking + evaluation settings.

    Controls where evaluation results are logged and the default repeat/threshold
    behaviour for multi-run evaluations. Consumed by whichever tracking backend
    is configured (MLflow by default; Langfuse / Phoenix when registered). All
    fields can be set via environment variables with the ``RAGPILL_`` prefix
    (e.g. ``RAGPILL_TRACKING_URI``, ``RAGPILL_REPEAT``).

    For MLflow auth, set ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``
    in the environment — mlflow reads those directly.

    Example:
        ```python
        from ragpill.settings import TrackingSettings

        settings = TrackingSettings(
            tracking_uri="http://mlflow.internal:5000",
            experiment_name="my_evaluation",
        )
        ```
    """

    model_config = SettingsConfigDict(env_prefix="RAGPILL_", env_file=".env", extra="ignore")

    tracking_uri: str | None = Field(
        None,
        description="Tracking server URI. None (default) uses a private temp SQLite store "
        "for MLflow (zero-server) or the backend's own env-derived destination for remote "
        "backends. Env: RAGPILL_TRACKING_URI.",
    )
    experiment_name: str = Field(
        "ragpill_experiment", description="Experiment / project name. Env: RAGPILL_EXPERIMENT_NAME."
    )
    run_description: str = Field(
        "RAGPill Evaluation Run", description="Description for the run. Env: RAGPILL_RUN_DESCRIPTION."
    )
    repeat: int = Field(
        default=1,
        ge=1,
        description="Default number of times to run each test case. Per-case overrides via TestCaseMetadata.repeat take precedence. Env: RAGPILL_REPEAT.",
    )
    threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Default minimum fraction of runs that must pass for a case to be considered passing. Per-case overrides via TestCaseMetadata.threshold take precedence. Env: RAGPILL_THRESHOLD.",
    )
    trace_fetch_timeout_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Max seconds to poll for a trace to be exported before giving up when attaching traces to evaluator context. Backends flush spans asynchronously, so a too-short budget leaves SpanBaseEvaluators without a trace. Env: RAGPILL_TRACE_FETCH_TIMEOUT_S.",
    )
    trace_fetch_poll_interval_s: float = Field(
        default=0.5,
        gt=0.0,
        description="Interval in seconds between trace-readiness polls within the trace-fetch timeout. Env: RAGPILL_TRACE_FETCH_POLL_INTERVAL_S.",
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

    model_config = SettingsConfigDict(env_prefix="RAGPILL_LLMJUDGE_", env_file=".env", extra="ignore")

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
            # base_url / api_key are optional: when unset, the OpenAI client
            # resolves them from OPENAI_BASE_URL / OPENAI_API_KEY (the documented
            # default path). Only a fully-missing API key fails — and that error
            # is raised by the OpenAI client at construction, not pre-empted here.
            self._cached_model = _get_pydantic_ai_llm_model(
                base_url=self.base_url,
                api_key=self.api_key.get_secret_value() if self.api_key else None,
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
