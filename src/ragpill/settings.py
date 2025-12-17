from pydantic import Field, SecretStr
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

    Example:
        ```python
        from ragpill.settings import LLMJudgeSettings

        settings = LLMJudgeSettings(
            model_name="gpt-4o",
            base_url="https://my-proxy.example.com/v1",
            api_key="sk-...",
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
