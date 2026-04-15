from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLFlowSettings(BaseSettings):
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


class LLMJudgeSettings(BaseSettings):
    """All settings related to LLMJudge evaluator."""

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
