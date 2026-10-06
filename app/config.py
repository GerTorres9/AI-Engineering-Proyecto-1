from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)
    redis_url: str = "redis://localhost:6379/0"
    api_token: SecretStr
    approval_token: SecretStr
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "tutor-preentrega-7"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_workspace_id: str | None = None
    worker_concurrency: int = Field(default=5, ge=1, le=32)
    job_timeout_seconds: int = Field(default=120, ge=30, le=3600)
    critical_output_tokens: int = Field(default=800, ge=1)

    @model_validator(mode="after")
    def validate_credentials(self):
        for secret in (self.api_token, self.approval_token, self.openai_api_key):
            if not secret.get_secret_value().strip():
                raise ValueError("Configura API_TOKEN, APPROVAL_TOKEN y OPENAI_API_KEY")
        if self.api_token.get_secret_value() == self.approval_token.get_secret_value():
            raise ValueError("API_TOKEN y APPROVAL_TOKEN deben ser distintos")
        if self.langsmith_tracing and (
            self.langsmith_api_key is None or not self.langsmith_api_key.get_secret_value().strip()
        ):
            raise ValueError("Configura LANGSMITH_API_KEY para el monitoreo activo")
        return self
