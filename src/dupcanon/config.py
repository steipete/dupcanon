from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dupcanon.judge_providers import normalize_judge_provider
from dupcanon.thinking import normalize_thinking_level


class Settings(BaseSettings):
    supabase_db_url: str | None = Field(default=None, validation_alias="SUPABASE_DB_URL")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    embedding_provider: str = Field(
        default="openai",
        validation_alias="DUPCANON_EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias="DUPCANON_EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=3072, validation_alias="DUPCANON_EMBEDDING_DIM")
    embed_batch_size: int = Field(default=32, validation_alias="DUPCANON_EMBED_BATCH_SIZE")
    embed_worker_concurrency: int = Field(
        default=2,
        validation_alias="DUPCANON_EMBED_WORKER_CONCURRENCY",
    )
    judge_provider: str = Field(default="openai-codex", validation_alias="DUPCANON_JUDGE_PROVIDER")
    judge_model: str = Field(
        default="gpt-5.1-codex-mini",
        validation_alias="DUPCANON_JUDGE_MODEL",
    )
    judge_thinking: str | None = Field(
        default=None,
        validation_alias="DUPCANON_JUDGE_THINKING",
    )
    judge_audit_cheap_provider: str = Field(
        default="gemini",
        validation_alias="DUPCANON_JUDGE_AUDIT_CHEAP_PROVIDER",
    )
    judge_audit_cheap_model: str | None = Field(
        default=None,
        validation_alias="DUPCANON_JUDGE_AUDIT_CHEAP_MODEL",
    )
    judge_audit_cheap_thinking: str | None = Field(
        default=None,
        validation_alias="DUPCANON_JUDGE_AUDIT_CHEAP_THINKING",
    )
    judge_audit_strong_provider: str = Field(
        default="openai",
        validation_alias="DUPCANON_JUDGE_AUDIT_STRONG_PROVIDER",
    )
    judge_audit_strong_model: str | None = Field(
        default=None,
        validation_alias="DUPCANON_JUDGE_AUDIT_STRONG_MODEL",
    )
    judge_audit_strong_thinking: str | None = Field(
        default=None,
        validation_alias="DUPCANON_JUDGE_AUDIT_STRONG_THINKING",
    )
    judge_worker_concurrency: int = Field(
        default=4,
        validation_alias="DUPCANON_JUDGE_WORKER_CONCURRENCY",
    )
    candidate_worker_concurrency: int = Field(
        default=4,
        validation_alias="DUPCANON_CANDIDATE_WORKER_CONCURRENCY",
    )
    artifacts_dir: Path = Field(
        default=Path(".local/artifacts"),
        validation_alias="DUPCANON_ARTIFACTS_DIR",
    )
    log_level: str = Field(default="INFO", validation_alias="DUPCANON_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, value: int) -> int:
        if value != 3072:
            msg = "DUPCANON_EMBEDDING_DIM must be 3072 in v1"
            raise ValueError(msg)
        return value

    @field_validator("embedding_provider")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gemini", "openai"}:
            msg = "DUPCANON_EMBEDDING_PROVIDER must be one of: gemini, openai"
            raise ValueError(msg)
        return normalized

    @field_validator(
        "judge_provider",
        "judge_audit_cheap_provider",
        "judge_audit_strong_provider",
    )
    @classmethod
    def normalize_judge_provider_setting(cls, value: str) -> str:
        return normalize_judge_provider(value, label="judge provider")

    @field_validator(
        "judge_thinking",
        "judge_audit_cheap_thinking",
        "judge_audit_strong_thinking",
    )
    @classmethod
    def normalize_judge_thinking(cls, value: str | None) -> str | None:
        return normalize_thinking_level(value)

    @field_validator(
        "embed_batch_size",
        "embed_worker_concurrency",
        "judge_worker_concurrency",
        "candidate_worker_concurrency",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            msg = "runtime concurrency/batch settings must be positive integers"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_embedding_provider_model_pair(self) -> Settings:
        if self.embedding_provider == "openai" and self.embedding_model.startswith("gemini-"):
            msg = (
                "DUPCANON_EMBEDDING_MODEL must be an OpenAI embedding model when "
                "DUPCANON_EMBEDDING_PROVIDER=openai"
            )
            raise ValueError(msg)
        return self


def load_settings(*, dotenv_path: str | Path | None = None) -> Settings:
    """Load settings from .env and environment variables using pydantic settings."""
    if dotenv_path is None:
        return Settings()

    settings_cls = cast(Any, Settings)
    return settings_cls(_env_file=dotenv_path)


def is_postgres_dsn(value: str | None) -> bool:
    if value is None:
        return False
    return value.startswith("postgresql://") or value.startswith("postgres://")


def postgres_dsn_help_text() -> str:
    return (
        "SUPABASE_DB_URL must be a Postgres DSN (postgresql://... or postgres://...), "
        "not your Supabase project HTTPS URL. "
        "If direct DB connections are unreachable on your network, prefer the Supabase IPv4 "
        "pooler DSN."
    )


def ensure_runtime_directories(settings: Settings) -> None:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
