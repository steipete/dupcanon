from __future__ import annotations

from pathlib import Path

import pytest

from dupcanon.config import Settings, is_postgres_dsn, load_settings


def test_load_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "a"
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-token")
    monkeypatch.setenv("DUPCANON_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DUPCANON_LOG_LEVEL", "debug")
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("DUPCANON_EMBEDDING_DIM", "3072")
    monkeypatch.setenv("DUPCANON_EMBED_BATCH_SIZE", "64")
    monkeypatch.setenv("DUPCANON_EMBED_WORKER_CONCURRENCY", "3")
    monkeypatch.setenv("DUPCANON_JUDGE_PROVIDER", "GEMINI")
    monkeypatch.setenv("DUPCANON_JUDGE_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("DUPCANON_JUDGE_THINKING", "low")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_CHEAP_PROVIDER", "openrouter")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_CHEAP_MODEL", "minimax/minimax-m2.5")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_CHEAP_THINKING", "minimal")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_STRONG_PROVIDER", "openai")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_STRONG_MODEL", "gpt-5-mini")
    monkeypatch.setenv("DUPCANON_JUDGE_AUDIT_STRONG_THINKING", "high")
    monkeypatch.setenv("DUPCANON_JUDGE_WORKER_CONCURRENCY", "5")
    monkeypatch.setenv("DUPCANON_CANDIDATE_WORKER_CONCURRENCY", "6")

    settings = load_settings()

    assert settings.supabase_db_url == "postgresql://localhost/test"
    assert settings.gemini_api_key == "gemini-key"
    assert settings.openai_api_key == "openai-key"
    assert settings.openrouter_api_key == "openrouter-key"
    assert settings.github_token == "gh-token"
    assert settings.artifacts_dir == artifacts_dir
    assert settings.log_level == "DEBUG"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.embedding_dim == 3072
    assert settings.embed_batch_size == 64
    assert settings.embed_worker_concurrency == 3
    assert settings.judge_provider == "gemini"
    assert settings.judge_model == "gemini-3-flash-preview"
    assert settings.judge_thinking == "low"
    assert settings.judge_audit_cheap_provider == "openrouter"
    assert settings.judge_audit_cheap_model == "minimax/minimax-m2.5"
    assert settings.judge_audit_cheap_thinking == "minimal"
    assert settings.judge_audit_strong_provider == "openai"
    assert settings.judge_audit_strong_model == "gpt-5-mini"
    assert settings.judge_audit_strong_thinking == "high"
    assert settings.judge_worker_concurrency == 5
    assert settings.candidate_worker_concurrency == 6


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DUPCANON_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("DUPCANON_LOG_LEVEL", raising=False)
    monkeypatch.delenv("DUPCANON_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("DUPCANON_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("DUPCANON_EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("DUPCANON_EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DUPCANON_EMBED_WORKER_CONCURRENCY", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_THINKING", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_CHEAP_PROVIDER", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_CHEAP_THINKING", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_STRONG_PROVIDER", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_STRONG_MODEL", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_AUDIT_STRONG_THINKING", raising=False)
    monkeypatch.delenv("DUPCANON_JUDGE_WORKER_CONCURRENCY", raising=False)
    monkeypatch.delenv("DUPCANON_CANDIDATE_WORKER_CONCURRENCY", raising=False)

    settings = load_settings(dotenv_path=tmp_path / "no-default.env")

    assert settings.supabase_db_url is None
    assert settings.gemini_api_key is None
    assert settings.openai_api_key is None
    assert settings.openrouter_api_key is None
    assert settings.github_token is None
    assert str(settings.artifacts_dir) == ".local/artifacts"
    assert settings.log_level == "INFO"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.embedding_dim == 3072
    assert settings.embed_batch_size == 32
    assert settings.embed_worker_concurrency == 2
    assert settings.judge_provider == "openai-codex"
    assert settings.judge_model == "gpt-5.1-codex-mini"
    assert settings.judge_thinking is None
    assert settings.judge_audit_cheap_provider == "gemini"
    assert settings.judge_audit_cheap_model is None
    assert settings.judge_audit_cheap_thinking is None
    assert settings.judge_audit_strong_provider == "openai"
    assert settings.judge_audit_strong_model is None
    assert settings.judge_audit_strong_thinking is None
    assert settings.judge_worker_concurrency == 4
    assert settings.candidate_worker_concurrency == 4


def test_settings_accepts_field_names_for_tests_and_callers() -> None:
    settings = Settings(supabase_db_url="postgresql://localhost/test")

    assert settings.supabase_db_url == "postgresql://localhost/test"


def test_is_postgres_dsn() -> None:
    assert is_postgres_dsn("postgresql://localhost/db")
    assert is_postgres_dsn("postgres://localhost/db")
    assert is_postgres_dsn("postgresql://user:pass@[2600:1f18::1]:5432/postgres")
    assert not is_postgres_dsn("https://example.supabase.co")


def test_embedding_dim_locked_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUPCANON_EMBEDDING_DIM", "1024")

    with pytest.raises(ValueError):
        load_settings()


def test_embedding_provider_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "invalid")

    with pytest.raises(ValueError):
        load_settings()


def test_openai_embedding_provider_requires_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "gemini-embedding-001")

    with pytest.raises(ValueError):
        load_settings()


def test_judge_thinking_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUPCANON_JUDGE_THINKING", "turbo")

    with pytest.raises(ValueError):
        load_settings()
