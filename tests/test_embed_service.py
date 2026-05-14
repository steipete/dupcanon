from __future__ import annotations

import threading

import pytest
from rich.console import Console

import dupcanon.embed_service as embed_service
from dupcanon.config import load_settings
from dupcanon.logging_config import get_logger
from dupcanon.models import EmbeddingItem, ItemType, TypeFilter


def test_build_embedding_text_applies_v1_limits() -> None:
    title = "t" * 400
    body = "b" * 9000

    text = embed_service.build_embedding_text(title=title, body=body)

    assert len(text) <= 8000
    first_line, _, remaining = text.partition("\n\n")
    assert len(first_line) == 300
    assert len(remaining) <= 7700


def test_run_embed_only_changed_skips_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {"upserts": 0, "embedded_texts": []}

    class FakeDatabase:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        def get_repo_id(self, repo):
            return 42

        def list_items_for_embedding(self, *, repo_id: int, type_filter: TypeFilter, model: str):
            return [
                EmbeddingItem(
                    item_id=1,
                    type=ItemType.ISSUE,
                    number=1,
                    title="unchanged",
                    body="body",
                    content_hash="h1",
                    embedded_content_hash="h1",
                ),
                EmbeddingItem(
                    item_id=2,
                    type=ItemType.ISSUE,
                    number=2,
                    title="changed",
                    body="body",
                    content_hash="h2",
                    embedded_content_hash="old",
                ),
            ]

        def upsert_embedding(self, **kwargs):
            captured["upserts"] += 1
            captured["last_upsert"] = kwargs

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            captured["embedded_texts"].extend(texts)
            return [[0.1] * 3072 for _ in texts]

    monkeypatch.setattr(embed_service, "Database", FakeDatabase)
    monkeypatch.setattr(embed_service, "GeminiEmbeddingsClient", FakeClient)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "gemini-embedding-001")

    stats = embed_service.run_embed(
        settings=load_settings(dotenv_path=tmp_path / "no-default.env"),
        repo_value="org/repo",
        type_filter=TypeFilter.ISSUE,
        only_changed=True,
        console=Console(),
        logger=get_logger("test"),
    )

    assert stats.discovered == 2
    assert stats.queued == 1
    assert stats.embedded == 1
    assert stats.skipped_unchanged == 1
    assert stats.failed == 0
    assert captured["upserts"] == 1
    assert len(captured["embedded_texts"]) == 1


def test_run_embed_batch_failure_falls_back_to_single_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {"upserts": 0, "batch_calls": 0, "single_calls": 0}

    class FakeDatabase:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        def get_repo_id(self, repo):
            return 42

        def list_items_for_embedding(self, *, repo_id: int, type_filter: TypeFilter, model: str):
            return [
                EmbeddingItem(
                    item_id=1,
                    type=ItemType.ISSUE,
                    number=1,
                    title="a",
                    body="body",
                    content_hash="h1",
                    embedded_content_hash=None,
                ),
                EmbeddingItem(
                    item_id=2,
                    type=ItemType.ISSUE,
                    number=2,
                    title="b",
                    body="body",
                    content_hash="h2",
                    embedded_content_hash=None,
                ),
            ]

        def upsert_embedding(self, **kwargs):
            captured["upserts"] += 1

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            if len(texts) > 1:
                captured["batch_calls"] += 1
                raise RuntimeError("batch failed")

            captured["single_calls"] += 1
            return [[0.1] * 3072]

    monkeypatch.setattr(embed_service, "Database", FakeDatabase)
    monkeypatch.setattr(embed_service, "GeminiEmbeddingsClient", FakeClient)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "gemini-embedding-001")

    stats = embed_service.run_embed(
        settings=load_settings(dotenv_path=tmp_path / "no-default.env"),
        repo_value="org/repo",
        type_filter=TypeFilter.ISSUE,
        only_changed=False,
        console=Console(),
        logger=get_logger("test"),
    )

    assert stats.discovered == 2
    assert stats.queued == 2
    assert stats.embedded == 2
    assert stats.failed == 0
    assert captured["upserts"] == 2
    assert captured["batch_calls"] == 1
    assert captured["single_calls"] == 2


def test_run_embed_parallel_workers_keep_db_writes_on_main_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    main_thread_id = threading.get_ident()
    captured = {"upserts": 0, "embed_thread_ids": set()}

    class FakeDatabase:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        def get_repo_id(self, repo):
            return 42

        def list_items_for_embedding(self, *, repo_id: int, type_filter: TypeFilter, model: str):
            return [
                EmbeddingItem(
                    item_id=1,
                    type=ItemType.ISSUE,
                    number=1,
                    title="a",
                    body="body",
                    content_hash="h1",
                    embedded_content_hash=None,
                ),
                EmbeddingItem(
                    item_id=2,
                    type=ItemType.ISSUE,
                    number=2,
                    title="b",
                    body="body",
                    content_hash="h2",
                    embedded_content_hash=None,
                ),
            ]

        def upsert_embedding(self, **kwargs):
            assert threading.get_ident() == main_thread_id
            captured["upserts"] += 1

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            thread_ids = captured["embed_thread_ids"]
            assert isinstance(thread_ids, set)
            thread_ids.add(threading.get_ident())
            return [[0.1] * 3072 for _ in texts]

    monkeypatch.setattr(embed_service, "Database", FakeDatabase)
    monkeypatch.setattr(embed_service, "GeminiEmbeddingsClient", FakeClient)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("DUPCANON_EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("DUPCANON_EMBED_WORKER_CONCURRENCY", "2")

    stats = embed_service.run_embed(
        settings=load_settings(dotenv_path=tmp_path / "no-default.env"),
        repo_value="org/repo",
        type_filter=TypeFilter.ISSUE,
        only_changed=False,
        console=Console(),
        logger=get_logger("test"),
    )

    assert stats.embedded == 2
    assert stats.failed == 0
    assert captured["upserts"] == 2


def test_run_embed_openai_provider_works(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = {"upserts": 0, "provider_client_inits": []}

    class FakeDatabase:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        def get_repo_id(self, repo):
            return 42

        def list_items_for_embedding(self, *, repo_id: int, type_filter: TypeFilter, model: str):
            return [
                EmbeddingItem(
                    item_id=1,
                    type=ItemType.ISSUE,
                    number=1,
                    title="a",
                    body="body",
                    content_hash="h1",
                    embedded_content_hash=None,
                )
            ]

        def upsert_embedding(self, **kwargs):
            captured["upserts"] += 1

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            inits = captured["provider_client_inits"]
            assert isinstance(inits, list)
            inits.append(kwargs)

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 3072 for _ in texts]

    monkeypatch.setattr(embed_service, "Database", FakeDatabase)
    monkeypatch.setattr(embed_service, "OpenAIEmbeddingsClient", FakeOpenAIClient)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/db")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "text-embedding-3-large")

    stats = embed_service.run_embed(
        settings=load_settings(dotenv_path=tmp_path / "no-default.env"),
        repo_value="org/repo",
        type_filter=TypeFilter.ISSUE,
        only_changed=False,
        console=Console(),
        logger=get_logger("test"),
    )

    assert stats.embedded == 1
    assert captured["upserts"] == 1
    inits = captured["provider_client_inits"]
    assert isinstance(inits, list)
    assert len(inits) == 1
    assert inits[0]["model"] == "text-embedding-3-large"


def test_run_embed_openai_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DUPCANON_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DUPCANON_EMBEDDING_MODEL", "text-embedding-3-large")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        embed_service.run_embed(
            settings=load_settings(dotenv_path=tmp_path / "no-default.env"),
            repo_value="org/repo",
            type_filter=TypeFilter.ISSUE,
            only_changed=False,
            console=Console(),
            logger=get_logger("test"),
        )
