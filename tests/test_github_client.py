from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

import dupcanon.github_client as github_client
from dupcanon.github_client import (
    GitHubClient,
    _extract_labels,
    _parse_datetime,
    _parse_http_status,
    _should_retry,
)
from dupcanon.models import ItemType, RepoRef, StateFilter


def test_parse_http_status_extracts_code() -> None:
    assert _parse_http_status("gh: HTTP 502 Bad Gateway") == 502


def test_github_client_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        GitHubClient(max_attempts=0)


def test_parse_http_status_returns_none_when_absent() -> None:
    assert _parse_http_status("some other error") is None


def test_should_retry_rules() -> None:
    assert _should_retry(None)
    assert _should_retry(429)
    assert _should_retry(500)
    assert _should_retry(503)
    assert not _should_retry(400)


def test_extract_labels_handles_mixed_formats() -> None:
    labels = _extract_labels([{"name": "bug"}, "help wanted", {"name": ""}, 123])

    assert labels == ["bug", "help wanted"]


def test_parse_datetime_none() -> None:
    assert _parse_datetime(None) is None


def test_parse_datetime_utc() -> None:
    parsed = _parse_datetime("2026-02-13T10:00:00Z")

    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed.year == 2026


def test_gh_api_paginated_collect_batches_and_flushes(monkeypatch) -> None:
    class _LineStream:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines

        def __iter__(self) -> _LineStream:
            return self

        def __next__(self) -> str:
            if not self.lines:
                raise StopIteration
            return self.lines.pop(0)

    class _ErrStream:
        def read(self) -> str:
            return ""

    class _Proc:
        def __init__(self, line_count: int) -> None:
            self.stdout = _LineStream([f'{{"id":{i}}}\n' for i in range(1, line_count + 1)])
            self.stderr = _ErrStream()

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        github_client.subprocess,
        "Popen",
        lambda *args, **kwargs: _Proc(205),
    )

    batches: list[int] = []
    client = GitHubClient(max_attempts=1)
    rows = client._gh_api_paginated_collect(
        "repos/org/repo/issues",
        params={"state": "all"},
        row_mapper=lambda row: row,
        on_batch_count=batches.append,
    )

    assert len(rows) == 205
    assert batches == [100, 100, 5]


def test_fetch_issues_with_since_uses_rest_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rest_collect(
        self, *, path, params, row_mapper, item_filter=None, on_batch_count=None
    ):
        captured["path"] = path
        captured["params"] = params
        captured["has_filter"] = item_filter is not None
        rows = [
            {
                "number": 1,
                "html_url": "https://example.test/1",
                "title": "old",
                "state": "open",
                "created_at": "2026-02-12T23:59:59Z",
            },
            {
                "number": 2,
                "html_url": "https://example.test/2",
                "title": "new",
                "state": "open",
                "created_at": "2026-02-13T00:00:00Z",
            },
        ]
        return [
            mapped
            for row in rows
            if item_filter is None or item_filter(row)
            if (mapped := row_mapper(row)) is not None
        ]

    monkeypatch.setattr(GitHubClient, "_parallel_rest_collect", fake_rest_collect)

    client = GitHubClient(max_attempts=1)
    result = client.fetch_issues(
        repo=RepoRef.parse("org/repo"),
        state=StateFilter.ALL,
        since=datetime(2026, 2, 13, tzinfo=UTC),
    )

    assert [item.number for item in result] == [2]
    assert captured["path"] == "repos/org/repo/issues"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["state"] == "all"
    assert params["sort"] == "created"
    assert params["direction"] == "asc"
    assert "since" in params
    assert captured["has_filter"]  # filters out PRs


def test_fetch_issues_without_since_uses_rest_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rest_collect(
        self, *, path, params, row_mapper, item_filter=None, on_batch_count=None
    ):
        captured["path"] = path
        captured["params"] = params
        captured["has_filter"] = item_filter is not None
        return []

    monkeypatch.setattr(GitHubClient, "_parallel_rest_collect", fake_rest_collect)

    client = GitHubClient(max_attempts=1)
    client.fetch_issues(
        repo=RepoRef.parse("org/repo"),
        state=StateFilter.OPEN,
        since=None,
    )

    assert captured["path"] == "repos/org/repo/issues"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["state"] == "open"
    assert "since" not in params
    assert captured["has_filter"]  # filters out PRs


def test_fetch_pulls_without_since_uses_rest_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rest_collect(
        self, *, path, params, row_mapper, item_filter=None, on_batch_count=None
    ):
        captured["path"] = path
        captured["params"] = params
        captured["has_filter"] = item_filter is not None
        return []

    monkeypatch.setattr(GitHubClient, "_parallel_rest_collect", fake_rest_collect)

    client = GitHubClient(max_attempts=1)
    client.fetch_pulls(
        repo=RepoRef.parse("org/repo"),
        state=StateFilter.CLOSED,
        since=None,
    )

    assert captured["path"] == "repos/org/repo/issues"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["state"] == "closed"
    assert captured["has_filter"]  # filters to PRs from issue-like endpoint


def test_fetch_pulls_with_since_uses_rest_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rest_collect(self, *, path, params, row_mapper, item_filter=None, on_batch_count=None):
        captured["path"] = path
        captured["params"] = params
        captured["has_filter"] = item_filter is not None
        rows = [
            {
                "number": 1,
                "html_url": "https://example.test/pull/1",
                "title": "old",
                "state": "open",
                "created_at": "2026-02-12T23:59:59Z",
                "pull_request": {},
            },
            {
                "number": 2,
                "html_url": "https://example.test/pull/2",
                "title": "new",
                "state": "open",
                "created_at": "2026-02-13T00:00:00Z",
                "pull_request": {},
                "labels": [{"name": "bug"}],
                "comments": 3,
            },
        ]
        return [
            mapped
            for row in rows
            if item_filter is None or item_filter(row)
            if (mapped := row_mapper(row)) is not None
        ]

    monkeypatch.setattr(GitHubClient, "_parallel_rest_collect", fake_rest_collect)

    client = GitHubClient(max_attempts=1)
    result = client.fetch_pulls(
        repo=RepoRef.parse("org/repo"),
        state=StateFilter.OPEN,
        since=datetime(2026, 2, 13, tzinfo=UTC),
    )

    assert [item.number for item in result] == [2]
    assert result[0].labels == ["bug"]
    assert result[0].comment_count == 3
    assert captured["path"] == "repos/org/repo/issues"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["state"] == "open"
    assert "since" in params
    assert captured["has_filter"]


def test_parallel_rest_collect_preserves_page_order(monkeypatch) -> None:
    def fake_fetch_rest_page(self, path, params, page_num, row_mapper, item_filter=None):
        if page_num == 1:
            time.sleep(0.02)
        return [page_num], 0 if page_num == 3 else 100

    monkeypatch.setattr(GitHubClient, "_fetch_rest_page", fake_fetch_rest_page)

    client = GitHubClient(max_attempts=1, fetch_workers=3)
    rows = client._parallel_rest_collect(
        path="repos/org/repo/issues",
        params={"state": "all"},
        row_mapper=lambda row: row,
    )

    assert rows == [1, 2, 3]


def test_fetch_pull_request_files_uses_paginated_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_collect(self, path, *, params, row_mapper, jq_expression=".[]", on_batch_count=None):
        captured["path"] = path
        captured["params"] = params
        captured["jq_expression"] = jq_expression

        rows = [
            {
                "filename": "src/main.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-old\n+new",
            },
            {
                "filename": "assets/logo.png",
                "status": "modified",
            },
        ]
        return [value for row in rows if (value := row_mapper(row)) is not None]

    monkeypatch.setattr(GitHubClient, "_gh_api_paginated_collect", fake_collect)

    client = GitHubClient(max_attempts=1)
    files = client.fetch_pull_request_files(repo=RepoRef.parse("org/repo"), number=12)

    assert len(files) == 2
    assert files[0].path == "src/main.py"
    assert files[0].patch is not None
    assert files[1].path == "assets/logo.png"
    assert files[1].patch is None
    assert captured["path"] == "repos/org/repo/pulls/12/files"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["per_page"] == 100
    assert captured["jq_expression"] == ".[]"


def test_fetch_maintainers_filters_by_permissions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_collect(self, path, *, params, row_mapper, jq_expression=".[]", on_batch_count=None):
        captured["path"] = path
        captured["params"] = params
        captured["jq_expression"] = jq_expression

        rows = [
            {"login": "alice", "permissions": {"admin": True}},
            {"login": "bob", "permissions": {"maintain": True}},
            {"login": "carol", "permissions": {"push": True}},
            {"login": "dave", "permissions": {"triage": True}},
            {"login": "eve", "permissions": {"pull": True}},
            {"login": "", "permissions": {"admin": True}},
        ]
        return [login for row in rows if (login := row_mapper(row)) is not None]

    monkeypatch.setattr(GitHubClient, "_gh_api_paginated_collect", fake_collect)

    client = GitHubClient(max_attempts=1)
    maintainers = client.fetch_maintainers(repo=RepoRef.parse("org/repo"))

    assert maintainers == {"alice", "bob", "carol"}
    assert captured["path"] == "repos/org/repo/collaborators"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["affiliation"] == "all"
    assert params["per_page"] == 100
    assert captured["jq_expression"] == ".[]"


def test_close_item_as_duplicate_uses_issue_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = "closed"
        stderr = ""

    def fake_run(cmd, *, check, capture_output, text):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(github_client.subprocess, "run", fake_run)

    client = GitHubClient(max_attempts=1)
    result = client.close_item_as_duplicate(
        repo=RepoRef.parse("org/repo"),
        item_type=ItemType.ISSUE,
        number=42,
        canonical_number=7,
    )

    assert result["status"] == "closed"
    assert result["item_type"] == "issue"
    assert result["number"] == 42

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["gh", "issue", "close", "42"]
    assert "--repo" in cmd
    assert "org/repo" in cmd
    assert "--comment" in cmd
    assert "#7" in cmd[-1]
