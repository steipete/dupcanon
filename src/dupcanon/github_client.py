from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from json import JSONDecodeError
from typing import Any, TypeVar
from urllib.parse import urlencode

from dupcanon.llm_retry import retry_delay_seconds, should_retry_http_status, validate_max_attempts
from dupcanon.models import (
    ItemPayload,
    ItemType,
    PullRequestFileChange,
    RepoMetadata,
    RepoRef,
    StateFilter,
)

_HTTP_STATUS_RE = re.compile(r"HTTP\s+(?P<code>\d{3})")
_T = TypeVar("_T")

_SEARCH_PER_PAGE = 100
_DEFAULT_FETCH_WORKERS = 3  # pages fetched concurrently per wave


class GitHubApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubNotFoundError(GitHubApiError):
    pass


def _parse_http_status(stderr: str) -> int | None:
    match = _HTTP_STATUS_RE.search(stderr)
    if match is None:
        return None
    return int(match.group("code"))


def _should_retry(status_code: int | None) -> bool:
    return should_retry_http_status(status_code)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _extract_labels(raw: Any) -> list[str]:
    labels: list[str] = []
    if not isinstance(raw, list):
        return labels

    for label in raw:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str) and name:
                labels.append(name)
        elif isinstance(label, str) and label:
            labels.append(label)

    return labels


class GitHubClient:
    def __init__(self, *, max_attempts: int = 5, fetch_workers: int = _DEFAULT_FETCH_WORKERS) -> None:
        validate_max_attempts(max_attempts)
        self.max_attempts = max_attempts
        self.fetch_workers = fetch_workers

    # ── Low-level gh CLI helpers ──────────────────────────────────────────

    def _gh_api(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        api_path = f"{path}?{query}" if query else path

        cmd = [
            "gh",
            "api",
            api_path,
            "--method",
            "GET",
            "-H",
            "Accept: application/vnd.github+json",
        ]

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )

            if proc.returncode == 0:
                return json.loads(proc.stdout)

            status_code = _parse_http_status(proc.stderr)
            message = proc.stderr.strip() or proc.stdout.strip() or "unknown gh api error"

            if status_code == 404:
                raise GitHubNotFoundError(message, status_code=status_code)

            error = GitHubApiError(message, status_code=status_code)
            last_error = error

            if attempt >= self.max_attempts or not _should_retry(status_code):
                raise error

            time.sleep(retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise GitHubApiError("unreachable gh api retry state")

    def _gh_api_paginated_collect(
        self,
        path: str,
        *,
        params: dict[str, Any] | None,
        row_mapper: Callable[[dict[str, Any]], _T | None],
        jq_expression: str = ".[]",
        on_batch_count: Callable[[int], None] | None = None,
    ) -> list[_T]:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        api_path = f"{path}?{query}" if query else path

        cmd = [
            "gh",
            "api",
            api_path,
            "--method",
            "GET",
            "-H",
            "Accept: application/vnd.github+json",
            "--paginate",
            "--jq",
            jq_expression,
        ]

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            mapped_rows: list[_T] = []
            pending = 0

            assert proc.stdout is not None
            for line in proc.stdout:
                raw = line.strip()
                if not raw:
                    continue

                try:
                    value = json.loads(raw)
                except JSONDecodeError as exc:
                    msg = "failed to decode paginated gh api object stream"
                    raise GitHubApiError(msg) from exc

                if not isinstance(value, dict):
                    continue

                mapped = row_mapper(value)
                if mapped is None:
                    continue

                mapped_rows.append(mapped)
                pending += 1
                if on_batch_count is not None and pending >= 100:
                    on_batch_count(pending)
                    pending = 0

            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()

            return_code = proc.wait()
            if return_code == 0:
                if on_batch_count is not None and pending > 0:
                    on_batch_count(pending)
                return mapped_rows

            status_code = _parse_http_status(stderr)
            message = stderr.strip() or "unknown gh api error"

            if status_code == 404:
                raise GitHubNotFoundError(message, status_code=status_code)

            error = GitHubApiError(message, status_code=status_code)
            last_error = error

            if attempt >= self.max_attempts or not _should_retry(status_code):
                raise error

            time.sleep(retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise GitHubApiError("unreachable gh api retry state")

    def _gh_graphql_paginated_collect(
        self,
        *,
        query: str,
        variables: dict[str, str],
        jq_expression: str,
        row_mapper: Callable[[dict[str, Any]], _T | None],
        on_batch_count: Callable[[int], None] | None = None,
    ) -> list[_T]:
        cmd = [
            "gh",
            "api",
            "graphql",
            "--paginate",
            "-f",
            f"query={query}",
            "--jq",
            jq_expression,
        ]
        for key in sorted(variables):
            cmd.extend(["-F", f"{key}={variables[key]}"])

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            mapped_rows: list[_T] = []
            pending = 0

            assert proc.stdout is not None
            for line in proc.stdout:
                raw = line.strip()
                if not raw:
                    continue

                try:
                    value = json.loads(raw)
                except JSONDecodeError as exc:
                    msg = "failed to decode paginated gh graphql object stream"
                    raise GitHubApiError(msg) from exc

                if not isinstance(value, dict):
                    continue

                mapped = row_mapper(value)
                if mapped is None:
                    continue

                mapped_rows.append(mapped)
                pending += 1
                if on_batch_count is not None and pending >= 100:
                    on_batch_count(pending)
                    pending = 0

            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()

            return_code = proc.wait()
            if return_code == 0:
                if on_batch_count is not None and pending > 0:
                    on_batch_count(pending)
                return mapped_rows

            status_code = _parse_http_status(stderr)
            message = stderr.strip() or "unknown gh graphql error"

            if status_code == 404:
                raise GitHubNotFoundError(message, status_code=status_code)

            error = GitHubApiError(message, status_code=status_code)
            last_error = error

            if attempt >= self.max_attempts or not _should_retry(status_code):
                raise error

            time.sleep(retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise GitHubApiError("unreachable gh graphql retry state")

    # ── Parallel REST API helpers ────────────────────────────────────────
    #
    # Uses standard REST endpoints (5000 req/hour) instead of search API
    # (30 req/min). Fetches pages in progressive parallel waves — fire N
    # pages at once, stop when any page returns < per_page items.

    def _fetch_rest_page(
        self,
        path: str,
        params: dict[str, Any],
        page_num: int,
        row_mapper: Callable[[dict[str, Any]], _T | None],
        item_filter: Callable[[dict[str, Any]], bool] | None = None,
    ) -> tuple[list[_T], int]:
        """Fetch one REST API page. Returns (mapped_items, raw_count)."""
        raw_items = self._gh_api(
            path,
            params={**params, "page": page_num, "per_page": _SEARCH_PER_PAGE},
        )
        if not isinstance(raw_items, list):
            return [], 0

        raw_count = len(raw_items)
        mapped: list[_T] = []
        for raw in raw_items:
            if item_filter is not None and not item_filter(raw):
                continue
            m = row_mapper(raw)
            if m is not None:
                mapped.append(m)
        return mapped, raw_count

    def _parallel_rest_collect(
        self,
        *,
        path: str,
        params: dict[str, Any],
        row_mapper: Callable[[dict[str, Any]], _T | None],
        item_filter: Callable[[dict[str, Any]], bool] | None = None,
        on_batch_count: Callable[[int], None] | None = None,
    ) -> list[_T]:
        """
        Progressive parallel REST API page fetching.

        Fetches pages in waves of N (= fetch_workers). Each wave fires N
        concurrent page requests. If ALL pages in a wave return per_page
        items, the next wave starts. Stops when any page returns fewer
        items (indicating end of data).

        Uses standard REST endpoints with 5000 req/hour rate limit — no
        search API abuse detection issues.
        """
        all_results: list[_T] = []
        lock = threading.Lock()
        wave_size = max(1, self.fetch_workers)
        current_start = 1

        with ThreadPoolExecutor(max_workers=wave_size) as executor:
            while True:
                pages = list(range(current_start, current_start + wave_size))
                found_end = False

                futures = {
                    executor.submit(
                        self._fetch_rest_page, path, params, p, row_mapper, item_filter
                    ): p
                    for p in pages
                }

                for future in as_completed(futures):
                    page_results, raw_count = future.result()
                    with lock:
                        all_results.extend(page_results)
                    if on_batch_count and page_results:
                        on_batch_count(len(page_results))
                    if raw_count < _SEARCH_PER_PAGE:
                        found_end = True

                if found_end:
                    break
                current_start += wave_size

        return all_results

    # ── Public fetch methods ──────────────────────────────────────────────

    def fetch_repo_metadata(self, repo: RepoRef) -> RepoMetadata:
        data = self._gh_api(f"repos/{repo.full_name()}")
        owner = data.get("owner") or {}
        return RepoMetadata(
            github_repo_id=int(data["id"]),
            org=str(owner.get("login") or repo.org),
            name=str(data.get("name") or repo.name),
        )

    def _since_created_qualifier(self, since: datetime) -> str:
        return since.astimezone(UTC).strftime("%Y-%m-%d")

    def _state_qualifier(self, state: StateFilter) -> str | None:
        if state == StateFilter.OPEN:
            return "is:open"
        if state == StateFilter.CLOSED:
            return "is:closed"
        return None

    def _issue_states_literal(self, state: StateFilter) -> str:
        if state == StateFilter.OPEN:
            return "[OPEN]"
        if state == StateFilter.CLOSED:
            return "[CLOSED]"
        return "[OPEN,CLOSED]"

    def _pr_states_literal(self, state: StateFilter) -> str:
        if state == StateFilter.OPEN:
            return "[OPEN]"
        if state == StateFilter.CLOSED:
            return "[CLOSED,MERGED]"
        return "[OPEN,CLOSED,MERGED]"

    def fetch_issues(
        self,
        *,
        repo: RepoRef,
        state: StateFilter,
        since: datetime | None,
        on_page_count: Callable[[int], None] | None = None,
    ) -> list[ItemPayload]:
        # REST API: /repos/{owner}/{repo}/issues returns issues + PRs mixed.
        # Filter out PRs client-side (entries with "pull_request" key).
        # Uses 5000 req/hour rate limit — no search API abuse detection.
        params: dict[str, Any] = {
            "state": state.value,
            "sort": "created",
            "direction": "asc",
        }
        if since is not None:
            params["since"] = since.isoformat()

        return self._parallel_rest_collect(
            path=f"repos/{repo.full_name()}/issues",
            params=params,
            row_mapper=self._to_issue_payload,
            item_filter=lambda raw: "pull_request" not in raw,
            on_batch_count=on_page_count,
        )

    def fetch_pulls(
        self,
        *,
        repo: RepoRef,
        state: StateFilter,
        since: datetime | None,
        on_page_count: Callable[[int], None] | None = None,
    ) -> list[ItemPayload]:
        # REST API: /repos/{owner}/{repo}/pulls returns only PRs.
        # 5000 req/hour rate limit — no search API abuse detection.
        params: dict[str, Any] = {
            "state": state.value,
            "sort": "created",
            "direction": "asc",
        }
        # Note: REST pulls endpoint doesn't have a "since" param.
        # For refresh with since, we fetch all and filter client-side.

        return self._parallel_rest_collect(
            path=f"repos/{repo.full_name()}/pulls",
            params=params,
            row_mapper=self._to_pr_payload,
            on_batch_count=on_page_count,
        )

    def fetch_item(self, *, repo: RepoRef, item_type: ItemType, number: int) -> ItemPayload:
        issue_like = self._gh_api(f"repos/{repo.full_name()}/issues/{number}")
        if item_type == ItemType.ISSUE:
            if "pull_request" in issue_like:
                msg = f"#{number} is a pull request, not an issue"
                raise GitHubNotFoundError(msg, status_code=404)
            return self._to_issue_payload(issue_like)

        pr = self._gh_api(f"repos/{repo.full_name()}/pulls/{number}")
        return self._to_pr_payload(pr, issue_like=issue_like)

    def fetch_pull_request_files(
        self, *, repo: RepoRef, number: int
    ) -> list[PullRequestFileChange]:
        def to_file_change(row: dict[str, Any]) -> PullRequestFileChange | None:
            filename = row.get("filename")
            if not isinstance(filename, str) or not filename:
                return None

            status = row.get("status")
            if not isinstance(status, str):
                status = None

            patch = row.get("patch")
            if not isinstance(patch, str):
                patch = None

            return PullRequestFileChange(path=filename, status=status, patch=patch)

        return self._gh_api_paginated_collect(
            f"repos/{repo.full_name()}/pulls/{number}/files",
            params={"per_page": 100},
            row_mapper=to_file_change,
            jq_expression=".[]",
        )

    def fetch_maintainers(self, *, repo: RepoRef) -> set[str]:
        def to_maintainer_login(row: dict[str, Any]) -> str | None:
            login = row.get("login")
            if not isinstance(login, str) or not login:
                return None

            permissions = row.get("permissions")
            if not isinstance(permissions, dict):
                return None

            if any(bool(permissions.get(key)) for key in ("admin", "maintain", "push")):
                return login
            return None

        maintainers = self._gh_api_paginated_collect(
            f"repos/{repo.full_name()}/collaborators",
            params={"affiliation": "all", "per_page": 100},
            row_mapper=to_maintainer_login,
            jq_expression=".[]",
        )

        return set(maintainers)

    def _run_gh_command_with_retry(self, cmd: list[str]) -> tuple[str, str]:
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )

            if proc.returncode == 0:
                return proc.stdout, proc.stderr

            status_code = _parse_http_status(proc.stderr)
            message = proc.stderr.strip() or proc.stdout.strip() or "unknown gh command error"
            error = GitHubApiError(message, status_code=status_code)
            last_error = error

            if attempt >= self.max_attempts or not _should_retry(status_code):
                raise error

            time.sleep(retry_delay_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise GitHubApiError("unreachable gh command retry state")

    def close_item_as_duplicate(
        self,
        *,
        repo: RepoRef,
        item_type: ItemType,
        number: int,
        canonical_number: int,
    ) -> dict[str, Any]:
        item_command = "issue" if item_type == ItemType.ISSUE else "pr"
        comment = (
            f"Closing as duplicate of #{canonical_number}. If this is incorrect, please contact us."
        )

        stdout, stderr = self._run_gh_command_with_retry(
            [
                "gh",
                item_command,
                "close",
                str(number),
                "--repo",
                repo.full_name(),
                "--comment",
                comment,
            ]
        )

        return {
            "status": "closed",
            "item_type": item_type.value,
            "number": number,
            "canonical_number": canonical_number,
            "comment": comment,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }

    # ── Row mappers ───────────────────────────────────────────────────────

    def _to_issue_payload_from_graphql(self, row: dict[str, Any]) -> ItemPayload:
        assignees = [
            str(node.get("login"))
            for node in (row.get("assignees") or {}).get("nodes", [])
            if isinstance(node, dict) and node.get("login")
        ]
        labels = [
            str(node.get("name"))
            for node in (row.get("labels") or {}).get("nodes", [])
            if isinstance(node, dict) and node.get("name")
        ]
        author = row.get("author") or {}

        raw_state = str(row.get("state") or "OPEN")
        state = StateFilter.OPEN if raw_state == "OPEN" else StateFilter.CLOSED

        return ItemPayload(
            type=ItemType.ISSUE,
            number=int(row["number"]),
            url=str(row.get("url") or ""),
            title=str(row.get("title") or ""),
            body=row.get("body"),
            state=state,
            author_login=author.get("login"),
            assignees=assignees,
            labels=labels,
            comment_count=int((row.get("comments") or {}).get("totalCount") or 0),
            review_comment_count=0,
            created_at_gh=_parse_datetime(row.get("createdAt")),
            updated_at_gh=_parse_datetime(row.get("updatedAt")),
            closed_at_gh=_parse_datetime(row.get("closedAt")),
        )

    def _to_pr_payload_from_graphql(self, row: dict[str, Any]) -> ItemPayload:
        assignees = [
            str(node.get("login"))
            for node in (row.get("assignees") or {}).get("nodes", [])
            if isinstance(node, dict) and node.get("login")
        ]
        labels = [
            str(node.get("name"))
            for node in (row.get("labels") or {}).get("nodes", [])
            if isinstance(node, dict) and node.get("name")
        ]
        author = row.get("author") or {}

        pr_state = str(row.get("state") or "OPEN")
        state = StateFilter.OPEN if pr_state == "OPEN" else StateFilter.CLOSED

        return ItemPayload(
            type=ItemType.PR,
            number=int(row["number"]),
            url=str(row.get("url") or ""),
            title=str(row.get("title") or ""),
            body=row.get("body"),
            state=state,
            author_login=author.get("login"),
            assignees=assignees,
            labels=labels,
            comment_count=int((row.get("comments") or {}).get("totalCount") or 0),
            review_comment_count=int((row.get("reviewThreads") or {}).get("totalCount") or 0),
            created_at_gh=_parse_datetime(row.get("createdAt")),
            updated_at_gh=_parse_datetime(row.get("updatedAt")),
            closed_at_gh=_parse_datetime(row.get("closedAt")),
        )

    def _to_issue_payload(self, row: dict[str, Any]) -> ItemPayload:
        assignees = [str(a.get("login")) for a in row.get("assignees", []) if isinstance(a, dict)]
        user = row.get("user") or {}

        return ItemPayload(
            type=ItemType.ISSUE,
            number=int(row["number"]),
            url=str(row.get("html_url") or ""),
            title=str(row.get("title") or ""),
            body=row.get("body"),
            state=StateFilter(str(row.get("state") or "open")),
            author_login=user.get("login"),
            assignees=assignees,
            labels=_extract_labels(row.get("labels")),
            comment_count=int(row.get("comments") or 0),
            review_comment_count=0,
            created_at_gh=_parse_datetime(row.get("created_at")),
            updated_at_gh=_parse_datetime(row.get("updated_at")),
            closed_at_gh=_parse_datetime(row.get("closed_at")),
        )

    def _to_pr_payload(
        self, row: dict[str, Any], *, issue_like: dict[str, Any] | None = None
    ) -> ItemPayload:
        assignees = [str(a.get("login")) for a in row.get("assignees", []) if isinstance(a, dict)]
        user = row.get("user") or {}

        issue_data = issue_like or {}
        labels_source = issue_data.get("labels") if issue_data else row.get("labels")

        return ItemPayload(
            type=ItemType.PR,
            number=int(row["number"]),
            url=str(row.get("html_url") or ""),
            title=str(row.get("title") or ""),
            body=row.get("body"),
            state=StateFilter(str(row.get("state") or "open")),
            author_login=user.get("login"),
            assignees=assignees,
            labels=_extract_labels(labels_source),
            comment_count=int(row.get("comments") or issue_data.get("comments") or 0),
            review_comment_count=int(row.get("review_comments") or 0),
            created_at_gh=_parse_datetime(row.get("created_at")),
            updated_at_gh=_parse_datetime(row.get("updated_at")),
            closed_at_gh=_parse_datetime(row.get("closed_at")),
        )
