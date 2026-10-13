"""Persist per-test start/finish records for diagnosing slow core shards."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _record(event: str, nodeid: str | None = None, duration_seconds: float | None = None) -> None:
    root = Path(os.environ.get("PYTEST_PROGRESS_DIR", ".pytest-progress"))
    root.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    path = root / f"{worker}.jsonl"
    row = {
        "event": event,
        "nodeid": nodeid,
        "time_unix": time.time(),
        "pid": os.getpid(),
    }
    if duration_seconds is not None:
        row["duration_seconds"] = duration_seconds
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def pytest_configure(config) -> None:
    _record("session-configured")


try:
    import pytest
except ImportError:  # pragma: no cover - pytest loads this plugin only in CI
    pytest = None


if pytest is not None:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(item, nextitem):
        start = time.monotonic()
        _record("test-start", item.nodeid)
        try:
            yield
        finally:
            _record("test-finish", item.nodeid, time.monotonic() - start)
