"""CI protocol controls: exhaustive files, fresh processes and failures."""
from types import SimpleNamespace

import pytest

from scripts import run_port_shard


def files(root, count):
    root.mkdir()
    paths = [root/f"test_{index:02d}.py" for index in range(count)]
    for path in paths:
        path.write_text("def test_placeholder(): pass\n")
    return paths


def invoke(monkeypatch, args, codes=None, *, fresh=True):
    calls = []
    args = ["--fresh-file-processes", *args] if fresh else args
    codes = iter(codes or [])
    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=next(codes, 0))
    monkeypatch.setattr(run_port_shard.sys, "argv", ["run_port_shard.py", *args])
    monkeypatch.setattr(run_port_shard.subprocess, "run", run)
    return run_port_shard.main(), calls


def assert_processes(calls, expected):
    assert [call[3] for call in calls] == [str(path) for path in expected]
    for call in calls:
        assert call[:3] == [run_port_shard.sys.executable, "-m", "pytest"]
        assert call[4:] == ["-q", "--tb=short", "--timeout=900", "-p", "no:cacheprovider"]


def test_whole_directory_runs_every_file_in_a_separate_process(tmp_path, monkeypatch):
    expected = files(tmp_path/"port", 4)
    status, calls = invoke(monkeypatch, [str(tmp_path/"port")])
    assert status == 0
    assert_processes(calls, expected)


def test_modulo_shards_preserve_disjoint_exhaustive_selection(tmp_path, monkeypatch):
    expected = files(tmp_path/"port", 7)
    observed = []
    for part in range(3):
        status, calls = invoke(monkeypatch, ["--part", str(part), "--parts", "3", str(tmp_path/"port")])
        assert status == 0
        assert_processes(calls, expected[part::3])
        observed.extend(call[3] for call in calls)
    assert sorted(observed) == [str(path) for path in expected]
    assert len(set(observed)) == len(observed)


def test_failure_propagates_and_later_files_still_run(tmp_path, monkeypatch):
    expected = files(tmp_path/"port", 3)
    status, calls = invoke(monkeypatch, [str(tmp_path/"port")], [1, 0, 2])
    assert status == 1
    assert_processes(calls, expected)


def test_overlapping_directories_are_rejected_before_running(tmp_path, monkeypatch):
    files(tmp_path/"port", 1)
    files(tmp_path/"port"/"nested", 1)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, [str(tmp_path/"port"), str(tmp_path/"port"/"nested")])
    assert error.value.code == 2


def test_empty_partition_is_rejected(tmp_path, monkeypatch):
    files(tmp_path/"port", 1)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--parts", "2", str(tmp_path/"port")])
    assert error.value.code == 2


def test_default_preserves_single_process_directory_invocation(tmp_path, monkeypatch):
    files(tmp_path/"port", 3)
    status, calls = invoke(monkeypatch, [str(tmp_path/"port")], fresh=False)
    assert status == 0 and len(calls) == 1
    assert_processes(calls, [tmp_path/"port"])
