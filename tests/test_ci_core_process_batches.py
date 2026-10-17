"""CI process boundaries retain exhaustive selections and fail closed."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_core_shard as core


def test_current_mirror_selections_match_actual_pytest_collection():
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-o', 'addopts=', '--collect-only', '-q', '--color=no',
         core.MIRROR_FILE], capture_output=True, text=True, check=True,
        env=dict(os.environ, PYTEST_ADDOPTS=''))
    collected = [line.strip() for line in result.stdout.splitlines()
                 if line.startswith(core.MIRROR_FILE+'::')]
    assert sorted(collected) == sorted(core.mirror_selections())


def test_inventory_partitions_every_selection_once():
    selections = core.inventory()
    groups = core.partition(selections)
    assert sorted(unit for group in groups for unit in group) == selections
    assert len(selections) == len(set(selections))
    mirrors = set(core.mirror_selections())
    assert core.MIRROR_FILE not in selections
    # The appended tail of Core6 has no mirrors. They remain distributed
    # over the original eight lanes rather than concentrated in one job.
    assert all(mirrors.intersection(group) for group in groups[:core.BASE_PARTS])
    for group in groups:
        pieces = core.batches(group)
        assert [unit for piece in pieces for unit in piece] == group
        assert all(0 < len(piece) <= core.BATCH_SIZE for piece in pieces)


@pytest.mark.parametrize('change', ['extra-test', 'duplicate-module', 'dynamic-list'])
def test_mirror_inventory_rejects_unaccounted_source_changes(tmp_path, monkeypatch, change):
    path = tmp_path / core.MIRROR_FILE
    path.parent.mkdir(parents=True)
    source = "MODULES = ['tests.test_one']\ndef test_port_mirror(modname): pass\n"
    if change == 'extra-test':
        source += 'def test_new(): pass\n'
    elif change == 'duplicate-module':
        source = source.replace("['tests.test_one']", "['tests.test_one', 'tests.test_one']")
    else:
        source = source.replace("['tests.test_one']", 'make_modules()')
    path.write_text(source)
    monkeypatch.chdir(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        core.mirror_selections()


def _fake_processes(monkeypatch, failed_batch=None):
    from coverage import CoverageData

    calls = []

    def run(command, env=None):
        calls.append(command)
        if command[2] == 'pytest':
            batch = len(calls)-1
            assert '-n' not in command and '--timeout=600' in command
            assert '--cov=chebfunjax' in command and '--cov-fail-under=0' in command
            data = CoverageData(basename=env['COVERAGE_FILE'])
            data.add_lines({str(Path(f'source_{batch}.py').resolve()): {batch+1}})
            data.write()
            return subprocess.CompletedProcess(command, 1 if batch == failed_batch else 0)
        assert command[2:4] == ['coverage', 'combine']
        output = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--data-file='))
        data = CoverageData(basename=output)
        for path in command[6:]:
            data.update(core.read_coverage(Path(path)))
        data.write()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(core.subprocess, 'run', run)
    return calls


def test_fresh_processes_combine_only_complete_successful_batches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    calls = _fake_processes(monkeypatch)
    selections = [f'tests/test_{i}.py' for i in range(core.BATCH_SIZE+1)]
    manifest = Path('core-manifest-0.json')
    assert core.run_batches(0, selections, manifest) == 0
    assert len(calls) == 3
    data = json.loads(manifest.read_text())
    core.validate_manifest(data, 0, selections, 'a'*40, tmp_path)


def test_failed_batch_cannot_be_combined_or_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = _fake_processes(monkeypatch, failed_batch=0)
    selections = [f'tests/test_{i}.py' for i in range(core.BATCH_SIZE+1)]
    manifest = Path('core-manifest-0.json')
    assert core.run_batches(0, selections, manifest) == 1
    assert len(calls) == 2
    assert json.loads(manifest.read_text())['returncode'] == 1
    assert not Path('.coverage.core-0').exists()


def test_corrupt_batch_coverage_cannot_be_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(command, env=None):
        assert command[2] == 'pytest'
        Path(env['COVERAGE_FILE']).write_bytes(b'not a coverage database')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(core.subprocess, 'run', run)
    assert core.run_batches(0, ['tests/test_one.py'], Path('manifest.json')) == 1
    assert not Path('.coverage.core-0').exists()


def test_union_check_rejects_silently_skipped_valid_input(tmp_path):
    from coverage import CoverageData

    paths = [tmp_path/'first', tmp_path/'second']
    for index, path in enumerate(paths):
        data = CoverageData(basename=str(path))
        data.add_lines({str(tmp_path/f'source_{index}.py'): {index+1}})
        data.write()
    with pytest.raises(AssertionError):
        core.verify_coverage_union(paths[0], paths)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'failed', 'sha', 'coverage'])
def test_aggregate_rejects_incomplete_or_stale_batch_receipts(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    _fake_processes(monkeypatch)
    selections = [f'tests/test_{i}.py' for i in range(core.BATCH_SIZE+1)]
    path = Path('core-manifest-0.json')
    assert core.run_batches(0, selections, path) == 0
    manifest = copy.deepcopy(json.loads(path.read_text()))
    if change == 'missing':
        manifest['batches'].pop()
    elif change == 'duplicate':
        manifest['batches'][1] = manifest['batches'][0]
    elif change == 'failed':
        manifest['batches'][0]['returncode'] = 1
    elif change == 'sha':
        manifest['sha'] = 'b'*40
    else:
        Path('.coverage.core-0').write_bytes(b'wrong coverage')
    with pytest.raises(AssertionError):
        core.validate_manifest(manifest, 0, selections, 'a'*40, tmp_path)


def test_deadline_split_preserves_other_lanes_and_complete_processes():
    selections = core.inventory()
    old = [selections[i::core.BASE_PARTS] for i in range(core.BASE_PARTS)]
    new = core.partition(selections)
    assert len(new) == core.PARTS == core.BASE_PARTS+1
    for part in range(core.BASE_PARTS):
        if part != core.SPLIT_PART:
            assert new[part] == old[part]
    assert new[core.SPLIT_PART]+new[-1] == old[core.SPLIT_PART]
    assert len(new[core.SPLIT_PART]) == core.SPLIT_AFTER_BATCHES*core.BATCH_SIZE
    assert core.batches(new[core.SPLIT_PART])+core.batches(new[-1]) == core.batches(old[core.SPLIT_PART])


def test_workflow_core_matrix_and_disk_ball_inventory_match_protocol():
    import ast
    import re

    workflow = Path('.github/workflows/ci.yml').read_text()
    parts = ast.literal_eval(re.search(r'^        part: (\[[^\n]+\])$', workflow, re.M)[1])
    assert parts == list(range(core.PARTS))
    disk = workflow.split('- name: disk\n', 1)[1].split('- name: ball\n', 1)[0]
    ball = workflow.split('- name: ball\n', 1)[1].split('- name:', 1)[0]
    assert re.findall(r'tests/test_matlab_port/(\w+)', disk) == ['diskfun', 'diskfunv']
    assert re.findall(r'tests/test_matlab_port/(\w+)', ball) == ['ballfun', 'ballfunv']


def test_aggregate_cannot_accept_original_eight_manifests_without_new_tail(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    for part in range(core.BASE_PARTS):
        (tmp_path/f'core-manifest-{part}.json').write_text('{}')
    checked = []
    monkeypatch.setattr(core, 'validate_manifest', lambda manifest, part, *args: checked.append(part))
    monkeypatch.setattr(sys, 'argv', ['run_core_shard.py', '--verify-and-combine', str(tmp_path)])

    def forbidden(*args, **kwargs):
        pytest.fail('Coverage combined before all nine manifests were present')

    monkeypatch.setattr(core.subprocess, 'run', forbidden)
    with pytest.raises(FileNotFoundError, match='core-manifest-8.json'):
        core.main()
    assert checked == list(range(core.BASE_PARTS))
