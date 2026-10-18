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


def test_recorded_ownership_preserves_lanes_and_complete_processes():
    baseline = json.loads(core.BASELINE_FILE.read_text())
    selections = sorted(name for group in baseline['groups'] for name in group)
    groups = core.partition(selections)
    assert groups == baseline['groups']
    assert len(groups) == core.PARTS == core.BASE_PARTS+4
    for group in groups:
        batches = core.batches(group)
        assert [name for batch in batches for name in batch] == group
        assert all(len(batch) == core.BATCH_SIZE for batch in batches[:-1])
        assert 0 < len(batches[-1]) <= core.BATCH_SIZE
    flat = [name for group in groups for name in group]
    assert len(flat) == len(set(flat)) == len(selections)
    assert sorted(flat) == selections


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


def test_new_core_selection_preserves_measured_lane_prefixes():
    selections = core.inventory()
    before = core.partition(selections)
    added = 'tests/test_chebfun1d/test_future_added_root_control.py'
    assert added not in selections
    after = core.partition(sorted(selections + [added]))
    for old, new in zip(before, after):
        assert new[:len(old)] == old
    assert sorted(name for group in after for name in group) == sorted(selections + [added])
    # The actual next root-control file must leave every anchor in its lane.
    actual_added = 'tests/test_chebfun1d/test_get_roots_for_breaks_source.py'
    baseline = json.loads(core.BASELINE_FILE.read_text())['groups']
    current = core.partition(sorted(set(selections + [actual_added])))
    for part, group in enumerate(baseline):
        retained = [name for name in group if name in selections]
        assert current[part][:len(retained)] == retained


def test_duplicate_core_baseline_fails_closed(tmp_path, monkeypatch):
    baseline = json.loads(core.BASELINE_FILE.read_text())
    baseline['groups'][1].append(baseline['groups'][0][0])
    path = tmp_path / 'invalid-baseline.json'
    path.write_text(json.dumps(baseline))
    monkeypatch.setattr(core, 'BASELINE_FILE', path)
    with pytest.raises(ValueError, match='baseline'):
        core.partition(core.inventory())


def test_core_baseline_covers_current_inventory():
    baseline = json.loads(core.BASELINE_FILE.read_text())
    anchors = [name for group in baseline['groups'] for name in group]
    assert sorted(anchors) == core.inventory(), (
        'Core inventory changed: run scripts/run_core_shard.py --update-baseline '
        'and review/stage core_shard_baseline.json with the tests.')


def test_aggregate_requires_new_core7_tail_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_SHA', 'a'*40)
    for part in range(core.PARTS-1):
        (tmp_path/f'core-manifest-{part}.json').write_text('{}')
    checked = []
    monkeypatch.setattr(core, 'validate_manifest', lambda manifest, part, *args: checked.append(part))
    monkeypatch.setattr(sys, 'argv', ['run_core_shard.py', '--verify-and-combine', str(tmp_path)])

    def forbidden(*args, **kwargs):
        pytest.fail('Coverage combined before the new Core7 tail was present')

    monkeypatch.setattr(core.subprocess, 'run', forbidden)
    with pytest.raises(FileNotFoundError, match='core-manifest-11.json'):
        core.main()
    assert checked == list(range(core.PARTS-1))
