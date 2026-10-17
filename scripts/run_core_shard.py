"""Partition core selections into fresh processes; retain aggregate coverage."""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_PARTS = 8
PARTS = 11
BATCH_SIZE = 12
SPLIT_PARTS = (6, 1, 2)
SPLIT_AFTER_BATCHES = 4
MIRROR_FILE = 'tests/test_coverage/test_port_mirrors.py'


def mirror_selections():
    """Expand the literal mirror parametrization so no shard owns it all.

    Fail closed if this file gains another test entry point or stops using
    its literal MODULES list. Such a change needs an inventory update.
    """
    tree = ast.parse(Path(MIRROR_FILE).read_text())
    tests = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name.startswith(('test_', 'Test'))]
    if tests != ['test_port_mirror']:
        raise ValueError('mirror test entry points changed; update core inventory')
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'MODULES'
                      for target in node.targets)]
    if len(values) != 1 or not isinstance(values[0], list):
        raise ValueError('expected one literal mirror MODULES list')
    modules = values[0]
    if not modules or not all(isinstance(name, str) and name
                              and all(part.isidentifier() for part in name.split('.'))
                              for name in modules) or len(set(modules)) != len(modules):
        raise ValueError('mirror module names must be unique nonempty dotted identifiers')
    return [f'{MIRROR_FILE}::test_port_mirror[{name}]' for name in modules]


def inventory():
    files = [str(p) for p in Path('tests').rglob('*.py')
             if 'test_matlab_port' not in p.parts and str(p) != MIRROR_FILE
             and (p.name.startswith('test_') or p.name.endswith('_test.py'))]
    return sorted(files + mirror_selections())


def partition(files):
    # Preserve the original eight lanes, splitting only complete batches.
    # Core6 was split earlier; run39059398761 also exhausted the90-minute
    # limit in Core1/2. Their first four batches cost about50minutes each;
    # give each remaining tail its own lane. Existing Core8 stays unchanged.
    groups = [files[i::BASE_PARTS] for i in range(BASE_PARTS)]
    split = BATCH_SIZE * SPLIT_AFTER_BATCHES
    for part in SPLIT_PARTS:
        remainder = groups[part][split:]
        groups[part] = groups[part][:split]
        groups.append(remainder)
    flat = [name for group in groups for name in group]
    if not all(groups) or len(flat) != len(set(flat)) or sorted(flat) != files:
        raise ValueError('core partition is not nonempty, disjoint and exhaustive')
    return groups


def batches(selected):
    return [selected[i:i+BATCH_SIZE] for i in range(0, len(selected), BATCH_SIZE)]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_coverage(path):
    """Validate the database rather than accepting a nonempty corrupt file."""
    from coverage import CoverageData

    data = CoverageData(basename=str(path))
    data.read()
    return data


def verify_coverage_union(output, inputs):
    """Reject a combine operation that silently skipped an invalid input."""
    lines = {}
    arcs = {}
    has_arcs = False
    for path in inputs:
        data = read_coverage(path)
        has_arcs = has_arcs or data.has_arcs()
        for name in data.measured_files():
            lines.setdefault(name, set()).update(data.lines(name) or ())
            arcs.setdefault(name, set()).update(data.arcs(name) or ())
    actual = read_coverage(output)
    assert actual.has_arcs() == has_arcs
    assert actual.measured_files() == set(lines)
    for name in lines:
        assert set(actual.lines(name) or ()) == lines[name]
        if has_arcs:
            assert set(actual.arcs(name) or ()) == arcs[name]


def validate_manifest(manifest, part, selected, sha, folder):
    """Reject missing, duplicated, failed or stale process coverage."""
    expected = batches(selected)
    assert manifest['schema_version'] == 2 and manifest['sha'] == sha
    assert manifest['part'] == part and manifest['selections'] == selected
    assert manifest['returncode'] == 0
    actual = manifest['batches']
    assert len(actual) == len(expected)
    for index, (record, selection) in enumerate(zip(actual, expected)):
        assert record['index'] == index and record['selections'] == selection
        assert record['returncode'] == 0
        assert record['coverage_validated'] is True
        assert len(record['coverage_sha256']) == 64
    coverage = folder / f'.coverage.core-{part}'
    assert coverage.stat().st_size > 0
    assert manifest['coverage_sha256'] == file_hash(coverage)


def run_batches(part, selected, manifest_path):
    """Retire each pytest/JAX process before starting the next selection.

    A single worker prevents overlapping large compilations. The mirror's
    source predicates, per-test timeouts, and aggregate coverage bound are
    unchanged. Process boundaries replace cumulative compiler-cache growth;
    they do not clear caches inside a scientific computation.
    """
    if not selected:
        raise ValueError('empty core selection')
    coverage = Path(f'.coverage.core-{part}')
    records = [{'index': index, 'selections': selection, 'returncode': None}
               for index, selection in enumerate(batches(selected))]
    manifest = {'schema_version': 2, 'sha': os.environ.get('GITHUB_SHA'),
                'part': part, 'selections': selected, 'batches': records,
                'returncode': None, 'coverage_sha256': None}

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    save()
    data_files = []
    for record in records:
        index = record['index']
        data = Path(f'.coverage.core-{part}-batch-{index}')
        # Never accept a coverage file left by an earlier invocation.
        if data.exists():
            raise FileExistsError(data)
        env = dict(os.environ, COVERAGE_FILE=str(data))
        progress = Path(env.get('PYTEST_PROGRESS_DIR', '.pytest-progress'))
        env['PYTEST_PROGRESS_DIR'] = str(progress / f'batch-{index}')
        command = [sys.executable, '-m', 'pytest', '-p', 'scripts.pytest_progress_plugin',
                   *record['selections'], '--ignore=tests/test_matlab_port', '-q',
                   '--tb=short', '--timeout=600', '--cov=chebfunjax',
                   '--cov-report=', '--cov-fail-under=0']
        print(f'Core shard {part+1}/{PARTS}, process {index+1}/{len(records)}: '
              f'{len(record["selections"])} selections', flush=True)
        result = subprocess.run(command, env=env)
        record['returncode'] = result.returncode
        save()
        if result.returncode == 0:
            if not data.is_file() or data.stat().st_size == 0:
                record['returncode'] = 1
                record['error'] = 'successful pytest process did not produce coverage'
                save()
            else:
                try:
                    measured = read_coverage(data)
                except Exception as error:
                    record['returncode'] = 1
                    record['error'] = f'invalid process coverage: {error}'
                else:
                    record['coverage_validated'] = True
                    record['coverage_sha256'] = file_hash(data)
                    record['measured_files'] = len(measured.measured_files())
                    data_files.append(data)
                save()
    if any(record['returncode'] != 0 for record in records):
        manifest['returncode'] = 1
        save()
        return 1
    result = subprocess.run([sys.executable, '-m', 'coverage', 'combine', '--keep',
                             '--data-file='+str(coverage), *map(str, data_files)])
    manifest['returncode'] = result.returncode
    if result.returncode == 0:
        try:
            verify_coverage_union(coverage, data_files)
            assert all(record['coverage_sha256'] == file_hash(path)
                       for record, path in zip(records, data_files))
        except Exception as error:
            manifest['returncode'] = 1
            manifest['error'] = f'incomplete coverage union: {error}'
        else:
            manifest['coverage_sha256'] = file_hash(coverage)
    save()
    return manifest['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', type=int, choices=range(PARTS))
    parser.add_argument('--verify-and-combine', type=Path)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    files = inventory()
    groups = partition(files)
    if args.inventory_only:
        print(json.dumps({'inventory': files, 'shards': groups}, indent=2))
        return 0
    if args.verify_and_combine:
        folder = args.verify_and_combine
        for i, selected in enumerate(groups):
            manifest = json.loads((folder / f'core-manifest-{i}.json').read_text())
            validate_manifest(manifest, i, selected, os.environ['GITHUB_SHA'], folder)
        subprocess.run([sys.executable, '-m', 'coverage', 'combine', str(folder)], check=True)
        # Reads existing tool.coverage.report.fail_under=79 unchanged.
        return subprocess.run([sys.executable, '-m', 'coverage', 'report']).returncode
    if args.part is None:
        parser.error('specify --part, --inventory-only or --verify-and-combine')
    selected = groups[args.part]
    manifest_path = Path(f'core-manifest-{args.part}.json')
    return run_batches(args.part, selected, manifest_path)


if __name__ == '__main__':
    raise SystemExit(main())
