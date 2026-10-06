"""Partition core test files; retain aggregate coverage threshold in CI."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PARTS = 3


def inventory():
    return sorted(str(p) for p in Path('tests').rglob('*.py')
                  if 'test_matlab_port' not in p.parts
                  and (p.name.startswith('test_') or p.name.endswith('_test.py')))


def partition(files):
    groups = [files[i::PARTS] for i in range(PARTS)]
    flat = [name for group in groups for name in group]
    if not all(groups) or len(flat) != len(set(flat)) or sorted(flat) != files:
        raise ValueError('core partition is not nonempty, disjoint and exhaustive')
    return groups


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
            assert manifest['sha'] == os.environ['GITHUB_SHA']
            assert manifest['part'] == i and manifest['files'] == selected
            assert manifest['returncode'] == 0
            assert (folder / f'.coverage.core-{i}').stat().st_size > 0
        subprocess.run([sys.executable, '-m', 'coverage', 'combine', str(folder)], check=True)
        # Reads existing tool.coverage.report.fail_under=79 unchanged.
        return subprocess.run([sys.executable, '-m', 'coverage', 'report']).returncode
    if args.part is None:
        parser.error('specify --part, --inventory-only or --verify-and-combine')
    selected = groups[args.part]
    manifest_path = Path(f'core-manifest-{args.part}.json')
    manifest = {'sha': os.environ.get('GITHUB_SHA'), 'part': args.part,
                'files': selected, 'returncode': None}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = dict(os.environ, COVERAGE_FILE=f'.coverage.core-{args.part}')
    command = [sys.executable, '-m', 'pytest', *selected,
               '--ignore=tests/test_matlab_port', '-q', '--tb=short', '--timeout=600',
               '-n', '2', '--cov=chebfunjax', '--cov-report=', '--cov-fail-under=0']
    print(f'Core shard {args.part+1}/{PARTS}: {len(selected)}/{len(files)} files', flush=True)
    result = subprocess.run(command, env=env)
    manifest['returncode'] = result.returncode
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
