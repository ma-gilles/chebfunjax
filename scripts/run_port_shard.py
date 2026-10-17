"""Run a complete port shard, optionally using a fresh process per file.

Process exit releases JAX compilation caches and allocator state between files.
All selections and the existing900-second per-test timeout are preserved.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--fresh-file-processes", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.parts < 1 or not 0 <= args.part < args.parts:
        parser.error("require parts >= 1 and 0 <= part < parts")
    files = []
    for name in args.paths:
        root = Path(name)
        if not root.is_dir():
            parser.error(f"shard directory does not exist: {name}")
        files.extend(root.rglob("test_*.py"))
    if len(files) != len(set(files)):
        parser.error("shard directories overlap")
    files = sorted(files)
    if not files or args.parts > len(files):
        parser.error("partition would contain no test files")
    # Modulo partition of a single sorted inventory is exhaustive and disjoint.
    selected = files[args.part::args.parts]
    print(
        f"Port shard {args.part + 1}/{args.parts}: "
        f"{len(selected)}/{len(files)} test files", flush=True,
    )
    if not args.fresh_file_processes:
        paths = args.paths if args.parts == 1 else [str(path) for path in selected]
        return subprocess.run([
            sys.executable, "-m", "pytest", *paths, "-q", "--tb=short",
            "--timeout=900", "-p", "no:cacheprovider",
        ]).returncode
    status = 0
    for index, path in enumerate(selected, 1):
        print(f"Port file {index}/{len(selected)}: {path}", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "pytest", str(path), "-q", "--tb=short",
            "--timeout=900", "-p", "no:cacheprovider",
        ])
        # Run every selection even after a failure; propagate the first code.
        if status == 0 and result.returncode:
            status = result.returncode
    return status



if __name__ == "__main__":
    raise SystemExit(main())
