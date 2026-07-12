"""Fail CI if a tests/test_matlab_port directory is missing from the
shard lists in .github/workflows/ci.yml (keeps the sharded port jobs
exhaustive as new module directories are added)."""

import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]
ci = (root / ".github" / "workflows" / "ci.yml").read_text()
listed = set(re.findall(r"tests/test_matlab_port/([\w]+)", ci))
actual = {
    p.name
    for p in (root / "tests" / "test_matlab_port").iterdir()
    if p.is_dir() and p.name != "__pycache__"
}
missing = sorted(actual - listed)
if missing:
    print(f"port-shard check FAILED; unsharded directories: {missing}")
    sys.exit(1)
print(f"port-shard check OK ({len(actual)} directories all sharded)")
