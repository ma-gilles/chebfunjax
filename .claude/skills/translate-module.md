# Skill: translate-module

Translate a MATLAB Chebfun module to Python/JAX for the chebfunjax project.

## Invocation

```
/translate-module U{XX} {module_path}
```

Example: `/translate-module U11 utils/transforms`

---

## §1. Isolated Clone Setup

**Every agent gets its own fresh clone.** Never reuse an existing checkout.
This allows multiple agents to work in parallel without conflicts.

```bash
# ── Generate unique agent ID ──
AGENT_ID="chebfunjax_$(date +%Y%m%d_%H%M%S)_$$"
WORKDIR="/scratch/gpfs/GILLES/mg6942/${AGENT_ID}"

# ── Clone ──
git clone git@github.com:ma-gilles/chebfunjax.git "$WORKDIR"
cd "$WORKDIR"

# ── Source project constants ──
source "$WORKDIR/project.conf"

# ── Isolate Python environment ──
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR="${SCRATCH}/tmp/${AGENT_ID}"
export PIXI_HOME="${SCRATCH}/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="${SCRATCH}/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

# ── Install ──
pixi install

# ── Provenance gate: verify correct environment ──
PIXI_PY="$WORKDIR/.pixi/envs/default/bin/python"
"$PIXI_PY" -c "
import chebfunjax, jax, pathlib, sys
pkg = pathlib.Path(chebfunjax.__file__).resolve()
assert '${AGENT_ID}' in str(pkg), f'Wrong chebfunjax: {pkg}'
print(f'chebfunjax: {pkg}')
print(f'JAX devices: {jax.devices()}')
print(f'Python: {sys.executable}')
print('Provenance gate PASSED')
"
```

**All subsequent commands in this skill use `$WORKDIR` as the working directory.**
Store `WORKDIR` and `AGENT_ID` — you will need them for Slurm scripts.

---

## §2. Prerequisites

Before starting, read these files in the clone:
1. **`PLAN.md`** — locked design decisions, code templates, quality gates
2. **`STATUS.md`** — check the unit's dependencies are `done` and nobody else has claimed it

---

## §3. Claim the Unit and Create Branch

```bash
cd "$WORKDIR"
git checkout -b translate/U${XX}-${SHORT_NAME}
```

Update `STATUS.md` in YOUR clone:
```
| U{XX} | {module} | in_progress | | ${AGENT_ID} | started YYYY-MM-DD |
```

---

## §4. Read the MATLAB Source

For every function listed in the unit's row in `PLAN.md`:

1. Read the MATLAB file in `$CHEBFUN_REF/` (from `project.conf`).
   This is a SHARED read-only directory — all agents read from it, never write to it.
2. Extract and record:
   - **Purpose**: the `%FUNCNAME   Description` line.
   - **Authors**: names from comments (e.g., "by Nick Hale, July 2011") or the generic
     "Copyright 2017 by The University of Oxford and The Chebfun Developers."
   - **Algorithm references**: paper citations from `DEVELOPER NOTES` section.
   - **"See also"**: related functions.
   - **Core algorithm**: understand the mathematical approach, not just the code.
3. If a chebpy equivalent exists, read it in `$CHEBPY_REF/src/chebpy/` (from `project.conf`).
   This is also shared and read-only.

---

## §5. Write MATLAB Reference Generator

Create a NEW per-module script: `$WORKDIR/matlab_harness/refs/{module}.m`.
**Do NOT edit the shared `generate_refs.m`** — it auto-discovers per-module scripts.

Generate diverse inputs covering:
- Small n (5, 10), medium n (32, 64), large n (128, 256, 1024)
- Edge cases: n=0, n=1, n=2
- Special inputs (e.g., random coefficients with fixed seed for transforms)
- Multiple output types (points, weights, matrices, etc.)

Run the generator:
```bash
source "$WORKDIR/project.conf"
module load $MATLAB_MODULE
cd "$WORKDIR"
matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/generate_refs.m')"
```

Verify the `.mat` file was created in `$WORKDIR/tests/references/`.

**Golden refs are committed.** After generating, `git add tests/references/<module>.mat`
and include it in your commit. Contributors without MATLAB rely on these committed files.
See `docs/testing.md` for the full policy.

---

## §6. Write the Python Implementation

Follow the conventions in `PLAN.md` Section 4.

**File header:**
```python
"""Module description.

Translated from MATLAB Chebfun (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
```

**For each function, the docstring MUST include a Provenance section:**

```python
def function_name(...):
    """One-line summary.

    Extended description preserving the mathematical explanation
    from the MATLAB source. Translate MATLAB syntax to Python in examples.

    Parameters
    ----------
    param1 : type
        Description.

    Returns
    -------
    result : type
        Description.

    Examples
    --------
    >>> result = function_name(input)
    >>> expected_output

    Provenance
    ----------
    MATLAB source : path/relative/to/chebfun/repo.m
    Chebfun commit: 7574c77
    Original authors: Name1, Name2 (or generic copyright)
    Algorithm:
        [1] Full citation from MATLAB source.
        [2] Another citation if applicable.

    See Also
    --------
    related_function_1, related_function_2
    """
```

**Implementation rules:**

1. Use `jax.numpy` everywhere. Never `import numpy` in library code.
2. Use `dtype=jnp.float64` explicitly for all array creation.
3. Functions that can be JIT-compiled should be decorated with `@jax.jit` or
   use `@eqx.filter_jit` if they take Module arguments.
4. Prefer `jax.lax.scan` for recurrences when it helps performance or JIT compatibility.
   Use Python loops when they're clearer and performance isn't critical.
5. Python naming: `snake_case` for functions, `PascalCase` for classes.
   MATLAB's `camelCase` → Python's `snake_case` (e.g., `baryWeights` → `bary_weights`).
6. MATLAB 1-indexing → Python 0-indexing. Double-check all index arithmetic.
7. MATLAB column vectors → Python 1D arrays (shape `(n,)`, not `(n, 1)`).
8. MATLAB `end` → Python `-1` or `len(x)-1`.
9. **The invariant is semantic and numerical equivalence, not mechanical fidelity
   to MATLAB structure.** You MAY restructure code for better JAX idioms, performance,
   or clarity. Justified deviations are encouraged when they improve the JAX version.
   Document deviations and verify identical numerical output.

**For classes (eqx.Module):**

1. Use `@classmethod` factories (`from_function`, `from_coeffs`, `from_values`),
   not complex `__init__` logic.
2. All fields are either `jax.Array` (dynamic) or `eqx.field(static=True)` (metadata).
3. Implement operator overloads as dunder methods (see PLAN.md Section 4.3).
4. Return new objects from every operation — never mutate.

---

## §7. Write Tests

Create test file: `$WORKDIR/tests/test_{subpackage}/test_{module}.py`

**Required test categories:**

```python
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.whatever import function_name


class TestFunctionName:
    """Tests for function_name."""

    # --- Tier 1: Mathematical properties (no MATLAB refs) ---

    def test_basic_correctness(self):
        """Known analytical result."""
        ...

    def test_edge_case_n0(self):
        """Empty input."""
        ...

    def test_edge_case_n1(self):
        """Single element."""
        ...

    def test_symmetry(self):
        """Mathematical symmetry property."""
        ...

    def test_exactness_for_polynomials(self):
        """Quadrature/interpolation exact for degree <= threshold."""
        ...

    # --- Tier 2: MATLAB cross-validation ---

    @pytest.mark.matlab
    def test_vs_matlab(self, matlab_fixture_name):
        """Compare against MATLAB Chebfun output."""
        for n in [5, 10, 32, 64, 128]:
            result = function_name(n)
            ref = matlab_fixture_name[f"key_n{n}"]
            npt.assert_allclose(
                np.array(result), ref,
                rtol=1e-12, atol=1e-14,
                err_msg=f"Mismatch at n={n}"
            )

    # --- JIT compatibility ---

    def test_jit_compatible(self):
        """Function works under jax.jit."""
        jitted = jax.jit(function_name)
        result = jitted(10)
        npt.assert_allclose(
            np.array(result), np.array(function_name(10)),
            rtol=1e-15
        )
```

**Test tolerance rules — ABSOLUTE RULE:**
- Default: `rtol=1e-12, atol=1e-14`
- **Accuracy must match or beat MATLAB.** If our result is less accurate than
  MATLAB Chebfun on the same input, that is a BUG — investigate and fix,
  do not relax the tolerance.
- To relax a tolerance, you MUST:
  1. Run the same computation in MATLAB and measure its error
  2. Prove our error is ≤ MATLAB's error
  3. Document the MATLAB error in a comment next to the relaxed tolerance
  4. The relaxed tolerance must still be tighter than MATLAB
- Small constant-factor differences from platform BLAS/LAPACK are OK
  (e.g., 1.3e-13 vs 1.0e-13). Order-of-magnitude differences are not.
- `cos(pi/2)` type issues (exact zero vs 6e-17): use `atol=1e-15`
- Never relax without measuring MATLAB first. "It's hard numerically" is
  not a reason — MATLAB does it, so can we.

---

## §8. Run Tests Locally and Fix

```bash
cd "$WORKDIR"

# Fast tests (no MATLAB refs, no GPU)
pixi run test-fast

# MATLAB comparison tests
pixi run test-matlab

# All tests
pixi run test-full

# Lint
pixi run lint
```

Iterate until ALL tests pass. Do not proceed with failing tests.

---

## §9. Slurm Submission (for GPU tests or heavy work)

For GPU benchmarks or integration tests, submit to Slurm.
**Critical:** hardcode `$WORKDIR` at script-generation time (unquoted EOF).

```bash
source "$WORKDIR/project.conf"
SCRIPT="${SLURM_LOG_DIR}/job_${AGENT_ID}.sh"

cat > "$SCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=chebfunjax-test
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${SLURM_LOG_DIR}/chebfunjax-test-%j.out

set -euo pipefail
WORKDIR="${WORKDIR}"
cd "\$WORKDIR"
source "\$WORKDIR/project.conf"

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="\${SCRATCH}/tmp/slurm_\${SLURM_JOB_ID}"
export PIXI_HOME="\${SCRATCH}/pixi_home/slurm_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="\${SCRATCH}/rattler_cache/slurm_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR" "\$PIXI_HOME" "\$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
PIXI_PY="\$(pixi run which python)"
export PATH="\$(dirname "\$PIXI_PY"):\$PATH"

# Provenance gate
"\$PIXI_PY" -c "import chebfunjax; print('OK:', chebfunjax.__file__)"

# Run tests
pixi run test-full
EOF

chmod +x "$SCRIPT"
JOB_ID=$(sbatch --parsable "$SCRIPT")
echo "Submitted job $JOB_ID for workdir $WORKDIR"
echo "Monitor: tail -f ${SLURM_LOG_DIR}/chebfunjax-test-${JOB_ID}.out"
```

Wait for completion:
```bash
squeue -u $USER | grep $JOB_ID
cat ${SLURM_LOG_DIR}/chebfunjax-test-${JOB_ID}.out
```

---

## §10. Write Benchmark (if applicable)

For compute-intensive functions (quadrature for large n, FFT-based transforms, etc.):

```python
# benchmarks/bench_{module}.py
# Follow template in PLAN.md Section 5.5
```

Run benchmark and record results:
```bash
cd "$WORKDIR"
pixi run -- python benchmarks/bench_{module}.py
```

Include timing table in PR description.

---

## §11. Commit on Branch

**Do NOT edit `tests/conftest.py` for new fixtures.** Use the generic `matlab_ref`
fixture with `@pytest.mark.parametrize("matlab_ref", ["your_module"], indirect=True)`.
See `docs/testing.md` for the pattern.

Commit on your branch:
```bash
cd "$WORKDIR"
git add src/chebfunjax/ tests/ matlab_harness/refs/ tests/references/ STATUS.md benchmarks/ 2>/dev/null

git commit -m "$(cat <<'COMMITEOF'
[U{XX}] Translate {module}: {list of functions}

Translated from MATLAB Chebfun (commit 7574c77):
- {function1}: description
- {function2}: description

All {N} tests pass including MATLAB cross-validation (rtol=1e-12).

Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.

Co-Authored-By: Claude <noreply@anthropic.com>
COMMITEOF
)"
```

---

## §12. Push Branch, Open PR, Auto-Merge

**Fully autonomous.** The agent pushes, opens a PR, and enables auto-merge.
CI is the sole reviewer. If all checks pass, the PR merges automatically.
No human approval required.

```bash
cd "$WORKDIR"
source project.conf
BRANCH="translate/U${XX}-${SHORT_NAME}"

# ── Rebase on latest main ──
git fetch origin main
git rebase origin/main
# If rebase conflicts: resolve, `git rebase --continue`, re-run tests

# ── Push branch ──
git push -u origin "$BRANCH"

# ── Create PR with auto-merge enabled ──
gh pr create \
  --title "[U${XX}] Translate ${MODULE}: ${SHORT_DESCRIPTION}" \
  --body "$(cat <<PRBODY
## Summary
- Translates MATLAB Chebfun functions to Python/JAX
- Functions: ${FUNC_LIST}
- Tests: ${N_TESTS} total (${N_UNIT} unit, ${N_MATLAB} MATLAB golden-ref)

## Functions translated

| Python function | MATLAB source | Original author | Accuracy vs MATLAB |
|----------------|--------------|-----------------|-------------------|
${FUNC_TABLE}

## CI gates (all enforced automatically)
- Lint (ruff)
- Unit tests + coverage ≥ 90%
- No numpy imports in library code
- Provenance docstrings present in all source files
- Golden .mat refs committed and valid
- MATLAB parity tests pass
PRBODY
)"

# ── Enable auto-merge (merges when all required checks pass) ──
gh pr merge "$BRANCH" --auto --squash
```

### What CI checks (the sole reviewer)

| Gate | Job | What it enforces |
|------|-----|-----------------|
| Lint | `lint` | `ruff check` passes, zero warnings |
| No numpy | `code-quality` | No `import numpy` in `src/` (only `jax.numpy`) |
| Provenance | `code-quality` | Every source file has a Provenance docstring section |
| Golden refs | `code-quality` | Every `@pytest.mark.matlab` test has a committed `.mat` file |
| Tests | `test` | All tests pass (unit + MATLAB golden-ref) |
| Coverage | `test` | ≥ 90% on new code |
| Ref integrity | `golden-refs` | All `.mat` files load without error |

### What happens

| Scenario | Result |
|----------|--------|
| All CI green | PR auto-merges to main. Done. |
| CI fails | PR stays open. Agent reads failure, fixes, pushes again. CI re-runs. |
| Rebase conflict | Agent resolves locally, force-pushes branch. CI re-runs. |

### If `gh` CLI is not available

Push the branch manually and create the PR on https://github.com/ma-gilles/chebfunjax/pulls.
Enable "auto-merge" in the GitHub UI after CI starts.

---

## §13. Post-Merge Summary

After successful merge, print a summary:

```
========================================
TRANSLATION COMPLETE: U{XX} {module}
========================================
Agent:     ${AGENT_ID}
Workdir:   ${WORKDIR}
Branch:    translate/U{XX}-{short-name}
Commit:    $(git log -1 --format='%h %s')

Functions translated:
  - function1 (from file1.m, by Author1)
  - function2 (from file2.m, by Author2)

Tests: {N} total, {K} unit, {L} MATLAB cross-validated
Accuracy: rtol < {best_rtol} vs MATLAB
Tolerance relaxations: {none | list with reasons}

Performance (if benchmarked):
  function1: CPU {X}ms, GPU {Y}ms, MATLAB {Z}ms

STATUS.md updated: U{XX} = done
========================================
```

### Cleanup

**Do NOT delete the workdir immediately** — keep it for a day in case of issues.
After verification, clean up:
```bash
source project.conf
rm -rf "$WORKDIR"
rm -rf "${SCRATCH}/tmp/${AGENT_ID}"
rm -rf "${SCRATCH}/pixi_home/${AGENT_ID}"
rm -rf "${SCRATCH}/rattler_cache/${AGENT_ID}"
```

---

## Common Pitfalls

### MATLAB → Python Gotchas

| MATLAB | Python/JAX | Trap |
|--------|-----------|------|
| `1:n` | `range(0, n)` or `jnp.arange(n)` | Off-by-one: MATLAB is 1-indexed |
| `A(end)` | `A[-1]` | |
| `A(:)` | `A.ravel()` | MATLAB is column-major (Fortran order) |
| `A.'` | `A.T` | Non-conjugate transpose |
| `A'` | `A.conj().T` | Conjugate transpose |
| `[a; b]` | `jnp.concatenate([a, b])` | Vertical concat |
| `[a, b]` | `jnp.concatenate([a, b])` | 1D arrays: same as vertical |
| `zeros(n, 1)` | `jnp.zeros(n)` | Use 1D, not column vector |
| `feval(f, x)` | `f(x)` | |
| `isa(x, 'double')` | `isinstance(x, (float, jnp.ndarray))` | |
| `isempty(x)` | `x.size == 0` or `len(x) == 0` | |
| `error(...)` | `raise ValueError(...)` | |
| `nargout` | Multiple return / separate functions | No nargout in Python |
| `persistent` / `global` | Module-level variable or class attribute | Avoid where possible |

### JAX-Specific Gotchas

1. **No in-place mutation**: `x[i] = v` → `x = x.at[i].set(v)` (returns new array).
2. **JIT + control flow**: `if x > 0` fails under JIT → use `jnp.where(x > 0, a, b)` or `jax.lax.cond`.
3. **JIT + dynamic shapes**: `x[:n]` where `n` is traced fails → use `jax.lax.dynamic_slice` or keep `n` static.
4. **Random numbers**: No `np.random.rand()` → use `jax.random.uniform(key, shape)` with explicit PRNG keys.
5. **Float64**: Always `dtype=jnp.float64`. JAX defaults to float32 if x64 isn't enabled.
6. **Print debugging under JIT**: Use `jax.debug.print("{x}", x=x)`, not `print()`.

### When the MATLAB Algorithm Won't Work in JAX

Sometimes a MATLAB algorithm uses patterns incompatible with JAX (dynamic allocation,
global state, etc.). In these cases:

1. **Document the deviation** in the docstring: "Note: deviates from MATLAB implementation because..."
2. **Preserve the same mathematical algorithm** — change the implementation, not the math.
3. **Verify identical numerical output** against MATLAB — the result must match even if the code differs.
4. **Common adaptations:**
   - MATLAB growing arrays → pre-allocate with max size + mask
   - MATLAB `while` with dynamic termination → Python `while` outside JIT
   - MATLAB `try/catch` → Python `try/except` (but not inside JIT)
   - MATLAB cell arrays → Python lists (outside JIT) or padded arrays (inside JIT)

---

## Quick Reference: Shared Paths (read-only, from project.conf)

| What | Variable | Value |
|------|----------|-------|
| MATLAB Chebfun source | `$CHEBFUN_REF` | see `project.conf` |
| Python chebpy reference | `$CHEBPY_REF` | see `project.conf` |
| MATLAB binary | `module load $MATLAB_MODULE` | see `project.conf` |
| Slurm logs | `$SLURM_LOG_DIR` | see `project.conf` |
| Scratch root | `$SCRATCH` | see `project.conf` |

## Quick Reference: Per-Agent Paths

| What | Where |
|------|-------|
| Agent workdir | `$WORKDIR` = `$SCRATCH/${AGENT_ID}/` |
| Agent pixi env | `$WORKDIR/.pixi/envs/default/bin/python` |
| Agent TMPDIR | `$SCRATCH/tmp/${AGENT_ID}/` |
| Agent pixi cache | `$SCRATCH/pixi_home/${AGENT_ID}/` |
| Agent rattler cache | `$SCRATCH/rattler_cache/${AGENT_ID}/` |
