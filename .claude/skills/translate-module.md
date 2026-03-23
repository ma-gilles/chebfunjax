# Skill: translate-module

Translate a MATLAB Chebfun module to Python/JAX for the chebfunjax project.

## Invocation

```
/translate-module U{XX} {module_path}
```

Example: `/translate-module U11 utils/transforms`

## Prerequisites

Before starting, verify:
1. You have read `PLAN.md` in the repo root (design decisions, templates, quality gates).
2. The unit's dependencies are marked `done` in `STATUS.md`.
3. No other agent is working on this unit (check `STATUS.md`).

## Step-by-Step Procedure

### Step 1: Claim the Unit

Update `STATUS.md`:
```
| U{XX} | {module} | in_progress | | {your-id} | started YYYY-MM-DD |
```

Create branch:
```bash
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
git checkout main && git pull
git checkout -b translate/U{XX}-{short-name}
```

### Step 2: Read the MATLAB Source

For every function listed in the unit's row in `PLAN.md`:

1. Open the MATLAB file in `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/`.
2. Extract and record:
   - **Purpose**: the `%FUNCNAME   Description` line.
   - **Authors**: names from comments (e.g., "by Nick Hale, July 2011") or the generic copyright.
   - **Algorithm references**: paper citations from `DEVELOPER NOTES` section.
   - **"See also"**: related functions.
   - **Core algorithm**: understand the mathematical approach, not just the code.
3. If a chebpy equivalent exists, read it in `/scratch/gpfs/GILLES/mg6942/chebpy_ref/src/chebpy/`.

### Step 3: Write MATLAB Reference Generator

Add test cases to `matlab_harness/generate_refs.m` for this module.
Generate diverse inputs covering:
- Small n (5, 10), medium n (32, 64), large n (128, 256, 1024)
- Edge cases: n=0, n=1, n=2
- Special inputs (e.g., random coefficients with fixed seed for transforms)
- Multiple output types (points, weights, matrices, etc.)

Run the generator:
```bash
module load matlab/R2025b
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"
```

Verify the `.mat` file was created in `tests/references/`.

### Step 4: Write the Python Implementation

Follow the templates in `PLAN.md` Section 4 exactly.

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

**For each function, the docstring MUST include:**

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

1. Use `jax.numpy` everywhere. Never `import numpy`.
2. Use `dtype=jnp.float64` explicitly for all array creation.
3. Functions that can be JIT-compiled should be decorated with `@jax.jit` or
   use `@eqx.filter_jit` if they take Module arguments.
4. Use `jax.lax.scan` for recurrences, not Python loops over array elements.
5. Python naming: `snake_case` for functions, `PascalCase` for classes.
   MATLAB's `camelCase` → Python's `snake_case` (e.g., `baryWeights` → `bary_weights`).
6. MATLAB 1-indexing → Python 0-indexing. Double-check all index arithmetic.
7. MATLAB column vectors → Python 1D arrays (shape `(n,)`, not `(n, 1)`).
8. MATLAB `end` → Python `-1` or `len(x)-1`.
9. Preserve algorithm structure. Don't "improve" the algorithm unless there's a
   demonstrable bug or JAX incompatibility. If you must diverge, document why.

**For classes (eqx.Module):**

1. Use `@classmethod` factories (`from_function`, `from_coeffs`, `from_values`),
   not complex `__init__` logic.
2. All fields are either `jax.Array` (dynamic) or `eqx.field(static=True)` (metadata).
3. Implement operator overloads as dunder methods (see PLAN.md Section 4.3).
4. Return new objects from every operation — never mutate.

### Step 5: Write Tests

Create test file: `tests/test_{subpackage}/test_{module}.py`

**Required test categories:**

```python
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

**Test tolerance rules:**
- Default: `rtol=1e-12, atol=1e-14`
- If you need to relax, add a comment explaining WHY
- `cos(pi/2)` type issues (exact zero vs 6e-17): use `atol=1e-15`
- Never use `rtol > 1e-8` without flagging it in the PR description

### Step 6: Run Tests and Fix

```bash
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun

# Fast tests (no MATLAB, no GPU)
pixi run test-fast

# MATLAB comparison tests
pixi run test-matlab

# All tests
pixi run test-full

# Lint
pixi run lint
```

Iterate until ALL tests pass. Do not merge with failing tests.

### Step 7: Write Benchmark (if applicable)

For compute-intensive functions (quadrature for large n, FFT-based transforms, etc.):

```python
# benchmarks/bench_{module}.py
# Follow template in PLAN.md Section 5.5
```

Run benchmark and record results:
```bash
pixi run -- python benchmarks/bench_{module}.py
```

Include timing table in PR description.

### Step 8: Update Status and Add to conftest

1. Add any new MATLAB fixtures to `tests/conftest.py`:
   ```python
   @pytest.fixture
   def matlab_transforms():
       return load_matlab_ref("transforms.mat")
   ```

2. Update `STATUS.md`:
   ```
   | U{XX} | {module} | in_review | #{pr} | {your-id} | N tests pass |
   ```

### Step 9: Commit and Push

```bash
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
git add src/chebfunjax/{path} tests/test_{path} matlab_harness/generate_refs.m tests/conftest.py STATUS.md

git commit -m "[U{XX}] Translate {module}: {list of functions}

Translated from MATLAB Chebfun (commit 7574c77):
- {function1}: description
- {function2}: description
...

All {N} tests pass including MATLAB cross-validation (rtol=1e-12).

Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git push -u origin translate/U{XX}-{short-name}
```

### Step 10: Create PR

Title: `[U{XX}] Translate {module}: {short description}`

Body:
```markdown
## Summary
- Translates MATLAB Chebfun {files} to Python/JAX
- {N} functions translated: {list}
- {M} tests: {K} pure Python, {L} MATLAB cross-validated

## Functions translated

| Python function | MATLAB source | Original author | Tests | Accuracy vs MATLAB |
|----------------|--------------|-----------------|-------|-------------------|
| `cheb2leg(c)` | `cheb2leg.m` | Alex Townsend | 8 | rtol < 1e-14 |
| `leg2cheb(c)` | `leg2cheb.m` | Alex Townsend & Nick Hale | 8 | rtol < 1e-14 |

## Performance (if benchmarked)

| Function | n | CPU (ms) | GPU (ms) | MATLAB (ms) |
|----------|---|---------|---------|------------|
| cheb2leg | 1024 | 0.5 | 0.1 | 0.8 |

## Test plan
- [ ] `pixi run test-fast` passes
- [ ] `pixi run test-matlab` passes
- [ ] `pixi run lint` passes
- [ ] No tolerance relaxations (or documented)
```

### Step 11: Update STATUS.md After Merge

```
| U{XX} | {module} | done | #{pr} | {your-id} | {N} tests, merged YYYY-MM-DD |
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

## Quick Reference: File Paths

| What | Where |
|------|-------|
| Our repo | `/scratch/gpfs/GILLES/mg6942/jaxchebfun/` |
| MATLAB Chebfun | `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/` |
| Python chebpy | `/scratch/gpfs/GILLES/mg6942/chebpy_ref/` |
| MATLAB refs | `tests/references/*.mat` |
| MATLAB harness | `matlab_harness/generate_refs.m` |
| Design plan | `PLAN.md` |
| Status tracker | `STATUS.md` |
| MATLAB binary | `module load matlab/R2025b` |
| pixi env | `.pixi/envs/default/bin/python` |
