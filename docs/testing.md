# Testing

## Three Tiers

### Tier 1: Unit tests (pure Python)
Mathematical properties, edge cases, JIT/vmap/grad compatibility. No external dependencies.

### Tier 2: MATLAB cross-validation (golden refs)
Compare against MATLAB Chebfun reference data stored as `.mat` files in `tests/references/`.

**Golden refs are committed to the repo.** Contributors do NOT need MATLAB to run tests.
Regeneration is a maintainer task (see below).

Missing `.mat` files cause test **failure**, not skip.

### Tier 3: Performance benchmarks
Timing: JIT compile time, steady-state CPU, steady-state GPU, MATLAB comparison.
Memory usage. Scaling curves for core functions.
Informational, not pass/fail — but results are recorded in every PR.

## Tolerances

```python
# Default (matches project.conf DEFAULT_RTOL / DEFAULT_ATOL):
np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-14)

# Relaxed (MUST document reason in a code comment):
np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)
```

Never widen tolerances without documenting why.

## JAX-Semantic Tests

Functions that are JIT-compiled should also be tested for:
- **JIT**: `jax.jit(f)(args)` matches `f(args)`
- **vmap**: `jax.vmap(f)(batched_args)` matches `[f(a) for a in args]`
- **grad**: `jax.grad(f)(x)` matches known analytical derivative
- **CPU/GPU parity**: results match across devices at `rtol=1e-12`

Not every function needs all four. Mark which apply in the test class docstring.

## Running Tests

```bash
pixi run test-fast     # Tier 1 only (no MATLAB, no GPU)
pixi run test-matlab   # Tier 2 (uses committed golden refs — no MATLAB needed)
pixi run test-full     # All tiers with coverage (must achieve ≥ 90%)
pixi run lint          # ruff check
```

## CI

GitHub Actions CI (`.github/workflows/ci.yml`) runs on every PR:
- Lint (ruff)
- Unit tests + coverage (fail_under=90%)
- Golden ref validation
- MATLAB parity tests (against committed `.mat` files)

PRs cannot merge until CI is green.

## MATLAB Reference Generation (Maintainer Task)

Each module has its own generator: `matlab_harness/refs/<module>.m`.
The runner `matlab_harness/generate_refs.m` auto-discovers and executes all of them.

```bash
source project.conf
module load $MATLAB_MODULE
matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/generate_refs.m')"
```

All generators use `rng(42)` for deterministic output.

Generated `.mat` files are committed to `tests/references/` so that contributors
without MATLAB can still run the full test suite. When adding a new module:
1. Create `matlab_harness/refs/<module>.m`
2. Run the generator
3. Commit the new `.mat` file alongside the code

## Test Fixture

Use the generic `matlab_ref` fixture (no per-module fixtures needed):

```python
@pytest.mark.matlab
@pytest.mark.parametrize("matlab_ref", ["quadrature"], indirect=True)
def test_vs_matlab(self, matlab_ref):
    for n in [5, 10, 32, 64]:
        result = my_function(n)
        npt.assert_allclose(np.array(result), matlab_ref[f"key_n{n}"],
                            rtol=1e-12, atol=1e-14)
```

Legacy named fixtures (e.g., `matlab_quadrature`) still work for existing tests.
