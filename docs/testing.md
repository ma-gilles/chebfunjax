# Testing

## Three Tiers

### Tier 1: Unit tests (pure Python)
Mathematical properties, edge cases, JIT compatibility. No external dependencies.

### Tier 2: MATLAB cross-validation
Compare against MATLAB Chebfun reference data stored as `.mat` files.
Reference data is generated per-module by scripts in `matlab_harness/refs/`.

**References are NOT optional.** If a module has MATLAB tests, missing `.mat` files
cause test FAILURE (not skip). Run `matlab_harness/generate_refs.m` to regenerate.

### Tier 3: Performance benchmarks
Timing vs MATLAB, GPU vs CPU speedup. Informational, not pass/fail.

## Tolerances

```python
# Default:
np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-14)

# Relaxed (MUST document reason):
np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)
# ^ e.g., rootfinding near double roots has reduced precision
```

Never widen tolerances without documenting why.

## Running Tests

```bash
pixi run test-fast     # Tier 1 only (no MATLAB, no GPU)
pixi run test-matlab   # Tier 2 only
pixi run test-full     # All tiers with coverage
pixi run lint          # ruff check
```

## MATLAB Reference Generation

Each module has its own generator: `matlab_harness/refs/<module>.m`.
Agents create new generators for their module — no shared file to conflict on.

```bash
module load matlab/R2025b
# Generate all:
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"
# Generate one:
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/refs/quadrature.m')"
```

All generators use `rng(42)` for reproducibility.
