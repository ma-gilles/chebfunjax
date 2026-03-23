# jaxchebfun — Chebfun in Python/JAX

## Project Goal
Translate the MATLAB Chebfun library (https://github.com/chebfun/chebfun) into Python
using JAX as the primary array backend. Every translated function must be tested against
MATLAB Chebfun reference outputs to machine precision (rtol ≤ 1e-12).

## Key References
- MATLAB Chebfun source: `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/`
- Python chebpy reference: `/scratch/gpfs/GILLES/mg6942/chebpy_ref/`
- MATLAB available: `module load matlab/R2025b`
- Chebfun Guide (the definitive spec): https://www.chebfun.org/docs/guide/

## Architecture Principles

### JAX-only backend
- Use `jax.numpy` (as `jnp`) for all array operations. No numpy/jax dual backend.
- JAX runs fine on CPU; no need for a numpy fallback.
- For functions JAX lacks, use `scipy` via `jax.pure_callback` if JIT needed,
  or plain scipy outside JIT boundaries.
- ALWAYS enable float64: `jax.config.update("jax_enable_x64", True)` (done in `__init__.py`).

### JIT boundaries
- Adaptive construction (while loops that check coefficient decay) CANNOT be JIT-compiled.
  Keep the adaptive loop in Python; JIT the inner kernels (FFT, evaluation, coefficient ops).
- Fixed-size operations (evaluation at points, differentiation, integration, inner products)
  SHOULD be JIT-compiled via `@jax.jit` or `jax.jit(f)`.
- Use `jax.lax.scan` for recurrences (e.g., Clenshaw evaluation).
- Use `jax.vmap` for batched operations over multiple chebfuns or evaluation points.

### Immutability
- All chebfun-like objects are immutable (JAX arrays are immutable).
- Operations return new objects, never mutate in place.
- Use frozen dataclasses or namedtuples for internal representations.

### Don't mirror MATLAB's file structure
- MATLAB `@chebfun/plus.m` → Python `Chebfun.__add__`. Don't create 299 files.
- Group methods by functionality (arithmetic, calculus, rootfinding, etc.) in the class.
- Only create separate modules when a method is a substantial standalone algorithm
  (e.g., `aaa.py`, `minimax.py`).

## Translation Workflow (per function/module)

1. **Read the MATLAB source** in `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/`.
2. **Read the corresponding chebpy implementation** if it exists, in `/scratch/gpfs/GILLES/mg6942/chebpy_ref/`.
3. **Read the Chebfun Guide** section for context on the algorithm.
4. **Translate to Python/JAX**, respecting the architecture principles above.
5. **Write Python tests** that compare against MATLAB outputs.
   - Generate MATLAB references: add cases to `matlab_harness/generate_refs.m` and run.
   - Also write property-based tests (mathematical invariants).
6. **Run tests** and verify `rtol ≤ 1e-12` vs MATLAB.

## Testing

### Generating MATLAB references
```bash
module load matlab/R2025b
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"
```

### Running tests
```bash
pixi run test-fast    # no GPU, no MATLAB refs needed
pixi run test-full    # everything (submit via Slurm for GPU tests)
pixi run test-matlab  # only tests comparing to MATLAB refs
```

### Test conventions
- One test file per source module: `tests/test_utils/test_quadrature.py` tests `src/jaxchebfun/utils/quadrature.py`.
- Use `@pytest.mark.matlab` for tests that need MATLAB reference data.
- Use `@pytest.mark.gpu` for tests that specifically need GPU.
- Use `@pytest.mark.slow` for tests that take > 10 seconds.
- Tolerance: `np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-14)`.
- NEVER widen tolerances without explicit approval — if precision is lost, find the bug.

## Git Workflow
- Never work on `main` directly. Branch: `translate/<module-name>`.
- One PR per module. Small, focused diffs.
- Every PR must have tests that pass.
- Rebase on main before pushing.

## Module Status Tracker

### Phase 1: Utilities (no internal deps)
- [ ] `utils/quadrature.py` — chebpts, legpts, jacpts, hermpts, lagpts, ultrapts
- [ ] `utils/transforms.py` — cheb2leg, leg2cheb, jac2cheb, etc.
- [ ] `utils/interpolation.py` — bary, trigBary, baryWeights, barymat
- [ ] `utils/diffmat.py` — diffmat, intmat, cumsummat
- [ ] `utils/polynomials.py` — chebpoly, legpoly, jacpoly, etc.
- [ ] `utils/approximation.py` — aaa, minimax, ratinterp, padeapprox
- [ ] `utils/misc.py` — standardChop, gridsample, seedRNG
- [ ] `domain.py` — interval representation
- [ ] `pref.py` — preferences

### Phase 2: Tech layer (deps: Phase 1)
- [ ] `tech/chebtech.py` — @chebtech + @chebtech1 + @chebtech2
- [ ] `tech/trigtech.py` — @trigtech

### Phase 3: Fun layer (deps: Phase 2)
- [ ] `fun/classicfun.py` + `fun/bndfun.py`
- [ ] `fun/unbndfun.py`
- [ ] `fun/singfun.py`
- [ ] `fun/deltafun.py`

### Phase 4: Chebfun 1D (deps: Phase 3) — THE BIG ONE
- [ ] `chebfun1d/chebfun.py` — construction, evaluation, arithmetic, calculus, roots, special funcs

### Phase 5: Discretization (deps: Phase 2)
- [ ] `discretization/` — chebcolloc, trigcolloc, ultraS, trigspec

### Phase 6: Operators (deps: Phase 4-5)
- [ ] `operators/linop.py`
- [ ] `operators/chebop.py`
- [ ] `operators/chebmatrix.py`

### Phase 7: 2D functions (deps: Phase 4)
- [ ] `chebfun2d/` — separableApprox + chebfun2 + chebfun2v
- [ ] `diskfun/`
- [ ] `spherefun/`

### Phase 8: 3D functions (deps: Phase 7)
- [ ] `chebfun3d/` + `ballfun/`

### Phase 9: Time integration (deps: Phase 4-5)
- [ ] `spin/`

### Phase 10: Autodiff (deps: Phase 4)
- [ ] `autodiff/`

## Slurm (for GPU tests)
```bash
#!/bin/bash
#SBATCH --partition=cryoem --account=amits
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp
mkdir -p "$TMPDIR"

cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
pixi run test-full
```

## What NOT to translate
- GUI: @chebgui, @chebguiController, @chebguiExporter* — use matplotlib for plotting
- Demos: chebguiDemos/ — write Jupyter notebooks instead
- MATLAB-specific: stringParser, chebguiWindow, Contents.m
