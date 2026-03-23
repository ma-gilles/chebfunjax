# chebfunjax — Chebfun in Python/JAX

## READ FIRST

Before doing ANY work, read these documents in order:
1. **This file** — quick reference and rules
2. **`PLAN.md`** — locked design decisions, code templates, module dependency graph, all translation units
3. **`STATUS.md`** — what's done, what's in progress, what's available
4. **`.claude/skills/translate-module.md`** — step-by-step procedure for translating a module

## Project Goal

Translate the MATLAB Chebfun library (https://github.com/chebfun/chebfun) into Python
using JAX as the primary array backend. Every translated function must be:
- Tested against MATLAB Chebfun reference outputs (rtol ≤ 1e-12)
- Credited to original Chebfun authors with provenance tracked
- Documented with NumPy-style docstrings preserving algorithm descriptions
- GPU-transparent via JAX (no device management in library code)

## Key References

- MATLAB Chebfun source (commit 7574c77): `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/`
- Python chebpy reference: `/scratch/gpfs/GILLES/mg6942/chebpy_ref/`
- MATLAB: `module load matlab/R2025b`
- Chebfun Guide: https://www.chebfun.org/docs/guide/
- Package name: `chebfunjax` (import as `import chebfunjax as cj`)
- Repo: `git@github.com:ma-gilles/chebfunjax.git`

## Architecture (Summary — see PLAN.md for details)

- **JAX-only**: `jax.numpy` everywhere, no numpy fallback, `float64` always
- **Equinox Modules**: all objects are frozen `eqx.Module` pytrees (JIT/vmap compatible)
- **Immutable**: operations return new objects, never mutate
- **JIT hot paths**: evaluation, differentiation, integration, rootfinding → `@jax.jit`
- **Python outer loops**: adaptive construction, convergence checks → plain Python
- **GPU transparent**: library never manages devices; users control via `jax.default_device`

## Rules

1. **Never import numpy in library code** — only `jax.numpy`. Tests may use numpy for comparison.
2. **Always `dtype=jnp.float64`** for array creation.
3. **Every public function has a Provenance section** in its docstring (MATLAB source, commit, authors, algorithm references). See PLAN.md Section 3.
4. **Never widen test tolerances** without documenting why.
5. **One agent per translation unit**. Never two agents on the same file.
6. **Branch naming**: `translate/U{XX}-{short-name}`.
7. **Check STATUS.md** before starting work to avoid conflicts.
8. **Run all tests** before pushing: `pixi run test-fast && pixi run test-matlab`.

## Quick Commands

```bash
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun

# Environment
pixi install
pixi run smoke                    # verify import works

# Tests
pixi run test-fast                # unit tests (no MATLAB, no GPU)
pixi run test-matlab              # MATLAB cross-validation
pixi run test-full                # everything
pixi run lint                     # ruff check

# MATLAB references
module load matlab/R2025b
matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('matlab_harness/generate_refs.m')"

# JAX device check
pixi run check-jax
```

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
