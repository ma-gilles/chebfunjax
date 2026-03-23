# chebfunjax — Chebfun in Python/JAX

## READ FIRST

Before doing ANY work, read these documents in order:
1. **This file** — quick reference and rules
2. **`PLAN.md`** — locked design decisions, code templates, module dependency graph, all translation units
3. **`STATUS.md`** — what's done, what's in progress, what's available
4. **`.claude/skills/translate-module.md`** — step-by-step procedure for translating a module
5. **`project.conf`** — all paths, accounts, and thresholds (source it, don't hardcode)

## Project Goal

Translate the MATLAB Chebfun library (https://github.com/chebfun/chebfun) into Python
using JAX as the primary array backend. Every translated function must be:
- Tested against MATLAB Chebfun reference outputs (rtol ≤ 1e-12)
- Credited to original Chebfun authors with provenance tracked
- Documented with NumPy-style docstrings preserving algorithm descriptions
- GPU-transparent via JAX (no device management in library code)

## Key References

All paths and constants are in `project.conf`. Source it:
```bash
source project.conf
```

| What | Variable | Default |
|------|----------|---------|
| Package name | `$PACKAGE_NAME` | `chebfunjax` |
| Repo SSH | `$REPO_URL` | `git@github.com:ma-gilles/chebfunjax.git` |
| MATLAB Chebfun | `$CHEBFUN_REF` | `/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref` |
| Chebfun commit | `$CHEBFUN_COMMIT` | `7574c77` |
| chebpy ref | `$CHEBPY_REF` | `/scratch/gpfs/GILLES/mg6942/chebpy_ref` |
| MATLAB module | `$MATLAB_MODULE` | `matlab/R2025b` |
| Slurm account | `$SLURM_ACCOUNT` | `amits` |
| Slurm partition | `$SLURM_PARTITION` | `cryoem` |
| Slurm logs | `$SLURM_LOG_DIR` | `/scratch/gpfs/GILLES/mg6942/slurmo` |
| Default rtol | `$DEFAULT_RTOL` | `1e-12` |

## Architecture (Summary — see PLAN.md for details)

- **JAX-only**: `jax.numpy` everywhere, no numpy fallback, `float64` always
- **Equinox Modules**: all objects are frozen `eqx.Module` pytrees (JIT/vmap compatible)
- **Immutable**: operations return new objects, never mutate
- **JIT hot paths**: evaluation, differentiation, integration, rootfinding → `@jax.jit`
- **Python outer loops**: adaptive construction, convergence checks → plain Python
- **GPU transparent**: library never manages devices; users control via `jax.default_device`

## Workflow (fully autonomous)

1. Agent clones repo into isolated workdir (see skill §1)
2. Agent works on a branch: `translate/U{XX}-{short-name}`
3. Agent pushes branch and opens PR with `--auto-merge` enabled
4. **CI is the sole reviewer** — no human approval required
5. CI passes → PR auto-merges to main. CI fails → agent fixes and re-pushes.

CI gates: lint, no-numpy-import, provenance docstrings, tests, coverage ≥ 90%, golden-ref validation.
See `.github/workflows/ci.yml` for details.

## Rules

1. **Never import numpy in library code** — only `jax.numpy`. Tests may use numpy for comparison.
2. **Always `dtype=jnp.float64`** for array creation.
3. **Every public function has a Provenance section** in its docstring. See PLAN.md §3.
4. **Never widen test tolerances** without documenting why.
5. **One agent per translation unit**. Never two agents on the same file.
6. **Branch naming**: `translate/U{XX}-{short-name}`.
7. **Check STATUS.md** before starting work to avoid conflicts.
8. **All PRs auto-merge after CI green** — CI is the sole reviewer.

## Quick Commands

```bash
source project.conf

# Environment
pixi install
pixi run smoke

# Tests
pixi run test-fast                # unit tests (no MATLAB, no GPU)
pixi run test-matlab              # MATLAB cross-validation (uses committed golden refs)
pixi run test-full                # everything with coverage

# Regenerate MATLAB refs (maintainer task, requires MATLAB)
module load $MATLAB_MODULE
matlab -batch "addpath('$CHEBFUN_REF'); run('matlab_harness/generate_refs.m')"

# Lint
pixi run lint

# JAX device check
pixi run check-jax
```

## Slurm (for GPU tests)

Agents use the Slurm template in the skill (§9). For quick manual use:

```bash
source project.conf  # sets SCRATCH, SLURM_ACCOUNT, SLURM_PARTITION, SLURM_LOG_DIR
WORKDIR="$(pwd)"     # must be set to your chebfunjax checkout

cat > "${SLURM_LOG_DIR}/job_gpu_test.sh" << EOF
#!/bin/bash
#SBATCH --partition=${SLURM_PARTITION} --account=${SLURM_ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${SLURM_LOG_DIR}/%j-%x.out

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${SCRATCH}/tmp/slurm_\${SLURM_JOB_ID}
mkdir -p "\${TMPDIR}"

cd ${WORKDIR}
pixi run test-full
EOF
sbatch "${SLURM_LOG_DIR}/job_gpu_test.sh"
```
