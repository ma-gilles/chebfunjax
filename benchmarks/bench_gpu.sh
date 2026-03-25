#!/bin/bash
#SBATCH --job-name=chebfunjax-gpu-bench
#SBATCH --partition=cryoem
#SBATCH --account=amits
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

# ---------------------------------------------------------------------------
# GPU benchmark: chebfunjax CPU vs GPU + vmap scaling
# ---------------------------------------------------------------------------

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp
mkdir -p "$TMPDIR"

REPO=/scratch/gpfs/GILLES/mg6942/jaxchebfun
BENCH_DIR="$REPO/benchmarks"
PIXI="$REPO/.pixi/envs/default/bin/python"

echo "========================================================"
echo "chebfunjax GPU benchmarks"
echo "Job: $SLURM_JOB_ID   Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================"

# Verify GPU is visible
nvidia-smi
"$PIXI" -c "import jax; print('JAX devices:', jax.devices())"

# ---- CPU run ----
echo ""
echo "========================================================"
echo "=== CPU benchmark ==="
echo "========================================================"
JAX_PLATFORMS=cpu "$PIXI" "$BENCH_DIR/bench_comparison.py" \
    --device cpu \
    --out "$BENCH_DIR/python_results_cpu.json"

# ---- GPU run ----
echo ""
echo "========================================================"
echo "=== GPU benchmark ==="
echo "========================================================"
"$PIXI" "$BENCH_DIR/bench_comparison.py" \
    --device gpu \
    --out "$BENCH_DIR/python_results_gpu.json"

# ---- GPU-specific vmap scaling benchmark ----
echo ""
echo "========================================================"
echo "=== GPU vmap scaling benchmark ==="
echo "========================================================"
"$PIXI" "$BENCH_DIR/bench_gpu_vmap.py" \
    --out "$BENCH_DIR/python_results_gpu_vmap.json"

# ---- Summary comparison ----
echo ""
echo "========================================================"
echo "=== CPU vs GPU comparison ==="
echo "========================================================"
"$PIXI" "$BENCH_DIR/summarize_results.py"

echo ""
echo "Done. $(date)"
