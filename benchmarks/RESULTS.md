# chebfunjax Benchmark Results

> Generated: 2026-03-25 15:10 UTC

## System

- **Python/JAX CPU device**: `TFRT_CPU_0`
- **GPU device**: `cuda:0`

## Notes

- **MATLAB** timings: warm loop averages from `matlab_timing.m` (R2025b,
  Chebfun commit 7574c77 equivalent).
- **Python CPU** timings: warm mean (after JIT compilation) from
  `bench_comparison.py --device cpu`.
- **Python GPU** timings: warm mean from `bench_comparison.py --device gpu`.
- *Speedup* = MATLAB_time / Python_warm_mean. Values > 1 mean chebfunjax is
  faster; values < 1 mean MATLAB is faster.
- First-call (JIT compile) times are **not** shown here; see the raw JSON
  files for first-call overhead.

## 1D Chebfun Results

| Operation | MATLAB (ms) | Python CPU (ms) | Python GPU (ms) | CPU speedup | GPU speedup |
|-----------|------------|----------------|----------------|-------------|-------------|
| construct sin(x) | 17.048 ms | 3.452 ms | 8.149 ms | 4.9x faster | 2.1x faster |
| eval f(0.5) single pt | 0.1675 ms | 0.1758 ms | 0.4669 ms | 1.0x slower | 2.8x slower |
| eval f(1000 pts) | 0.0842 ms | 0.1329 ms | 0.3877 ms | 1.6x slower | 4.6x slower |
| diff sin(10x) | 1.063 ms | 2.935 ms | 5.671 ms | 2.8x slower | 5.3x slower |
| sum exp(-x²) | 0.0798 ms | 0.4791 ms | 1.021 ms | 6.0x slower | 12.8x slower |
| roots sin(5πx) | 3.217 ms | 0.8091 ms | 1.065 ms | 4.0x faster | 3.0x faster |

### 1D Construction Scaling (fixed degree)

| n (degree) | MATLAB (ms) | CPU (ms) | GPU (ms) | CPU speedup | GPU speedup |
|-----------|------------|---------|---------|-------------|-------------|
| n=10     | 13.787 ms | 1.195 ms | 2.267 ms | 12x faster | 6.1x faster |
| n=100    | 4.435 ms | 1.205 ms | 2.279 ms | 3.7x faster | 1.9x faster |
| n=1000   | 10.054 ms | 1.275 ms | 2.435 ms | 7.9x faster | 4.1x faster |
| n=10000  | N/A | 1.226 ms | 2.391 ms | N/A | N/A |

## 2D Chebfun2 Results

| Operation | MATLAB (ms) | Python CPU (ms) | Python GPU (ms) | CPU speedup | GPU speedup |
|-----------|------------|----------------|----------------|-------------|-------------|
| construct cos(x+y) | 135.514 ms | 17.719 ms | 40.156 ms | 7.6x faster | 3.4x faster |
| eval (1000 pts) | 0.8518 ms | 0.1463 ms | 0.8042 ms | 5.8x faster | 1.1x faster |
| diff (x-direction) | 0.6711 ms | 30.790 ms | 59.026 ms | 45.9x slower | 88.0x slower |
| sum2 | 0.2978 ms | 1.691 ms | 3.939 ms | 5.7x slower | 13.2x slower |
| norm | 1.950 ms | 36.341 ms | 69.431 ms | 18.6x slower | 35.6x slower |

## 3D Chebfun3 Results

| Operation | MATLAB (ms) | Python CPU (ms) | Python GPU (ms) | CPU speedup | GPU speedup |
|-----------|------------|----------------|----------------|-------------|-------------|
| construct cos(x+y+z) | 337.310 ms | 38.251 ms | 76.266 ms | 8.8x faster | 4.4x faster |
| eval (500 pts) | 38.083 ms | 0.1468 ms | 1.153 ms | 259x faster | 33x faster |

## GPU & vmap Benchmarks

### GPU Evaluation Throughput (N points)

| N points | Mean (ms) | Throughput (pts/s) |
|---------|---------|-------------------|
| 100 | 0.3615 ms | 2.77e+05 |
| 1000 | 0.3864 ms | 2.59e+06 |
| 10000 | 0.3928 ms | 2.55e+07 |
| 100000 | 0.4419 ms | 2.26e+08 |
| 1000000 | 0.8273 ms | 1.21e+09 |

### vmap Batch Eval (32 functions, 1000 pts each)

| Mode | Mean (ms) | Speedup |
|------|---------|--------|
| loop (sequential) | 37.509 ms | 1.0x |
| vmap (batched) | 1.725 ms | 21.7x |

### 2D Chebfun2 Grid Evaluation Throughput

| Grid | N total pts | Mean (ms) | Throughput (pts/s) |
|------|------------|---------|-------------------|
| 50×50 | 2500 | 0.7039 ms | 3.55e+06 |
| 100×100 | 10000 | 0.7021 ms | 1.42e+07 |
| 200×200 | 40000 | 0.7298 ms | 5.48e+07 |
| 500×500 | 250000 | 0.7931 ms | 3.15e+08 |

## How to Reproduce

```bash
# 1. MATLAB timings (requires MATLAB R2025b)
cd /scratch/gpfs/GILLES/mg6942/jaxchebfun
module load matlab/R2025b
matlab -batch "run('benchmarks/matlab_timing.m')"

# 2. Python CPU benchmarks
pixi run python benchmarks/bench_comparison.py --device cpu

# 3. GPU benchmarks (Slurm)
sbatch benchmarks/bench_gpu.sh
# outputs saved to benchmarks/python_results_{cpu,gpu}.json
# and benchmarks/python_results_gpu_vmap.json

# 4. Re-generate this table
pixi run python benchmarks/summarize_results.py
```

## Raw Data

- [`matlab_results.json`](matlab_results.json) — MATLAB reference timings
- [`python_results_cpu.json`](python_results_cpu.json) — Python/JAX CPU (warm + first-call)
- [`python_results_gpu.json`](python_results_gpu.json) — Python/JAX GPU (warm + first-call)
- [`python_results_gpu_vmap.json`](python_results_gpu_vmap.json) — GPU vmap scaling
