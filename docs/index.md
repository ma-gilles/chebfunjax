# chebfunjax

**Chebfun in Python, powered by JAX.**

chebfunjax is a faithful translation of the [MATLAB Chebfun](https://www.chebfun.org/) library to Python,
using [JAX](https://github.com/jax-ml/jax) as the numerical backend.
It brings Chebfun's elegant spectral-method interface to the Python ecosystem while
adding GPU acceleration, JIT compilation, automatic differentiation, and
vectorized batch evaluation.

---

## What is Chebfun?

Chebfun represents smooth functions as Chebyshev series with *adaptive* degree:
given a callable, it finds the minimum number of Chebyshev coefficients needed to
represent that function to machine precision (roughly 1e-15 in float64).
All the usual calculus operations — differentiation, integration, rootfinding,
ODE solving — are then performed algebraically on the coefficients, giving
spectral accuracy automatically.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Adaptive approximation** | Automatically determines the polynomial degree needed to represent a function to machine precision |
| **Full calculus** | Differentiation, antiderivative, definite integral, norms — all spectrally accurate |
| **Rootfinding** | Finds all real roots of a function on an interval via the companion-matrix eigenvalue method |
| **ODE/BVP solving** | `Chebop` class solves linear and nonlinear boundary-value problems via spectral collocation |
| **PDE time-stepping** | `SpinOp` solves periodic PDEs (Allen-Cahn, KdV, NLS) using ETDRK4 |
| **2D and 3D functions** | `Chebfun2` (low-rank on rectangles), `Diskfun`, `Spherefun`, `Chebfun3` (Tucker) |
| **GPU-ready** | All evaluation operations are JIT-compiled and run on GPU transparently |
| **JAX-native** | `jax.grad`, `jax.jit`, and `jax.vmap` work on Chebfun evaluation |

---

## Comparison with MATLAB Chebfun

| | MATLAB Chebfun | chebfunjax |
|---|---|---|
| Language | MATLAB | Python |
| Backend | MATLAB built-ins | JAX (CPU/GPU) |
| JIT compilation | No | Yes (`jax.jit`) |
| Automatic differentiation | No | Yes (`jax.grad`) |
| Batch evaluation | No | Yes (`jax.vmap`) |
| GPU support | No | Yes |
| float64 | Default | Forced on by default |
| MATLAB parity | — | ~90%, MATLAB-validated at rtol ≤ 1e-12 |

---

## What's Included

| Domain | Classes |
|--------|---------|
| 1D functions | `Chebfun`, `Chebtech2`, `Trigtech` |
| 1D bounded [a,b] | `Bndfun` |
| 1D unbounded | `Unbndfun` |
| 1D singular | `Singfun`, `Deltafun` |
| 2D rectangles | `Chebfun2`, `SeparableApprox` |
| 2D disk | `Diskfun` |
| 2D sphere | `Spherefun` |
| 3D cuboids | `Chebfun3` (Tucker decomposition) |
| ODE/BVP solving | `Linop`, `Chebop` |
| PDE time-stepping | `SpinOp`, ETDRK4 |
| 2D PDE | `Chebop2` (Poisson, Helmholtz) |
| Discretization | `ChebColloc`, `UltraS` |
| Rational approx | `aaa` (AAA algorithm) |

---

## Quick Example

```python
import jax.numpy as jnp
import chebfunjax as cj

# Approximate sin(x) on [-1, 1]
f = cj.chebfun(jnp.sin)
print(f)                    # Chebfun with 14 coefficients

# Evaluate, integrate, find roots
print(f(0.5))               # 0.4794255386042...
print(f.sum())              # ~0.0  (integral of sin on [-1,1])
print(f.roots())            # [0.]

# Differentiation
fp = f.diff()               # derivative  (= cos)
F  = f.cumsum()             # antiderivative

# Arithmetic
g = cj.chebfun(jnp.cos)
h = f**2 + g**2             # sin^2 + cos^2  (= 1 to machine precision)
print(h.norm() - jnp.sqrt(2.0))   # ~0.0
```

---

## Architecture

- **JAX-only backend** — `jax.numpy` everywhere, GPU-transparent
- **Equinox Modules** — immutable pytree objects, JIT/vmap compatible
- **float64 always** — spectral methods need double precision
- **MATLAB-validated** — every function tested against MATLAB Chebfun at `rtol ≤ 1e-12`

See the [JAX Contract](jax-contract.md) page for which operations are JIT/grad/vmap-safe
and [API Reference](api.md) for the full class and function listing.

---

## Credits

This project is a derivative work of [Chebfun](https://www.chebfun.org/).
Each function tracks provenance back to the original MATLAB source (commit `7574c77`),
preserving author credits and algorithm references.

Licensed under BSD-3-Clause. Original Chebfun is BSD-2-Clause. See the
[LICENSE](https://github.com/ma-gilles/chebfunjax/blob/main/LICENSE) file.

### References

- [Chebfun Guide](https://www.chebfun.org/docs/guide/)
- Trefethen, *Approximation Theory and Approximation Practice* (2013)
- Driscoll, Hale, & Trefethen, "Chebfun Guide" (2014)
