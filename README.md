# chebfunjax

Chebfun in Python, powered by JAX.

A **complete translation** of the [MATLAB Chebfun](https://github.com/chebfun/chebfun) library
to Python using [JAX](https://github.com/jax-ml/jax) as the numerical backend,
enabling GPU acceleration, JIT compilation, automatic differentiation, and
vectorized operations (`vmap`) for spectral methods and function approximation.

This includes all major Chebfun features: 1D/2D/3D function approximation (Chebfun, Chebfun2, Chebfun3),
ODE/BVP/eigenvalue solvers (Chebop, Linop), PDE time-stepping (SpinOp with ETDRK4 and IMEX schemes),
functions on disks, spheres, and balls (Diskfun, Spherefun, Ballfun), rational approximation (AAA, minimax),
operator automatic differentiation, and more.
See the [full examples gallery](https://ma-gilles.github.io/chebfunjax/) with 350+ translated examples.

## Installation

```bash
git clone https://github.com/ma-gilles/chebfunjax.git
cd chebfunjax
pip install -e .
```

Or with [pixi](https://pixi.sh) (recommended, handles JAX+CUDA automatically):

```bash
git clone https://github.com/ma-gilles/chebfunjax.git
cd chebfunjax
pixi install
pixi run smoke   # verify: "chebfunjax imported OK"
```

## Quick Start

```python
import jax.numpy as jnp
import chebfunjax as cj

# 1D: approximate sin(x) on [-1, 1]
f = cj.chebfun(jnp.sin)
print(f)                              # 14 Chebyshev coefficients
print('f(0.5) =', f(0.5))            # evaluate
print('integral =', f.sum())          # definite integral (= 0 by symmetry)
print('roots =', f.roots())           # find zeros

# Calculus
fp = f.diff()                         # derivative (= cos)
F = f.cumsum()                        # antiderivative

# Arithmetic
g = cj.chebfun(jnp.cos)
h = f**2 + g**2                       # sin^2 + cos^2 = 1

# Special functions
g = cj.exp(f)                         # exp(sin(x))

# Custom domain
f2 = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print('integral of sin on [0,pi] =', f2.sum())  # = 2.0

# 2D: approximate cos(x+y) on [-1,1]^2
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
print('double integral =', g2.sum2())

# 3D: approximate cos(x+y+z) on [-1,1]^3
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
g3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
print('triple integral =', g3.sum3())

# ODE solving: u'' = -1, u(-1) = u(1) = 0
from chebfunjax.operators import Chebop
N = Chebop(lambda x, u: u.diff(2), domain=[-1, 1])
N.lbc = 0
N.rbc = 0
u = N.solve(-1)  # u = (1 - x^2) / 2
```

## JAX Features

All evaluation operations are JIT-compiled, differentiable, and vectorizable:

```python
import jax

# JIT-compiled evaluation
fast_f = jax.jit(lambda x: f(x))

# Automatic differentiation
df_dx = jax.grad(lambda x: f(x))(0.5)  # = cos(0.5)

# Batched evaluation
xs = jnp.linspace(-1, 1, 1000)
ys = jax.vmap(lambda x: f(x))(xs)
```

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
| Discretization | `ChebColloc`, `UltraS` |
| Rational approx | `aaa` (AAA algorithm) |

See [STATUS.md](STATUS.md) for detailed progress.

## Architecture

- **JAX-only backend** — `jax.numpy` everywhere, GPU-transparent
- **Equinox Modules** — immutable pytree objects, JIT/vmap compatible
- **float64 always** — spectral methods need double precision
- **MATLAB-validated** — every function tested against MATLAB Chebfun at `rtol ≤ 1e-12`

See [docs/architecture.md](docs/architecture.md) for design decisions and
[docs/jax-contract.md](docs/jax-contract.md) for what is/isn't JIT/grad/vmap-safe.

## Credits

This project is a derivative work of [Chebfun](https://www.chebfun.org/).
Each function tracks provenance back to the original MATLAB source
(commit `7574c77`), preserving author credits and algorithm references.

Licensed under BSD-3-Clause. Original Chebfun is BSD-2-Clause. See [LICENSE](LICENSE).

## References

- [Chebfun Guide](https://www.chebfun.org/docs/guide/)
- Trefethen, *Approximation Theory and Approximation Practice* (2013)
- Driscoll, Hale, & Trefethen, "Chebfun Guide" (2014)
