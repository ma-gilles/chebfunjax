# API Reference

This page summarises the main public classes and functions in chebfunjax.
See the source files in `src/chebfunjax/` for full docstrings.

---

## Top-Level (`chebfunjax` / `cj`)

```python
import chebfunjax as cj
```

### Factory functions

| Name | Signature | Description |
|------|-----------|-------------|
| `chebfun` | `chebfun(f, domain=None, n=None)` | Construct a 1D Chebfun from a callable. Adaptive by default; pass `n` for fixed degree. |
| `chebfun2` | `chebfun2(f, domain=None)` | Construct a 2D Chebfun2 from a bivariate callable. |

### Special functions (top-level aliases)

Each `cj.func(f)` is equivalent to `f.func()`.

| `cj.sin` | `cj.cos` | `cj.exp` | `cj.log` | `cj.sqrt` |
|----------|----------|----------|----------|-----------|
| `cj.abs` | `cj.sign` | `cj.sinh` | `cj.cosh` | `cj.tanh` |
| `cj.asin` | `cj.acos` | `cj.atan` | | |

---

## 1D Chebfun (`chebfunjax.chebfun1d`)

### `Chebfun`

Piecewise smooth function approximation on a domain.

```python
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
```

**Construction**

| Method | Description |
|--------|-------------|
| `chebfun(f, domain, n)` | Adaptive or fixed-degree approximation of a callable |
| `Chebfun.from_values(values, domain)` | Interpolate from values at Chebyshev-2 points |
| `Chebfun.from_coeffs(coeffs, domain)` | Build from Chebyshev coefficients |

**Evaluation**

| Method | Description |
|--------|-------------|
| `f(x)` | Evaluate at scalar or array `x`. JIT/vmap/grad-safe. |

**Calculus**

| Method | Signature | Description |
|--------|-----------|-------------|
| `.diff` | `f.diff(k=1)` | k-th derivative |
| `.cumsum` | `f.cumsum()` | Antiderivative with F(a) = 0 |
| `.sum` | `f.sum()` | Definite integral over domain |
| `.mean` | `f.mean()` | Mean value over domain |
| `.norm` | `f.norm(p=2)` | Lp norm |
| `.inner` | `f.inner(g)` | L2 inner product |

**Roots and extrema**

| Method | Description |
|--------|-------------|
| `.roots()` | All real roots on the domain, sorted |
| `.max()` | Returns `(x_max, f_max)` |
| `.min()` | Returns `(x_min, f_min)` |

**Arithmetic**

All standard Python operators are supported:
`f + g`, `f - g`, `f * g`, `f / g`, `f ** k`, and their scalar variants.

**Special functions (methods)**

`.sin()`, `.cos()`, `.exp()`, `.log()`, `.sqrt()`, `.abs()`, `.sign()`,
`.sinh()`, `.cosh()`, `.tanh()`, `.asin()`, `.acos()`, `.atan()`

---

## Tech Layer (`chebfunjax.tech`)

Low-level Chebyshev representations.  Most users interact with `Chebfun` instead.

### `Chebtech2`

```python
from chebfunjax.tech.chebtech import Chebtech2
```

Chebyshev expansion on the reference interval [-1, 1] using Chebyshev points
of the second kind.

| Method | Description |
|--------|-------------|
| `Chebtech2.from_function(f, n)` | Build from callable (adaptive or fixed n) |
| `Chebtech2.from_coeffs(c)` | Build from coefficient array |
| `Chebtech2.from_values(v)` | Build from values at Chebyshev-2 points |
| `.coeffs` | Chebyshev coefficients (array) |
| `.values` | Values at Chebyshev-2 points (array) |
| `.diff(k)` | Derivative |
| `.sum()` | Definite integral on [-1, 1] |
| `.roots()` | All real roots on [-1, 1] |

### `Trigtech`

Trigonometric (Fourier) expansion for periodic functions.

```python
from chebfunjax.tech.trigtech import Trigtech
```

---

## Utilities (`chebfunjax.utils`)

### Quadrature

```python
from chebfunjax.utils.quadrature import (
    chebpts, legpts, jacpts, hermpts, lagpts,
    ultrapts, radaupts, lobpts, trigpts,
)
```

| Function | Description |
|----------|-------------|
| `chebpts(n)` | Chebyshev-2 points on [-1,1]. JIT-safe with static `n`. |
| `legpts(n)` | Gauss-Legendre points and weights |
| `jacpts(n, a, b)` | Gauss-Jacobi points and weights |
| `hermpts(n)` | Gauss-Hermite points and weights |
| `lagpts(n)` | Gauss-Laguerre points and weights |

### Transforms

```python
from chebfunjax.utils.transforms import (
    vals2coeffs, coeffs2vals, cheb2leg, leg2cheb,
    cheb2jac, jac2cheb,
)
```

| Function | Description |
|----------|-------------|
| `vals2coeffs(v)` | DCT-based values → coefficients |
| `coeffs2vals(c)` | DCT-based coefficients → values |
| `cheb2leg(c)` | Chebyshev to Legendre coefficients |
| `leg2cheb(c)` | Legendre to Chebyshev coefficients |

### Interpolation

```python
from chebfunjax.utils.interpolation import bary, bary_weights, barymat
```

| Function | Description |
|----------|-------------|
| `bary(x, values, points, w)` | Barycentric interpolation |
| `bary_weights(n)` | Barycentric weights for Chebyshev-2 points |

### Differentiation / Integration Matrices

```python
from chebfunjax.utils.diffmat import diffmat, cumsummat, intmat
```

### Polynomials

```python
from chebfunjax.utils.polynomials import chebpoly, legpoly, jacpoly
```

### Rational Approximation

```python
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.minimax import minimax
from chebfunjax.utils.ratapprox import ratinterp, padeapprox
```

| Function | Description |
|----------|-------------|
| `aaa(f, z)` | AAA rational approximation (Adaptive Antoulas-Anderson) |
| `minimax(f, n, m)` | Best polynomial/rational approximation (Remez exchange) |
| `ratinterp(f, m, n)` | Rational interpolation |

---

## Operators (`chebfunjax.operators`)

### `Linop`

Linear differential operator for BVPs.

```python
from chebfunjax.operators.linop import Linop
```

### `Chebop`

User-friendly nonlinear BVP solver.

```python
from chebfunjax.operators import Chebop
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `.lbc` | float, callable, or str | Left boundary condition |
| `.rbc` | float, callable, or str | Right boundary condition |

| Method | Description |
|--------|-------------|
| `.solve(rhs)` | Solve `N[u] = rhs` |
| `N \ rhs` | Alias for `.solve(rhs)` |
| `.eigs(k)` | Compute first `k` eigenvalues and eigenfunctions |

### `Chebop2`

2D PDE solver on rectangles.

```python
from chebfunjax.operators.chebop2 import Chebop2
```

| Method | Description |
|--------|-------------|
| `.solve(rhs)` | Solve `L[u] = rhs` with boundary conditions |

---

## 2D Functions (`chebfunjax.chebfun2d`)

### `Chebfun2`

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
```

| Method | Description |
|--------|-------------|
| `Chebfun2.from_function(f, domain)` | Build from bivariate callable |
| `g(x, y)` | Evaluate at scalar or array points |
| `.diff(nx, ny)` | Partial derivative ∂^(nx+ny) / ∂x^nx ∂y^ny |
| `.sum2()` | Double integral over domain |
| `.sum(dim)` | Integrate over one variable |
| `.max2()` | Global maximum `(x, y, val)` |
| `.min2()` | Global minimum `(x, y, val)` |
| `.rank` | Low-rank approximation rank |

### `SeparableApprox`

```python
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
```

### `Chebfun2v`

2D vector field.

```python
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
```

---

## 3D Functions (`chebfunjax.chebfun3d`)

### `Chebfun3`

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
```

| Method | Description |
|--------|-------------|
| `Chebfun3.from_function(f, domain)` | Build from trivariate callable |
| `g(x, y, z)` | Evaluate at scalar or array points |
| `.diff(nx, ny, nz)` | Partial derivative |
| `.sum3()` | Triple integral over domain |

---

## PDE Time-Stepping (`chebfunjax.spin`)

### `SpinOp` (1D)

```python
from chebfunjax.spin.spinop import SpinOp
from chebfunjax.spin.solver import spin
```

| Built-in key | PDE |
|---|---|
| `'ac'` | Allen-Cahn: u_t = ε·u_xx + u - u³ |
| `'kdv'` | KdV: u_t + u·u_x + u_xxx = 0 |
| `'nls'` | Nonlinear Schrödinger |
| `'ks'` | Kuramoto-Sivashinsky |

```python
u, t, x = spin('ac', tspan=(0.0, 100.0))
```

### `SpinOp2`, `SpinOp3`, `SpinOpSphere` (2D/3D/sphere)

```python
from chebfunjax.spin.spinop2 import SpinOp2
from chebfunjax.spin.spinop3 import SpinOp3
from chebfunjax.spin.spinopsphere import SpinOpSphere
```

---

## Disk and Sphere Functions

### `Diskfun`, `Diskfunv`

```python
from chebfunjax.diskfun import Diskfun, Diskfunv
```

### `Spherefun`, `Spherefunv`

```python
from chebfunjax.spherefun import Spherefun, Spherefunv
```

---

## Discretization

```python
from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2
from chebfunjax.discretization.ultras import UltraS
from chebfunjax.discretization.trigcolloc import TrigColloc
```

---

## Automatic Differentiation (Fréchet)

```python
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.autodiff.treevar import TreeVar
```

`ADChebfun` provides exact Fréchet derivatives of Chebfun operators for use
in Newton iterations; `TreeVar` is the symbolic linearization backend.
