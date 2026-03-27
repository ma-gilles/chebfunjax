# Chapter 10: Nonlinear ODEs and Initial-Value Problems

*Based on [Chebfun Guide Chapter 10](https://www.chebfun.org/docs/guide/guide10.html)*

## 10.1 Introduction

Chapter 7 described chebfunjax's capabilities for solving linear ordinary differential equations. This chapter extends those capabilities to:

- **Nonlinear boundary-value problems** (BVPs), solved by Newton iteration.
- **Initial-value problems** (IVPs), solved either by global spectral collocation or by time-stepping methods.

The key references are Birkisson (2014), Birkisson & Driscoll (2012), and Trefethen, Birkisson & Driscoll (2018, *Exploring ODEs*).

## 10.2 Nonlinear Boundary-Value Problems

### The `Chebop` Approach

For nonlinear problems, the `Chebop` class uses exactly the same syntax as for linear problems. The system automatically detects nonlinearity and applies Newton iteration.

```python
from chebfunjax.operators.chebop import Chebop
import jax.numpy as jnp

# 0.001*u'' - u^3 = 0, u(-1) = 1, u(1) = -1
N = Chebop(lambda x, u: 0.001 * u.diff(2) - u**3, domain=(-1.0, 1.0))
N.lbc = 1.0
N.rbc = -1.0
u = N.solve(0.0)
```

### How Newton Iteration Works

For a nonlinear operator $\mathcal{N}[u] = f$, chebfunjax applies Newton's method in function space:

1. Start from an initial guess $u_0$ (default: the zero function).
2. At each step $k$, compute the residual $r_k = \mathcal{N}[u_k] - f$.
3. Linearize $\mathcal{N}$ around $u_k$ to obtain the Frechet derivative (Jacobian) $J_k$.
4. Solve the linear correction equation $J_k \delta u = -r_k$ using spectral collocation.
5. Update: $u_{k+1} = u_k + \delta u$.
6. Repeat until $\|\delta u\| < \texttt{newton\_tol}$.

The Jacobian is computed by one of two methods:
- **Symbolic linearization** via `ADChebfun` (automatic differentiation on chebfuns), when available.
- **Finite-difference linearization**: perturbing each collocation-point basis vector and observing the change in the operator output.

### Newton Solver Parameters

The `Chebop.solve` method accepts parameters controlling the Newton iteration:

```python
u = N.solve(
    f=0.0,            # right-hand side
    n=64,             # fixed discretization size (None = adaptive)
    max_iter=15,      # maximum Newton iterations
    newton_tol=5e-13, # convergence tolerance for ||delta_u||
    tol=1e-10,        # coefficient decay tolerance for adaptive sizing
)
```

### Linearity Detection

Before starting Newton iteration, `Chebop` automatically checks whether the operator is linear:

1. **Exact detection** via symbolic AD (`detect_linearity`): checks whether the Frechet derivative is constant.
2. **Numerical probe**: evaluates the operator on the zero function, a probe function $p$, and $2p$, then checks whether $\mathcal{N}[2p] - \mathcal{N}[0] \approx 2(\mathcal{N}[p] - \mathcal{N}[0])$.

If the operator is linear, `Chebop` delegates to `Linop.solve` (direct spectral solve) without Newton iteration.

## 10.3 Examples of Nonlinear BVPs

### A Boundary Layer Problem

$$\varepsilon u'' + xu = e^x, \quad u(-1) = 0, \quad u(1) = 1$$

with $\varepsilon = 0.0001$:

```python
N = Chebop(
    lambda x, u: 0.0001 * u.diff(2) + x * u,
    domain=(-1.0, 1.0),
)
N.lbc = 0.0
N.rbc = 1.0
u = N.solve(lambda x: jnp.exp(x))
```

### Carrier's Problem

$$\varepsilon u'' + 2(1-x^2)u + u^2 = 1, \quad u(-1) = u(1) = 0$$

with $\varepsilon = 0.01$. This problem has multiple solutions depending on the initial guess:

```python
import chebfunjax as cj

ep = 0.01
N = Chebop(
    lambda x, u: ep * u.diff(2) + 2 * (1 - x**2) * u + u**2,
    domain=(-1.0, 1.0),
)
N.lbc = 0.0
N.rbc = 0.0

# Default initial guess (zero function) finds one solution
u1 = N.solve(1.0)
```

### The Bratu Equation

$$u'' + \lambda e^u = 0, \quad u(0) = u(1) = 0$$

```python
lam = 1.0
N = Chebop(
    lambda x, u: u.diff(2) + lam * cj.exp(u),
    domain=(0.0, 1.0),
)
N.lbc = 0.0
N.rbc = 0.0
u = N.solve(0.0)
```

## 10.4 Initial-Value Problems with `ivp`

An IVP is a special case of a BVP where all boundary conditions are imposed at the left endpoint. The `ivp` function provides a convenient wrapper:

```python
from chebfunjax.chebfun1d.ode import ivp

# u' = u^2, u(0) = 0.95, t in [0, 1]
# Exact solution: u(t) = 0.95 / (1 - 0.95*t)
u = ivp(
    lambda t, u: u.diff() - u**2,
    domain=(0.0, 1.0),
    ic=[0.95],
)
exact = lambda t: 0.95 / (1.0 - 0.95 * t)
print(f"u(0.5) = {float(u(jnp.float64(0.5))):.10f}")
print(f"exact  = {exact(0.5):.10f}")
```

### Multiple Initial Conditions

For higher-order equations, provide a list of initial values $[u(a), u'(a), u''(a), \ldots]$:

```python
# u'' + u = 0, u(0) = 1, u'(0) = 0 => u = cos(t)
u = ivp(
    lambda t, u: u.diff(2) + u,
    domain=(0.0, 10.0),
    ic=[1.0, 0.0],
)
print(f"u(pi) = {float(u(jnp.float64(jnp.pi))):.10f}")
print(f"cos(pi) = {float(jnp.cos(jnp.pi)):.10f}")
```

### Chebop for IVPs

You can also use `Chebop` directly by setting only `lbc`:

```python
N = Chebop(lambda t, u: u.diff(2) + u, domain=(0.0, 100.0))
N.lbc = [1.0, 0.0]   # u(0) = 1, u'(0) = 0
u = N.solve(0.0)
```

## 10.5 Time-Stepping Methods: `ode45`, `ode78`, `ode89`

For IVPs where global spectral methods may struggle (e.g., chaotic systems, stiff problems, very long time intervals), chebfunjax provides ODE integrators that wrap SciPy's `solve_ivp` and return the result as a Chebfun:

- **`ode78`**: 7(8)-order Runge-Kutta (via SciPy's DOP853).
- **`ode89`**: 8(9)-order Runge-Kutta at very tight tolerances.

```python
import chebfunjax as cj

# u' = u, u(0) = 1 => u = exp(t)
sol = cj.ode78(
    lambda t, y: y,
    tspan=(0.0, 1.0),
    y0=jnp.array([1.0]),
)
print(f"u(1) = {float(sol(jnp.float64(1.0))):.10f}")
print(f"exp(1) = {float(jnp.exp(jnp.float64(1.0))):.10f}")
```

### Controlling Tolerance

```python
# Higher accuracy with ode89
sol = cj.ode89(
    lambda t, y: y,
    tspan=(0.0, 1.0),
    y0=jnp.array([1.0]),
    rtol=1e-12,
    atol=1e-14,
)
```

### Comparison of Methods

| Method  | Order | Default `rtol` | Default `atol` | Best for                     |
|---------|-------|----------------|----------------|------------------------------|
| `ode78` | 7(8)  | $10^{-8}$      | $10^{-10}$     | General non-stiff problems   |
| `ode89` | 8(9)  | $10^{-10}$     | $10^{-12}$     | High-accuracy requirements   |

## 10.6 Nonlinear IVP Example: The Van der Pol Oscillator

The Van der Pol equation is a classic nonlinear oscillator:

$$\varepsilon u'' = (1 - u^2)u' - u, \quad u(0) = 3, \quad u'(0) = 0.$$

```python
ep = 0.05
N = Chebop(
    lambda t, u: ep * u.diff(2) - (1 - u**2) * u.diff() + u,
    domain=(0.0, 20.0),
)
N.lbc = [3.0, 0.0]
u = N.solve(0.0, n=256)
```

## 10.7 BVP Convenience Functions: `bvp4c` and `bvp5c`

Chebfunjax provides `bvp4c` and `bvp5c` functions that mirror the MATLAB interfaces. These wrap the standard `bvp` solver with collocation refinement:

```python
from chebfunjax.chebfun1d.ode import bvp4c, bvp5c

# u'' = -1, u(-1) = 0, u(1) = 0
u = bvp4c(
    lambda x, u: u.diff(2),
    domain=(-1.0, 1.0),
    lbc=0.0,
    rbc=0.0,
    f=-1.0,
)
print(f"u(0) = {float(u(jnp.float64(0.0))):.10f}")
```

The key difference from `bvp` is that `bvp4c`/`bvp5c` enable collocation refinement by default: if the initial solve is not sufficiently resolved (tail coefficients have not decayed below `tol`), the discretization size is doubled and the solve is repeated.

| Function | Default `tol` | Refinement doublings |
|----------|---------------|---------------------|
| `bvp4c`  | $10^{-6}$     | 2                   |
| `bvp5c`  | $10^{-6}$     | 3                   |

## 10.8 Eigenvalue Problems for Nonlinear Operators

The `Chebop.eigs` method linearizes a nonlinear operator around the zero function and computes eigenvalues of the resulting linear operator:

```python
N = Chebop(lambda x, u: u.diff(2) + u, domain=(0.0, float(jnp.pi)))
N.lbc = 0.0
N.rbc = 0.0
lam = N.eigs(k=4)
print("Eigenvalues:", lam)
```

For a truly nonlinear operator, the eigenvalues describe the linearization at $u = 0$, which may not be physically meaningful for all problems.

## 10.9 PDE Solvers

Chebfunjax includes method-of-lines PDE solvers for time-dependent problems on 1D spatial domains:

```python
import chebfunjax as cj

# Heat equation: u_t = u_xx on [-1, 1]
# with Dirichlet BCs u(-1,t) = u(1,t) = 0
u0 = cj.chebfun(lambda x: jnp.exp(-20 * x**2))

# pdeSolve integrates in time using the method of lines
import jax.numpy as jnp
t_span = jnp.linspace(0.0, 0.1, 50)
# u = cj.pdeSolve(pdefun, t_span, u0, ...)
```

The `pde15s` module provides additional PDE solving capabilities based on implicit time-stepping.

## 10.10 References

- C. M. Bender and S. A. Orszag, *Advanced Mathematical Methods for Scientists and Engineers*, McGraw-Hill, 1978.

- A. Birkisson, *Numerical Solution of Nonlinear Boundary Value Problems for ODEs in the Continuous Framework*, D.Phil. thesis, University of Oxford, 2014.

- A. Birkisson and T. A. Driscoll, "Automatic Frechet differentiation for the numerical solution of boundary-value problems," *ACM Trans. Math. Softw.* 38 (2012), 1-26.

- L. N. Trefethen, A. Birkisson, and T. A. Driscoll, *Exploring ODEs*, SIAM, 2018.
