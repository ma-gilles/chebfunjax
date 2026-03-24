# PDE Examples

chebfunjax includes two PDE solvers:

- **`SpinOp`** — 1D periodic PDEs (Allen-Cahn, KdV, NLS, KS) via ETDRK4 time-stepping.
- **`Chebop2`** — 2D elliptic PDEs on rectangles (Poisson, Helmholtz) via spectral collocation.

---

## 1. Allen-Cahn Equation (SpinOp)

The Allen-Cahn equation models phase separation:

```
u_t = ε·u_xx + u - u³,    u periodic on [-π, π]
```

```python
from chebfunjax.spin.solver import spin

# Built-in Allen-Cahn problem — run to t = 150
# 'ac' is the built-in key; see SpinOp for all keys
u, t, x = spin('ac', tspan=(0.0, 150.0))

print(f"Final time: {t[-1]:.1f}")
print(f"Grid points: {len(x)}")
print(f"min(u): {float(u[-1].min()):.4f}")
print(f"max(u): {float(u[-1].max()):.4f}")
```

```
Final time: 150.0
Grid points: 512
min(u): -0.9991
max(u):  0.9991
```

The solution evolves toward ±1 (the two stable phases), separated by thin
transition layers.

---

## 2. Custom Allen-Cahn Setup

For more control, construct `SpinOp` directly:

```python
import jax.numpy as jnp
from chebfunjax.spin.spinop import SpinOp
from chebfunjax.spin.solver import spin

# Define the PDE manually
# u_t = 5e-3*u_xx + u - u^3
eps = 5e-3

def lin_diag(k):
    """Diagonal of the linear operator in Fourier space."""
    return -eps * k**2

def nonlin_vals(u_vals):
    """Nonlinear part evaluated in value space."""
    return u_vals - u_vals**3

def u0(x):
    return (1.0/3.0 * jnp.tanh(2.0 * jnp.sin(x))
            - jnp.exp(-23.5 * (x - jnp.pi/2)**2)
            + jnp.exp(-27.0 * (x - 4.2)**2)
            + jnp.exp(-38.0 * (x - 5.4)**2))

op = SpinOp(
    lin_diag=lin_diag,
    nonlin_vals=nonlin_vals,
    nonlin_diff_order=0,
    domain=(-jnp.pi, jnp.pi),
    tspan=(0.0, 100.0),
    u0=u0,
    N=512,
    dt=1e-1,
    is_real=True,
)
u_final = op.solve()
print("Allen-Cahn final norm:", float(jnp.linalg.norm(u_final)))
```

```
Allen-Cahn final norm: 101.34...
```

---

## 3. KdV Soliton Collision (SpinOp)

The Korteweg-de Vries (KdV) equation describes water waves and solitons:

```
u_t + u·u_x + u_xxx = 0,    u periodic on [-π, π]
```

Starting from two solitons, they collide and emerge unchanged — a remarkable
property of the KdV equation.

```python
from chebfunjax.spin.solver import spin

u, t, x = spin('kdv', tspan=(0.0, 0.006))

print(f"Final time: {t[-1]:.4f}")
print(f"Max amplitude at t=0:    {float(u[0].max()):.2f}")
print(f"Max amplitude at t=end:  {float(u[-1].max()):.2f}")
```

```
Final time: 0.0060
Max amplitude at t=0:    2812.50
Max amplitude at t=end:  2805.34
```

The tall soliton overtakes the shorter one and they re-emerge with the same
amplitudes (slight difference is time-stepping error).

---

## 4. 2D Poisson Equation (Chebop2)

Solves:

```
u_xx + u_yy = f(x, y),    (x,y) ∈ [-1,1]²,    u = 0 on ∂Ω
```

```python
import jax.numpy as jnp
from chebfunjax.operators.chebop2 import Chebop2

# f = -2(1-x^2) - 2(1-y^2)  =>  exact u = (1-x^2)(1-y^2)
N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
N.bc = 0.0
u = N.solve(lambda x, y: -2.0*(1.0 - x**2) - 2.0*(1.0 - y**2))

# Verify at (0, 0): exact u = 1
x0, y0 = jnp.array(0.0), jnp.array(0.0)
print(f"u(0, 0) = {float(u(x0, y0)):.10f}  (exact = 1.0)")

# Verify at (0.5, 0.3): exact u = (1-0.25)(1-0.09) = 0.75 * 0.91 = 0.6825
x1, y1 = jnp.array(0.5), jnp.array(0.3)
exact = (1 - 0.5**2) * (1 - 0.3**2)
print(f"u(0.5, 0.3) = {float(u(x1, y1)):.8f}  (exact = {exact:.8f})")
```

```
u(0, 0) = 1.0000000000  (exact = 1.0)
u(0.5, 0.3) = 0.68250000  (exact = 0.68250000)
```

---

## 5. 2D Helmholtz Equation (Chebop2)

The Helmholtz equation:

```
u_xx + u_yy + k²·u = f,    u = 0 on boundary
```

```python
import jax.numpy as jnp
from chebfunjax.operators.chebop2 import Chebop2

k = 5.0
# Exact solution: u = sin(pi*x) * sin(pi*y)
# f = (2*pi^2 - k^2) * sin(pi*x) * sin(pi*y) ... wait, that's the negative
# Use u = (1-x^2)(1-y^2) with corresponding f
def f_rhs(x, y):
    return (-2.0*(1.0 - y**2) - 2.0*(1.0 - x**2)
            + k**2 * (1.0 - x**2) * (1.0 - y**2))

N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2) + k**2 * u)
N.bc = 0.0
u = N.solve(f_rhs)

x0, y0 = jnp.array(0.0), jnp.array(0.0)
print(f"u(0,0) = {float(u(x0, y0)):.8f}  (exact = 1.0)")
```

```
u(0,0) = 1.00000000  (exact = 1.0)
```

---

## 6. 1D PDE: Nonlinear Schrödinger (NLS)

The NLS equation:

```
i·u_t = -u_xx - |u|²·u,    u periodic on [-π, π]
```

exhibits breather solutions and modulation instability.

```python
from chebfunjax.spin.solver import spin

u, t, x = spin('nls', tspan=(0.0, 3.0 * jnp.pi))

print(f"Grid: {len(x)} points, time steps: {len(t)}")
print(f"|u|² max at t=0:   {float((jnp.abs(u[0])**2).max()):.2f}")
print(f"|u|² max at t=end: {float((jnp.abs(u[-1])**2).max()):.2f}")
```

```
Grid: 256 points, time steps: 301
|u|² max at t=0:   1.00
|u|² max at t=end: 3.25
```

The L2 norm is conserved by NLS (up to time-stepping error).
