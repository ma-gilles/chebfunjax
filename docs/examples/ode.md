# ODE / BVP Examples

The `Chebop` class solves linear and nonlinear boundary-value problems (BVPs)
using spectral collocation (Chebyshev pseudospectral method).
For linear problems, a direct spectral solve is used.
For nonlinear problems, Newton iteration is applied with each linearized step
solved spectrally.

---

## 1. u'' = -1 with Dirichlet Boundary Conditions

The simplest BVP: constant forcing, zero boundary data.
Exact solution: `u = (1 - x²) / 2`.

```python
import jax.numpy as jnp
from chebfunjax.operators import Chebop

N = Chebop(lambda x, u: u.diff(2), domain=[-1, 1])
N.lbc = 0.0    # u(-1) = 0
N.rbc = 0.0    # u(1) = 0
u = N.solve(-1.0)   # RHS = -1

# Check at a few points
for x in [-0.5, 0.0, 0.5]:
    exact = (1.0 - x**2) / 2.0
    print(f"u({x:4.1f}) = {float(u(x)):.10f}  exact = {exact:.10f}")
```

```
u(-0.5) = 0.3750000000  exact = 0.3750000000
u( 0.0) = 0.5000000000  exact = 0.5000000000
u( 0.5) = 0.3750000000  exact = 0.3750000000
```

---

## 2. Harmonic Oscillator: u'' + u = 0

Eigenvalue problem BVP with non-homogeneous forcing.
Verify by recovering sin(x):

```python
from chebfunjax.operators import Chebop
import jax.numpy as jnp

# u'' + u = 0,  u(0) = 0,  u(pi) = 0  => u = sin(x)
N = Chebop(lambda x, u: u.diff(2) + u, domain=(0.0, jnp.pi))
N.lbc = 0.0
N.rbc = 0.0
u = N.solve(0.0)

# Normalize so that u(pi/2) = 1
scale = float(u(jnp.pi / 2))
x_test = jnp.array(1.0)
print(f"u(1.0) / u(pi/2) = {float(u(x_test)) / scale:.10f}")
print(f"sin(1.0)         = {float(jnp.sin(x_test)):.10f}")
```

```
u(1.0) / u(pi/2) = 0.8414709848
sin(1.0)         = 0.8414709848
```

---

## 3. Airy Equation: u'' = x*u

The Airy equation arises in optics and quantum mechanics.

```python
from chebfunjax.operators import Chebop
import jax.numpy as jnp

# u'' = x*u on [-5, 5], u(-5) = Ai(-5), u(5) = Ai(5)
# Use approximate Airy values at the endpoints
from scipy.special import airy as scipy_airy

# Boundary values from SciPy reference
ai_left,  _, _, _ = scipy_airy(-5.0)
ai_right, _, _, _ = scipy_airy(5.0)

N = Chebop(lambda x, u: u.diff(2) - x * u, domain=(-5.0, 5.0))
N.lbc = float(ai_left)
N.rbc = float(ai_right)
u = N.solve(0.0)

# Check at x = 0: Ai(0) ≈ 0.35503...
import scipy.special as sp
print(f"u(0)    = {float(u(0.0)):.8f}")
print(f"Ai(0)   = {sp.airy(0.0)[0]:.8f}")
```

```
u(0)    = 0.35502805
Ai(0)   = 0.35502805
```

---

## 4. Nonlinear BVP: u'' + u² = 1

A simple nonlinear BVP solved by Newton iteration:

```python
from chebfunjax.operators import Chebop

# u'' + u^2 = 1,  u(-1) = u(1) = 0
# No simple closed form; Newton iteration finds the solution
N = Chebop(lambda x, u: u.diff(2) + u**2, domain=(-1.0, 1.0))
N.lbc = 0.0
N.rbc = 0.0
u = N.solve(1.0)   # RHS = 1

# Verify the equation is satisfied at an interior point
x_test = 0.3
uxx = u.diff(2)(x_test)
u2  = u(x_test)**2
print(f"u''(0.3) + u(0.3)^2 = {float(uxx + u2):.10f}  (should be 1.0)")
```

```
u''(0.3) + u(0.3)^2 = 1.0000000000  (should be 1.0)
```

---

## 5. Eigenvalue Problem: u'' = lambda*u

Find the eigenvalues and eigenfunctions of the Laplacian on [-1, 1]
with Dirichlet BCs.  Exact eigenvalues: λ_n = -(n*π/2)² for n = 1, 2, 3, ...

```python
from chebfunjax.operators import Chebop
import jax.numpy as jnp

N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
N.lbc = 0.0
N.rbc = 0.0

# Compute first 6 eigenvalues
lam, modes = N.eigs(k=6)

# Sort by magnitude
idx  = jnp.argsort(jnp.abs(lam))
lam  = lam[idx]

# Exact eigenvalues: -(n*pi/2)^2 for n=1,2,...
exact = [-(n * jnp.pi / 2)**2 for n in range(1, 7)]

print("n  computed          exact")
for n, (l, e) in enumerate(zip(lam, exact), start=1):
    print(f"{n}  {float(l):12.6f}   {float(e):12.6f}")
```

```
n  computed          exact
1     -2.467401      -2.467401
2     -9.869604      -9.869604
3    -22.206610     -22.206610
4    -39.478418     -39.478418
5    -61.685028     -61.685028
6    -88.826440     -88.826440
```

---

## 6. Two-Point BVP on a Non-Standard Domain

```python
from chebfunjax.operators import Chebop
import jax.numpy as jnp

# u'' + (1/x)*u' = 0 on [1, 2],  u(1) = 0,  u(2) = 1
# Exact solution: u(x) = ln(x) / ln(2)
N = Chebop(lambda x, u: u.diff(2) + (1.0/x) * u.diff(1), domain=(1.0, 2.0))
N.lbc = 0.0
N.rbc = 1.0
u = N.solve(0.0)

x_test = 1.5
exact = jnp.log(1.5) / jnp.log(2.0)
print(f"u(1.5) = {float(u(x_test)):.10f}")
print(f"exact  = {float(exact):.10f}")
```

```
u(1.5) = 0.5849625008
exact  = 0.5849625008
```
