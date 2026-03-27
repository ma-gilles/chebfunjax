# Bessel Equation BVP

**Inspired by [Chebfun](https://www.chebfun.org/) examples (ode-linear/BesselBvp)**

---

Bessel's differential equation of order $\nu$:

$$x^2 y'' + x y' + (x^2 - \nu^2) y = 0$$

has solutions $J_\nu(x)$ and $Y_\nu(x)$. As a boundary value problem on
$[\epsilon, R]$ (avoiding the singularity at 0), chebfun can solve it directly.

## BVP formulation

Rewrite as $y'' + \frac{1}{x}y' + (1 - \nu^2/x^2)y = 0$ and impose boundary
conditions $y(\epsilon) = J_0(\epsilon)$, $y(R) = J_0(R)$:

```python
import numpy as np
import scipy.special
import scipy.linalg

# Finite difference solution for J_0 on [0.1, 10]
n = 500
x = np.linspace(0.1, 10, n)
h = x[1] - x[0]

# D^2 + (1/x)*D + (1 - nu^2/x^2)*I = 0, nu=0
D2 = (np.diag(np.ones(n-1), -1) - 2*np.diag(np.ones(n)) + np.diag(np.ones(n-1), 1)) / h**2
D1 = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)) / (2*h)
L = D2 + np.diag(1/x) @ D1 + np.diag(1 - 0/x**2)

# BCs: y(0.1) = J0(0.1), y(10) = J0(10)
L[0, :] = 0; L[0, 0] = 1
L[-1, :] = 0; L[-1, -1] = 1
rhs = np.zeros(n)
rhs[0] = scipy.special.j0(0.1)
rhs[-1] = scipy.special.j0(10.0)

y = np.linalg.solve(L, rhs)
err = np.max(np.abs(y - scipy.special.j0(x)))
print(f"Bessel BVP error: {err:.2e}")
```

![Bessel equation BVP solution](../../images/ode-linear/bessel_bvp.png)
