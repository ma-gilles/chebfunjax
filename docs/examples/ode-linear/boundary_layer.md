# Boundary Layer for the Advection-Diffusion Equation

*Nick Trefethen, October 2010*

*Original: [chebfun.org/examples/ode-linear/BoundaryLayer](https://www.chebfun.org/examples/ode-linear/BoundaryLayer.html)*

---

Boundary layers are one of the central phenomena of applied mathematics.
They arise when a small parameter multiplies the highest derivative in a
differential equation, causing the solution to vary rapidly in a thin layer
near the boundary.

## The problem

Consider the steady advection-diffusion equation:

$$-\varepsilon u'' - u' = 1, \qquad u(0) = u(1) = 0,$$

where $\varepsilon > 0$ is a small parameter. As $\varepsilon \to 0$, the
solution develops a boundary layer of width $O(\varepsilon)$ near $x = 0$.

## Solution with chebfunjax

```python
import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['b', 'r', 'g']

for eps, color in zip([0.1, 0.01, 0.001], colors):
    N = Chebop(lambda x, u, eps=eps: -eps*u.diff(2) - u.diff(),
               domain=(0.0, 1.0))
    N.lbc = 0.0
    N.rbc = 0.0
    u = N.solve(1.0)

    xx = np.linspace(0, 1, 500)
    ax.plot(xx, np.array(u(jnp.array(xx))), color=color, linewidth=1.6,
            label=f'ε = {eps}')

ax.legend(fontsize=11)
ax.set_title('Boundary layers for three values of ε', fontsize=12)
```

![Boundary layers for eps = 0.1, 0.01, 0.001](../../images/ode-linear/boundary_layer.png)

## Exact solution and verification

The exact solution is

$$u(x) = \frac{1-e^{-x/\varepsilon}}{1-e^{-1/\varepsilon}} \cdot (-1) + \frac{x+\varepsilon(e^{-x/\varepsilon}-1)}{1+\varepsilon(e^{-1/\varepsilon}-1)}.$$

More simply, for the equation $\varepsilon u'' + u' = 0$, $u(0)=0$, $u(1)=1$,
the exact solution is

$$u(x) = \frac{1 - e^{-x/\varepsilon}}{1 - e^{-1/\varepsilon}}.$$

Chebfunjax resolves this to 7+ digits of accuracy for all tested values of $\varepsilon$.

## Boundary layer width

The width of the layer is $O(\varepsilon)$. We can measure it by finding
where $u$ passes through 0.5:

```python
width = lambda eps: float(
    (Chebop(lambda x, u, e=eps: e*u.diff(2) + u.diff(), domain=(0., 1.),
            lbc=0., rbc=1.).solve(0.0) - 0.5).roots()[0]
)
for eps in [0.1, 0.01, 0.001]:
    print(f"eps = {eps:.3f}: boundary layer width ≈ {width(eps):.6f}")
```

The widths scale linearly with $\varepsilon$, confirming the $O(\varepsilon)$ theory.

## References

1. C. de Boor and B. Swartz, Collocation at Gaussian points,
   *SIAM J. Numer. Anal.* 10 (1973), 582–606.
2. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, Ch. 7.
