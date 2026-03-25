# Poisson equation

*chebfunjax team*

## Overview

Solves the 1D Poisson equation

$$-u'' = f(x), \quad u(-1) = u(1) = 0$$

for several right-hand sides including $f(x) = 1$ (exact: $(1-x^2)/2$)
and $f(x) = e^x \cos(5x)$ (no closed form).

```python
from chebfunjax.operators.chebop import Chebop

dom = (-1.0, 1.0)
N = Chebop(lambda x, u: -u.diff(2), domain=dom)
N.lbc = 0.0; N.rbc = 0.0

# f = 1, exact u = (1-x^2)/2
u1 = N.solve(1.0)

# f = exp(x)*cos(5x)
u2 = N.solve(lambda x: jnp.exp(x) * jnp.cos(5*x))
```

## Results

For $f = 1$, the numerical solution matches $(1-x^2)/2$ to machine precision.

![Poisson equation](../../images/ode-linear/poisson_equation.png)
