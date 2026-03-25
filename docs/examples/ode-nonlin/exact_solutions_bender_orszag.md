# Exact solutions of nonlinear ODEs from Bender and Orszag

*Nick Trefethen, December 2010*

[Chebfun example](https://www.chebfun.org/examples/ode-nonlin/exactsolutionsbenderorszag.html)

## Overview

Reproduces exact solutions from Bender & Orszag's *Advanced Mathematical Methods*,
including the Bernoulli equation $y' = y - y^2$, the Riccati equation, and
the separable equation $y' = y(1-y)$.

```python
from chebfunjax.operators.chebop import Chebop

# Bernoulli: y' = y - y^2, y(0) = 0.5
dom = (0.0, 3.0)
N = Chebop(lambda x, u: u.diff() - u + u**2, domain=dom)
N.lbc = 0.5
u = N.solve(0.0)
# Exact: u = 1/(1 + e^{-x})
```


![Exact solutions of nonlinear ODEs from Bender and Orszag](../../images/ode-nonlin/exact_solutions_bender_orszag.png)
