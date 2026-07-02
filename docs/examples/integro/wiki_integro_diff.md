# Wikipedia Integro-Differential Equation

**Source:** `integro/WikiIntegroDiff.m` — Mark Richardson, September 2010
**Python:** `examples/integro/wiki_integro_diff.py`
**Original MATLAB:** https://www.chebfun.org/examples/integro/WikiIntegroDiff.html

## Problem

Solve the integro-differential equation:
```
u'(x) + 2u(x) + 5 ∫₀ˣ u(t) dt = 1,   u(0) = 0
```

Exact solution: `u(x) = (1/2) e^{-x} sin(2x)`.

## Solution method

Differentiating the IDE once gives the equivalent second-order ODE:
```
u''(x) + 2u'(x) + 5u(x) = 0
```
with initial conditions `u(0) = 0`, `u'(0) = 1`
(from the original IDE at `x=0`: `u'(0) + 2·0 + 5·0 = 1`).

This second-order ODE is solved via `scipy.integrate.solve_ivp` as a
first-order system:

```python
from scipy.integrate import solve_ivp
import numpy as np

def rhs(x, y):
    u, up = y
    return [up, -2*up - 5*u]

sol = solve_ivp(rhs, [0, 5], [0.0, 1.0], rtol=1e-10, atol=1e-12)
```

## Verification

The solution is verified against the exact `u(x) = (1/2) e^{-x} sin(2x)`
to machine precision (max error < 1e-6).

Additionally, the original IDE is verified by computing the cumulative
integral numerically.

## Plots

![WikiIntegroDiff](../../images/integro/wiki_integro_diff.png)

Left: exact solution vs numerical.
Right: pointwise error (machine precision).
