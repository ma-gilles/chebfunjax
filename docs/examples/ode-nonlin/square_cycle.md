# A square limit cycle

*Nick Trefethen, May 2019*

[Chebfun example](https://www.chebfun.org/examples/ode-nonlin/SquareCycle.html)

## Overview

A specially constructed 2D system exhibits a square limit cycle (due to
Johnson and Tucker). The system has four saddle equilibria connected by
heteroclinic orbits forming a square.

```python
from scipy.integrate import solve_ivp

def square_cycle_rhs(t, xy):
    x, y = xy
    # Van der Pol-type with square symmetry
    return [y * (1 - x**2 - y**2) + x * (x**2 - y**2),
            -x * (1 - x**2 - y**2) + y * (x**2 - y**2)]
```


![A square limit cycle](../../images/ode-nonlin/square_cycle.png)

## Figures (chebfun.org parity)

![SquareCycle figure 1](../../images/ode-nonlin/SquareCycle_01.png)

![SquareCycle figure 2](../../images/ode-nonlin/SquareCycle_02.png)

![SquareCycle figure 3](../../images/ode-nonlin/SquareCycle_03.png)
