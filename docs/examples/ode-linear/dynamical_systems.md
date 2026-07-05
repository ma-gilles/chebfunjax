# Classification of linear dynamical systems

*Georges Klein, March 2013*

[Chebfun example](https://www.chebfun.org/examples/ode-linear/DynamicalSystems.html)

## Overview

Classifies 2D linear dynamical systems $\dot{\mathbf{x}} = A \mathbf{x}$
by the nature of their equilibrium at the origin: stable/unstable node,
spiral, center, or saddle — determined by the eigenvalues of $A$.

## Method

For several canonical matrices $A$, integrates trajectories using
`scipy.integrate.solve_ivp` and plots the phase portraits.

```python
import numpy as np
from scipy.integrate import solve_ivp

# Stable spiral: A has complex eigenvalues with negative real part
A_stable = np.array([[-1, 2], [-2, -1]])
def f_spiral(t, y): return A_stable @ y
sol = solve_ivp(f_spiral, [0, 10], [1, 0], rtol=1e-10)
```


![Classification of linear dynamical systems](../../images/ode-linear/dynamical_systems.png)

## Figures (chebfun.org parity)

![DynamicalSystems figure 1](../../images/ode-linear/DynamicalSystems_01.png)

![DynamicalSystems figure 2](../../images/ode-linear/DynamicalSystems_02.png)

![DynamicalSystems figure 3](../../images/ode-linear/DynamicalSystems_03.png)

![DynamicalSystems figure 4](../../images/ode-linear/DynamicalSystems_04.png)

![DynamicalSystems figure 5](../../images/ode-linear/DynamicalSystems_05.png)

![DynamicalSystems figure 6](../../images/ode-linear/DynamicalSystems_06.png)

![DynamicalSystems figure 7](../../images/ode-linear/DynamicalSystems_07.png)

![DynamicalSystems figure 8](../../images/ode-linear/DynamicalSystems_08.png)

![DynamicalSystems figure 9](../../images/ode-linear/DynamicalSystems_09.png)

![DynamicalSystems figure 10](../../images/ode-linear/DynamicalSystems_10.png)

![DynamicalSystems figure 11](../../images/ode-linear/DynamicalSystems_11.png)
