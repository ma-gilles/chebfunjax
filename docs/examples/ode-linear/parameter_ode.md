# A parameter-dependent ODE with breakpoints

*Asgeir Birkisson, January 2012*

[Chebfun example](https://www.chebfun.org/examples/ode-linear/ParameterODE.html)

## Overview

Solves $(a(x, s) u')' = 1$ with $u(\pm 1) = 0$, where $a(x, s)$
is a parameter-dependent piecewise function. The exact solution is
$u = \log(a)/(8s)$ in a special case. Demonstrates how Chebop handles
problems with a continuous parameter $s$.

```python
from chebfunjax.operators.chebop import Chebop

dom = (-1.0, 1.0)
for s in [0.5, 1.0, 2.0]:
    def a_func(x, _s=s):
        return 1.0 + _s * x
    N = Chebop(
        lambda x, u: (a_func(x) * u.diff()).diff(),
        domain=dom)
    N.lbc = 0.0; N.rbc = 0.0
    u = N.solve(1.0)
```


![A parameter-dependent ODE with breakpoints](../../images/ode-linear/parameter_ode.png)

## Figures (chebfun.org parity)

![ParameterODE figure 1](../../images/ode-linear/ParameterODE_01.png)

![ParameterODE figure 2](../../images/ode-linear/ParameterODE_02.png)

![ParameterODE figure 3](../../images/ode-linear/ParameterODE_03.png)

![ParameterODE figure 4](../../images/ode-linear/ParameterODE_04.png)

![ParameterODE figure 5](../../images/ode-linear/ParameterODE_05.png)

![ParameterODE figure 6](../../images/ode-linear/ParameterODE_06.png)

![ParameterODE figure 7](../../images/ode-linear/ParameterODE_07.png)

![ParameterODE figure 8](../../images/ode-linear/ParameterODE_08.png)

![ParameterODE figure 9](../../images/ode-linear/ParameterODE_09.png)

![ParameterODE figure 10](../../images/ode-linear/ParameterODE_10.png)

![ParameterODE figure 11](../../images/ode-linear/ParameterODE_11.png)

![ParameterODE figure 12](../../images/ode-linear/ParameterODE_12.png)

![ParameterODE figure 13](../../images/ode-linear/ParameterODE_13.png)
