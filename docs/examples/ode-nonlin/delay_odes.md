# Delay differential equations in Chebfun

*Nick Hale, June 2022*

[Chebfun example](https://www.chebfun.org/examples/ode-nonlin/DelayDifferentialEquations.html)

## Overview

Solves delay differential equations (DDEs) including the pantograph equation

$$y'(t) = -y(t/2), \quad y(0) = 1$$

whose exact solution is $y(t) = \sum_{k} c_k t^{\alpha_k}$.
Implemented via forward Euler stepping with interpolation for the delayed term.

```python
import numpy as np

dt = 1e-4
T = 4.0
t_arr = np.arange(0, T + dt, dt)
y_arr = np.ones(len(t_arr))
for i in range(1, len(t_arr)):
    t_delay = t_arr[i-1] / 2.0
    y_delay = np.interp(t_delay, t_arr[:i], y_arr[:i])
    y_arr[i] = y_arr[i-1] - dt * y_delay
```


![Delay differential equations in Chebfun](../../images/ode-nonlin/delay_odes.png)

## Figures (chebfun.org parity)

![DelayDifferentialEquations figure 1](../../images/ode-nonlin/DelayDifferentialEquations_01.png)

![DelayDifferentialEquations figure 2](../../images/ode-nonlin/DelayDifferentialEquations_02.png)

![DelayDifferentialEquations figure 3](../../images/ode-nonlin/DelayDifferentialEquations_03.png)

![DelayDifferentialEquations figure 4](../../images/ode-nonlin/DelayDifferentialEquations_04.png)

![DelayDifferentialEquations figure 5](../../images/ode-nonlin/DelayDifferentialEquations_05.png)

![DelayDifferentialEquations figure 6](../../images/ode-nonlin/DelayDifferentialEquations_06.png)

![DelayDifferentialEquations figure 7](../../images/ode-nonlin/DelayDifferentialEquations_07.png)

![DelayDifferentialEquations figure 8](../../images/ode-nonlin/DelayDifferentialEquations_08.png)

![DelayDifferentialEquations figure 9](../../images/ode-nonlin/DelayDifferentialEquations_09.png)

![DelayDifferentialEquations figure 10](../../images/ode-nonlin/DelayDifferentialEquations_10.png)

![DelayDifferentialEquations figure 11](../../images/ode-nonlin/DelayDifferentialEquations_11.png)

![DelayDifferentialEquations figure 12](../../images/ode-nonlin/DelayDifferentialEquations_12.png)

![DelayDifferentialEquations figure 13](../../images/ode-nonlin/DelayDifferentialEquations_13.png)

![DelayDifferentialEquations figure 14](../../images/ode-nonlin/DelayDifferentialEquations_14.png)

![DelayDifferentialEquations figure 15](../../images/ode-nonlin/DelayDifferentialEquations_15.png)

![DelayDifferentialEquations figure 16](../../images/ode-nonlin/DelayDifferentialEquations_16.png)

![DelayDifferentialEquations figure 17](../../images/ode-nonlin/DelayDifferentialEquations_17.png)

![DelayDifferentialEquations figure 18](../../images/ode-nonlin/DelayDifferentialEquations_18.png)

![DelayDifferentialEquations figure 19](../../images/ode-nonlin/DelayDifferentialEquations_19.png)

![DelayDifferentialEquations figure 20](../../images/ode-nonlin/DelayDifferentialEquations_20.png)

![DelayDifferentialEquations figure 21](../../images/ode-nonlin/DelayDifferentialEquations_21.png)

![DelayDifferentialEquations figure 22](../../images/ode-nonlin/DelayDifferentialEquations_22.png)

![DelayDifferentialEquations figure 23](../../images/ode-nonlin/DelayDifferentialEquations_23.png)

![DelayDifferentialEquations figure 24](../../images/ode-nonlin/DelayDifferentialEquations_24.png)
