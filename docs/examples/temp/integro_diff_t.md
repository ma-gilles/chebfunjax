# Time-Dependent Integro-Differential Equation

**Original:** [pde/IntegroDiffT](https://github.com/chebfun/examples/blob/master/pde/IntegroDiffT.m)
**Author(s):** Nick Hale, October 2010

---

This example demonstrates how to solve a time-dependent integro-differential
equation using Chebfun's `pde15s` command.

## The equation

The PDE is

$$u_t = 0.02\,u''(x) + \left(\int_{-1}^{1} u(\xi)\,d\xi\right)\!\left(\int_{-1}^{x} u(\xi)\,d\xi\right),$$

with homogeneous Dirichlet boundary conditions $u(-1) = u(1) = 0$.

The first integral $\int_{-1}^{1} u\,d\xi$ is the **total mass** (a scalar),
while the second $\int_{-1}^{x} u\,d\xi$ is a **running integral** (a
function of $x$). Their product introduces a nonlocal coupling that drives
growth: as the total mass increases, the nonlinear source term grows, which
in turn increases the total mass further.

## Initial condition and solution

The initial condition is a pulse centred at $x = -0.5$:

$$u_0(x) = (1 - x^2)\,\exp\!\bigl(-30(x + 0.5)^2\bigr).$$

The solution is computed on $[-1,1]$ for $t \in [0, 4]$ and displayed as a
waterfall plot. The pulse broadens and grows as the nonlocal feedback takes
hold.

This example can also be found as the "Integro-differential equation" demo
among the PDE-Scalar demos of Chebfun's CHEBGUI.

## Code

```python
from examples.temp.integro_diff_t import run
run()
```

## Output

![Time-Dependent Integro-Differential Equation](../../images/temp/integro_diff_t.png)
