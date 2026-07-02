# The Carrier Equation

*Asgeir Birkisson, October 2010*

*Original: [chebfun.org/examples/ode-nonlin/Carrier](https://www.chebfun.org/examples/ode-nonlin/Carrier.html)*

---

The Carrier equation is a nonlinear boundary value problem from Bender and
Orszag [1]:

$$\varepsilon u'' + 2(1-x^2)u + u^2 = 1, \qquad u(-1) = 0,\; u(1) = 0.$$

This equation has **multiple solutions**, and Newton's method (the default
solver in chebfunjax) will find different solutions depending on the initial
guess.

## Finding a solution with chebfunjax

```python
import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
import jax.numpy as jnp
import numpy as np

eps = 0.01
N = Chebop(lambda x, u: eps*u.diff(2) + 2*(1-x**2)*u + u**2,
           domain=(-1.0, 1.0))
N.lbc = 0.0
N.rbc = 0.0
# Initial guess: parabola
N.init = cj.chebfun(lambda x: 2*(x**2 - 1))

u = N.solve(1.0)
print(f"Solution length: {len(u)}")
print(f"Residual: {float((eps*u.diff(2) + 2*(1-cj.chebfun(lambda x:x)**2)*u + u*u - 1).norm()):.2e}")
```

![Carrier equation solutions from two different initial guesses](../../images/ode-nonlin/carrier_equation.png)

## Multiple solutions

Starting from a different initial guess finds a different solution:

```python
x = cj.chebfun(lambda t: t)
N.init = cj.chebfun(lambda t: 2*(t**2 - 1)*(1 - 2.0/(1 + 20*t**2)))
u2 = N.solve(1.0)
```

Both solutions satisfy the ODE to near machine precision, demonstrating
that the nonlinear boundary value problem has multiple branches.

## Alternative boundary conditions

Chebfunjax makes it easy to experiment with different boundary conditions:

```python
# u(-1) = 1, u'(1) + u(1) = 0
N.lbc = 1.0
N.rbc = lambda u: u.diff() + u   # Robin-type condition
u3 = N.solve(1.0)
```

## References

1. C. Bender and S. A. Orszag, *Advanced Mathematical Methods for Scientists
   and Engineers*, Springer, 1999, §9.7.
2. A. Birkisson and T. Driscoll, Automatic Fréchet differentiation for the
   numerical solution of boundary-value problems, *ACM TOMS* 38 (2012), 26.
