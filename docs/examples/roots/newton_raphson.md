# Newton–Raphson Method

*Original: [chebfun.org/examples/roots/](https://www.chebfun.org/examples/roots/)*

---

Newton's method finds roots of $f(x) = 0$ by iterating:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}.$$

With Chebfun, $f$ and $f'$ are both high-accuracy polynomial approximations,
so Newton iterations converge with **spectral accuracy**.

## Newton's method for $\cos(x) = x$

The fixed point of $\cos$ satisfies $x^* = \cos(x^*)$ — equivalently,
$g(x) = \cos(x) - x = 0$:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

g = cj.chebfun(lambda x: jnp.cos(x) - x, domain=(-2.0, 2.0))
roots = np.array(g.roots())
print(f"Dottie number: x = {roots[0]:.15f}")
print(f"  cos(x) - x = {float(jnp.cos(roots[0]) - roots[0]):.2e}")
```

```
Dottie number: x = 0.739085133215161
  cos(x) - x = 0.00e+00
```

## Convergence of Newton iterates

For a simple root, Newton's method converges **quadratically** — doubling
the number of correct digits at each step:

```python
x = 0.5  # initial guess
g = cj.chebfun(lambda x: jnp.cos(x) - x, domain=(-2.0, 2.0))
gp = g.diff()
dottie = 0.739085133215160641655...

errors = []
for _ in range(8):
    x_new = x - float(g(jnp.array(x))) / float(gp(jnp.array(x)))
    errors.append(abs(x_new - dottie))
    x = x_new
```

![Newton-Raphson convergence to the Dottie number](../../../images/roots/newton_raphson.png)

The error plot shows quadratic convergence: the log-error approximately doubles
at each iteration (the curve bends sharply down).

## References

1. J. M. Ortega and W. C. Rheinboldt, *Iterative Solution of Nonlinear Equations
   in Several Variables*, Academic Press, 1970.
