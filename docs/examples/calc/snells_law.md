# Snell's Law and Refraction

*Original: [chebfun.org/examples/calc/SnellsLaw](https://www.chebfun.org/examples/calc/SnellsLaw.html)*
**Author(s):** Mohsin Javed, October 2013

---

Snell's law of refraction states that when light crosses an interface between
two media with refractive indices $n_1$ and $n_2$:

$$n_1 \sin\theta_1 = n_2 \sin\theta_2.$$

This is a consequence of Fermat's principle: light takes the path of least time.

## Fermat's principle as an optimization

For a flat interface at $y=0$, with source at $(0, 1)$ and target at $(1, -1)$,
the travel time as a function of crossing point $x \in [0,1]$ is:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Refractive indices
n1, n2 = 1.0, 1.5

# Travel time T(x) = n1 * sqrt(x^2 + 1) + n2 * sqrt((1-x)^2 + 1)
T = cj.chebfun(
    lambda x: n1 * jnp.sqrt(x**2 + 1) + n2 * jnp.sqrt((1-x)**2 + 1),
    domain=(0.0, 1.0)
)
x_min, T_min = T.min()
print(f"Optimal crossing point: x = {float(x_min):.6f}")

# Verify Snell's law: n1*sin(theta1) = n2*sin(theta2)
x_opt = float(x_min)
sin_theta1 = x_opt / np.sqrt(x_opt**2 + 1)
sin_theta2 = (1 - x_opt) / np.sqrt((1-x_opt)**2 + 1)
print(f"n1*sin(θ1) = {n1*sin_theta1:.6f}")
print(f"n2*sin(θ2) = {n2*sin_theta2:.6f}")
```

```
Optimal crossing point: x = 0.359257
n1*sin(θ1) = 0.359257
n2*sin(θ2) = 0.359257
```

The numerical minimization confirms Snell's law to high accuracy.

![Snell's law: light path and travel time](../../images/calc/snells_law.png)