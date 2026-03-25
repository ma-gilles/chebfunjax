# Bird Flight Optimization

*Original: [chebfun.org/examples/calc/](https://www.chebfun.org/examples/calc/)*

---

A bird flying between two points chooses a path that minimizes total energy.
For flight over water (high drag) vs. land (low drag), the optimal strategy
is to fly at an angle — an application of Fermat-like variational principles.

## Energy-minimizing crossover

Suppose a bird flies from a shore point to an island. Its energy cost per
unit distance is $c_w$ over water and $c_l$ over land (with $c_w > c_l$).
The total energy $E(x)$ as a function of the coastal crossing point is:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Bird starts at (0, d) over land, ends at (L, 0) over water
d = 1.0   # perpendicular distance to shore
L = 3.0   # horizontal distance to destination
cw, cl = 2.0, 1.0  # relative cost: water 2x land

# E(x) = cl * x + cw * sqrt((L-x)^2 + d^2), x in [0, L]
E = cj.chebfun(
    lambda x: cl * x + cw * jnp.sqrt((L - x)**2 + d**2),
    domain=(0.0, L)
)
x_opt, E_min = E.min()
print(f"Optimal x = {float(x_opt):.6f}")
print(f"Min energy = {float(E_min):.6f}")

# Critical angle from Snell's-law analogy
import numpy as np
sin_theta = cl / cw  # Snell's law condition
x_snell = L - d * sin_theta / np.sqrt(1 - sin_theta**2)
print(f"Snell's law prediction: x = {x_snell:.6f}")
```

![Bird flight optimal path and energy landscape](../../../images/calc/bird_flight_optimization.png)

## Connection to Snell's law

This problem is mathematically identical to Snell's law of refraction:
the "refractive index" is the energy cost per unit distance. The optimal
crossing angle satisfies:

$$c_l \sin\theta_w = c_w \sin\theta_l,$$

the continuous analog of Fermat's principle.
