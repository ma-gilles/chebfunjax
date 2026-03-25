# Optimizing a bird's flight path

**Toby Driscoll, November 2012**

---

Birds expend energy both flying over water (high cost) and flying over land
(lower cost). Given two shoreline points, the optimal path minimises total
energy. This is a continuous optimisation problem solved elegantly with Chebfun.

## The setup

A bird starts at point $A$ on shore and must reach point $B$ also on shore, with
a stretch of water in between. Let the water width be $d$ and the horizontal
distance be $L$. If the bird crosses the water at angle $\theta$ from the shore,
the total energy cost is

$$
E(\theta) = c_w \frac{d}{\sin\theta} + c_l \left(L - \frac{d}{\tan\theta}\right),
$$

where $c_w > c_l$ are the per-unit-distance costs over water and land.

## chebfunjax computation

```python
import jax.numpy as jnp
import chebfunjax as cj

d = 1.0; L = 4.0; cw = 2.0; cl = 1.0

def energy(theta):
    return cw * d / jnp.sin(theta) + cl * (L - d / jnp.tan(theta))

# Build chebfun on (0, pi/2) — open interval to avoid singularities
f = cj.chebfun(energy, domain=(0.05, jnp.pi / 2 - 0.05))

# Find the minimum
x_min, f_min = f.min()
theta_opt = float(x_min)
print(f"Optimal angle: {jnp.degrees(theta_opt):.4f} degrees")
print(f"Minimum energy: {float(f_min):.6f}")
```

The optimal angle satisfies $\cos\theta^* = c_l/c_w$, giving
$\theta^* = \arccos(c_l/c_w)$.

## Gallery

![Bird flight optimisation](../../../examples/calc/bird_flight_optimization.png)

Energy as a function of crossing angle, with the minimum marked.
