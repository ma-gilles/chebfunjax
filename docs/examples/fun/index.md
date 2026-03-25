# Fun/Miscellaneous Examples

A collection of examples showcasing the fun side of chebfunjax: Koch snowflakes,
parametric curves, and creative applications.

---

## Audible chebfuns

**Source:** `fun/AudibleChebfuns.m`

Chebfun coefficients can encode and synthesize audio waveforms.

---

## Koch snowflake

**Source:** `fun/KochSnowflake.m`

The Koch snowflake is a fractal curve.  Its perimeter diverges but its area
converges — both can be computed analytically.

---

## Piecewise linear functions

**Source:** `fun/PiecewiseLinear.m`

Chebfunjax can represent piecewise linear functions exactly with breakpoints.

```python
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.domain import Domain

# Tent function: max(0, 1 - |x|)
f = cj.chebfun(lambda x: jnp.maximum(0, 1 - jnp.abs(x)),
               domain=Domain([-1, 0, 1]))
print(f(jnp.array(0.0)))   # 1.0
print(f(jnp.array(0.5)))   # 0.5
```

---

## Encryption with Chebfun

**Source:** `fun/Encryption.m`

Using Chebyshev coefficients as a key for simple encryption demonstrations.

---

## Hello World

**Source:** `fun/HelloWorld.m`

Plotting text as a chebfun trajectory in the complex plane.
