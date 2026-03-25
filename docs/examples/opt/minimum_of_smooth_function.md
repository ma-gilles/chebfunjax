# Minimum of a Smooth Function

*Original: [chebfun.org/examples/opt/](https://www.chebfun.org/examples/opt/)*

---

Finding the minimum of a smooth function with Chebfun is both simple and
highly accurate. The `min()` method returns the global minimum and its location.

## Single-variable minimization

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# f(x) = cos(x) + cos(3x)/3 + cos(5x)/5 on [0, 5]
f = cj.chebfun(
    lambda x: jnp.cos(x) + jnp.cos(3*x)/3 + jnp.cos(5*x)/5,
    domain=(0.0, 5.0)
)
x_min, f_min = f.min()
print(f"Global min: f({float(x_min):.8f}) = {float(f_min):.8f}")
# Also find all local minima via f'(x) = 0 with f''(x) > 0
fp = f.diff()
fpp = f.diff(2)
critical = np.array(fp.roots())
for xc in critical:
    if float(fpp(jnp.array(xc))) > 0:
        print(f"  Local min at x = {xc:.4f}, f = {float(f(jnp.array(xc))):.6f}")
```

## Minimization with multiple local minima

For functions with many local minima, Chebfun's global optimization is
reliable because the polynomial approximation captures all features:

```python
g = cj.chebfun(lambda x: jnp.sin(20*x) + jnp.sin(2*x), domain=(0.0, 4.0))
x_gmin, g_gmin = g.min()
print(f"Global min: f({float(x_gmin):.6f}) = {float(g_gmin):.6f}")

# All local minima
gp = g.diff()
gpp = g.diff(2)
local_mins = [(xc, float(g(jnp.array(xc))))
              for xc in np.array(gp.roots())
              if float(gpp(jnp.array(xc))) > 0]
print(f"Number of local minima: {len(local_mins)}")
```

![Minimum of smooth function](../../../images/opt/minimum_of_smooth_function.png)
