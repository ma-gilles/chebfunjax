# Differentiation in 2D

*Original: [chebfun.org/examples/approx2/Differentiation](https://www.chebfun.org/examples/approx2/Differentiation.html)*

---

Partial derivatives of a Chebfun2 are computed by differentiating the Chebyshev
expansion in the corresponding variable, giving spectral accuracy.

## Partial derivatives

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# f(x,y) = sin(pi*x) * exp(y)
f = cj.chebfun2(lambda x, y: jnp.sin(jnp.pi * x) * jnp.exp(y))

# ∂f/∂x = pi * cos(pi*x) * exp(y)
fx = f.diff(0)  # differentiate with respect to x
# ∂f/∂y = sin(pi*x) * exp(y)
fy = f.diff(1)  # differentiate with respect to y

# Evaluate at a point
x0, y0 = jnp.array(0.3), jnp.array(0.5)
print(f"∂f/∂x(0.3, 0.5) = {float(fx(x0, y0)):.8f}")
print(f"  exact: {float(jnp.pi * jnp.cos(jnp.pi * 0.3) * jnp.exp(0.5)):.8f}")
print(f"∂f/∂y(0.3, 0.5) = {float(fy(x0, y0)):.8f}")
print(f"  exact: {float(jnp.sin(jnp.pi * 0.3) * jnp.exp(0.5)):.8f}")
```

![Partial derivatives of sin(πx)exp(y)](../../../images/approx2/differentiation_2d.png)

## Gradient and Laplacian

```python
# Gaussian: f(x,y) = exp(-(x^2 + y^2))
g = cj.chebfun2(lambda x, y: jnp.exp(-(x**2 + y**2)))
gx = g.diff(0)   # ∂g/∂x = -2x * exp(-(x^2+y^2))
gy = g.diff(1)   # ∂g/∂y = -2y * exp(-(x^2+y^2))
gxx = gx.diff(0) # ∂²g/∂x² = (4x²-2) * exp(-(x^2+y^2))
gyy = gy.diff(1) # ∂²g/∂y² = (4y²-2) * exp(-(x^2+y^2))
```

![Second derivatives and Laplacian](../../../images/approx2/differentiation_2d_f2.png)
