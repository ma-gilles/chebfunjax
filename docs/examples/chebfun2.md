# 2D Functions: Chebfun2

`Chebfun2` represents bivariate functions on rectangles `[a,b] × [c,d]` as
a sum of rank-1 terms (low-rank approximation):

```
f(x, y) ≈ Σ_k  c_k · u_k(x) · v_k(y)
```

where each `u_k`, `v_k` is a 1D Chebfun.  This is the Chebfun2 *skeleton
approximation* (Townsend & Trefethen 2013).

*See also: [Chebfun approx2 examples](https://www.chebfun.org/examples/approx2/)*

---

## 1. Approximating cos(x + y)

```python
import jax.numpy as jnp
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

g = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
print(g)
print(f"g(0.3, 0.5) = {float(g(jnp.array(0.3), jnp.array(0.5))):.12f}")
print(f"cos(0.8)    = {float(jnp.cos(0.8)):.12f}")
```

```
Chebfun2 on [-1,1] x [-1,1], rank 2
g(0.3, 0.5) = 0.696706709348
cos(0.8)    = 0.696706709347
```

Only 2 terms are needed because `cos(x+y) = cos(x)cos(y) - sin(x)sin(y)` is
exactly rank 2.

---

## 2. Double Integral

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

g = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
integral = g.sum2()
print(f"double integral of cos(x+y) on [-1,1]^2 = {float(integral):.12f}")
# Exact: integral = (sin(1)+sin(-1))^2 - (integral of cos on [-1,1])^2
# = 4*sin(1)^2*... = -4*sin(1)*cos(1) = -2*sin(2) ≈ -1.8185...
```

```
double integral of cos(x+y) on [-1,1]^2 = -1.818594853...
```

---

## 3. The Peaks Function

The MATLAB `peaks` function — a combination of Gaussians:

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

def peaks(x, y):
    return (3*(1-x)**2 * jnp.exp(-x**2 - (y+1)**2)
            - 10*(x/5 - x**3 - y**5) * jnp.exp(-x**2 - y**2)
            - 1/3 * jnp.exp(-(x+1)**2 - y**2))

g = Chebfun2.from_function(peaks, domain=[-3, 3, -3, 3])
print(g)

# Find the maximum value
x_max, y_max, f_max = g.max2()
print(f"max at ({float(x_max):.3f}, {float(y_max):.3f}) = {float(f_max):.4f}")
```

```
Chebfun2 on [-3,3] x [-3,3], rank 18
max at (-0.009, 1.582) = 8.1062
```

---

## 4. Gradient and Laplacian

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

# f(x,y) = exp(-(x^2+y^2))
g = Chebfun2.from_function(lambda x, y: jnp.exp(-(x**2 + y**2)))

# Partial derivatives
gx = g.diff(1, 0)   # ∂f/∂x
gy = g.diff(0, 1)   # ∂f/∂y

# Verify at (0.3, 0.4):
# df/dx = -2x * exp(-(x^2+y^2))
x0, y0 = jnp.array(0.3), jnp.array(0.4)
r2 = 0.3**2 + 0.4**2
exact_x = -2*0.3 * jnp.exp(-r2)
exact_y = -2*0.4 * jnp.exp(-r2)

print(f"gx(0.3,0.4) = {float(gx(x0,y0)):.10f}")
print(f"exact       = {float(exact_x):.10f}")
print(f"gy(0.3,0.4) = {float(gy(x0,y0)):.10f}")
print(f"exact       = {float(exact_y):.10f}")
```

```
gx(0.3,0.4) = -0.4649839...
exact       = -0.4649839...
gy(0.3,0.4) = -0.6199786...
exact       = -0.6199786...
```

---

## 5. Integration over a Rectangle

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

# Integrate x^2 * y^2 on [0,1] x [0,1]
# Exact: (1/3) * (1/3) = 1/9
g = Chebfun2.from_function(lambda x, y: x**2 * y**2, domain=[0, 1, 0, 1])
print(f"integral x^2*y^2 = {float(g.sum2()):.12f}")
print(f"exact (1/9)       = {1.0/9.0:.12f}")
```

```
integral x^2*y^2 = 0.111111111111
exact (1/9)       = 0.111111111111
```

---

## 6. Sum Along One Direction

`.sum(dim)` integrates over one variable, returning a 1D Chebfun:

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

g = Chebfun2.from_function(lambda x, y: jnp.sin(x) * jnp.cos(y))

# Integrate over y: result should be sin(x) * integral_y(cos y)
# = sin(x) * 2*sin(1)
g_x = g.sum(dim=1)   # integrate over y, get function of x only

x0 = jnp.array(0.5)
expected = float(jnp.sin(0.5) * 2.0 * jnp.sin(1.0))
print(f"(sum over y)(0.5) = {float(g_x(x0)):.10f}")
print(f"expected          = {expected:.10f}")
```

```
(sum over y)(0.5) = 0.79824...
expected          = 0.79824...
```

---

## 7. Low-Rank Rank Structure

Separable functions are represented exactly with very few ranks:

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
import jax.numpy as jnp

# Rank-1: f(x,y) = exp(x) * cos(y)
r1 = Chebfun2.from_function(lambda x, y: jnp.exp(x) * jnp.cos(y))
print(f"exp(x)*cos(y): rank = {r1.rank}")

# Rank-2: cos(x+y)
r2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
print(f"cos(x+y):      rank = {r2.rank}")

# Higher rank: exp(-x^2 - y^2)
rg = Chebfun2.from_function(lambda x, y: jnp.exp(-x**2 - y**2))
print(f"exp(-x^2-y^2): rank = {rg.rank}")
```

```
exp(x)*cos(y): rank = 1
cos(x+y):      rank = 2
exp(-x^2-y^2): rank = 12
```

---

## Gallery

Figures generated automatically from `examples/approx2/` and `examples/opt/`.

### exp(x+y) surface

![exp(x+y) surface](../../images/approx2/smooth_functions_2d_exp.png)

### cos(x+y²) contour

![cos(x+y²) contour](../../images/approx2/smooth_functions_2d_cos.png)

### Franke's function

![Franke's function](../../images/approx2/smooth_functions_2d_franke.png)

### Rank-1 and rank-2 functions

![Rank of functions](../../images/approx2/rank_of_functions.png)

![Rank contour](../../images/approx2/rank_of_functions_contour.png)

### 2-D integration

![Integration 2D](../../images/approx2/integration_2d.png)

### 2-D differentiation

![Differentiation 2D](../../images/approx2/differentiation_2d.png)

### Global minimum in 2D

![Global minimum 2D](../../images/opt/global_minimum_2d.png)
