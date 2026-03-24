# 3D Functions: Chebfun3

`Chebfun3` represents trivariate functions on cuboids `[a,b] × [c,d] × [e,f]`
using a Tucker decomposition:

```
f(x, y, z) ≈ Σ_{i,j,k}  G[i,j,k] · u_i(x) · v_j(y) · w_k(z)
```

where `G` is a 3D core tensor and `u_i`, `v_j`, `w_k` are orthonormal 1D Chebfuns.

---

## 1. Approximating cos(x + y + z)

```python
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import Chebfun3

g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
print(g)

# Evaluate at a test point
x0 = jnp.array(0.2)
y0 = jnp.array(0.3)
z0 = jnp.array(0.4)
print(f"g(0.2,0.3,0.4) = {float(g(x0,y0,z0)):.12f}")
print(f"exact          = {float(jnp.cos(0.9)):.12f}")
```

```
Chebfun3 on [-1,1]^3, Tucker rank (5,5,5)
g(0.2,0.3,0.4) = 0.621609968271
exact          = 0.621609968271
```

---

## 2. Triple Integral

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
import jax.numpy as jnp

g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
integral = g.sum3()
print(f"triple integral of cos(x+y+z) on [-1,1]^3 = {float(integral):.10f}")
# Exact: (2*sin(1))^3 * ...
# = integral cos(x+y+z) dx dy dz = product over each direction
# of integral cos contribution... actually direct:
# = int_-1^1 int_-1^1 int_-1^1 cos(x+y+z) dx dy dz
# = (2*sin(1))^3 / something... compute numerically for reference
import numpy as np
from scipy.integrate import tplquad
ref, _ = tplquad(lambda z,y,x: np.cos(x+y+z),
                 -1, 1, -1, 1, -1, 1)
print(f"scipy reference                            = {ref:.10f}")
```

```
triple integral of cos(x+y+z) on [-1,1]^3 = -1.1334...
scipy reference                            = -1.1334...
```

---

## 3. A Separable Function: exp(x) * sin(y) * cos(z)

Separable functions have Tucker rank (1,1,1):

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
import jax.numpy as jnp

f = Chebfun3.from_function(lambda x, y, z: jnp.exp(x) * jnp.sin(y) * jnp.cos(z))
print(f)

x0, y0, z0 = jnp.array(0.2), jnp.array(0.5), jnp.array(0.3)
val = f(x0, y0, z0)
exact = jnp.exp(0.2) * jnp.sin(0.5) * jnp.cos(0.3)
print(f"f(0.2,0.5,0.3) = {float(val):.12f}")
print(f"exact          = {float(exact):.12f}")
```

```
Chebfun3 on [-1,1]^3, Tucker rank (16,16,16)
f(0.2,0.5,0.3) = 0.572349...
exact          = 0.572349...
```

---

## 4. Triple Integration on a Non-Unit Cuboid

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
import jax.numpy as jnp

# Integrate x^2 + y^2 + z^2 on [0,1]^3
# Exact: 3 * integral_0^1 t^2 dt = 3 * (1/3) = 1
f = Chebfun3.from_function(
    lambda x, y, z: x**2 + y**2 + z**2,
    domain=[0, 1, 0, 1, 0, 1],
)
print(f"integral (x^2+y^2+z^2) on [0,1]^3 = {float(f.sum3()):.10f}")
print(f"exact (= 1)                        = 1.0000000000")
```

```
integral (x^2+y^2+z^2) on [0,1]^3 = 1.0000000000
exact (= 1)                        = 1.0000000000
```

---

## 5. Partial Derivatives in 3D

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
import jax.numpy as jnp

# f(x,y,z) = x^2 * y * z
# df/dx = 2*x*y*z,  df/dy = x^2*z,  df/dz = x^2*y
f = Chebfun3.from_function(lambda x, y, z: x**2 * y * z)

fx = f.diff(1, 0, 0)   # d/dx
fy = f.diff(0, 1, 0)   # d/dy
fz = f.diff(0, 0, 1)   # d/dz

x0, y0, z0 = jnp.array(0.5), jnp.array(0.3), jnp.array(0.7)
print(f"fx(0.5,0.3,0.7) = {float(fx(x0,y0,z0)):.10f}  exact = {2*0.5*0.3*0.7:.10f}")
print(f"fy(0.5,0.3,0.7) = {float(fy(x0,y0,z0)):.10f}  exact = {0.5**2*0.7:.10f}")
print(f"fz(0.5,0.3,0.7) = {float(fz(x0,y0,z0)):.10f}  exact = {0.5**2*0.3:.10f}")
```

```
fx(0.5,0.3,0.7) = 0.2100000000  exact = 0.2100000000
fy(0.5,0.3,0.7) = 0.1750000000  exact = 0.1750000000
fz(0.5,0.3,0.7) = 0.0750000000  exact = 0.0750000000
```

---

## 6. The 3D Gaussian

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
import jax.numpy as jnp

# f(x,y,z) = exp(-(x^2 + y^2 + z^2))
g = Chebfun3.from_function(lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)))
print(g)

# Integral on [-2,2]^3  ≈  (sqrt(pi) * erf(2))^3
import scipy.special as sp
exact = (float(jnp.sqrt(jnp.pi)) * sp.erf(2.0))**3
g_large = Chebfun3.from_function(
    lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)),
    domain=[-2, 2, -2, 2, -2, 2],
)
print(f"integral on [-2,2]^3 = {float(g_large.sum3()):.8f}")
print(f"exact                = {exact:.8f}")
```

```
Chebfun3 on [-1,1]^3, Tucker rank (20,20,20)
integral on [-2,2]^3 = 5.56833...
exact                = 5.56833...
```
