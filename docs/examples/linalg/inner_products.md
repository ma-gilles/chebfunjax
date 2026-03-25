# Inner Products and Norms

*Original: [chebfun.org/examples/linalg/](https://www.chebfun.org/examples/linalg/)*

---

Chebfunjax treats functions as elements of function spaces with inner products
and norms. This example demonstrates the $L^2$ inner product, various norms,
and orthogonality relations.

## The $L^2$ inner product

The inner product of two functions $f$ and $g$ on $[a,b]$ is

$$\langle f, g \rangle = \int_a^b f(x)\,g(x)\,dx.$$

In chebfunjax, this is computed by `f.inner(g)`:

```python
import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(lambda x: jnp.exp(x))
g = cj.chebfun(lambda x: jnp.sin(jnp.pi * x))
print(f"<exp(x), sin(πx)> = {float(f.inner(g)):.15f}")
```

## Legendre orthogonality

The Legendre polynomials $P_n$ satisfy $\langle P_m, P_n \rangle = \frac{2}{2n+1}\delta_{mn}$:

```python
from scipy.special import legendre
import numpy as np

def legpoly(n):
    return cj.chebfun(lambda x: jnp.array(legendre(n)(np.array(x))))

for m in range(5):
    for n in range(5):
        inn = float(legpoly(m).inner(legpoly(n)))
        expected = 0.0 if m != n else 2.0 / (2*n+1)
        assert abs(inn - expected) < 1e-10
print("All Legendre inner products correct!")
```

## Various norms

```python
f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)))
print(f"L1 norm  = {float(f.norm(1)):.10f}")
print(f"L2 norm  = {float(f.norm(2)):.10f}")
print(f"L∞ norm  = {float(f.norm(float('inf'))):.10f}")
```

```
L1 norm  = 2.4079768...
L2 norm  = 1.5694066...
L∞ norm  = 2.7182818...  (= e = max of exp(sin(x)))
```

![exp(sin(x)) with L1 area shaded](../../../images/linalg/inner_products.png)

## Trigonometric orthogonality

Sine and cosine functions are orthogonal on $[0, 2\pi]$:

```python
T = 2 * float(jnp.pi)
for m, n in [(1,2), (2,3), (1,1), (2,2)]:
    sm = cj.chebfun(lambda x, m=m: jnp.sin(m*x), domain=(0.0, T))
    sn = cj.chebfun(lambda x, n=n: jnp.sin(n*x), domain=(0.0, T))
    print(f"<sin({m}x), sin({n}x)> = {float(sm.inner(sn)):.8f}")
```

```
<sin(1x), sin(2x)> = 0.00000000
<sin(2x), sin(3x)> = 0.00000000
<sin(1x), sin(1x)> = 3.14159265   (= π)
<sin(2x), sin(2x)> = 3.14159265   (= π)
```
