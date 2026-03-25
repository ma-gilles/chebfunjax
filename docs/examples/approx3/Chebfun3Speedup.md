# Chebfun3 Construction Speedup

*Behnam Hashemi, Christoph Strössner, and Nick Trefethen, March 2023*

*Original: [Chebfun3 speedups — Chebfun](https://www.chebfun.org/examples/approx3/Chebfun3Speedup.html)*

---

## The Chebfun3f Algorithm

Chebfun3 was introduced by Behnam Hashemi in 2016 [2] and represents
trivariate functions $f(x,y,z)$ in Tucker format. Christoph Strössner,
in his PhD work at EPFL with Sergey Dolgov and Daniel Kressner, found a
faster Tucker construction algorithm [1] — the **Chebfun3f** algorithm —
that identifies fibers in each of the three directions more directly.

In chebfunjax, the Chebfun3f algorithm is the default constructor.

## Hard Functions (Not Low Rank)

The function $f(x,y,z) = \tanh(k(x+y+z))$ is not of low Tucker rank,
and its rank grows with $k$:

```python
from chebfunjax.chebfun3d.chebfun3 import chebfun3
import jax.numpy as jnp

for k in [1, 2, 4, 8]:
    f = chebfun3(lambda x, y, z, _k=k: jnp.tanh(_k * (x + y + z)))
    print(f"k={k}: Tucker rank = {f.rank}")
```

```
k=1: Tucker rank = (5, 5, 5)
k=2: Tucker rank = (11, 11, 11)
k=4: Tucker rank = (23, 23, 23)
k=8: Tucker rank = (47, 47, 47)
```

## Easy Functions (Low Rank)

The 3D Runge function $f(x,y,z) = \frac{1}{1+k(x^2+y^2+z^2)}$ has
low Tucker rank for all values of $k$, because it separates as

$$f(x,y,z) = \sum_{n=0}^\infty (-k)^n (x^2+y^2+z^2)^n$$

and $(x^2+y^2+z^2)^n$ has Tucker rank $(n+1, n+1, n+1)$.

```python
for k in [1, 8, 32, 64]:
    f = chebfun3(lambda x, y, z, _k=k: 1.0 / (1.0 + _k*(x**2+y**2+z**2)))
    print(f"k={k}: Tucker rank = {f.rank}")
```

```
k=1:  Tucker rank = (10, 10, 10)
k=8:  Tucker rank = (15, 15, 15)
k=32: Tucker rank = (15, 15, 15)
k=64: Tucker rank = (16, 16, 16)
```

## Sometimes the Classical Approach Is Better

For functions that separate as $f(x,y,z) = g(x,y) \cdot h(z)$,
the fiber-based algorithm still works but may use slightly more evaluations
than a direct slice-Tucker approach. The chebfunjax implementation uses the
Chebfun3f algorithm consistently.

```python
f_mixed = chebfun3(lambda x, y, z: jnp.tanh(10*(x+y)) * jnp.cos(z))
print(f"tanh(10(x+y))*cos(z): rank = {f_mixed.rank}")
```

![Tucker ranks for hard vs easy functions](../../../images/approx3/Chebfun3Speedup.png)

## References

1. S. Dolgov, D. Kressner, and C. Strössner, Functional Tucker approximation
   using Chebyshev interpolation, *SIAM J. Sci. Comput.* 43 (2021), A2190–A2210.

2. B. Hashemi and L. N. Trefethen, Chebfun in three dimensions,
   *SIAM J. Sci. Comput.* 39 (2017), C341–C363.

3. L. N. Trefethen, Chebfun timings for tough 1D/2D/3D functions,
   Chebfun Example, April 2015.

4. L. N. Trefethen, Cubature, approximation, and isotropy in the hypercube,
   *SIAM Review* 59 (2017), 469–491.
