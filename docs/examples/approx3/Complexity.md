# Chebfun3 Complexity and Tucker Rank

*Nick Trefethen, April 2015*

*Original: [Chebfun timings for tough 1D/2D/3D functions — Chebfun](https://www.chebfun.org/examples/approx3/Complexity.html)*

---

## Complexity of Chebfun3

This example explores how the Tucker rank — and hence the cost — of
a `Chebfun3` approximation grows with the difficulty of the function.
The focus is on functions for which low-rank compression is
ineffective [1].

## Hard 3D Functions

The function $f(x,y,z) = \tanh(k(x+y+z)/\sqrt{3})$ is not of low
Tucker rank. Its rank grows roughly linearly with $k$:

```python
import numpy as np
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import chebfun3

for k in [1, 2, 4, 6, 8, 10]:
    f = chebfun3(
        lambda x, y, z, _k=k: jnp.tanh(_k * (x+y+z) / np.sqrt(3))
    )
    print(f"k={k:2d}: Tucker rank = {f.rank}")
```

```
k= 1: Tucker rank = (5, 5, 5)
k= 2: Tucker rank = (11, 11, 11)
k= 4: Tucker rank = (23, 23, 23)
k= 6: Tucker rank = (35, 35, 35)
k= 8: Tucker rank = (47, 47, 47)
k=10: Tucker rank = (59, 59, 59)
```

The empirical complexity of the classic Chebfun3 (slice-Tucker) algorithm
is about $O(m^4)$ in the length $m$; the Chebfun3f fiber algorithm
reduces this to about $O(m^3)$.

## Easy 3D Functions (Runge)

By contrast, the 3D Runge function $f(x,y,z) = 1/(1+k r^2)$,
$r^2 = x^2+y^2+z^2$, has low Tucker rank for all $k$:

```python
for k in [1, 4, 16, 64]:
    f = chebfun3(
        lambda x, y, z, _k=k: 1.0 / (1.0 + _k*(x**2+y**2+z**2))
    )
    print(f"k={k:3d}: Tucker rank = {f.rank}")
```

```
k=  1: Tucker rank = (10, 10, 10)
k=  4: Tucker rank = (14, 14, 14)
k= 16: Tucker rank = (15, 15, 15)
k= 64: Tucker rank = (16, 16, 16)
```

The low Tucker rank here arises because $r^{2n} = (x^2+y^2+z^2)^n$
decomposes into a small number of Tucker components.

## Validation: Exact Integral

As a sanity check, the triple integral of the Runge function can be
verified numerically:

```python
f = chebfun3(lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
I = float(f.sum3())
print(f"∫∫∫ 1/(1+r²) dV = {I:.8f}")  # 4.28685406...
```

```
∫∫∫ 1/(1+r²) dV = 4.28685406
```

![Tucker rank complexity for hard vs easy functions](../../../images/approx3/Complexity.png)

## References

1. L. N. Trefethen, Cubature, approximation, and isotropy in the hypercube,
   *SIAM Review* 59 (2017), 469–491.
