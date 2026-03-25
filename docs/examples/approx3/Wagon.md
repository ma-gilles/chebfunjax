# Low-Rank Representation of Wagon's Function

*Behnam Hashemi, July 2016*

*Original: [Low-rank representation of Wagon's function — Chebfun](https://www.chebfun.org/examples/approx3/Wagon.html)*

---

## Wagon's Function

Stan Wagon [1] suggested the problem of finding the global minimum of the
three-dimensional function

$$f(x,y,z) = e^{\sin(50x)} + \sin(60e^y)\sin(60z)
+ \sin(70\sin x)\cos(10z) + \sin(\sin(80y))
- \sin(10(x+z)) + \tfrac{x^2+y^2+z^2}{4}.$$

Despite its apparent complexity, this function has a surprisingly low
Tucker rank:

```python
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import chebfun3

f = chebfun3(lambda x, y, z:
    jnp.exp(jnp.sin(50*x))
    + jnp.sin(60*jnp.exp(y)) * jnp.sin(60*z)
    + jnp.sin(70*jnp.sin(x)) * jnp.cos(10*z)
    + jnp.sin(jnp.sin(80*y))
    - jnp.sin(10*(x+z))
    + (x**2 + y**2 + z**2) / 4
)
print(f.rank)  # (4, 3, 5)
```

The Tucker rank $(4, 3, 5)$ is tiny: we need only 4 column fibers,
3 row fibers, and 5 tube fibers to represent $f$ to machine precision.

## Global Minimum

Wagon's function achieves its global minimum around $f \approx -2.72$:

```python
import numpy as np

n_grid = 50
xs = np.linspace(-1, 1, n_grid)
XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
vals = np.array(f(XX, YY, ZZ))
idx = np.unravel_index(np.argmin(vals), vals.shape)
x0, y0, z0 = xs[idx[0]], xs[idx[1]], xs[idx[2]]
print(f"min ≈ f({x0:.3f}, {y0:.3f}, {z0:.3f}) = {vals[idx]:.6f}")
```

```
min ≈ f(-0.388, 0.061, -0.020) = -2.722852
```

## Tucker Factor Fibers

The Tucker representation exposes the underlying structure.
The **columns** $f_1(x), \ldots, f_4(x)$ are the $x$-direction fibers,
each a smooth univariate function requiring hundreds of Chebyshev coefficients:

```python
import jax.numpy as jnp
t_ref = jnp.linspace(-1, 1, 200)
for i, col in enumerate(f.cols):
    n_coeffs = len(col.coeffs)
    print(f"col[{i}]: {n_coeffs} Chebyshev coefficients")
```

```
col[0]: 705 Chebyshev coefficients
col[1]: 705 Chebyshev coefficients
col[2]: 705 Chebyshev coefficients
col[3]: 705 Chebyshev coefficients
```

This highlights an important distinction: the **rank** of a function
can be small (4, 3, 5 here) while its **1D complexity** is high
(each fiber needs ~705 Chebyshev terms).

![Wagon's function Tucker factor fibers and coefficient decay](../../../images/approx3/Wagon.png)

## References

1. F. Bornemann, D. Laurie, S. Wagon, and J. Waldvogel,
   *The SIAM 100-Digit Challenge: a Study in High-Accuracy Numerical Computing*,
   SIAM, 2004.

2. B. Hashemi, L. N. Trefethen, Chebfun in three dimensions,
   *SIAM J. Sci. Comput.* 39 (2017), C341–C363.
