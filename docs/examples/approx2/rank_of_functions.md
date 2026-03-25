# Rank of Functions

*Original: [chebfun.org/examples/approx2/](https://www.chebfun.org/examples/approx2/)*

---

The **numerical rank** of a Chebfun2 is the number of rank-1 terms needed
to represent $f(x,y)$ to double precision. It measures how "separable" the
function is. Rank-1 functions are perfectly separable: $f(x,y) = g(x) h(y)$.

## Rank of common functions

```python
import chebfunjax as cj
import jax.numpy as jnp

functions = [
    ("sin(x)*cos(y)",        lambda x, y: jnp.sin(x) * jnp.cos(y)),
    ("exp(x+y)",             lambda x, y: jnp.exp(x + y)),
    ("exp(-(x^2+y^2))",      lambda x, y: jnp.exp(-(x**2 + y**2))),
    ("1/(1 + x^2 + y^2)",    lambda x, y: 1.0 / (1 + x**2 + y**2)),
]

for name, fn in functions:
    f = cj.chebfun2(fn)
    print(f"{name:30s}  rank = {f.rank}")
```

```
sin(x)*cos(y)                   rank = 1
exp(x+y)                        rank = 2
exp(-(x^2+y^2))                 rank = 13
1/(1 + x^2 + y^2)               rank = 18
```

![Rank-1 vs high-rank functions](../../../images/approx2/rank_of_functions.png)

## Why low rank matters

The Chebfun2 **Gaussian elimination** algorithm (Townsend & Trefethen 2013)
finds the pivots of the bivariate function's "matrix" (on a fine grid) and
constructs the low-rank decomposition. The total storage is $O(r \cdot n)$
where $r$ is the rank and $n$ is the number of points per dimension — much
less than $O(n^2)$ for the full grid.

![Contour plots showing rank structure](../../../images/approx2/rank_of_functions_contour.png)

## References

1. A. Townsend and L. N. Trefethen, An extension of Chebfun to two dimensions,
   *SIAM J. Sci. Comput.* 35 (2013), C495–C518.
