# Smooth 2D Functions

*Original: [chebfun.org/examples/approx2/](https://www.chebfun.org/examples/approx2/)*

---

Chebfun2 approximates smooth functions of two variables on rectangles using a
**low-rank decomposition**. The function $f(x,y)$ is represented as a sum of
rank-1 terms (outer products of 1D Chebyshev expansions):

$$f(x,y) \approx \sum_{k=1}^r c_k \, u_k(x) \, v_k(y).$$

The number of terms $r$ — the **numerical rank** — measures how "separable"
the function is. For analytic functions, $r$ grows slowly or stays bounded.

## Franke's function

Franke's function is a standard test case in scattered data interpolation:

```python
import chebfunjax as cj
import jax.numpy as jnp

def franke(x, y):
    return (0.75 * jnp.exp(-((9*x-2)**2 + (9*y-2)**2)/4)
          + 0.75 * jnp.exp(-((9*x+1)**2)/49 - (9*y+1)/10)
          + 0.5  * jnp.exp(-((9*x-7)**2 + (9*y-3)**2)/4)
          - 0.2  * jnp.exp(-(9*x-4)**2 - (9*y-7)**2))

f = cj.chebfun2(franke)
print(f"Numerical rank: {f.rank}")
print(f"∬ Franke dA = {float(f.sum()):.8f}")
```

![Franke's function and Gaussian on [-1,1]^2](../../../images/approx2/smooth_functions_2d_franke.png)

## Trigonometric function

```python
g = cj.chebfun2(lambda x, y: jnp.cos(3*x) * jnp.sin(5*y))
print(f"cos(3x)sin(5y): rank = {g.rank}")  # rank 1 — perfectly separable!
```

![cos(3x)*sin(5y) and exp(-x^2-y^2)](../../../images/approx2/smooth_functions_2d_cos.png)

## Notes

- **Rank-1 functions** like $\cos(3x)\sin(5y)$ are represented exactly with one term.
- **Gaussian** $e^{-(x^2+y^2)}$ has rank ~15 for double precision.
- The Franke function is less separable and requires higher rank.
