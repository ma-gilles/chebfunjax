# Chebfun2: Smooth 2D Function Approximation

*Original: [chebfun.org/examples/approx2/](https://www.chebfun.org/examples/approx2/)*

---

Chebfun2 extends the Chebfun philosophy to functions of two variables on
rectangles. Just as a 1D Chebfun is a polynomial approximation in $x$,
a Chebfun2 is a *bivariate polynomial approximation* — a sum of rank-1
terms (outer products of 1D Chebyshev expansions).

## Creating a Chebfun2

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Gaussian on the square [-1,1]^2
f = cj.chebfun2(lambda x, y: jnp.exp(-(x**2 + y**2)))

# Evaluate at a point
val = float(f(jnp.array(0.5), jnp.array(0.3)))
exact = float(jnp.exp(-(0.5**2 + 0.3**2)))
print(f"f(0.5, 0.3) = {val:.12f}  (exact: {exact:.12f})")
```

```
f(0.5, 0.3) = 0.711770322763  (exact: 0.711770322763)
```

## Integration

The `sum` method computes the double integral:

```python
import scipy.special

# ∬ exp(-(x^2+y^2)) dA over [-1,1]^2 = pi * erf(1)^2
integral = float(f.sum())
exact_int = np.pi * scipy.special.erf(1.0)**2
print(f"∬ f dA = {integral:.10f}  (exact: {exact_int:.10f})")
```

```
∬ f dA = 2.2309851414  (exact: 2.2309851414)
```

![Gaussian on [-1,1]^2: contour and 3D surface](../../images/approx2/chebfun2_basics.png)

## Symmetry

The integral of an odd function over a symmetric domain is zero:

```python
g = cj.chebfun2(lambda x, y: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y))
print(f"∬ sin(πx)cos(πy) dA = {float(g.sum()):.2e}")  # → ≈ 0
```

```
∬ sin(πx)cos(πy) dA = 7.68e-33
```

## Notes

Chebfun2 uses a **Gaussian elimination** (low-rank) algorithm to represent
$f(x,y)$ as a sum of rank-1 terms. This is memory-efficient for functions
that are well-approximated by low-rank matrices. For more details, see the
[approximation 2D examples](smooth_functions_2d.md).