# Polynomial Approximation

*Original: [chebfun.org/examples/approx/](https://www.chebfun.org/examples/approx/)*

---

Polynomial approximation is one of the most fundamental topics in numerical
analysis. Chebfun uses **Chebyshev interpolants** — polynomials that pass
through the function values at Chebyshev nodes — which near-optimally approximate
smooth functions in the $L^\infty$ norm.

## Degree and error

For $f(x) = e^x$, a degree-$n$ Chebyshev interpolant achieves error $\approx e/(4(n+1))!$:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(jnp.exp)
print(f"exp(x) on [-1,1]: degree {len(f)-1}")

# Evaluate error at 1000 points
x_test = np.linspace(-1, 1, 1000)
fvals = np.array(f(jnp.array(x_test)))
err = np.max(np.abs(fvals - np.exp(x_test)))
print(f"Max error: {err:.2e}")
```

```
exp(x) on [-1,1]: degree 16
Max error: 2.22e-16
```

## Equioscillation (Chebyshev property)

The best polynomial approximation of degree $n$ to $f$ in $L^\infty$ has an
error that **equioscillates** at $n+2$ points. The Chebyshev interpolant
achieves this near-optimally.

## Non-analytic functions

For $f(x) = |x|$, the Chebyshev coefficients decay only like $O(n^{-2})$
(because $f$ is only $C^0$):

```python
g = cj.chebfun(jnp.abs)
print(f"|x|: degree {len(g)-1}")
```

```
|x|: degree 112
```

![Polynomial approximation error and coefficients](../../../images/approx/polynomial_approximation.png)

![Chebyshev coefficient comparison](../../../images/approx/polynomial_approximation_coeffs.png)

## References

1. L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2013.
2. T. J. Rivlin, *An Introduction to the Approximation of Functions*, Dover, 1981.
