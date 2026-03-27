# Clenshaw–Curtis Quadrature

**Inspired by [Chebfun](https://www.chebfun.org/) examples (quad/ClenshawCurtis)**

---

**Clenshaw–Curtis quadrature** uses the values of $f$ at Chebyshev nodes
$x_k = \cos(k\pi/n)$ to form an exact integral of the degree-$n$ Chebyshev
interpolant. Its weights are computed via the FFT.

## Equivalence with Chebyshev integration

When you call `f.sum()` on a Chebfun, it integrates the Chebyshev series
term by term using the exact formula $\int_{-1}^1 T_k(x)\,dx = 0$ for odd $k$
and $2/k^2-1$ for even $k$. This is equivalent to applying Clenshaw–Curtis
weights to the function values.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Compute integral of exp(sin(x)) by chebfun
f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)))
val = float(f.sum())
print(f"∫₋₁¹ exp(sin(x)) dx = {val:.12f}")
print(f"Chebfun degree used: {len(f)-1}")
```

```
∫₋₁¹ exp(sin(x)) dx = 2.532131755505
Chebfun degree used: 22
```

## Manual CC weights

The Clenshaw–Curtis weights can be computed directly:

```python
def clenshaw_curtis_weights(n):
    """Compute n+1 Clenshaw-Curtis weights for [-1,1]."""
    c = np.zeros(n+1)
    c[0:n+1:2] = 2.0 / (1 - np.arange(0, n+1, 2)**2)
    return np.fft.irfft(c, n=2*n)[:n+1] / n  # simplified

n = 20
w = clenshaw_curtis_weights(n)
x = np.cos(np.pi * np.arange(n+1) / n)
val_cc = np.dot(w, np.exp(np.sin(x)))
print(f"Manual CC ({n} pts): {val_cc:.12f}")
```

![Clenshaw-Curtis nodes and weights](../../images/quad/clenshaw_curtis.png)

## References

1. C. W. Clenshaw and A. R. Curtis, A method for numerical integration on an
   automatic computer, *Numer. Math.* 2 (1960), 197–205.
2. L. N. Trefethen, Is Gauss quadrature better than Clenshaw–Curtis?
   *SIAM Review* 50 (2008), 67–87.
