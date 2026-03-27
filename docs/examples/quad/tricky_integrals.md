# Tricky Integrals

*Original: [chebfun.org/examples/quad/TrickyIntegrands](https://www.chebfun.org/examples/quad/TrickyIntegrands.html)*

---

Some integrals are challenging for standard quadrature because of near-singularities,
oscillatory integrands, or endpoint behavior. Chebfun handles many of these
gracefully because the Chebyshev nodes cluster near the endpoints.

## Highly oscillatory integrands

For $f(x) = \sin(100x)$ on $[-1,1]$, the integral is exactly 0, but
a naive quadrature rule needs $O(100)$ points to resolve the oscillations:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(lambda x: jnp.sin(100 * x))
val = float(f.sum())
print(f"∫ sin(100x) dx = {val:.2e}  (exact: 0)")
print(f"Degree: {len(f)-1}")
```

```
∫ sin(100x) dx = 1.11e-16  (exact: 0)
Degree: 203
```

Chebfun automatically uses enough Chebyshev points to resolve 100 oscillations.

## Near-endpoint singularities

For $f(x) = \log(1+x)$ near $x=-1$ (where $\log\to-\infty$), chebfun resolves
the singularity because Chebyshev nodes cluster near the endpoints:

```python
g = cj.chebfun(lambda x: jnp.log(1 + x + 1e-14), domain=(-1.0+1e-10, 1.0))
# ∫₋₁¹ log(1+x) dx = 2*log(2) - 2 ≈ -0.6137...
val = float(g.sum())
exact = 2*np.log(2) - 2
print(f"∫ log(1+x) dx ≈ {val:.8f}  (exact: {exact:.8f})")
```

## Dawson's integral

Dawson's integral $D(x) = e^{-x^2} \int_0^x e^{t^2}\,dt$ arises in quantum
mechanics and can be computed via `cumsum`:

```python
h = cj.chebfun(lambda t: jnp.exp(t**2), domain=(0.0, 2.0))
H = h.cumsum()  # H(x) = ∫₀ˣ exp(t²) dt
dawson_approx = cj.chebfun(lambda x: jnp.exp(-x**2) * H(x), domain=(0.0, 2.0))
```

![Tricky integral convergence and examples](../../images/quad/tricky_integrals.png)
