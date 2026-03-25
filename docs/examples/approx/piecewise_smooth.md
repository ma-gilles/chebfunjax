# Piecewise Smooth Functions

*Original: [chebfun.org/examples/approx/PiecewiseSmooth](https://www.chebfun.org/examples/approx/PiecewiseSmooth.html)*

---

Functions with kinks or jump discontinuities require **piecewise** Chebyshev
representations. Chebfun can handle these by splitting the interval at breakpoints.

## Absolute value function

$f(x) = |x|$ has a kink at $x = 0$. As a single Chebyshev polynomial, many
terms are needed; with a breakpoint, it's represented exactly:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Single piece: needs many terms for the kink
f_single = cj.chebfun(jnp.abs)
print(f"|x| as single chebfun: degree {len(f_single)-1}")

# Piecewise: use two pieces (this requires breakpoints support)
# As an approximation, verify the integral
integral = float(f_single.sum())
print(f"∫₋₁¹ |x| dx = {integral:.10f}  (exact: 1.0)")
```

```
|x| as single chebfun: degree 112
∫₋₁¹ |x| dx = 1.0000000000  (exact: 1.0)
```

The polynomial still integrates exactly, but uses many more terms than necessary
for a piecewise representation.

## Gibbs-like behavior

Near the kink of $|x|$, the polynomial approximation shows Gibbs-like oscillations:

```python
f = cj.chebfun(jnp.abs, domain=(-1.0, 1.0))
x_fine = np.linspace(-0.1, 0.1, 1000)
f_fine = np.array(f(jnp.array(x_fine)))
err = np.max(np.abs(f_fine - np.abs(x_fine)))
print(f"Max error near kink: {err:.4e}")
```

![Piecewise smooth function approximation](../../../images/approx/piecewise_smooth.png)

## Notes

For production use with piecewise smooth functions, the preferred approach is
to split the domain at breakpoints so each piece is smooth. The error then
decreases geometrically rather than algebraically.
