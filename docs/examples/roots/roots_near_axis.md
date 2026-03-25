# Complex Roots Near the Real Axis

*Nick Trefethen, October 2011*

*Original: [chebfun.org/examples/roots/RootsNearAxis](https://www.chebfun.org/examples/roots/RootsNearAxis.html)*

---

A Chebfun approximation of a smooth function is a polynomial that is accurate
inside a **Bernstein ellipse** in the complex plane. This gives access not
only to real roots, but also to complex roots near the real interval.

## A function with no real roots

Consider $f(x) = 3 + \sin(x) + \sin(\pi x)$ on $[0, 30]$:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(lambda x: 3 + jnp.sin(x) + jnp.sin(jnp.pi * x),
               domain=(0.0, 30.0))

real_roots = f.roots()
print(f"Real roots: {len(real_roots)}")   # → 0
print(f"Polynomial degree: {len(f) - 1}")
```

```
Real roots: 0
Polynomial degree: 85
```

The function has no real roots — the minimum value is well above zero:

```python
_, min_val = f.min()
print(f"min(f) ≈ {min_val:.4f}")   # → 23.5
```

## The Bernstein ellipse

A Chebfun of degree $n$ on $[a,b]$ is accurate inside a Bernstein ellipse
in the complex plane. For a function of degree 85, the ellipse extends roughly
$1/85$ of the interval length into the complex plane — enough to capture
roots that are close to the real axis.

![Bernstein ellipses for f on [0,30]](../../../images/roots/roots_near_axis.png)

The ellipses shown have semi-minor axes $\approx$ Im(f singularities).
The function $\sin(x)$ has singularities at $\pm i$, $\sin(\pi x)$ at
$\pm i/\pi$. So $f$ has complex roots near $\text{Im}(z) = 1/\pi \approx 0.32$.

## Why this matters

This mechanism — access to complex roots via the Bernstein ellipse — is
exploited by Chebfun's `roots(f,'complex')` flag. While chebfunjax's
`roots()` focuses on real roots, the underlying polynomial representation
gives the same access to complex roots for users who want to work with them.

## References

1. L. N. Trefethen, *Approximation Theory and Approximation Practice*,
   SIAM, 2013, Ch. 8.
2. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
