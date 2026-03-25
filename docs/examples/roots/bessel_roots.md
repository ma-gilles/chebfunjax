# Bessel Function Roots

*Original: [chebfun.org/examples/roots/BesselRoots](https://www.chebfun.org/examples/roots/BesselRoots.html)*

---

The Bessel function $J_0(x)$ has infinitely many real positive roots, which
are important in physics (cylindrical waveguides, heat conduction in cylinders).
Chebfun finds them all on a given interval.

## Roots of $J_0$ on $[0, 30]$

```python
import chebfunjax as cj
import jax.numpy as jnp
import scipy.special
import numpy as np

J0 = cj.chebfun(lambda x: jnp.array(scipy.special.j0(np.array(x))),
                domain=(0.0, 30.0))
roots = np.sort(np.array(J0.roots()))
print(f"Number of roots of J_0 on [0, 30]: {len(roots)}")

# Compare with known values from scipy
exact = scipy.special.jn_zeros(0, len(roots))
err = np.max(np.abs(roots - exact))
print(f"Max error vs scipy: {err:.2e}")
```

```
Number of roots of J_0 on [0, 30]: 9
Max error vs scipy: 2.13e-13
```

## First 5 roots

| $k$ | $j_{0,k}$ (chebfunjax) | $j_{0,k}$ (exact) | error |
|-----|------------------------|-------------------|-------|
| 1 | 2.4048255577 | 2.4048255577 | 2.4e-14 |
| 2 | 5.5200781103 | 5.5200781103 | 3.6e-14 |
| 3 | 8.6537279129 | 8.6537279129 | 2.1e-14 |
| 4 | 11.7915344391 | 11.7915344391 | 1.8e-13 |
| 5 | 14.9309177086 | 14.9309177086 | 2.1e-13 |

## Asymptotic spacing

For large $x$, the roots of $J_0$ are approximately equally spaced with
spacing $\pi$ (the asymptotic period of $J_0$):

```python
spacings = np.diff(roots)
print(f"Mean spacing: {np.mean(spacings):.6f}  (π ≈ {np.pi:.6f})")
```

![Bessel function roots on [0,30]](../../../images/roots/bessel_roots.png)
