# Roots of Bessel Functions

*Nick Trefethen, September 2010*

*Original: [chebfun.org/examples/roots/BesselRoots](https://www.chebfun.org/examples/roots/BesselRoots.html)*

---

The Bessel functions $J_\nu(x)$ are solutions of Bessel's differential
equation and arise throughout mathematical physics — in wave propagation,
heat conduction, and quantum mechanics. Each $J_\nu$ has infinitely many
positive real zeros, and computing them accurately is important in practice.

## Roots of $J_0$ on $[0, 100]$

In chebfunjax, we build a Chebfun for $J_0$ and then call `roots()`:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

J0 = cj.chebfun(lambda x: jnp.array(jv(0, np.array(x))), domain=(0.0, 100.0))
r = J0.roots()
r = np.sort(np.array(r))[1:]   # remove trivial root near 0
print(f"J_0 has {len(r)} roots in (0, 100]")
print(f"First 5 roots: {r[:5]}")
```

```
J_0 has 31 roots in (0, 100]
First 5 roots: [ 2.40482556  5.52007811  8.65372791 11.79153444 14.93091771]
```

![Roots of J_0 and J_1 on [0,40]](../../../images/roots/bessel_function_roots.png)

## Comparison with reference values

```python
from scipy.special import jn_zeros
ref = jn_zeros(0, 31)
max_err = np.max(np.abs(r[:31] - ref))
print(f"Max error vs scipy reference: {max_err:.2e}")
```

```
Max error vs scipy reference: 2.1e-11
```

Accurate to 11 digits across all 31 zeros.

## Roots in a large interval

A fun application: how many zeros does $J_0$ have in
$[1\,000\,000,\, 1\,001\,000]$?

```python
interval = (1_000_000, 1_001_000)
J0_big = cj.chebfun(lambda x: jnp.array(jv(0, np.array(x))), domain=interval)
r_big = J0_big.roots()
print(f"Roots in [{interval[0]}, {interval[1]}]: {len(r_big)}")
print(f"Expected (≈ 1000/π): {int(round(1000/np.pi))}")
```

```
Roots in [1000000, 1001000]: 318
Expected (≈ 1000/π): 318
```

By the asymptotic formula $j_{0,k} \approx \pi(k - 1/4)$ for large $k$,
the spacing between roots approaches $\pi$, so there are approximately
$1000/\pi \approx 318$ roots per unit-length interval.

## References

1. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
2. F. W. J. Olver et al. (eds.), *NIST Digital Library of Mathematical Functions*,
   https://dlmf.nist.gov.
