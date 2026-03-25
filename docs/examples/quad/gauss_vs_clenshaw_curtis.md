# Gauss and Clenshaw-Curtis Quadrature

*Nick Trefethen, September 2010*

*Original: [chebfun.org/examples/quad/GaussClenCurt](https://www.chebfun.org/examples/quad/GaussClenCurt.html)*

---

Two of the most important quadrature rules for smooth functions are
**Gauss-Legendre** and **Clenshaw-Curtis** quadrature. Both achieve
geometric (exponential) convergence for analytic functions, but they differ
in their nodes and derivation.

## The two rules

**Gauss-Legendre** uses the $n$ zeros of the Legendre polynomial $P_n$
as nodes, and achieves exactness for polynomials of degree up to $2n-1$.

**Clenshaw-Curtis** uses the Chebyshev points $x_k = \cos(k\pi/n)$ for
$k=0,1,\ldots,n$ as nodes. It achieves exactness for polynomials of degree
up to $n$, but in practice converges as fast as Gauss for smooth functions.

## Comparing convergence

For the highly oscillatory function $f(x) = x\sin(2e^{2\sin(2e^{2x})})$:

```python
import chebfunjax as cj
import jax.numpy as jnp
from scipy.special import roots_legendre
import numpy as np

f_fn = lambda x: x * np.sin(2 * np.exp(2 * np.sin(2 * np.exp(2 * x))))
f_jax = lambda x: x * jnp.sin(2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x))))

# Reference integral from Chebfun
fc = cj.chebfun(f_jax)
I_exact = float(fc.sum())
print(f"Reference integral = {I_exact:.15f}")
print(f"Chebfun degree = {len(fc)}")
```

```
Reference integral = -0.231857749048...
Chebfun degree = 220
```

![Convergence of Gauss and Clenshaw-Curtis](../../../images/quad/gauss_vs_clenshaw_curtis.png)

## The Trefethen–Bornemann result

A surprising theorem by Xiang and Bornemann shows that both Gauss and
Clenshaw-Curtis converge at the same **algebraic rate** for functions
with endpoint singularities or algebraic behavior — not just geometric
convergence for analytic functions.

For a function with Chebyshev coefficients decaying as $O(n^{-p})$, the
quadrature error is $O(n^{-(p+1)})$ — **one order better** than the
coefficient decay would suggest. This is because the high-frequency
Chebyshev components contribute near-zero integrals.

## Notes

- In chebfunjax, the integral `f.sum()` uses the Clenshaw-Curtis rule
  implicitly — it integrates the Chebyshev expansion analytically.
- Both rules converge geometrically for analytic functions.
- For rough functions (algebraic singularities), both rules converge
  algebraically at one order faster than the coefficient decay.

## References

1. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, Ch. 12.
2. S. Xiang and F. Bornemann, On the convergence rates of Gauss and
   Clenshaw-Curtis quadrature, *SIAM J. Numer. Anal.* 50 (2012), 2685–2704.
