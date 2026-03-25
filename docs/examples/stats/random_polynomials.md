# Random Polynomials

*Original: [chebfun.org/examples/roots/RandomPolynomials](https://www.chebfun.org/examples/roots/RandomPolynomials.html)*

---

The distribution of roots of random polynomials is a beautiful subject
connecting probability theory, complex analysis, and numerical computation.
For polynomials with Gaussian random coefficients, the roots cluster near
the unit circle in the complex plane — and on the real axis, they follow
the arcsine distribution.

## Random Chebfuns

A natural way to generate "random smooth functions" is to take random
Chebyshev coefficients with a controlled decay rate:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(42)
n = 50

# Coefficients decay as exp(-k/10) — gives smooth but wiggly functions
coeffs = rng.standard_normal(n) * np.exp(-np.arange(n) / 10.0)
f = cj.chebfun.from_coeffs(jnp.array(coeffs))
roots = np.sort(np.array(f.roots()))
print(f"Degree {len(f)-1} polynomial has {len(roots)} real roots")
```

## Arcsine distribution of real roots

Over many trials, the real roots follow the **arcsine distribution**:

$$\rho(x) = \frac{1}{\pi\sqrt{1-x^2}}, \quad x \in (-1,1).$$

This is the same distribution as the Chebyshev points themselves — the
equilibrium measure of $[-1,1]$.

```python
all_roots = []
for trial in range(30):
    coeffs = rng.standard_normal(n) * np.exp(-np.arange(n) / 10.0)
    f = cj.chebfun.from_coeffs(jnp.array(coeffs))
    roots = np.array(f.roots())
    all_roots.extend(roots[(roots > -1) & (roots < 1)].tolist())

# Fraction in (-0.5, 0.5) should be 2*arcsin(0.5)/pi = 1/3
fraction = np.mean((np.array(all_roots) > -0.5) & (np.array(all_roots) < 0.5))
print(f"Fraction in (-0.5, 0.5): {fraction:.4f}  (expected: 0.3333)")
```

![Random polynomials and arcsine root distribution](../../../images/stats/random_polynomials.png)

The histogram of root locations (right panel) closely follows the arcsine
density (red curve), confirming the theoretical prediction.

## Why the arcsine distribution?

The arcsine distribution is the equilibrium measure of the interval $[-1,1]$
in logarithmic potential theory. It arises naturally as the limiting distribution
of roots of polynomials whose coefficients are i.i.d. random variables.

## References

1. I. E. Pritsker and R. S. Varga, The Szegő curve, zero distribution and
   weighted approximation, *Trans. AMS* 349 (1997), 4085–4105.
2. A. Edelman and E. Kostlan, How many zeros of a random polynomial are real?
   *Bull. AMS* 32 (1995), 1–37.
