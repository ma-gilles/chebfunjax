# Fourier Coefficients

*Original: [chebfun.org/examples/fourier/FourierCoefficients](https://www.chebfun.org/examples/fourier/FourierCoefficients.html)*

---

The Chebyshev and Fourier coefficients of a function encode its smoothness.
For an analytic function, both decay geometrically fast; for a $C^k$ function,
they decay like $n^{-k}$.

## Coefficient decay

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

# Analytic function: exponential decay
f = cj.chebfun(lambda x: jnp.exp(jnp.cos(x)), domain=(0.0, 2*float(jnp.pi)))
c = np.array(f.coeffs())
print(f"Degree: {len(c)-1}")
print(f"Max coeff: {np.max(np.abs(c)):.4e}")
print(f"Last coeff: {np.abs(c[-1]):.4e}")
```

```
Degree: 38
Max coeff: 2.2796e+00
Last coeff: 1.3878e-17
```

The Chebyshev coefficients of $e^{\cos x}$ decay geometrically, with machine
precision reached at degree ~38. The ratio between successive coefficients
is approximately $e^{-1/\rho}$ where $\rho$ is the Bernstein ellipse parameter.

## Aliasing

When a function is sampled at $n$ points, all modes $k = m \cdot n \pm j$
contribute to the $j$-th alias. For smooth functions this is negligible;
for non-smooth functions it can cause visible error.

![Fourier coefficient decay and aliasing](../../../images/fourier/fourier_coefficients.png)

## Parseval's theorem

For $f \in L^2$, the sum of squared Fourier (or Chebyshev) coefficients
equals the squared $L^2$ norm:

```python
g = cj.chebfun(lambda x: jnp.sin(3*x) + 0.5*jnp.cos(5*x))
c = np.array(g.coeffs())
parseval_sum = np.sum(c**2) / 2  # factor for Chebyshev normalization
l2_norm_sq = float(g.norm(2))**2
print(f"Parseval: sum c_k^2 ≈ ||g||^2: {abs(parseval_sum - l2_norm_sq) < 1e-10}")
```
