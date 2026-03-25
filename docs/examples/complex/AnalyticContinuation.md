# Analytic continuation via rational approximation

**Nick Trefethen, March 2013**

---

A function analytic on $[-1, 1]$ can in principle be continued to a larger
region in the complex plane. The size of the *Bernstein ellipse* — the largest
ellipse with foci at $\pm 1$ in which the function is analytic — determines
how fast the Chebyshev coefficients decay.

## Coefficient decay and the Bernstein ellipse

For $f(x) = \tanh(x)$, the nearest singularities are at $z = \pm i\pi/2$,
giving Bernstein ellipse parameter $\rho = \pi/2 + \sqrt{(\pi/2)^2 - 1}
\approx 2.48$. The coefficients decay like $\rho^{-n}$:

```python
import jax.numpy as jnp
import numpy as np
import chebfunjax as cj

f = cj.chebfun(jnp.tanh)
c = f.coeffs           # Chebyshev coefficients (property, not method)

import matplotlib.pyplot as plt
plt.semilogy(np.abs(c))
rho = np.pi/2 + np.sqrt((np.pi/2)**2 - 1)
print(f"Bernstein rho ≈ {rho:.4f}")
```

## Runge's function

The function $1/(1 + 25x^2)$ on $[-1, 1]$ has singularities at $z = \pm i/5$,
giving a much smaller Bernstein ellipse ($\rho \approx 1.02$) and therefore
slower coefficient decay:

```python
f_runge = cj.chebfun(lambda x: 1.0 / (1.0 + 25*x**2))
c_runge = f_runge.coeffs

# Compare decay rates
n_vals = np.arange(len(c))
plt.semilogy(n_vals, np.abs(c_runge), label=r'$1/(1+25x^2)$')
plt.semilogy(n_vals, np.abs(c),       label=r'$\tanh(x)$')
```

## Gallery

![Analytic continuation](../../../examples/complex/analytic_continuation.png)

Chebyshev coefficient magnitudes for $\tanh(x)$ (fast decay) and $1/(1+25x^2)$
(slow decay due to nearby poles at $\pm i/5$).
