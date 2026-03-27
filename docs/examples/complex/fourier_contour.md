# Fourier transforms using contour integrals

**Mohsin Javed, July 2013**

[Original MATLAB source](https://github.com/chebfun/examples/blob/master/complex/FourierContour.m)

---

The Fourier transform of a function with poles can be evaluated exactly using
the residue theorem. For $\omega > 0$, close the contour in the lower
half-plane; for $\omega < 0$, in the upper half-plane.

## Rational function: $2a/(a^2 + z^2)$

The function $f(z) = 2a/(a^2 + z^2)$ has poles at $z = \pm ia$. For
$\omega > 0$, the contour encloses the pole at $z = -ia$ (negatively oriented),
giving

$$
\hat{f}(\omega) = \frac{1}{2\pi}\int_{-\infty}^{\infty}
  \frac{2a}{a^2+z^2}\,e^{-i\omega z}\,dz
= e^{-a\omega} / a \cdot a = e^{-a|\omega|}.
$$

Wait — the exact answer is $\hat{f}(\omega) = e^{-a|\omega|}$:

```python
import numpy as np
import jax.numpy as jnp
import chebfunjax as cj

a = 2.0
# Verify numerically via Chebfun on a large symmetric interval
for omega in [0.5, 1.0, 2.0]:
    L = 40.0
    integrand = lambda x: (2*a/(a**2 + x**2)) * np.exp(-1j*omega*x) / (2*np.pi)
    f = cj.chebfun(lambda x: (2*a/(a**2 + x**2)) * jnp.exp(-omega*x) / (2*jnp.pi),
                   domain=(-L, L))
    # Use real part only (imaginary part should vanish)
    I_num  = float(f.sum())
    I_exact = np.exp(-a * abs(omega))
    print(f"ω={omega}: numerical = {I_num:.8f}, exact = {I_exact:.8f}")
```

## Non-rational: $x e^{-|x|}$

The two-sided FT of $x e^{-|x|}$ equals $\frac{1}{(1-i\omega)^2} -
\frac{1}{(1+i\omega)^2}$, verifiable at $\omega = 1$ where the exact value
is $i$:

```python
omega = 1.0
exact_val = 1/(1 - 1j*omega)**2 - 1/(1 + 1j*omega)**2
print(f"Exact FT at ω=1: {exact_val}")   # = 1j
```

## Gallery

![Fourier contour](../../images/complex/fourier_contour.png)

*Left*: Fourier transform $e^{-a|\omega|}$ vs numerical estimate.
*Right*: Contour in the complex plane with pole marked.
