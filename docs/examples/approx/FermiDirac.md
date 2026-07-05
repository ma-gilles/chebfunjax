# Rational Approximation of the Fermi-Dirac Function

*Nick Trefethen, July 2019*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/FermiDirac.html)

## The Fermi-Dirac function

The Fermi-Dirac distribution $f(x) = 1/(1 + e^x)$ arises in quantum mechanics
and electronic structure theory.  It has a smooth but rapid transition near
$x=0$.  Rational approximation is much more efficient than polynomials here:
the poles of $f$ lie at $x = i\pi(2k+1)$ for $k \in \mathbb{Z}$.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp

xs = jnp.linspace(-10.0, 10.0, 500)
ys = 1.0 / (1.0 + jnp.exp(xs))
r, pol, res, zer, *_ = aaa(ys, xs)
import numpy as np
err = np.max(np.abs(np.asarray(ys) - np.real(r(np.asarray(xs)))))
print(f"AAA type: ({len(pol)-1}, {len(pol)-1}), max err: {err:.2e}")
```

![Rational Approximation of the Fermi-Dirac Function](../../images/approx/FermiDirac.png)

## Figures (chebfun.org parity)

![FermiDirac figure 1](../../images/approx/FermiDirac_01.png)

![FermiDirac figure 2](../../images/approx/FermiDirac_02.png)

![FermiDirac figure 3](../../images/approx/FermiDirac_03.png)

![FermiDirac figure 4](../../images/approx/FermiDirac_04.png)

![FermiDirac figure 5](../../images/approx/FermiDirac_05.png)

![FermiDirac figure 6](../../images/approx/FermiDirac_06.png)
