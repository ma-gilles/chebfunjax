# Nonlinear periodic ODE

*Hadrien Montanelli, December 2014*

[Chebfun example](https://www.chebfun.org/examples/ode-nonlin/FourierCollocationNonLin.html)

## Overview

Solves the nonlinear periodic ODE

$$u' - u\cos(u) = \cos(4x), \quad u \text{ periodic on } [0, 2\pi]$$

using Fourier spectral collocation with `N.bc = "periodic"`.

```python
import numpy as np

# periodic nonlinear ODE u' + u^3 = cos t solved by Newton on a
# Fourier grid (Chebop bc='periodic' is a library backlog item)
n = 128
xg = np.linspace(0, 2*np.pi, n, endpoint=False)
dx = xg[1] - xg[0]
col = np.zeros(n)
kk = np.arange(1, n)
col[1:] = 0.5 * (-1.0)**kk / np.tan(kk * dx / 2)
Dp = -np.column_stack([np.roll(np.concatenate([[0.0], col[1:]]), k)
                       for k in range(n)])
u = 0.5 * np.cos(xg)
for _ in range(30):
    du = np.linalg.solve(Dp + np.diag(3*u**2),
                         -(Dp @ u + u**3 - np.cos(xg)))
    u += du
    if np.max(np.abs(du)) < 1e-13:
        break
print(f"Newton residual: {np.max(np.abs(Dp @ u + u**3 - np.cos(xg))):.2e}")
```


![Nonlinear periodic ODE](../../images/ode-nonlin/fourier_nonlin.png)

## Figures (chebfun.org parity)

![FourierCollocationNonLin figure 1](../../images/ode-nonlin/FourierCollocationNonLin_01.png)

![FourierCollocationNonLin figure 2](../../images/ode-nonlin/FourierCollocationNonLin_02.png)
