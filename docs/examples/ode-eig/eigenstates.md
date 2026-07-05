# Eigenstates of the Schrodinger equation

*Nick Trefethen, January 2012*

[Chebfun example](https://www.chebfun.org/examples/ode-eig/Eigenstates.html)

## Overview

Computes quantum mechanical eigenstates for the time-independent Schrodinger
equation:

$$-u'' + V(x) u = E u, \quad u(a) = u(b) = 0$$

For the harmonic oscillator $V(x) = x^2/2$, exact eigenvalues are
$E_n = n + 1/2$. The double-well potential $V(x) = x^4 - 2x^2$ is also studied.

```python
from chebfunjax.operators.chebop import Chebop

dom = (-6.0, 6.0)
L_harm = Chebop(lambda x, u: -u.diff(2) + x**2/2.0*u, domain=dom)
L_harm.lbc = 0.0; L_harm.rbc = 0.0
lams = L_harm.eigs(k=6)
# Exact: E_n = n + 0.5
```


![Eigenstates of the Schrodinger equation](../../images/ode-eig/eigenstates.png)

## Figures (chebfun.org parity)

![Eigenstates figure 1](../../images/ode-eig/Eigenstates_01.png)

![Eigenstates figure 2](../../images/ode-eig/Eigenstates_02.png)

![Eigenstates figure 3](../../images/ode-eig/Eigenstates_03.png)

![Eigenstates figure 4](../../images/ode-eig/Eigenstates_04.png)

![Eigenstates figure 5](../../images/ode-eig/Eigenstates_05.png)

![Eigenstates figure 6](../../images/ode-eig/Eigenstates_06.png)

![Eigenstates figure 7](../../images/ode-eig/Eigenstates_07.png)

![Eigenstates figure 8](../../images/ode-eig/Eigenstates_08.png)

![Eigenstates figure 9](../../images/ode-eig/Eigenstates_09.png)

![Eigenstates figure 10](../../images/ode-eig/Eigenstates_10.png)
