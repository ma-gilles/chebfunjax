# Two electrons orbiting symmetrically about a nucleus

*Jeremy Fleury and Nick Trefethen, June 2016*

[Chebfun example](https://www.chebfun.org/examples/ode-nonlin/TwoElectrons.html)

## Overview

Models two electrons in symmetric orbits around a proton, including
the electron-electron repulsion and Coulomb attraction:

$$H = \frac{p_r^2}{2} + \frac{p_\theta^2}{2r^2} - \frac{Z}{r} + \frac{1}{2r}$$

For circular orbits, the equilibrium radius $r_0$ and angular momentum
$p_\theta = \sqrt{(Z - 1/4)r_0}$ are computed.

```python
import numpy as np
from scipy.integrate import solve_ivp

Z = 1.0; r0 = 2.0
ptheta0 = np.sqrt((Z - 0.25) * r0)

def two_electron_rhs(t, state):
    r, theta, pr, ptheta = state
    return [pr, ptheta/r**2,
            ptheta**2/r**3 - Z/r**2 + 1.0/(2*r)**2,
            0.0]
```


![Two electrons orbiting symmetrically about a nucleus](../../images/ode-nonlin/two_electrons.png)

## Figures (chebfun.org parity)

![TwoElectrons figure 1](../../images/ode-nonlin/TwoElectrons_01.png)

![TwoElectrons figure 2](../../images/ode-nonlin/TwoElectrons_02.png)

![TwoElectrons figure 3](../../images/ode-nonlin/TwoElectrons_03.png)

![TwoElectrons figure 4](../../images/ode-nonlin/TwoElectrons_04.png)

![TwoElectrons figure 5](../../images/ode-nonlin/TwoElectrons_05.png)

![TwoElectrons figure 6](../../images/ode-nonlin/TwoElectrons_06.png)

![TwoElectrons figure 7](../../images/ode-nonlin/TwoElectrons_07.png)

![TwoElectrons figure 8](../../images/ode-nonlin/TwoElectrons_08.png)

![TwoElectrons figure 9](../../images/ode-nonlin/TwoElectrons_09.png)

![TwoElectrons figure 10](../../images/ode-nonlin/TwoElectrons_10.png)
