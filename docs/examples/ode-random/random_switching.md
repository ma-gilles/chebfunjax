# Linear ODEs with Random Switching

**Original MATLAB:** [ode-random/RandomSwitching](https://www.chebfun.org/examples/ode-random/RandomSwitching.html)
**Author:** Nick Trefethen (May 2017)

## Overview

When an ODE switches randomly between two coefficient matrices, the behavior
depends on the switching rate. Remarkably, even if each matrix is individually
stable, intermediate switching rates can lead to net amplification.

This example follows the Lawley-Mattingly-Reed phenomenon [1].

## Mathematical Background

**Scalar case:** $y' = \text{sign}(f) \cdot y$ switches between growth and decay.

**Matrix case:** switch between $y' = Ay$ and $y' = By$ with
$$A = \begin{pmatrix} -1 & 5 \\ 0 & -1 \end{pmatrix}, \quad B = \begin{pmatrix} -1 & 0 \\ -5 & -1 \end{pmatrix}$$

Both have eigenvalues $-1$ (stable), but intermediate switching can lead to
exponential growth. The three regimes:

- **Slow switching** ($\lambda = 3$): dominated by individual stability → decay
- **Intermediate** ($\lambda = 1$): resonance between the matrices → possible growth
- **Fast switching** ($\lambda = 1/3$): governed by average matrix $(A+B)/2$ → stable decay

## Code

```python
import chebfunjax as cj
import numpy as np

f_fn = cj.randnfun(lam, domain=[0,40], seed=1)
f_switch = 5.0 * (1 + np.sign(f_vals)) / 2.0  # ∈ {0, 5}

def rhs(t, uv):
    fi = np.interp(t, t_eval, f_switch)
    return [-uv[0] + fi*uv[1], -uv[1] - (5-fi)*uv[0]]
```

## References

[1] S. D. Lawley, J. C. Mattingly, and M. C. Reed, Sensitivity to switching
rates in stochastically switched PDEs, *Commun. Math. Sci.* 12 (2014), 1343-1352.

## Results

![Random switching ODEs](../../images/ode-random/random_switching.png)
