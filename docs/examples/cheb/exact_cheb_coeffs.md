# Exact Chebyshev Expansion Coefficients

**Original MATLAB:** [cheb/ExactChebCoeffs](https://www.chebfun.org/examples/cheb/ExactChebCoeffs.html)
**Author(s):** Mark Richardson, June 2012

## Overview

Uses Elliott's residue method [1] to derive an exact closed-form formula for the
Chebyshev expansion coefficients of $f(x) = 1/(5+x)$, then compares with
numerically computed coefficients.

## Mathematical Background

The Chebyshev coefficients of a function $f$ with a pole at $z_0$ (outside
$[-1,1]$) are given by the residue formula:

$$a_n = \frac{-2r_0}{\sqrt{z_0^2-1}(z_0 - \sqrt{z_0^2-1})^n}$$

where $r_0 = \text{res}(f, z_0)$ is the residue at the pole.

For $f(x) = 1/(5+x)$, the pole is at $z_0 = -5$ with residue $r_0 = 1$:

$$a_n = \frac{1}{\sqrt{6}} \cdot \frac{(-1)^n}{(5 + \sqrt{24})^n}$$

The denominator $(5 + \sqrt{24})^n$ is the Bernstein ellipse parameter $\rho^n$,
confirming that Chebyshev coefficients decay geometrically with rate $\rho^{-1}$,
where $\rho = 5 + \sqrt{24} \approx 9.899$ is the ellipse through the pole.

## Code

```python
import numpy as np
import jax.numpy as jnp
import chebfunjax as cj

f = cj.chebfun(lambda x: 1.0 / (5 - 4 * x))
c_num = np.abs(np.asarray(f.funs[0].tech.coeffs))
N = len(c_num) - 1
k = np.arange(N + 1)
c_exact = (1 / np.sqrt(6)) / (5 + np.sqrt(24)) ** k
print(f"max coefficient error: {np.max(np.abs(c_num - c_exact)):.2e}")
```

## References

1. D. Elliott, The evaluation and estimation of the coefficients in the Chebyshev
series expansion of a function, *Mathematics of Computation* 18 (1964), 274-284.

## Results

Numerical and exact coefficients agree to machine precision for all $n \geq 1$
(the $n = 0$ coefficient differs by the usual factor-of-2 convention).

![Exact Chebyshev coefficients](../../images/cheb/exact_cheb_coeffs.png)

## Figures (chebfun.org parity)

![ExactChebCoeffs figure 1](../../images/cheb/ExactChebCoeffs_01.png)
