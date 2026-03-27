# Generalized Polynomial Chaos

**Original:** [stats/GeneralizedPolynomialChaos](https://github.com/chebfun/examples/blob/master/stats/GeneralizedPolynomialChaos.m)
**Author(s):** Toby Driscoll, December 2011

---

Generalized polynomial chaos (gPC) is a powerful way to represent stochastic
quantities with spectral accuracy. This example demonstrates the technique in
one dimension.

## Strong approximation

When the density of a random variable $Y$ is known explicitly and can be
reparameterized in terms of a standard random variable $Z$, a polynomial
approximation in $Z$ reproduces $Y$ accurately. The gPC method expresses the
approximation using orthogonal polynomials based on the density of $Z$, so
that the approximation reduces to a simple least-squares (Fourier) projection.

### Lognormal example with Hermite basis

Suppose $Y$ is a lognormal variable (i.e., $\log Y$ is normal with mean $\mu$
and variance $\sigma^2$). If $Z$ is a standard normal variable, then
$Y = \exp(\mu + \sigma Z)$. A standard gPC approximation uses **Hermite
polynomials** $H_n(z)$, which satisfy the three-term recurrence

$$H_{n+1}(z) = z\,H_n(z) - n\,H_{n-1}(z),$$

and are orthogonal with respect to the Gaussian weight
$\rho(z) = e^{-z^2/2}$. On a domain truncated to $\pm 10$ standard
deviations, orthogonality can be verified numerically.

The expansion coefficients are obtained by solving the normal equations. Because
the Gram matrix $G_{ij} = \langle H_i, H_j \rangle_\rho$ is diagonal by
orthogonality, inversion is trivial. For $\mu = 1$ and $\sigma = 0.5$, the
coefficients decrease rapidly, reflecting spectral accuracy.

Alternatively, one can skip orthogonal polynomials entirely and use the
Chebyshev basis with a weighted least-squares solve. Although Chebyshev
polynomials are not orthogonal under the Gaussian weight, they are
well-conditioned enough for the job.

## Weak approximation

In practice, the explicit parameterization $Y(Z)$ is often unknown, but the
distribution $F_Y(y) = P[Y \le y]$ is known. The trick is to reparameterize
through a uniform variable $U$: both $F_Y(Y)$ and $F_Z(Z)$ are uniformly
distributed on $[0,1]$, so $Y = F_Y^{-1}(F_Z(Z))$.

For an exponential distribution $F_Y(y) = 1 - e^{-y}$, this composition gives
an expression for $Y$ in terms of $Z$ that can be approximated as before.
The choice of weight function affects both the parameterization and the quality
of the approximation -- a Gaussian weight deemphasizes the tails, while an
inverse-square weight retains more information at the extremes.

## References

1. D. Xiu, *Numerical Methods for Stochastic Computations*, Princeton
   University Press, 2010.

```python
from examples.stats.generalized_polynomial_chaos import run
run()
```

![Generalized Polynomial Chaos](../../images/stats/generalized_polynomial_chaos.png)