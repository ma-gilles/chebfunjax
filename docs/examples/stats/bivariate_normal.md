# Bivariate Normal Distribution

**Original:** [stats/BivariateNormalDistribution](https://www.chebfun.org/examples/stats/BivariateNormalDistribution.html)
**Author(s):** Alex Townsend, March 2013

---

The bivariate normal distribution is a standard example for probability density
functions of continuous random variables, coupling two normal random variables
$X$ and $Y$ through a correlation parameter $\rho$.

## The joint density

The joint PDF of two standard normals with means $\mu_1, \mu_2$, standard
deviations $\sigma_1, \sigma_2$, and correlation $\rho$ is

$$p(x, y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}
\exp\!\left(-\frac{z(x,y)}{2(1-\rho^2)}\right),$$

where

$$z(x,y) = \frac{(x-\mu_1)^2}{\sigma_1^2}
 - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}
 + \frac{(y-\mu_2)^2}{\sigma_2^2}.$$

As a proper density, it integrates to 1 over the plane. On a truncated domain
$[-10,10]^2$ the integral is indistinguishable from 1 to machine precision.

## Marginal distributions

Integrating out $y$ yields the marginal distribution of $X$:

$$p_X(x) = \int_{-\infty}^{\infty} p(x, y)\,dy.$$

A fundamental property of the bivariate normal is that each marginal is itself a
univariate normal distribution -- $p_X$ is $N(\mu_1, \sigma_1^2)$ -- which can
be verified numerically to high accuracy.

## Conditional distributions

Given a known realization of $X = x_0$, the conditional distribution of $Y$ is

$$Y \mid X = x_0 \;\sim\; N\!\left(\mu_2 + \frac{\sigma_2}{\sigma_1}\rho(x_0 - \mu_1),\;
(1 - \rho^2)\sigma_2^2\right).$$

This conditional density can be computed numerically as $p(x_0, y) / p_X(x_0)$
and compared to the exact formula. For $\rho = 0.5$ and $x_0 = \pi/6$, the
error is at the level of machine precision.

## References

1. [Multivariate normal distribution](http://en.wikipedia.org/wiki/Multivariate_normal_distribution) (Wikipedia)

```python
from examples.stats.bivariate_normal import run
run()
```

![Bivariate Normal Distribution](../../images/stats/bivariate_normal.png)