# Probability Distributions

**Original:** [stats/ProbabilityConvolution](https://www.chebfun.org/examples/stats/ProbabilityConvolution.html)
**Author(s):** Nick Trefethen, 2012

---

This example demonstrates computing with several standard probability
distributions -- normal, beta, gamma, and Cauchy -- using Chebfun to evaluate
PDFs, CDFs, moments, and convolutions.

## Normal distribution

The normal (Gaussian) distribution with mean $\mu$ and standard deviation
$\sigma$ has PDF

$$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

Key properties: $E[X] = \mu$, $\mathrm{Var}[X] = \sigma^2$, and the integral
over $(-\infty, \infty)$ equals 1. These are verified numerically to full
precision by constructing a chebfun on a sufficiently large interval.

## Convolution of normals

The sum of two independent normals $N(\mu_1, \sigma_1^2)$ and
$N(\mu_2, \sigma_2^2)$ is again normal:

$$N(\mu_1, \sigma_1^2) * N(\mu_2, \sigma_2^2) = N(\mu_1 + \mu_2,\, \sigma_1^2 + \sigma_2^2).$$

This is verified by FFT-based convolution and comparison with the exact PDF.

## Beta distribution

The beta distribution on $[0,1]$ with shape parameters $a, b > 0$ has PDF

$$f(x; a, b) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)},$$

where $B(a,b)$ is the beta function. The mean is $a/(a+b)$. Various shape
parameters produce symmetric, skewed, U-shaped, or uniform densities:

- $B(1,1)$: uniform on $[0,1]$
- $B(2,2)$: symmetric, bell-shaped
- $B(2,5)$: right-skewed
- $B(0.5,0.5)$: U-shaped (arcsine distribution)

## Central Limit Theorem preview

The uniform distribution $U(0,1)$ has mean $1/2$ and variance $1/12$. By the
Central Limit Theorem, the sum of $n$ independent uniform random variables
(the Irwin--Hall distribution) converges to $N(n/2, n/12)$ as $n$ grows.
Even the sum of two or three uniforms is already close to Gaussian in shape.

```python
from examples.stats.probability_distributions import run
run()
```

## Output

![Probability Distributions](../../images/stats/probability_distributions.png)
