# Least Squares Fitting

**Original:** [stats/LeastSquares](https://www.chebfun.org/examples/stats/LeastSquares.html)
**Author(s):** Nick Trefethen, October 2011

---

In MATLAB, the standard command for least-squares fitting by a polynomial to
discrete data is `polyfit`, which returns a coefficient vector in the monomial
basis. Chebfun provides an overloaded `polyfit` that returns the polynomial as a
chebfun instead.

## Discrete polynomial fitting

Given $n$ noisy data points $(x_i, y_i)$ sampled from a function, we can fit a
polynomial of degree $d$ in the least-squares sense: minimize

$$\sum_{i=1}^n \bigl(p(x_i) - y_i\bigr)^2$$

over all polynomials $p$ of degree at most $d$.

For example, fitting a degree-10 polynomial to 100 points sampled from the
Runge function $1/(1 + 25x^2)$ with added Gaussian noise produces a smooth
approximation that captures the overall shape while filtering out the noise.

## Continuous polynomial fitting

Chebfun also supports **continuous** least-squares fitting. Given a chebfun $f$,
`polyfit(f, d)` returns the degree-$d$ polynomial $p^*$ that minimizes

$$\int_{-1}^{1} \bigl(p(x) - f(x)\bigr)^2\,dx.$$

This is the best $L^2$ polynomial approximation of $f$. For a jagged function
like $|x + 0.2| - 0.5\,\mathrm{sign}(x - 0.5)$, the degree-10 continuous
least-squares fit provides a smooth polynomial approximation in a single
command.

```python
from examples.stats.least_squares import run
run()
```

![Least Squares Fitting](../../images/stats/least_squares.png)