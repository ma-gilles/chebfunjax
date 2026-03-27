# Expectations of Distributions

**Original:** [stats/Expectations](https://www.chebfun.org/examples/stats/Expectations.html)
**Author(s):** Mark Richardson, May 2011

---

This example uses Chebfun to solve probability distribution problems from the
textbook by Spiegel, Schiller, and Srinivasan [1].

## 1. Expectation of a random variable

Suppose a continuous random variable $X$ has the probability density function

$$f(x) = 2e^{-2x}, \quad x \ge 0, \qquad f(x) = 0, \quad x < 0.$$

We seek $E(X)$ and $E(X^2)$.

Since $e^{-2x}$ decays rapidly, it suffices to work on $[0, 40]$.
The density integrates to 1 (as required for any PDF), and

$$E(X) = \int_0^\infty x\,f(x)\,dx = \frac{1}{2}, \qquad
E(X^2) = \int_0^\infty x^2\,f(x)\,dx = \frac{1}{2}.$$

Both are computed to full precision by integrating the corresponding chebfun
products.

## 2. Mean, median, and mode

The PDF of a continuous random variable $X$ is

$$g(x) = \frac{4x(9 - x^2)}{81}, \quad 0 \le x \le 3,$$

and zero otherwise. We find:

**(a) Mean.** The mean is $E(X) = \int_0^3 x\,g(x)\,dx = 1.6$.

**(b) Median.** The median $a$ satisfies $P(X \le a) = 1/2$. Using the
cumulative distribution function $G(x) = \int_0^x g(t)\,dt$ and solving
$G(a) = 1/2$ gives

$$a = \sqrt{9 - \tfrac{9\sqrt{2}}{2}} \approx 1.624.$$

**(c) Mode.** The mode is the location of the global maximum of $g$, found
at $x = \sqrt{3} \approx 1.732$.

All three measures -- mean, median, and mode -- are distinct, reflecting the
skewness of the distribution.

## References

1. M. Spiegel, J. Schiller, and R. Srinivasan, *Schaum's Outlines: Probability
   and Statistics*, 3rd ed., 2009.

```python
from examples.stats.expectations import run
run()
```

## Output

![Expectations of Distributions](../../images/stats/expectations.png)
