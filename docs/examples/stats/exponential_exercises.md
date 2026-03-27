# Exponential Distribution Exercises

**Original:** [stats/ExponentialExercises](https://github.com/chebfun/examples/blob/master/stats/ExponentialExercises.m)
**Author(s):** Jie Gao and Nick Trefethen, May 2013

---

This example uses Chebfun to solve problems involving the exponential
distribution from the textbook [1]. The exponential distribution models the
length of the time interval between successive events in a Poisson process and
is widely used to describe lifetimes of various kinds.

## Conditional probability

The PDF of the exponential distribution with parameter $\lambda$ is

$$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0.$$

For mean 2 (so $\lambda = 1/2$), we compute $P[X < 1 \mid X < 2]$ using Bayes'
theorem:

$$P[X < 1 \mid X < 2] = \frac{P[X < 1]}{P[X < 2]} = \frac{F(1)}{F(2)},$$

where $F$ is the CDF.

## Memorylessness

A key property of the exponential distribution is **memorylessness**:

$$P[X > s + t \mid X > s] = P[X > t], \quad s, t > 0.$$

For example, $P[X > 8 \mid X > 5] = P[X > 3]$. This can be verified
numerically to machine precision.

## Numerical variant

Replacing the linear exponent by a power $\log 2$, i.e., using the density
$f(x) \propto \exp(-\lambda\, x^{\log 2})$, we normalize by hand and compute
the conditional probability $P[X < 1 \mid X < 3]$ numerically.

## Application: light bulb reliability

Suppose the lifespan $T$ (in hundreds of hours) of a light bulb is exponentially
distributed with failure rate $\lambda = 0.2$:

$$f(t) = 0.2\,e^{-0.2\,t}, \quad t > 0.$$

The **reliability** at time $t$ is $\mathrm{Rel}(t) = P[T > t] = 1 - F(t)$.
The probability of lasting more than 700 hours is $P[T > 7]$, and the time at
which reliability drops to 10% can be found by solving $1 - F(d) = 0.1$.

## Finding $\lambda$ from a median condition

If $P[X \le 1] = P[X > 1]$, then the median equals 1, so
$\lambda = \ln 2 \approx 0.6931$. The variance is then $1/\lambda^2 = 1/(\ln 2)^2$.

A further numerical variant replaces the exponent with
$|\lambda\, x^{13/5}\,\ln(x + 1/2)|$, introducing a singularity at $x = 1/2$.
The mode, mean, and standard deviation of this exotic distribution are computed
numerically.

## References

1. A. M. Mood, F. A. Graybill, and D. Boes, *Introduction to the Theory of
   Statistics*, McGraw-Hill, 1974.
2. J. M. Horgan, Chapter 17: Applications of the Exponential Distribution.

```python
from examples.stats.exponential_exercises import run
run()
```

## Output

![Exponential Distribution Exercises](../../images/stats/exponential_exercises.png)
