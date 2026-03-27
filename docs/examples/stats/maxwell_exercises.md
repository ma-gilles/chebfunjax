# Maxwell Distribution Exercises

**Original:** [stats/MaxwellExercises](https://github.com/chebfun/examples/blob/master/stats/MaxwellExercises.m)
**Author(s):** Jie Gao, September 2013

---

This example uses Chebfun to solve a problem involving the Maxwell distribution
from the textbook [1]. Maxwell distributions frequently arise in physics,
particularly for describing the distribution of molecular speeds in an ideal gas
at a given temperature.

## The Maxwell distribution

The Maxwell (or Maxwell--Boltzmann) distribution has the PDF

$$f(x; b) = \frac{\sqrt{2}}{b^3\sqrt{\pi}}\,x^2 \exp\!\left(-\frac{x^2}{2b^2}\right), \quad x \ge 0.$$

The parameter $b$ is related to temperature in kinetic theory. The exact mean
and variance are

$$E[X] = 2b\sqrt{\frac{2}{\pi}}, \qquad \mathrm{Var}[X] = b^2\left(3 - \frac{8}{\pi}\right).$$

For $b = 2.3$, the mean, variance, and standard deviation can be computed
numerically by integrating $x\,f(x)$ and $x^2\,f(x)$ and compared to the exact
formulas, matching to full precision.

## Application: molecular speed distribution

In kinetic theory, the Maxwell distribution describes the speed distribution of
gas molecules. For a gas with parameter $b = 3.7$:

- The probability that a particle's speed lies between 2.9 and 3.1 m/s is
  $P[2.9 < X < 3.1] = F(3.1) - F(2.9)$, computed via the CDF.
- The **average speed** is $\langle v \rangle = E[X]$.
- The **average speed squared** $\langle v^2 \rangle = E[X^2]$ gives the
  average kinetic energy via $K = \tfrac{1}{2}m\langle v^2 \rangle$.

## References

1. A. M. Mood, F. A. Graybill, and D. Boes, *Introduction to the Theory of
   Statistics*, McGraw-Hill, 1974.
2. J. Baliga, Maxwell-Boltzmann Distribution.

```python
from examples.stats.maxwell_exercises import run
run()
```

![Maxwell Distribution Exercises](../../images/stats/maxwell_exercises.png)