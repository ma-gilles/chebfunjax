# Smooth Random Walk

**Original:** [stats/SmoothRandomWalk](https://www.chebfun.org/examples/stats/SmoothRandomWalk.html)
**Author(s):** Nick Trefethen, February 2017

---

By integrating coin flips in one or more dimensions, we get a **random walk**,
which becomes Brownian motion in the limit of infinitely many infinitely small
steps. Chebfun's `randnfun` command enables exploration of a smooth continuous
analogue of this process.

## Construction

Working in 2D with a complex variable for convenience, we plot the indefinite
integral of a complex random function scaled by $(dx)^{-1/2}$:

$$g(t) = \int_{-1}^{t} f(s)\,ds,$$

where $f$ is a smooth random function with characteristic length scale $dx$.
The path $g(t)$ traces a smooth curve in the complex plane, with red dots
marking the initial and end points.

## Convergence to Brownian motion

Dividing the characteristic length $dx$ by 4 repeatedly (three times, from
$dx = 0.1$ down to $dx \approx 0.0016$), the smooth random walk approaches
**Brownian motion**. The paths become increasingly wiggly at fine scales while
maintaining the same large-scale wandering behavior.

This convergence is analyzed rigorously in [1], which establishes the connection
between smooth random functions (defined via finite Fourier series with random
coefficients) and Gaussian processes.

## References

1. S. Filip, A. Javeed, and L. N. Trefethen, Smooth random functions, random
   ODEs, and Gaussian processes, *SIAM Rev.* 61 (2019), 185--205.

```python
from examples.stats.smooth_random_walk import run
run()
```

## Output

![Smooth Random Walk](../../images/stats/smooth_random_walk.png)
