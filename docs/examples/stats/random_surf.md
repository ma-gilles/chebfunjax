# Random Surfaces

**Original:** [stats/RandomSurf](https://www.chebfun.org/examples/stats/RandomSurf.html)
**Author(s):** Nick Trefethen, May 2019

---

This example demonstrates smooth random functions on the unit disk, generated
via finite Fourier series with random coefficients. These are examples of
**Gaussian random fields**, with applications spanning many scientific
disciplines.

## Construction

A smooth random function on the unit disk is generated using `randnfundisk`
with a characteristic length scale parameter (here 0.1). Adding a deterministic
paraboloid $p(r) = 2 - 4r^2$ creates an interesting combined surface.

## Visualization

The combined function can be displayed in several ways:

- **Zebra mode:** alternating light and dark bands across the surface produce
  a striking visual pattern that highlights level sets.
- **Contour plot:** traditional contour lines with color mapping reveal the
  topography.
- **Surface plot:** a 3D rendering with lighting shows the undulating
  structure created by combining smooth randomness with the parabolic base.

## Applications

Random surfaces of this kind have been studied since Longuet-Higgins in 1957
and appear throughout science:

- **Oceanography** -- modeling sea surface waves
- **Biology** -- evolution of biological systems in random media
- **Cosmology** -- statistics of peaks in the cosmic microwave background
- **Condensed matter physics** -- critical points of Gaussian fields
- **Arctic climate** -- fractal geometry of melt ponds

The mathematical theory is developed in detail by Adler and Taylor [1] and
the connection to smooth random functions is described by Filip, Javeed, and
Trefethen [7].

## References

1. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer, 2009.
2. J. M. Bardeen et al., The statistics of peaks of Gaussian random fields,
   *Astrophys. J.* 304 (1986), 15--61.
3. S. Filip, A. Javeed, and L. N. Trefethen, Smooth random functions, random
   ODEs, and Gaussian processes, *SIAM Rev.* 61 (2019), 185--205.
4. M. S. Longuet-Higgins, The statistical analysis of a random, moving
   surface, *Phil. Trans. Roy. Soc. Lond. A* 429 (1957), 321--387.

```python
from examples.stats.random_surf import run
run()
```

![Random Surfaces](../../images/stats/random_surf.png)