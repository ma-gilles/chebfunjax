# Computing with an Atmospheric Dataset in Spherefun

**Original:** [sphere/AtmosphericTemperature](https://www.chebfun.org/examples/sphere/AtmosphericTemperature.html)
**Author(s):** Alex Townsend and Grady Wright, May 2016

---

## Introduction

Spherefun is a part of Chebfun for computing with functions on the sphere.
The underlying approximation scheme is based on representing functions on
the sphere by certain structure-preserving low rank approximants [1].
Mathematically, most functions are of infinite rank on the sphere, with a
notable exception being spherical harmonic functions. However, many
functions are numerically of low rank, and when a function is of low rank,
Spherefun is surprisingly efficient.

## Is surface air temperature of low rank?

We apply Spherefun to a global atmospheric temperature dataset on a
$529\times1024$ latitude-longitude grid containing the temperature (in
Kelvin) of Earth on 12 July 2005, obtained from the National Oceanic and
Atmospheric Administration (NOAA) Earth System Research Laboratory.

Spherefun computes the numerical rank of the temperature field as
approximately 185. Since the dataset has dimensions $529\times1024$, the
low rank representation achieves some useful compression, although the
results are not as dramatic as one often sees for smooth functions [2].

## Investigating the temperature

After converting to Celsius by subtracting 273.15, we can ask several
questions:

- **Mean temperature**: computed by `mean2(f)`.
- **Pole temperatures**: the data was taken during Northern hemisphere
  summer, so the North Pole is warmer than the South Pole.
- **Equatorial temperature**: plotted as a function of longitude.
- **Isolines**: contour lines at 5-degree intervals reveal the global
  temperature structure.
- **Zonal mean**: the average temperature at each latitude, plotted as a
  function of co-latitude $\theta$.

## Poisson solver

The steady heat profile with an external source (and no internal sinks or
sources) can be computed by solving Poisson's equation on the sphere.
The solution only makes sense if the right-hand side has mean zero.

## Scale-space selection using Gaussian filtering

It is common to smooth data by applying a Gaussian filter, which provides
a means of analysing data at various scales without introducing artificial
structures [3]. The `smooth` command in Spherefun uses a Gaussian filter
with parameter $\sigma$ (measured in radians at the equator). For
example, smoothing at scales of 2, 10, and 20 degrees reveals features
that are robust over multiple scales.

## References

1. A. Townsend, H. Wilber, and G. B. Wright, Computing with functions
   in spherical and polar geometries I. The sphere, _SIAM J. Sci. Comp._,
   2016.

2. A. Townsend, _Computing with Functions in Two Dimensions_, PhD Thesis,
   University of Oxford, 2014.

3. K. Marvel, D. Ivanova, and K. E. Taylor, Scale space methods for
   climate model analysis, _J. Geophys. Res. Atmospheres_, 118,
   5082--5097, 2013.

4. M. K. Chung, K. M. Dalton, and R. J. Davidson, Tensor-based cortical
   surface morphometry via weighted spherical harmonic representation,
   _IEEE Trans. On Medical Imag._, 27, 1143--1151, 2008.


![Atmospheric Temperature](../../images/sphere/atmospheric_temperature.png)

## Code

```python
from examples.sphere.atmospheric_temperature import run
run()
```
