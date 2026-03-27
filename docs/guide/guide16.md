# Chapter 16: Diskfun

*Based on [Chebfun Guide Chapter 16](https://www.chebfun.org/docs/guide/guide16.html)*

Diskfun is the chebfunjax module for computing with functions on the unit disk $\{(x,y) : x^2 + y^2 \le 1\}$. It represents functions in polar coordinates $(\theta, r)$ using a low-rank Chebyshev-Fourier expansion, maintaining spectral accuracy while handling the coordinate singularity at the origin.

## 16.1 Introduction

A `Diskfun` approximates a function $f(\theta, r)$ on the unit disk, where $\theta \in [-\pi, \pi]$ is the angle and $r \in [0, 1]$ is the radius. In Cartesian coordinates, the relationship is $x = r\cos\theta$, $y = r\sin\theta$.

```python
import jax.numpy as jnp
import numpy as np
from chebfunjax.diskfun import Diskfun

# A simple function on the disk (in polar coordinates)
f = Diskfun.from_function(lambda theta, r: r**2 * jnp.cos(2 * theta))
print(f)  # Diskfun(rank=..., n_plus=..., n_minus=...)
```

The `rank` property tells you the number of terms in the low-rank decomposition. The `n_plus` and `n_minus` counts reflect the BMC-II (block mirror-centrosymmetric) structure used internally.

### Cartesian functions

To work with functions defined in Cartesian coordinates, convert inside the lambda:

```python
# f(x, y) = x^2 - y^2 in polar coordinates: r^2*(cos^2(theta) - sin^2(theta)) = r^2*cos(2*theta)
f = Diskfun.from_function(lambda theta, r: r**2 * jnp.cos(2 * theta))
```

## 16.2 Evaluation

A `Diskfun` is callable with arguments $(\theta, r)$:

```python
# Evaluate at a single point
val = f(0.0, 0.5)
print(val)  # 0.25 * cos(0) = 0.25

# Evaluate at multiple points
thetas = jnp.linspace(0, 2 * jnp.pi, 20)
rs = jnp.full(20, 0.5)
vals = f(thetas, rs)
```

Evaluation is JIT-compiled and compatible with `jax.vmap` and `jax.grad`.

## 16.3 Integration

The `sum()` method computes the definite integral over the unit disk with the polar area element $r\, dr\, d\theta$:

$$\texttt{f.sum()} = \int_0^{2\pi} \int_0^1 f(\theta, r)\, r\, dr\, d\theta$$

```python
# Integral of 1 over the disk should be pi
one = Diskfun.from_function(lambda theta, r: jnp.ones_like(r))
print(one.sum())  # ~ 3.14159265...

# Integral of r^2 over the disk: integral r^2 * r dr dtheta = 2*pi * (1/4) = pi/2
f = Diskfun.from_function(lambda theta, r: r**2)
print(f.sum())  # ~ pi/2 = 1.5707963...
```

The integral exploits the low-rank structure: only the "plus" terms contribute (minus terms integrate to zero due to antisymmetry). For each plus term:

$$\int\!\!\int f\, r\, dr\, d\theta = \sum_{j \in \text{plus}} \frac{1}{d_j} \left(\int_0^1 c_j(r)\, r\, dr\right) \left(\int_{-\pi}^{\pi} \text{row}_j(\theta)\, d\theta\right)$$

## 16.4 The BMC-II Structure

The key algorithmic idea behind `Diskfun` is the *doubled-up* representation. A function $f(\theta, r)$ on $[-\pi, \pi] \times [0, 1]$ is extended to $[-\pi, \pi] \times [-1, 1]$ using the identity

$$f(\theta, -r) = f(\theta + \pi, r)$$

This extension has *block mirror-centrosymmetric* (BMC-II) structure, which means it splits into "plus" (even) and "minus" (odd) components under the $\pi$-shift in $\theta$:

$$F_+(\theta, r) = \tfrac{1}{2}[f(\theta + \pi, r) + f(\theta, r)]$$
$$F_-(\theta, r) = \tfrac{1}{2}[f(\theta + \pi, r) - f(\theta, r)]$$

The construction algorithm performs Gaussian elimination with 2x2 block pivoting on these components, yielding a low-rank decomposition where:

- Column slices $c_j(r)$ are `Chebtech2` objects (Chebyshev polynomials on $[-1, 1]$)
- Row slices $\text{row}_j(\theta)$ are `Trigtech` objects (trigonometric polynomials on $[-\pi, \pi]$)

## 16.5 Low-Rank Representation

The internal representation is:

$$f(\theta, r) \approx \sum_j \frac{1}{d_j}\, c_j(r)\, \text{row}_j(\theta)$$

The `rank` property gives the total number of terms:

```python
# Simple functions have low rank
f = Diskfun.from_function(lambda theta, r: r * jnp.cos(theta))
print(f.rank)  # small rank

# More complex functions need higher rank
g = Diskfun.from_function(
    lambda theta, r: jnp.exp(-10 * (r * jnp.cos(theta) - 0.3)**2
                               - 10 * (r * jnp.sin(theta) - 0.4)**2))
print(g.rank)
```

## 16.6 Cylindrical Harmonics

The cylindrical harmonics are the eigenfunctions of the Laplace operator on the disk. They are $J_m(j_{mn} r)\, e^{im\theta}$, where $J_m$ is the Bessel function of the first kind and $j_{mn}$ is its $n$-th zero. These can be approximated as Diskfun objects:

```python
from scipy.special import jn_zeros

# Cylindrical harmonic J_2(j_{2,1} * r) * cos(2*theta)
j21 = jn_zeros(2, 1)[0]  # first zero of J_2

from scipy.special import jv
harmonic = Diskfun.from_function(
    lambda theta, r: jv(2, j21 * r) * jnp.cos(2 * theta))
print(harmonic)
```

## 16.7 Vector-Valued Functions: Diskfunv

The `Diskfunv` class represents 2-component vector fields on the disk:

```python
from chebfunjax.diskfun import Diskfunv

# A vector field F = (f, g) on the disk
f_comp = Diskfun.from_function(lambda theta, r: -r * jnp.sin(theta))
g_comp = Diskfun.from_function(lambda theta, r: r * jnp.cos(theta))
F = Diskfunv(f_comp, g_comp)
print(F)
```

`Diskfunv` supports:
- Evaluation: `F(theta, r)` returns a tuple `(f_val, g_val)`
- Dot product: `F.dot(G)` returns a `Diskfun`
- Norm: `F.norm()` returns a `Diskfun` (the pointwise magnitude)
- Arithmetic: `F + G`, `F - G`, `c * F`

## 16.8 Construction Details

The `Diskfun.from_function` constructor follows a two-phase algorithm:

1. **Phase 1 (rank determination)**: Sample the function on a doubled-up Chebyshev-Fourier grid, split into plus/minus blocks, and perform GE with 2x2 block pivoting to identify pivot locations and estimate the rank.

2. **Phase 2 (slice resolution)**: Evaluate the function along the skeleton slices at increasing resolution until the Chebyshev coefficients (for radial slices) and Fourier coefficients (for angular slices) decay below tolerance.

The algorithm automatically handles the coordinate singularity at $r = 0$ (the origin) via a pole-removal step.

## 16.9 References

1. A. Townsend, H. Wilber, and G. Wright, "Computing with functions on spherical and polar geometries II: The disk", *SIAM J. Sci. Comput.*, 39(5), C238--C262, 2017.

2. H. Wilber, "Computing numerically with functions on the sphere and disk", PhD thesis, Boise State University, 2016.
