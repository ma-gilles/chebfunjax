# Chapter 17: Spherefun

*Based on [Chebfun Guide Chapter 17](https://www.chebfun.org/docs/guide/guide17.html)*

Spherefun is the chebfunjax module for computing with functions on the surface of the unit sphere $S^2 = \{(x,y,z) : x^2 + y^2 + z^2 = 1\}$. It uses a low-rank Fourier-Fourier representation with the BMC-I (block mirror-centrosymmetric) structure to maintain spectral accuracy while handling the coordinate singularities at the poles.

## 17.1 Introduction

A `Spherefun` approximates a function $f(\lambda, \theta)$ on the unit sphere, where $\lambda \in [-\pi, \pi]$ is the longitude (azimuth) and $\theta \in [0, \pi]$ is the colatitude (polar angle, measured from the north pole). The Cartesian coordinates are:

$$x = \cos\lambda\, \sin\theta, \quad y = \sin\lambda\, \sin\theta, \quad z = \cos\theta$$

```python
import jax.numpy as jnp
import numpy as np
from chebfunjax.spherefun import Spherefun

# A spherical harmonic Y_1^0 = cos(theta)
f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
print(f)  # Spherefun(rank=..., n_plus=..., n_minus=...)
```

### Cartesian functions on the sphere

To approximate a function given in Cartesian form, convert coordinates inside the lambda:

```python
# f(x, y, z) = x*y on the sphere
g = Spherefun.from_function(
    lambda lam, theta: jnp.cos(lam) * jnp.sin(theta) * jnp.sin(lam) * jnp.sin(theta))
```

## 17.2 Evaluation

A `Spherefun` is callable with arguments $(\lambda, \theta)$:

```python
# Evaluate at the equator (theta = pi/2) at longitude lambda = 0
val = f(0.0, jnp.pi / 2)
print(val)  # cos(pi/2) ~ 0

# Evaluate at the north pole
print(f(0.0, 0.0))  # cos(0) = 1

# Vectorized evaluation
lams = jnp.linspace(-jnp.pi, jnp.pi, 50)
thetas = jnp.full(50, jnp.pi / 2)
vals = f(lams, thetas)
```

Evaluation is JIT-compiled, vmap-safe, and grad-safe.

## 17.3 Integration

The `sum()` method computes the surface integral with the standard area element $\sin\theta\, d\theta\, d\lambda$:

$$\texttt{f.sum()} = \int_{-\pi}^{\pi} \int_0^{\pi} f(\lambda, \theta)\, \sin\theta\, d\theta\, d\lambda$$

```python
# Integral of 1 over the sphere = 4*pi
one = Spherefun.from_function(lambda lam, theta: jnp.ones_like(theta))
print(one.sum())  # ~ 12.566370614... = 4*pi

# Integral of cos(theta) over the sphere = 0 (odd symmetry)
f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
print(f.sum())  # ~ 0
```

Only the "plus" terms (in the BMC-I decomposition) contribute to the integral. For each plus term:

$$\int\!\!\int f\, \sin\theta\, d\theta\, d\lambda = \sum_{j \in \text{plus}} \frac{1}{d_j} \left(\int_0^{\pi} c_j(\theta)\, \sin\theta\, d\theta\right) \left(\int_{-\pi}^{\pi} \text{row}_j(\lambda)\, d\lambda\right)$$

## 17.4 The BMC-I Structure

The key idea behind `Spherefun` is the *doubled Fourier sphere* (DFS) method. A function $f(\lambda, \theta)$ on the sphere is extended to a periodic function on $[-\pi, \pi]^2$ by exploiting the identity:

$$f(\lambda + \pi, \pi - \theta) = f(\lambda, \theta)$$

This doubled function has BMC-I structure, which splits into plus and minus components under the $\pi$-shift in longitude:

$$F_+(\lambda, \theta) = \tfrac{1}{2}[f(\lambda + \pi, \theta) + f(\lambda, \theta)]$$
$$F_-(\lambda, \theta) = \tfrac{1}{2}[f(\lambda + \pi, \theta) - f(\lambda, \theta)]$$

The construction uses GE with 2x2 block pivoting on these components, with automatic pole removal at $\theta = 0$ and $\theta = \pi$.

## 17.5 Low-Rank Representation

The internal representation is:

$$f(\lambda, \theta) \approx \sum_j \frac{1}{d_j}\, c_j(\theta)\, \text{row}_j(\lambda)$$

where:
- $c_j(\theta)$ are `Trigtech` objects (trigonometric polynomials on the doubled domain $[-\pi, \pi]$)
- $\text{row}_j(\lambda)$ are `Trigtech` objects (trigonometric polynomials on $[-\pi, \pi]$)
- $d_j$ are scalar pivot values

```python
# Low-rank function
f = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
print(f.rank)  # very small rank

# Higher-rank function
g = Spherefun.from_function(
    lambda lam, theta: jnp.exp(-10 * (jnp.cos(lam) * jnp.sin(theta))**2))
print(g.rank)
```

## 17.6 Spherical Harmonics

The spherical harmonics $Y_l^m(\lambda, \theta)$ are the eigenfunctions of the Laplace-Beltrami operator on the sphere. They can be constructed as `Spherefun` objects:

```python
from scipy.special import sph_harm

# Y_2^1 spherical harmonic
# Note: scipy uses (m, l, phi, theta) with phi=longitude, theta=colatitude
l, m = 2, 1
Y21 = Spherefun.from_function(
    lambda lam, theta: jnp.real(
        sph_harm(m, l, lam, theta)))
print(Y21.rank)
```

These satisfy $\Delta_S Y_l^m = -l(l+1) Y_l^m$ where $\Delta_S$ is the Laplace-Beltrami operator.

## 17.7 Vector-Valued Functions: Spherefunv

The `Spherefunv` class represents 2-component vector fields on the sphere:

```python
from chebfunjax.spherefun import Spherefunv

# Two components
f_comp = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
g_comp = Spherefun.from_function(lambda lam, theta: jnp.sin(lam))
F = Spherefunv(f_comp, g_comp)
print(F)
```

`Spherefunv` supports:
- Evaluation: `F(lam, theta)` returns a tuple `(f_val, g_val)`
- Dot product: `F.dot(G)` returns a `Spherefun`
- Norm: `F.norm()` returns a `Spherefun`
- Arithmetic: `F + G`, `F - G`, `c * F`

## 17.8 Construction Details

The `Spherefun.from_function` constructor follows the same two-phase algorithm as `Diskfun`, adapted for spherical geometry:

1. **Phase 1**: Sample on a doubled-up colatitude-longitude grid, split into BMC-I plus/minus blocks, and perform GE with block pivoting. The pole rows ($\theta = 0$ and $\theta = \pi$) are removed before rank determination and handled via a separate pole-removal step.

2. **Phase 2**: Evaluate skeleton slices at increasing resolution until Fourier coefficients decay below tolerance.

The algorithm avoids the artificial oversampling near the poles that plagues naive tensor-product methods on the sphere.

## 17.9 Coordinate Conventions

The chebfunjax `Spherefun` follows the MATLAB Chebfun convention:
- $\lambda$ (first argument) is the longitude in $[-\pi, \pi]$
- $\theta$ (second argument) is the colatitude in $[0, \pi]$, measured from the north pole

This matches the standard physics convention. Note that some references use $\phi$ for longitude and $\theta$ for colatitude, while others reverse them.

## 17.10 References

1. A. Townsend, H. Wilber, and G. Wright, "Computing with functions on spherical and polar geometries I: The sphere", *SIAM J. Sci. Comput.*, 38(4), C403--C425, 2016.

2. H. Wilber, "Computing numerically with functions on the sphere and disk", PhD thesis, Boise State University, 2016.

3. A. Townsend and L. N. Trefethen, "An extension of Chebfun to two dimensions", *SIAM J. Sci. Comput.*, 35(6), C495--C518, 2013.
