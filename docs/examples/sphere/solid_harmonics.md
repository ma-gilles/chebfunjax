# Solid Harmonics

**Original:** [sphere/SolidHarmonics](https://www.chebfun.org/examples/sphere/SolidHarmonics.html)
**Author(s):** Nicolas Boulle and Alex Townsend, May 2019

---

## Introduction

Solid harmonics are solutions of the Laplace equation in spherical
coordinates:

$$
\nabla^2\phi = \frac{1}{r^2}\left[\frac{\partial}{\partial r}
\left(r^2\frac{\partial\phi}{\partial r}\right)
+ \frac{1}{\sin\theta}\frac{\partial}{\partial\theta}
\left(\sin\theta\frac{\partial\phi}{\partial\theta}\right)
+ \frac{1}{\sin^2\theta}\frac{\partial^2\phi}{\partial\lambda^2}
\right] = 0.
$$

This relationship holds because the spherical harmonics $Y_l^m$ are
eigenfunctions of the surface Laplace (Laplace-Beltrami) operator:

$$
\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}
\left(\sin\theta\frac{\partial Y_l^m}{\partial\theta}\right)
+ \frac{1}{\sin^2\theta}\frac{\partial^2 Y_l^m}{\partial\lambda^2}
= -l(l+1)Y_l^m.
$$

Substituting $\phi = F(r)Y_l^m$ into the Laplace equation gives
$F(r) = Ar^l$ or $F(r) = Br^{-(l+1)}$.

## Regular and irregular solid harmonics

The **regular solid harmonics** are

$$
R_l^m(r,\lambda,\theta) = a_{lm}\,r^l\,Y_l^m(\lambda,\theta),
$$

which vanish at the origin, and the **irregular solid harmonics** are

$$
I_l^m(r,\lambda,\theta) = a_{lm}\,\frac{Y_l^m(\lambda,\theta)}{r^{l+1}},
$$

which have a singularity at the origin. In this example, "solid harmonics"
refers to the regular ones. They are normalized so that their 2-norm over
the ball is 1:

$$
\int_B |R_l^m|^2\,dV = 1,
$$

which gives $a_{lm} = \sqrt{2l+3}$.

## Solid harmonics in Ballfun

Solid harmonics can be constructed using the `solharm` command. For
example, $R_4^2$ can be constructed and verified to be an eigenfunction
of the Laplace operator with zero eigenvalue.

The solid harmonics are orthonormal on the ball with respect to the
standard $L^2$ inner product:

$$
\int_B R_l^m\,R_{l'}^{m'}\,dV = \delta_{ll'}\delta_{mm'}.
$$

## Computing solid harmonic coefficients

A fast and stable algorithm for computing the solid harmonics is
implemented in Ballfun, requiring $\mathcal{O}(l\log l)$ operations for
degree $l$. The algorithm uses the Modified Forward Column (MFC) method
[3] to compute the Fourier coefficients of the associated Legendre
polynomial $P_l^m$, avoiding the overflow that occurs in the standard
Forward Column recursion for large degrees $l > 1900$ [2].

## References

1. O. L. Colombo, Numerical methods for harmonic analysis on the sphere,
   report DGS-310, Ohio State University, 1981.

2. D. M. Gleason, Partial sums of Legendre series via Clenshaw summation,
   _Manuscr. Geod._, 10 (1985), pp. 115--130.

3. S. A. Holmes and W. E. Featherstone, A unified approach to the
   Clenshaw summation and the recursive computation of very high degree
   and order normalised associated Legendre functions, _Journal of
   Geodesy_, 76 (2002), pp. 279--299.

## Code

```python
from examples.sphere.solid_harmonics import run
run()
```

## Output

![Solid Harmonics](../../images/sphere/solid_harmonics.png)
