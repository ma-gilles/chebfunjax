# Solving the Heat Equation on the Unit Sphere

**Original:** [sphere/SphereHeatConduction](https://www.chebfun.org/examples/sphere/SphereHeatConduction.html)
**Author(s):** Alex Townsend and Grady Wright, May 2016

---

## Introduction

Spherefun has about 100 commands for computing with scalar- and
vector-valued functions [1]. There is also functionality for solving
partial differential equations with the `poisson` and `helmholtz`
commands. In this example, we show how the latter can be used to solve the
heat equation on the sphere using an implicit time-stepping scheme.

## The heat equation on the sphere

The heat equation on the sphere is

$$
u_t = \alpha\nabla^2 u,
$$

where $\nabla^2$ is the surface Laplacian (Laplace-Beltrami operator) and
$\alpha > 0$ is the coefficient of thermal diffusivity. The initial
condition is $u(\lambda,\theta,0) = u_0(\lambda,\theta)$, where
$-\pi\leq\lambda\leq\pi$ is longitude and $0\leq\theta\leq\pi$ is
co-latitude.

## Implicit BDF2 time discretization

We discretize using the second-order backward differentiation formula
(BDF2). Replacing $u_t$ by $(3u_{n+1} - 4u_n + u_{n-1})/(2\Delta t)$
and rearranging gives a Helmholtz equation at each time step:

$$
\nabla^2 u_{n+1} + K^2 u_{n+1} = \frac{K^2}{3}(4u_n - u_{n-1}),
$$

where $K^2 = -3/(2\Delta t\,\alpha)$. This can be solved efficiently
with Spherefun's `helmholtz` command [2].

## Example with an analytic solution

The initial condition is the "soccer ball" function, a sum of spherical
harmonics:

$$
u_0 = Y_6^0 + \sqrt{\tfrac{14}{11}}\,Y_6^5.
$$

Since $Y_l^m$ is an eigenfunction of the surface Laplacian with eigenvalue
$-l(l+1)$, the exact solution is

$$
u(\lambda,\theta,t) = e^{-42\alpha t}\,u_0(\lambda,\theta),
$$

which provides a benchmark for the numerical scheme. With
$\alpha = 1/42$ and $\Delta t = 0.01$, the BDF2 method (bootstrapped by
one step of backward Euler) computes the solution to time $t = 1$ with
the temporal error dominating over spatial discretization error.

## A more complicated example

For an initial condition with no closed-form solution -- a sum of five
Gaussian bumps placed at random locations on the sphere -- the heat
equation smooths the bumps and they spread and merge over time. Since the
sphere has no boundary, the total amount of heat is conserved: the mean of
the solution at any time equals the mean of the initial condition.

A contour at the mean value $\overline{u}_0$ can be tracked through the
evolution, confirming that the numerical scheme preserves the mean value
property to machine precision.

## References

1. A. Townsend, H. Wilber, and G. B. Wright, Computing with functions
   in polar and spherical geometries I. The sphere, _SIAM J. Sci. Comp._,
   2016.

2. A. Townsend and G. B. Wright, Fast spectral methods for partial
   differential equations in spherical and polar geometries, manuscript
   in preparation, 2016.


![Solving the Heat Equation on the Unit Sphere](../../images/sphere/sphere_heat_conduction.png)

## Code

```python
from examples.sphere.sphere_heat_conduction import run
run()
```
