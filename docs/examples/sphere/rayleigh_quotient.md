# Rayleigh Quotient and the Maximum Principle for Eigenvalues

**Original:** [sphere/RayleighQuotientExample](https://www.chebfun.org/examples/sphere/RayleighQuotientExample.html)
**Author(s):** Grady Wright, February 2017

---

## Introduction

The Rayleigh quotient plays a key role in the study of eigenvalues of
symmetric matrices. If $A$ is a real $n\times n$ symmetric matrix, the
Rayleigh quotient is

$$
r(\mathbf{x}) = \frac{\mathbf{x}^T A\,\mathbf{x}}{\mathbf{x}^T\mathbf{x}},
$$

for any $n$-dimensional real vector $\mathbf{x}\neq 0$. A key property
is that if $\mathbf{x}$ is an eigenvector of $A$, then $r(\mathbf{x})$
gives the corresponding eigenvalue.

## The restricted Rayleigh quotient on the 2-sphere

Restricting to unit vectors $\|\mathbf{x}\|=1$, the Rayleigh quotient
simplifies to

$$
q(\mathbf{x}) = \mathbf{x}^T A\,\mathbf{x}.
$$

For a $3\times3$ symmetric matrix, this is a scalar function defined on
the 2-sphere $S^2$, which Spherefun can compute with and visualize.

## Maximum principle

The following theorem tells us that the eigenvalues of $A$ are given by
the maximum value of $q$ on certain subspaces of the sphere [2]:

**Theorem** (Maximum principle). The largest eigenvalue $\lambda_1$ of $A$
is

$$
\lambda_1 = \max_{\|\mathbf{x}\|=1} q(\mathbf{x}),
$$

and the maximizer $\mathbf{x}_1$ is the corresponding eigenvector. The
remaining eigenvalues $\lambda_2 \geq \lambda_3$ are given by

$$
\lambda_k = \max_{\|\mathbf{x}\|=1} \bigl\{q(\mathbf{x}) \;\big|\;
\langle\mathbf{x},\mathbf{x}_j\rangle = 0,\; j=1,\ldots,k-1\bigr\}.
$$

## Demonstration on the 2-sphere

The largest eigenvalue $\lambda_1$ is found using Spherefun's `max2`
command, and the result agrees with NumPy's `eig` to machine precision.

The next two eigenvalues lie on the great circle formed by the plane
normal to $\mathbf{x}_1$ passing through the origin. By restricting $q$
to this great circle (a 1D chebfun in the angle parameter), we find
$\lambda_2$ as the maximum, and $\lambda_3$ at a shift of $\pi/2$.

## Eigenvalues and the vanishing gradient

Another property of the restricted Rayleigh quotient is that eigenvalues
occur where the surface gradient of $q$ vanishes [2]. The zero-level
curves of the three components of $\nabla q$ all pass through the
eigenvector locations.

## References

1. J. P. Keener, _Principles of Applied Mathematics: Transformation and
   Approximation_, Westview Press, 2000.

2. L. N. Trefethen and D. Bau, III, _Numerical Linear Algebra_, SIAM,
   1997.

## Code

```python
from examples.sphere.rayleigh_quotient import run
run()
```

## Output

![Rayleigh Quotient on the Sphere](../../images/sphere/rayleigh_quotient.png)
