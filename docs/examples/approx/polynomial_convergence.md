# Chebyshev Interpolation of Oscillatory Entire Functions

*Mark Richardson, October 2011*

*Original: [chebfun.org/examples/approx/Entire](https://www.chebfun.org/examples/approx/Entire.html)*

---

In this example we explore the approximation properties of Chebyshev
interpolation for entire functions — that is, functions that are analytic
everywhere in the complex plane.

## Analytic functions and the Bernstein ellipse

The key concept is the **Bernstein $r$-ellipse**: the image of the circle
$|z|=r$ under the mapping $x=(z+z^{-1})/2$. If a function $f$ is analytic
on $[-1,1]$ and analytically continuable to the closed $r$-ellipse $E_r$
for some $r>1$, then the $\infty$-norm error from polynomial interpolation
in $n+1$ Chebyshev points satisfies

$$\|f - p_n\|_\infty \leq \frac{4M}{r^n(r-1)},$$

where $M = \max_{z\in E_r} |f(z)|$.
This gives **geometric (exponential) convergence** in $n$.

## Oscillatory entire functions

When $f$ is entire — analytic everywhere — the convergence must hold for
*every* $r>1$, giving a rate faster than geometric. For example, the function
$f(x) = \sin(N\pi x)$ is entire, and as $N$ grows we need more Chebyshev
coefficients to resolve the oscillations.

In chebfunjax, the number of coefficients needed grows linearly with $N$:

```python
import chebfunjax as cj
import jax.numpy as jnp

for N in [1, 2, 4, 8, 16, 32]:
    f = cj.chebfun(lambda x, N=N: jnp.sin(N * jnp.pi * x))
    print(f"N={N:3d}: degree = {len(f)-1}")
```

```
N=  1: degree =   3
N=  2: degree =   5
N=  4: degree =  11
N=  8: degree =  23
N= 16: degree =  47
N= 32: degree =  95
```

The degree grows roughly as $N\pi / \log\rho$ for the optimal $\rho$.

![Degree vs N and coefficient decay](../../../images/approx/polynomial_convergence.png)

## Coefficient decay: analytic vs non-analytic

For an analytic function like $e^x$, the Chebyshev coefficients decay
geometrically (the left panel above). For the non-analytic function $|x|$,
convergence is only algebraic — the coefficients decay as $k^{-2}$,
reflecting the corner at the origin (right panel).

```python
f_exp = cj.chebfun(lambda x: jnp.exp(x))
print(f"exp(x): {len(f_exp)} coefficients")

f_abs = cj.chebfun(lambda x: jnp.abs(x), n=256)
print(f"|x|:    {len(f_abs)} coefficients")
```

The message is clear: **smooth, analytic functions are much cheaper to
approximate** than functions with corners or kinks.

## References

1. L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2013.
