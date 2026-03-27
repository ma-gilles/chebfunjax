# Chapter 13: Chebfun2: Integration and Differentiation

*Based on [Chebfun Guide Chapter 13](https://www.chebfun.org/docs/guide/guide13.html)*

This chapter covers integration and differentiation operations for `Chebfun2`, including double integrals, partial integrals, norms, and partial derivatives.

## 13.1 Double Integration with sum2

The `sum2()` method computes the definite double integral of a `Chebfun2` over its domain:

$$\texttt{f.sum2()} = \int_{x_a}^{x_b} \int_{y_a}^{y_b} f(x, y)\, dy\, dx$$

```python
import jax.numpy as jnp
from chebfunjax.chebfun2d import chebfun2

f = chebfun2(lambda x, y: jnp.sin(10 * x * y))
print(f.sum2())
```

The computation exploits the low-rank structure. If $f \approx \sum_j d_j\, c_j(y)\, r_j(x)$, then

$$\int\!\!\int f\, dy\, dx = \sum_j d_j \left(\int c_j(y)\, dy\right) \left(\int r_j(x)\, dx\right)$$

Each 1D integral is computed exactly from the Chebyshev coefficients, so the result is spectrally accurate.

## 13.2 Partial Integration with sum

The `sum(dim=...)` method integrates over a single variable:

- `f.sum(dim=1)` integrates over $y$, returning a function of $x$ only.
- `f.sum(dim=2)` integrates over $x$, returning a function of $y$ only.

```python
f = chebfun2(lambda x, y: jnp.sin(10 * x * y))

# Integrate over y: g(x) = integral_{-1}^{1} f(x,y) dy
g = f.sum(dim=1)

# g is still a Chebfun2 object, but with collapsed columns
# Evaluate g at a few x-values
xs = jnp.array([0.0, 0.5, 1.0])
print(g(xs, jnp.zeros_like(xs)))  # y-argument is ignored
```

Computing `f.sum(dim=1).sum(dim=2)` is equivalent to `f.sum2()`:

```python
s1 = f.sum2()
s2 = f.sum(dim=1).sum2()
print(jnp.abs(s1 - s2))  # should be near machine epsilon
```

Calling `f.sum()` with no arguments is equivalent to `f.sum2()`.

## 13.3 Norm and Mean

### L2 norm

The `norm()` method computes the Frobenius (L2) norm:

$$\|f\|_2 = \sqrt{\int_{x_a}^{x_b} \int_{y_a}^{y_b} |f(x,y)|^2\, dy\, dx}$$

```python
f = chebfun2(lambda x, y: jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y))
print(f.norm())  # L2 norm
```

The computation uses the low-rank structure directly:

$$\|f\|_2^2 = \sum_{j,k} d_j d_k \langle c_j, c_k \rangle \langle r_j, r_k \rangle$$

where $\langle \cdot, \cdot \rangle$ denotes the L2 inner product on the physical interval.

### Mean value

The mean value of $f$ over the domain can be computed from `sum2()` divided by the area:

```python
f = chebfun2(lambda x, y: jnp.exp(x + y), domain=(0.0, 1.0, 0.0, 1.0))
area = 1.0  # (1-0) * (1-0)
mean_val = f.sum2() / area
print(mean_val)  # mean of exp(x+y) on [0,1]^2
```

## 13.4 Partial Differentiation

The `diff(dim, k)` method computes partial derivatives:

- `f.diff(dim=1)` computes $\partial f / \partial y$
- `f.diff(dim=2)` computes $\partial f / \partial x$
- `f.diff(dim=1, k=2)` computes $\partial^2 f / \partial y^2$

```python
f = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))

# Partial derivatives
fx = f.diff(dim=2)  # df/dx = cos(x)*cos(y)
fy = f.diff(dim=1)  # df/dy = -sin(x)*sin(y)

# Verify at a point
x0, y0 = 0.5, 0.3
print(f"fx({x0},{y0}) = {fx(x0, y0):.15f}")
print(f"exact       = {jnp.cos(x0) * jnp.cos(y0):.15f}")
```

### Laplacian

The Laplacian $\nabla^2 f = f_{xx} + f_{yy}$ can be computed by summing the two second derivatives:

```python
f = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))
fxx = f.diff(dim=2, k=2)
fyy = f.diff(dim=1, k=2)

# Laplacian at a point: should be -2*sin(x)*cos(y)
x0, y0 = 0.5, 0.3
lap_val = fxx(x0, y0) + fyy(x0, y0)
exact = -2.0 * jnp.sin(x0) * jnp.cos(y0)
print(f"Laplacian = {lap_val:.15f}")
print(f"exact     = {exact:.15f}")
```

### Cauchy-Riemann verification

For an analytic function $f(z) = u(x,y) + iv(x,y)$, the Cauchy-Riemann equations state $u_x = v_y$ and $u_y = -v_x$. We can verify this:

```python
# f(z) = exp(z) = exp(x)*cos(y) + i*exp(x)*sin(y)
u = chebfun2(lambda x, y: jnp.exp(x) * jnp.cos(y))
v = chebfun2(lambda x, y: jnp.exp(x) * jnp.sin(y))

ux = u.diff(dim=2)
vy = v.diff(dim=1)
uy = u.diff(dim=1)
vx = v.diff(dim=2)

# Check Cauchy-Riemann at (0.5, 0.3)
x0, y0 = 0.5, 0.3
print(f"u_x = {ux(x0, y0):.15f},  v_y = {vy(x0, y0):.15f}")
print(f"u_y = {uy(x0, y0):.15f}, -v_x = {-vx(x0, y0):.15f}")
```

## 13.5 How Differentiation Works

Each `Chebfun2` stores its approximation as

$$f(x,y) \approx \sum_j d_j\, c_j(y)\, r_j(x)$$

When we differentiate with respect to $y$ (dim=1), we differentiate each column slice:

$$\frac{\partial f}{\partial y} \approx \sum_j d_j\, c_j'(y)\, r_j(x)$$

The chain rule from the affine map $y = \frac{y_b - y_a}{2} t + \frac{y_a + y_b}{2}$ introduces a scale factor $\frac{2}{y_b - y_a}$ per derivative.

Similarly, differentiation with respect to $x$ (dim=2) acts on the row slices while leaving the column slices unchanged.

## 13.6 Integration via the Low-Rank Structure

The key computational advantage of the `Chebfun2` representation is that integrals and derivatives decompose into independent 1D operations. For a rank-$k$ function on a rectangle:

- **Double integral** costs $O(k)$ 1D integrations (two per rank-1 term).
- **Partial derivative** costs $O(k)$ 1D differentiations.
- **Norm** costs $O(k^2)$ inner products.

This is much cheaper than operating on a full grid, especially when $k$ is small.

## 13.7 References

1. A. Townsend and L. N. Trefethen, "An extension of Chebfun to two dimensions", *SIAM J. Sci. Comput.*, 35(6), C495--C518, 2013.

2. O. A. Carvajal, F. W. Chapman, and K. O. Geddes, "Hybrid symbolic-numeric integration in multiple dimensions via tensor-product series", *Proc. ISSAC*, 2005.
