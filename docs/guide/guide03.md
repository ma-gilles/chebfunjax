# Chapter 3: Rootfinding and Minima and Maxima

*Based on [Chebfun Guide Chapter 3](https://www.chebfun.org/docs/guide/guide03.html) by Lloyd N. Trefethen*

## 3.1 The `roots` Method

Chebfunjax can find all the real zeros of a function on its domain.  The
`roots` method returns a sorted JAX array of all roots:

```python
import jax.numpy as jnp
import chebfunjax as cj

x = cj.chebfun(lambda x: x)
f = x**3 + x**2 - x
r = f.roots()
print(r)
# [0.         0.61803399]   (plus root at x = -1.618... outside domain?)
```

### Algorithm

The rootfinding algorithm is based on the Boyd-Battles method.  For each smooth
piece of the chebfun, the zeros of the underlying Chebyshev polynomial are
computed as eigenvalues of the *colleague matrix* -- the Chebyshev analogue of
the companion matrix for monomials.

For high-degree polynomials (above about 50), the algorithm recursively
subdivides the interval and finds roots on smaller pieces, then combines them.
This keeps the eigenvalue problems small and numerically stable.

### Example: zeros of sin(kx)

```python
# Zeros of sin(10*x) on [0, 10]
f = cj.chebfun(lambda x: jnp.sin(10 * x), domain=[0, 10])
r = f.roots()
print(f"Number of roots: {len(r)}")
print(r[:5])  # first five roots: approximately 0, pi/10, 2*pi/10, ...
```

Since $\sin(10x) = 0$ when $10x = k\pi$, i.e., $x = k\pi/10$, there should
be about $10 \cdot 10/\pi \approx 32$ roots in $[0, 10]$.

### Solving nonlinear equations

The `roots` method naturally solves nonlinear equations.  To find all solutions
of $\cos(x) = x$ on $[-5, 5]$:

```python
g = cj.chebfun(lambda x: jnp.cos(x) - x, domain=[-5, 5])
solutions = g.roots()
print(solutions)
# Should find the single solution near x = 0.739...
```

In general, to solve $f(x) = g(x)$, construct the chebfun $h = f - g$ and
call `h.roots()`.

## 3.2 Operations That Introduce Breakpoints

Several operations take smooth chebfuns as input and produce piecewise-smooth
chebfuns as output.  They do this by using `roots` internally to locate the
points where smoothness breaks down, then inserting those points as breakpoints.

### abs

The absolute value $|f(x)|$ has a kink wherever $f(x) = 0$:

```python
x = cj.chebfun(lambda x: x)
g = x.abs()
print(g)
# Chebfun column (2 smooth pieces)
#        interval       length     endpoint values
# [      -1,       0]        2       1.00     0.00
# [       0,       1]        2       0.00     1.00
```

The single-piece chebfun representing $x$ has become a 2-piece chebfun
representing $|x|$, with a breakpoint at $x = 0$.

### sign

The sign function creates discontinuities:

```python
x = cj.chebfun(lambda x: x)
s = x.sign()
print(float(s(-0.5)))   # -1
print(float(s(0.5)))    #  1
```

### Combining abs with more complex functions

```python
x = cj.chebfun(lambda x: x, domain=[0, 2 * jnp.pi])
f = cj.sin(8 * x)
g = f.abs()
print(g)   # multiple pieces
```

## 3.3 Local Extrema

Local extrema of a function occur at the roots of its derivative (and
possibly at breakpoints or domain endpoints).  There are two ways to find them:

### Method 1: Roots of the derivative

```python
f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2), domain=[0, 10])
fp = f.diff()
critical_points = fp.roots()
print(f"Number of critical points: {len(critical_points)}")
```

To classify each critical point as a minimum or maximum, check the sign of
the second derivative:

```python
fpp = f.diff(2)
for cp in critical_points[:5]:
    cp_val = float(f(cp))
    fpp_val = float(fpp(cp))
    label = "min" if fpp_val > 0 else "max" if fpp_val < 0 else "inflection"
    print(f"  x = {float(cp):.6f},  f(x) = {cp_val:.6f},  type = {label}")
```

### Method 2: The minandmax method

The `minandmax` method finds the global minimum and maximum simultaneously:

```python
f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2), domain=[0, 15])
(x_min, f_min), (x_max, f_max) = f.minandmax()
print(f"Global min: f({x_min:.6f}) = {f_min:.6f}")
print(f"Global max: f({x_max:.6f}) = {f_max:.6f}")
```

The algorithm finds all roots of the derivative, evaluates the function there
and at the endpoints of each piece, and returns the overall min and max.

## 3.4 Global Extrema: min and max

The `min` and `max` methods are convenience wrappers around `minandmax`:

```python
f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(x**2), domain=[0, 15])

x_min, f_min = f.min()
x_max, f_max = f.max()

print(f"min: f({x_min:.6f}) = {f_min:.6f}")
print(f"max: f({x_max:.6f}) = {f_max:.6f}")
```

Both methods return a tuple `(x_opt, f_opt)` giving the location and value
of the extremum.

### Example: finding the minimum of a Runge-type function

```python
f = cj.chebfun(lambda x: 1.0 / (1 + 25 * x**2))
x_max, f_max = f.max()
print(f"Maximum of Runge function: f({x_max:.6f}) = {f_max:.6f}")
# Maximum is at x = 0, f(0) = 1
```

## 3.5 Norm Computations

The `norm` method leverages rootfinding internally for certain norm types:

### The 1-norm (L1 norm)

$$\|f\|_1 = \int_a^b |f(x)|\,dx$$

This is computed by first calling `abs()` (which uses `roots` to find sign
changes), then integrating:

```python
f = cj.chebfun(jnp.sin, domain=[0, 4 * jnp.pi])
print(f"||sin||_1 on [0, 4*pi] = {float(f.norm(1)):.6f}")
# Should be 8.0  (4 half-periods, each contributing 2)
```

### The infinity norm

$$\|f\|_\infty = \max_{x \in [a,b]} |f(x)|$$

This uses the extrema-finding machinery:

```python
f = cj.chebfun(jnp.sin, domain=[0, 4 * jnp.pi])
print(f"||sin||_inf on [0, 4*pi] = {float(f.norm(jnp.inf)):.6f}")
# Should be 1.0
```

### The 2-norm

$$\|f\|_2 = \sqrt{\int_a^b |f(x)|^2\,dx} = \sqrt{\langle f, f \rangle}$$

The 2-norm is computed via the inner product, which uses Clenshaw-Curtis
quadrature -- no rootfinding needed:

```python
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print(f"||sin||_2 on [0, pi] = {float(f.norm(2)):.6f}")
# sqrt(pi/2) = 1.2533...
```

## 3.6 Rootfinding for Piecewise Chebfuns

When a chebfun has multiple pieces, `roots()` finds the zeros of each piece
independently and combines the results.  Duplicate roots at breakpoints (which
might be found by two adjacent pieces) are automatically deduplicated:

```python
# A piecewise function with known roots
f = cj.chebfun(lambda x: jnp.sin(x), domain=[0, jnp.pi, 2 * jnp.pi])
r = f.roots()
print(r)
# Should find roots at 0, pi, and 2*pi
```

## 3.7 Finding Intersections

To find where two functions intersect, take their difference and find its
roots:

```python
f = cj.chebfun(jnp.sin, domain=[0, 10])
g = cj.chebfun(jnp.cos, domain=[0, 10])
h = f - g   # sin(x) - cos(x)
crossings = h.roots()
print(f"sin(x) = cos(x) at {len(crossings)} points in [0, 10]:")
for xc in crossings:
    print(f"  x = {float(xc):.6f},  f(x) = {float(f(xc)):.6f}")
```

The intersections of $\sin(x)$ and $\cos(x)$ occur at $x = \pi/4 + k\pi$.

## 3.8 Example: Optimization of a Multimodal Function

Here is a more substantial example -- finding the global minimum of a function
with many local minima:

```python
f = cj.chebfun(
    lambda x: jnp.sin(x) + jnp.sin(3 * x) / 3 + jnp.cos(7 * x) / 5,
    domain=[0, 20]
)

# All critical points
fp = f.diff()
crit = fp.roots()
print(f"Number of critical points: {len(crit)}")

# Global min and max
x_min, f_min = f.min()
x_max, f_max = f.max()
print(f"Global min: f({x_min:.6f}) = {f_min:.6f}")
print(f"Global max: f({x_max:.6f}) = {f_max:.6f}")
```

## 3.9 Summary

| Method | Description | Returns |
|---|---|---|
| `f.roots()` | All zeros in $[a, b]$ | `jax.Array` of root locations |
| `f.min()` | Global minimum | `(x_min, f_min)` |
| `f.max()` | Global maximum | `(x_max, f_max)` |
| `f.minandmax()` | Both min and max | `((x_min, f_min), (x_max, f_max))` |
| `f.abs()` | Absolute value (with breakpoints) | Piecewise `Chebfun` |
| `f.sign()` | Sign function (with breakpoints) | Piecewise `Chebfun` |
| `f.norm(p)` | $L^p$ norm | scalar |

### Key algorithmic ideas

1. **Colleague matrix eigenvalues**: Roots of Chebyshev polynomials are found
   via the eigenvalues of a companion-like matrix.
2. **Recursive subdivision**: High-degree polynomials are split into smaller
   intervals for stable rootfinding.
3. **Derivative-based optimization**: Extrema are found by rootfinding on the
   derivative, combined with endpoint evaluation.
4. **Automatic breakpoint insertion**: Operations like `abs` and `sign` use
   `roots` to locate sign changes and insert breakpoints for a smooth
   piecewise representation.

## 3.10 References

- J. P. Boyd, *Computing zeros on a real interval through Chebyshev expansion
  and polynomial rootfinding*, SIAM J. Numer. Anal. 40 (2002), 1666-1682.
- Z. Battles, *Numerical Linear Algebra for Continuous Functions*, D.Phil.
  thesis, Oxford, 2006.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013.
